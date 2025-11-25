from gymnasium import spaces
import numpy as np
import warnings
import torch as th
import torch.nn.functional as F
import os
from typing import Any, ClassVar, Optional, TypeVar, Union

from rl_microservice_decomposer.configs.daytrader.config_daytrader_dqn import DaytraderConfig
from rl_microservice_decomposer.callbacks.atlas_callback import CustomCallback
from rl_microservice_decomposer.replay_buffer.prio_replay_buffer import PrioritizedReplayBuffer
from rl_microservice_decomposer.agent.dqn_embedder import MixedInputFeaturesExtractor

from stable_baselines3 import DQN

from gymnasium.wrappers import FlattenObservation
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import LinearSchedule, get_parameters_by_name, polyak_update
from stable_baselines3.dqn.policies import CnnPolicy, DQNPolicy, MlpPolicy, MultiInputPolicy, QNetwork
from stable_baselines3.common.noise import ActionNoise

SelfDQN = TypeVar("SelfDQN", bound="DQN")


class MaskedDQN(OffPolicyAlgorithm):
    policy_aliases: ClassVar[dict[str, type[BasePolicy]]] = {
        "MlpPolicy": MlpPolicy,
        "CnnPolicy": CnnPolicy,
        "MultiInputPolicy": MultiInputPolicy,
    }
    # Linear schedule will be defined in `_setup_model()`
    exploration_schedule: Schedule
    q_net: QNetwork
    q_net_target: QNetwork
    policy: DQNPolicy

    def __init__(
        self,
        policy: Union[str, type[DQNPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 1e-4,
        buffer_size: int = 1_000_000,  # 1e6
        learning_starts: int = 100,
        batch_size: int = 32,
        tau: float = 1.0,
        gamma: float = 0.99,
        train_freq: Union[int, tuple[int, str]] = 4,
        gradient_steps: int = 1,
        replay_buffer_class: Optional[type[PrioritizedReplayBuffer]] = None,
        replay_buffer_kwargs: Optional[dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        n_steps: int = 1,
        target_update_interval: int = 10000,
        exploration_fraction: float = 0.1,
        exploration_initial_eps: float = 1.0,
        exploration_final_eps: float = 0.05,
        max_grad_norm: float = 10,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ) -> None:
        super().__init__(
            policy,
            env,
            learning_rate,
            buffer_size,
            learning_starts,
            batch_size,
            tau,
            gamma,
            train_freq,
            gradient_steps,
            action_noise=None,  # No action noise
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            optimize_memory_usage=optimize_memory_usage,
            n_steps=n_steps,
            policy_kwargs=policy_kwargs,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            seed=seed,
            sde_support=False,
            supported_action_spaces=(spaces.Discrete,),
            support_multi_env=True,
        )

        self.exploration_initial_eps = exploration_initial_eps
        self.exploration_final_eps = exploration_final_eps
        self.exploration_fraction = exploration_fraction
        self.target_update_interval = target_update_interval
        
        # For updating the target network with multiple envs:
        self._n_calls = 0
        self.max_grad_norm = max_grad_norm
        
        # "epsilon" for the epsilon-greedy exploration
        self.exploration_rate = 0.0
        self.last_action_mask = None


        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super()._setup_model()
        self._create_aliases()
        # Copy running stats
        self.batch_norm_stats = get_parameters_by_name(self.q_net, ["running_"])
        self.batch_norm_stats_target = get_parameters_by_name(self.q_net_target, ["running_"])
        self.exploration_schedule = LinearSchedule(
            self.exploration_initial_eps,
            self.exploration_final_eps,
            self.exploration_fraction,
        )

        if self.n_envs > 1:
            if self.n_envs > self.target_update_interval:
                warnings.warn(
                    "The number of environments used is greater than the target network "
                    f"update interval ({self.n_envs} > {self.target_update_interval}), "
                    "therefore the target network will be updated after each call to env.step() "
                    f"which corresponds to {self.n_envs} steps."
                )

    def _create_aliases(self) -> None:
        self.q_net = self.policy.q_net
        self.q_net_target = self.policy.q_net_target

    def _on_step(self) -> None:
        """
        Update the exploration rate and target network if needed.
        This method is called in ``collect_rollouts()`` after each step in the environment.
        """
        self._n_calls += 1
        # Account for multiple environments
        # each call to step() corresponds to n_envs transitions
        if self._n_calls % max(self.target_update_interval // self.n_envs, 1) == 0:
            polyak_update(self.q_net.parameters(), self.q_net_target.parameters(), self.tau)
            # Copy running stats, see GH issue #996
            polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

        self.exploration_rate = self.exploration_schedule(self._current_progress_remaining)
        self.logger.record("rollout/exploration_rate", self.exploration_rate)

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update learning rate according to schedule
        self._update_learning_rate(self.policy.optimizer)

        losses = []
        for _ in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]
            # For n-step replay, discount factor is gamma**n_steps (when no early termination)
            discounts = replay_data.discounts if replay_data.discounts is not None else self.gamma

            with th.no_grad():
                # Compute the next Q-values using the target network
                next_q_values = self.q_net_target(replay_data.next_observations)
                # Follow greedy policy: use the one with the highest value
                next_q_values, _ = next_q_values.max(dim=1)
                # Avoid potential broadcast issue
                next_q_values = next_q_values.reshape(-1, 1)
                # 1-step TD target
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * discounts * next_q_values

            # Get current Q-values estimates
            current_q_values = self.q_net(replay_data.observations)

            # Retrieve the q-values for the actions from the replay buffer
            current_q_values = th.gather(current_q_values, dim=1, index=replay_data.actions.long())

            td_errors = (current_q_values - target_q_values).abs().detach().cpu().numpy().flatten()

            # Compute Huber loss (less sensitive to outliers)
            loss = F.smooth_l1_loss(current_q_values, target_q_values, reduction='none')
            weighted_loss = (loss.flatten() * replay_data.weights).mean()
            
            losses.append(weighted_loss.item())

            # Update priorities in replay buffer
            if hasattr(replay_data, 'indices'):
                self.replay_buffer.update_priorities(replay_data.indices, td_errors)

            # Optimize the policy
            self.policy.optimizer.zero_grad()
            weighted_loss.backward()
            # Clip gradient norm
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        # Increase update counter
        self._n_updates += gradient_steps
    
    def _get_current_action_mask_from_env(self):
        """Get action mask from current environment state"""
        try:
            # Access environment through VecEnv wrapper
            if hasattr(self.env, 'envs') and len(self.env.envs) > 0:
                env = self.env.envs[0]  # Get first environment
                
                # Navigate through wrappers to get your SequentialEnv
                while hasattr(env, 'env'):
                    env = env.env
                
                # Get action mask from your environment
                if hasattr(env, '_get_action_mask'):
                    mask = env._get_action_mask()    
                    return mask.astype(bool)
                elif hasattr(env, 'action_mask_buffer'):
                    mask = env.action_mask_buffer.astype(bool)
                    return env.action_mask_buffer.astype(bool) 

            return None
        except Exception as e:
            print(f"Could not get current action mask: {e}")
            return None

    def predict(self, observation, state=None, episode_start=None, deterministic=False):

        action_mask = self.last_action_mask
        if action_mask is None:
            action_mask = self._get_current_action_mask_from_env()
        # Handle exploration with action masking
        if not deterministic and np.random.rand() < self.exploration_rate:
            return self._sample_masked_action(action_mask), state
        
        # Greedy action with masking
        with th.no_grad():
            if len(observation.shape) == 1:
                observation = observation.reshape(1, -1)
            
            q_values = self.q_net(th.FloatTensor(observation).to(self.device))
            
            if action_mask is not None:
                q_values_masked = q_values.clone()
                mask_tensor = th.tensor(action_mask, device=q_values.device, dtype=th.bool)
                
                if len(mask_tensor.shape) == 1:
                    mask_tensor = mask_tensor.reshape(1, -1)
                
                q_values_masked[~mask_tensor] = -th.inf
                action = q_values_masked.argmax(dim=-1)
            else:
                action = q_values.argmax(dim=-1)
        
        return action.cpu().numpy(), state

    def _sample_action(
        self,
        learning_starts: int,
        action_noise: Optional[ActionNoise] = None,
        n_envs: int = 1,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Override to ensure action masking even during warmup phase"""
        
        # Always use our predict method, even during warmup
        unscaled_action, _ = self.predict(self._last_obs, deterministic=False)
        
        # Handle action noise if needed (probably not relevant for discrete actions)
        if isinstance(self.action_space, spaces.Box):
            scaled_action = self.policy.scale_action(unscaled_action)
            if action_noise is not None:
                scaled_action = np.clip(scaled_action + action_noise(), -1, 1)
            buffer_action = scaled_action
            action = self.policy.unscale_action(scaled_action)
        else:
            # Discrete case - no scaling needed
            buffer_action = unscaled_action
            action = buffer_action
        
        return action, buffer_action

    def _store_transition(self, replay_buffer, buffer_action, new_obs, reward, done, infos):
        # Extract and store action mask from info
        if isinstance(infos, list) and len(infos) > 0:
            info = infos[0]
        else:
            info = infos
            
        if 'action_mask' in info:
            self.last_action_mask = info['action_mask'].astype(bool)

        if done:
            self.last_action_mask = None
        
        super()._store_transition(replay_buffer, buffer_action, new_obs, reward, done, infos)

    def _sample_masked_action(self, action_mask):
        if action_mask is None:
            return np.array([self.action_space.sample()])
        
        valid_actions = np.where(action_mask)[0]
        
        if len(valid_actions) > 0:
            action = np.random.choice(valid_actions)
        else:
            action = 0  # Fallback
        return np.array([action])
    
    def learn(
        self: SelfDQN,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        tb_log_name: str = "DQN",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfDQN:
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )

    def _excluded_save_params(self) -> list[str]:
        return [*super()._excluded_save_params(), "q_net", "q_net_target"]

    def _get_torch_save_params(self) -> tuple[list[str], list[str]]:
        state_dicts = ["policy", "policy.optimizer"]

        return state_dicts, []

class DQNDecompositionAgent:
    def __init__(
        self,
        config: DaytraderConfig,
        adjacency_matrix: np.ndarray,
        env_factory,
        n_envs: int = 4,
        seed: int = 42,
        use_gpu: bool = True,
        normalize_env: bool = False,
        tensorboard_log: Optional[str] =None,
        model_save_path: str = "./models/",
        embedded_input: bool = False
    ):
        
        self.env_factory = env_factory
        self.n_envs = n_envs
        self.seed = seed
        self.normalize_env = normalize_env
        self.model_save_path = model_save_path
        self.tensorboard_log = tensorboard_log
        self.custom_callback = CustomCallback()
        self.config = config

        # Embedding-specific parameters
        self.n_classes = adjacency_matrix.shape[0]
        self.n_max_services = self.config.environment.max_microservices
        if hasattr(self.config.dqn_agent, 'embedding_dim'):
            self.embedding_dim = self.config.dqn_agent.embedding_dim
        
        # Create directories
        os.makedirs(model_save_path, exist_ok=True)
        
        # Setup device
        if use_gpu and th.cuda.is_available():
            self.device = th.device("cuda")
            print(f"Using GPU: {th.cuda.get_device_name()}")
        else:
            self.device = th.device("cpu")
            print("Using CPU")
        
        # Create environments
        self.env = self.env_factory()
        
        # Allows embedding of input features if specified
        if embedded_input:
            policy_kwargs = dict(
                net_arch=self.config.dqn_agent.architecture,
                activation_fn=self.config.dqn_agent.activation_fn,
                # Embedding-specific kwargs if embedding is enabled
                features_extractor_class=MixedInputFeaturesExtractor,
                features_extractor_kwargs={ 
                    "n_classes": self.n_classes,
                    "max_n_services": self.n_max_services,
                    "embedding_dim": self.embedding_dim,
                    "observation_components": self.config.environment.observation_components
                }
            )
        else:
            policy_kwargs = dict(
                net_arch=self.config.dqn_agent.architecture,
                activation_fn=self.config.dqn_agent.activation_fn,
            )
        
        # DQN model parameters
        self.dqn_params = {
            "learning_rate": self.config.dqn_agent.learning_rate,
            "buffer_size": self.config.dqn_agent.buffer_size,
            "learning_starts": self.config.dqn_agent.learning_starts,
            "batch_size": self.config.dqn_agent.batch_size,
            "tau": self.config.dqn_agent.tau,
            "gamma": self.config.dqn_agent.gamma,
            "train_freq": self.config.dqn_agent.train_freq,
            "gradient_steps": self.config.dqn_agent.gradient_steps,
            "replay_buffer_class": PrioritizedReplayBuffer,
            "replay_buffer_kwargs": {
                "alpha": self.config.dqn_agent.prioritized_replay_alpha,
                "beta": self.config.dqn_agent.prioritized_replay_beta,
                "eps": self.config.dqn_agent.prioritized_replay_eps,
                "handle_timeout_termination": False,
                "n_steps": self.config.dqn_agent.n_steps,
                "gamma": self.config.dqn_agent.gamma
            },
            # "optimize_memory_usage": True,
            "target_update_interval": self.config.dqn_agent.target_update_interval,
            "exploration_fraction": self.config.dqn_agent.exploration_fraction,
            "exploration_initial_eps": self.config.dqn_agent.exploration_initial_eps,
            "exploration_final_eps": self.config.dqn_agent.exploration_final_eps,
            "max_grad_norm": self.config.dqn_agent.grad_clip,
            "policy_kwargs": policy_kwargs,
            "verbose": 0
        }
        
        # Initialize model
        self.model = None
        self.is_trained = False

    def create_model(self):
        wrapped_env = FlattenObservation(self.env)
        self.model = MaskedDQN("MlpPolicy", wrapped_env, **self.dqn_params)

    def get_embedding_weights(self):
        """
        Get the learned embedding weights for analysis.
        Returns the embedding matrix of shape [n_services + 1, embedding_dim]
        """
        if self.model is None:
            raise ValueError("Model not created yet. Call create_model() first.")
        
        # Access the embedding layer through the policy's features extractor
        features_extractor = self.model.policy.features_extractor
        if hasattr(features_extractor, 'ms_embedding'):
            return features_extractor.ms_embedding.weight.data.cpu().numpy()
        else:
            raise ValueError("No embedding layer found in the model.")
    
    def train(self, total_timesteps: 1_000_000, progress_bar: bool = True):

        if self.model is None:
            self.create_model()    

        print(f"Starting training for {total_timesteps} timesteps...")
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=self.custom_callback,
            progress_bar=progress_bar
        )

        self.is_trained = True

        episode_data, training_metrics = self.custom_callback.get_training_metrics()

        results = {
            'mean_decomposition_quality_last_N': training_metrics['mean_decomposition_quality_last_N'],
            'best_decomposition_quality': training_metrics['best_decomposition_quality'],
            'std_decomposition_quality': np.std(training_metrics['decomposition_qualities']),
            'decomposition_qualities': training_metrics['decomposition_qualities'],
            'best_mean_decomposition_quality_N': training_metrics['best_mean_decomposition_quality_N'],
            'episode_data': episode_data
        }

        return results
    
    def close(self):
        """Close environments"""
        if self.env:
            self.env.close()
    