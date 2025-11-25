from typing import Dict, List
from dataclasses import dataclass, field
from torch import nn

from rl_microservice_decomposer.configs.utils.config_utils import BaseConfig

@dataclass    
class SeqEnvironmentConfig(BaseConfig):
    """Configuration for the decomposition environment"""
    initial_state: str = "all_together"  # "all_separate" or "all_together", what the decomposition looks like
    max_microservices: int = 90
    final_reward_scaling_factor: int= 100 # Need to scale reward to counter sparse rewards from modularity
    reward_type: str = 'step_by_step' # end_of_episode / step_by_step
    observation_components: Dict[str, bool] = field(default_factory=lambda: {
        'include_decomposition_state': True,
        'include_current_class_info': True,
        'include_quality_metrics': False,
        'embed_decomposition': True
    })
    action_masking: bool = True 
    action_masking_type: str = 'DQN'
    max_classes_in_service: int = 200

    reward_frequency: int = 50 # How often to reward the agent (in steps) - n_step should be aligned with this

@dataclass
class DQNAgentConfig(BaseConfig):
    """Configuration for the RL agent"""
    # Network Architecture
    architecture: list = field(default_factory=lambda: [16384, 8192, 4096, 2048, 1024])
    activation_fn: str = nn.ReLU  # Fixed typo

    # Learning Parameters
    learning_rate: float = 1.4e-05
    batch_size: int = 4096
    grad_clip: float = 50.0
    grad_clip_by: str = 'global_norm'
    td_error_loss_fn: str = 'huber'
    n_steps: int = 52 # should be >= reward_freq  
    target_update_interval: int = 20260
    gamma: float = 0.998
    tau: float = 0.5
    train_freq: tuple = (2, 'episode')
    gradient_steps: int = 8

    # Exploration
    exploration_type: str = "EpsilonGreedy"
    exploration_initial_eps: float = 1
    exploration_final_eps: float = 0.08
    exploration_fraction: float = 0.75
    learning_starts: int = 5000

    # Replay
    buffer_size: int = 101300
    prioritized_replay_alpha: float = 0.55
    prioritized_replay_beta: float = 1.0
    prioritized_replay_eps: float = 1e-06

    # Embedding
    embedding_dim: int = 2

    # Other
    use_gpu: bool = True

@dataclass
class TrainingConfig(BaseConfig):
    """Configuration for training process"""
    total_timesteps: int = 15_000_000 # Total training steps
    eval_freq: int = 10000        # How often to evaluate performance
    save_freq: int = 50000        # How often to save the model
    log_interval: int = 1000       # How often to log progress

class LegacyConfig:
    """Main configuration class that combines all settings"""
    
    def __init__(self):
        self.environment = SeqEnvironmentConfig()
        self.dqn_agent = DQNAgentConfig()
        self.training = TrainingConfig()


