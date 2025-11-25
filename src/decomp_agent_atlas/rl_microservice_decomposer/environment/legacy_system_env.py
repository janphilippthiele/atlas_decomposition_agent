import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Any, Tuple, Optional
import sys
import os
from collections import Counter

from rl_microservice_decomposer.configs.legacy_system.config_legacy_system_dqn import LegacyConfig
from rl_microservice_decomposer.metrics.legacy_system_metrics import LegacySystemMetricsCalculator

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

class SequentialEnvLegacySystem(gym.Env):
    """
    Environment for the trainings on legacy systems.
    """
    def __init__(self, adjacency_matrix: np.ndarray, config: LegacyConfig, dataset_name: str):
        
        super().__init__()

        self.adjacency_matrix = adjacency_matrix
        self.config = config
        self.n_classes = adjacency_matrix.shape[0]

        # Env config setup
        self.embedding_dim = self.config.dqn_agent.embedding_dim
        self.reward_scaling_factor = self.config.environment.final_reward_scaling_factor
        self.include_quality_metrics = self.config.environment.observation_components.get('include_quality_metrics', False)
        self.reward_frequency = self.config.environment.reward_frequency

        #self.state_manager = StateManager(adjacency_matrix)
        self.max_microservices = self.config.environment.max_microservices
        self.legacy_metrics_calculator = LegacySystemMetricsCalculator(adjacency_matrix, dataset_name)

        self.action_space = spaces.Discrete(self.config.environment.max_microservices)

        obs_size = self._calculate_observation_space_size()

        # Regular Box space for other algorithms
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(obs_size,),
            dtype=np.float32
        )
    
        # Env variables
        self.current_observation = None
        self.current_decomposition = None
        self.current_class_idx = 0
        self.episode_count = 0
        self.highest_service_used = 0
        self.prev_decomposition_quality = 0.0

        # Performance Tracking
        self.action_history = [] # This is also the decomposition
        self.episode_rewards = []

        # Initial Buffers
        self.action_mask_buffer = np.zeros(self.max_microservices, dtype=np.int8)

        self.reset()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        
        # apply the action to the current_decomposition
        microservice_id = action + 1

        self.action_history.append(microservice_id)

        self.current_decomposition[self.current_class_idx] = microservice_id

        self.highest_service_used = max(self.highest_service_used, microservice_id)

        self.current_class_idx += 1

        # end when all elements have been assigned to a microservice
        done = self.current_class_idx >= self.n_classes

        # An implementation that allows to include the quality metrics in the observation (quality state augmentation)
        if self.include_quality_metrics:
            state_metrics = self._calculate_reward(self.current_decomposition, done)

            current_decomposition_quality = 0.1 * state_metrics['decomposition_quality']
            reward = current_decomposition_quality - self.prev_decomposition_quality

            self.prev_decomposition_quality = current_decomposition_quality

            info = dict(
                episode_complete=False,
            )

            if done:
                reward = reward + state_metrics['decomposition_quality'] * self.reward_scaling_factor # more general term than modularity
                info = dict(
                    episode_complete=True,
                    episode_reward = reward,
                    decomposition=self.action_history,
                    episode_count=self.episode_count,
                    decomposition_quality=state_metrics.get('decomposition_quality', None),
                    state_metrics=dict(
                        density=state_metrics.get('density', None),
                        coupling=state_metrics.get('coupling', None)
                    ),
                )

        # Standard implementation with reward only at the end of the episode or at intervals that was used in thesis experiments
        elif done or self.current_class_idx % self.reward_frequency == 0:
            
            state_metrics = self._calculate_reward(self.current_decomposition, done)
            reward = state_metrics['decomposition_quality']

            self.episode_rewards.append(reward)
            info = dict(
                episode_complete=False,
                )
            if done:

                state_metrics = self._calculate_reward(self.current_decomposition, done)
                reward = state_metrics['decomposition_quality'] * self.reward_scaling_factor
                self.episode_rewards.append(reward)

                info = dict(
                    episode_complete=True,
                    final_reward=reward,
                    all_rewards=self.episode_rewards,
                    decomposition=self.action_history,
                    episode_count=self.episode_count,
                    decomposition_quality=state_metrics.get('decomposition_quality', None),
                    state_metrics=state_metrics,
                    observation_size=self.current_observation.shape[0]
                )
            
        else:
            reward = 0.0
            info = dict(
                episode_complete=False,
                )
       
        info['action_mask'] = self.action_mask_buffer.copy()

        if not done:
            self.current_observation = self._get_observation(current_class_idx=self.current_class_idx, state_metrics=None)

        return self.current_observation, reward, done, False, info

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        if seed is not None:
            np.random.seed(seed)

        # Create starting decomposition
        self.current_decomposition = np.zeros(self.n_classes, dtype=np.int32)
        self.action_mask_buffer = np.zeros(self.max_microservices, dtype=np.int8)
        
        # Counting
        self.current_class_idx = 0
        self.episode_count += 1

        # Reset state metrics for quality state augmentation
        state_metrics = dict(
            read_write_densities = np.zeros(self.max_microservices, dtype=np.float32),
            call_densities = np.zeros(self.max_microservices, dtype=np.float32),
            external_edges_ratios = np.zeros(self.max_microservices, dtype=np.float32),
            hub_concentrations = np.zeros(self.max_microservices, dtype=np.float32),
            ned_penalty = np.atleast_1d(0).astype(np.float32)
        )

        if self.include_quality_metrics:
            self.legacy_metrics_calculator.reset_cache()

        # Reset performance tracking
        self.episode_rewards = []
        self.action_history = []

        # Reset episode variables
        self.highest_service_used = 0
        self.prev_modularity = 0.0
        
        self.current_observation = self._get_observation(current_class_idx=0, state_metrics=state_metrics)

        info = {'action_mask': self.action_mask_buffer}

        return self.current_observation, info
    
    def _calculate_reward(self, state: np.ndarray, done) -> float:
        state_metrics = self.legacy_metrics_calculator.calculate_state_metrics(state)
        return state_metrics

    def _calculate_observation_space_size(self) -> int:
        "Calculate the total observation space size based on config"
        size = self.n_classes + 2*self.n_classes # decomp_state + row + col vector

        if self.include_quality_metrics:
            size += 4 * self.max_microservices + 1

        return size

    def _get_observation(self, current_class_idx: int, state_metrics: Dict[str, float]) -> np.ndarray:
        """Build observation space vector"""
        observation_parts = []
        
        # Decomposition state
        observation_parts.append(self.current_decomposition)
        # Row vector: relationships from current class to all others
        class_row = self.adjacency_matrix[current_class_idx, :]
        # Col vector: relationships from all others to current class
        class_col = self.adjacency_matrix[:, current_class_idx]
        
        observation_parts.extend([class_row, class_col])

        if self.include_quality_metrics:
            rw_densities = state_metrics['read_write_densities']
            call_densities = state_metrics['call_densities']
            external_edges_ratios = state_metrics['external_edges_ratios']
            hub_concentrations = state_metrics['hub_concentrations']
            ned = np.atleast_1d(state_metrics['ned_penalty']).astype(np.float32)

            rw_densities = np.concatenate([rw_densities, np.zeros(self.max_microservices - len(rw_densities))])
            call_densities = np.concatenate([call_densities, np.zeros(self.max_microservices - len(call_densities))])
            external_edges_ratios = np.concatenate([external_edges_ratios, np.zeros(self.max_microservices - len(external_edges_ratios))])
            hub_concentrations = np.concatenate([hub_concentrations, np.zeros(self.max_microservices - len(hub_concentrations))])

            observation_parts.extend([rw_densities, call_densities, external_edges_ratios, hub_concentrations, ned])

        observation = np.concatenate(observation_parts).astype(np.float32)
        self.action_mask = self._get_action_mask()
 
        return observation

    def _get_action_mask(self) -> np.ndarray:
        """Generate action mask based on current state"""
        self.action_mask_buffer.fill(0)

        if self.current_class_idx >= self.n_classes:
            # episode done, no valid actions
            return self.action_mask_buffer.astype(np.int8)
        
        # Normal logic: allow reusing existing microservices + one new one
        for i in range(self.highest_service_used):
            self.action_mask_buffer[i] = 1


        ms_count = Counter(self.current_decomposition[self.current_decomposition > 0])
        for microservice_id in ms_count:
            if ms_count[microservice_id]>=self.config.environment.max_classes_in_service:
                self.action_mask_buffer[microservice_id-1]=0

        
        # Allow creating a new microservice if we haven't reached the limit
        if self.highest_service_used < self.max_microservices:
            self.action_mask_buffer[self.highest_service_used] = 1

        return self.action_mask_buffer.astype(np.int8)
    
    def action_mask(self):
        """Required method for ActionMasker wrapper"""
        return getattr(self, 'action_mask_buffer', np.ones(self.max_microservices, dtype=bool))