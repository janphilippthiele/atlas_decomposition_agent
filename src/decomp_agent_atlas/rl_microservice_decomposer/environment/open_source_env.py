import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Any, Tuple, Optional
import sys
import os
from collections import Counter

from rl_microservice_decomposer.configs.roller.config_roller_dqn import RollerConfig
from rl_microservice_decomposer.metrics.open_source_metrics import SparseVectorizedMetricsCalculator

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

class SequentialEnvOS(gym.Env):
    """
    Environment for the trainings on open-source applications.

    - Uses structural modularity for reward
    - Uses Custom Modularity Callback
    """
    def __init__(self, adjacency_matrix: np.ndarray, config: RollerConfig):
        
        super().__init__()

        self.adjacency_matrix = adjacency_matrix
        self.config = config
        self.n_classes = adjacency_matrix.shape[0]

        # Env config setup
        self.include_decomp_state =  self.config.environment.observation_components.get('include_decomposition_state', True)
        self.include_cc_info = self.config.environment.observation_components.get('include_current_class_info', True)
        self.include_quality_metrics = self.config.environment.observation_components.get('include_quality_metrics', False)
        self.embed_decomposition = self.config.environment.observation_components.get('embed_decomposition', False)
        self.one_hot_encoding = self.config.environment.one_hot_encoding
        self.reward_scaling_factor = self.config.environment.reward_scaling_factor

        #self.state_manager = StateManager(adjacency_matrix)
        self.max_microservices = self.config.environment.max_microservices
        self.open_source_metrics_calculator = SparseVectorizedMetricsCalculator(adjacency_matrix, self.max_microservices)

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
        
        if done:
            state_metrics = self._calculate_reward(self.current_decomposition, done)
            reward =state_metrics['decomposition_quality'] # more general term than modularity
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
        
        if not done:
            reward = 0.0
            info = dict(
                episode_complete=False,
            )
            state_metrics = dict()
            self.current_observation = self._get_observation(current_class_idx=self.current_class_idx, state_metrics=state_metrics)

        info['action_mask'] = self.action_mask_buffer.copy()

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

        # Reset state metrics for initial observation
        state_metrics = dict(
            densities=np.zeros(self.max_microservices),
            couplings=np.zeros(self.max_microservices),
        )
        # Reset performance tracking
        self.episode_rewards = []
        self.action_history = []
        self.proxy_rewards = []

        # Reset episode variables
        self.highest_service_used = 0
        self.prev_decomposition_quality = 0.0
        
        self.current_observation = self._get_observation(current_class_idx=0, state_metrics=state_metrics)

        info = {'action_mask': self.action_mask_buffer}

        return self.current_observation, info

    
    def _calculate_reward(self, state: np.ndarray, done) -> float:

        state_metrics = self.open_source_metrics_calculator.calculate_state_metrics(state)
        return state_metrics
            

    def _encode_state_one_hot(self, state: np.ndarray) -> np.ndarray:
        """
        Using one hot encoding on smaller applications.
        Roller will use embeddings.
        """
        if self.one_hot_encoding:
            one_hot = np.zeros((self.n_classes, self.max_microservices), dtype=np.float32)
            for i, microservice_id in enumerate(state):
                if microservice_id > 0:  # 0 means unassigned
                    # Convert to 0-based index for one-hot encoding
                    one_hot[i, microservice_id - 1] = 1.0                        
            return one_hot.flatten()
        else:
            return state

    def _calculate_observation_space_size(self) -> int:
        "Calculate the total observation space size based on config"
        size=0

        if self.one_hot_encoding:
            if self.include_decomp_state:
                size += self.n_classes * self.max_microservices # ohe
            
            if self.include_cc_info:
                size += 2 * self.n_classes # row + col vector
            
            if self.include_quality_metrics:
                size += 2 * self.max_microservices 

        elif self.embed_decomposition:
            size += self.n_classes

            if self.include_cc_info:
                size += 2 * self.n_classes # row + col vector

            if self.include_quality_metrics:
                size += 2 * self.max_microservices 

        print(f"Calculated obs size: {size}")
        return size

    def _get_observation(self, current_class_idx: int, state_metrics: Dict[str, float]) -> np.ndarray:
        """Build observation space vector"""
        observation_parts = []

        if self.include_decomp_state:
            if self.one_hot_encoding:
                observation_parts.append(self._encode_state_one_hot(self.current_decomposition.copy()))
            else:
                observation_parts.append(self.current_decomposition)
        if  self.include_cc_info:
            # Row vector: relationships from current class to all others
            class_row = self.adjacency_matrix[current_class_idx, :]
            # Col vector: relationships from all others to current class
            class_col = self.adjacency_matrix[:, current_class_idx]
            observation_parts.extend([class_row, class_col])
        if self.include_quality_metrics:
            density_qualities = state_metrics['densities']
            coupling_qualities = state_metrics['couplings']

            density_qualities = np.concatenate([density_qualities, np.zeros(self.max_microservices - len(density_qualities))])
            coupling_qualities = np.concatenate([coupling_qualities, np.zeros(self.max_microservices - len(coupling_qualities))])

            observation_parts.extend([density_qualities, coupling_qualities])
        
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