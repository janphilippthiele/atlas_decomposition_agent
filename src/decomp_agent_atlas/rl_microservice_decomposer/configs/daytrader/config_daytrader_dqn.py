# This file is used to configure the RL-Agent tool's hyperparameters

from typing import Dict, List
from dataclasses import dataclass, field
import torch as th
import json
import os
import sys
from torch import nn

from rl_microservice_decomposer.configs.utils.config_utils import BaseConfig

@dataclass    
class SeqEnvironmentConfig(BaseConfig):
    """Configuration for the decomposition environment"""
    initial_state: str = "all_together"  # "all_separate" or "all_together", what the decomposition looks like
    max_microservices: int = 16
    reward_scaling_factor: int= 1
    reward_type: str = 'end_of_episode' # end_of_episode / step_by_step
    observation_components: Dict[str, bool] = field(default_factory=lambda: {
        'include_decomposition_state': True,
        'include_current_class_info': True,
        'include_quality_metrics': False
    })
    action_masking: bool = True 
    max_classes_in_service: int = 19
    one_hot_encoding: bool = True

@dataclass
class DQNAgentConfig(BaseConfig):
    """Configuration for the RL agent"""
    # Network Architecture
    architecture: list = field(default_factory=lambda: [512, 256])
    activation_fn: str = nn.ReLU

    # Learning Parameters
    learning_rate: float = 0.005
    batch_size: int = 1028
    grad_clip: float = 0.5
    grad_clip_by: str = 'global_norm'
    td_error_loss_fn: str = 'huber'
    n_steps: int = 20
    gradient_steps: int = 2
    train_freq: int = (1, "episode")
    target_update_interval: int = 74
    gamma: float = 0.9995
    tau: float = 1

    # Exploration
    exploration_type: str = "EpsilonGreedy"
    exploration_initial_eps: float = 1.0
    exploration_final_eps: float = 0.02
    learning_starts: int = 10000
    exploration_fraction: float = 0.8

    # Replay Buffer (Prioritized Experience Replay)
    buffer_size: int = 10000
    prioritized_replay_alpha: float = 0.9
    prioritized_replay_beta: float = 1.5
    prioritized_replay_eps: float = 1e-06

    # Other
    use_gpu: bool = True

@dataclass
class TrainingConfig(BaseConfig):
    """Configuration for training process"""
    total_timesteps: int = 250_000  # Total training steps
    eval_freq: int = 10000        # How often to evaluate performance
    save_freq: int = 50000        # How often to save the model
    log_interval: int = 1000       # How often to log progress

class DaytraderConfig:
    """Main configuration class that combines all settings"""
    
    def __init__(self):
        self.environment = SeqEnvironmentConfig()
        self.dqn_agent = DQNAgentConfig()
        self.training = TrainingConfig()
        
            
