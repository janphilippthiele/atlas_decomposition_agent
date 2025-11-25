from rl_microservice_decomposer.configs.plants.config_plants_dqn import PlantsConfig
from rl_microservice_decomposer.configs.jpetstore.config_jpetstore_dqn import JpetstoreConfig
from rl_microservice_decomposer.configs.daytrader.config_daytrader_dqn import DaytraderConfig
from rl_microservice_decomposer.configs.roller.config_roller_dqn import RollerConfig
# Import customized config class for more datasets

def get_tuning_config(trial, agent_type = 'dqn', n_classes = None, dataset_name = None):
    
    if 'plants' in dataset_name:

        trial_config: PlantsConfig = PlantsConfig()
        
        print('DQN config suggested')
        trial_config.environment.action_masking = True

        trial_config.dqn_agent.learning_rate = trial.suggest_loguniform("lr", 1e-5, 1e-3)
        trial_config.dqn_agent.batch_size = trial.suggest_categorical("batch_size", [256, 512, 1024, 2048])
        trial_config.dqn_agent.train_freq =  trial.suggest_categorical("train_freq", [(n_classes), (n_classes*2), (n_classes*3)]) # therefore expressed as episodes
        trial_config.dqn_agent.tau = trial.suggest_uniform("tau", 0.9, 1)
        trial_config.dqn_agent.exploration_final_eps = trial.suggest_uniform("exploration_final_eps", 0.01, 0.15)
        trial_config.dqn_agent.exploration_fraction = trial.suggest_uniform("exploration_fraction", 0.5, 0.8)
        trial_config.dqn_agent.buffer_size = trial.suggest_categorical("buffer_size", [2500, 5000, 7500])
        trial_config.dqn_agent.prioritized_replay_alpha = trial.suggest_uniform("prioritized_alpha", 0.4, 1.0)
        trial_config.dqn_agent.prioritized_replay_beta = trial.suggest_uniform("prioritized_beta", 0.4, 1.6)
        trial_config.dqn_agent.learning_starts = trial_config.dqn_agent.buffer_size
        trial_config.dqn_agent.grad_clip = trial.suggest_categorical("max_grad_norm", [1, 2, 5])
        trial_config.dqn_agent.prioritized_replay_alpha = trial.suggest_uniform("alpha", 0.4, 1.0)
        trial_config.dqn_agent.prioritized_replay_beta = trial.suggest_uniform("beta", 0.55, 1.6)

        trial_config.dqn_agent.architecture = trial.suggest_categorical("net_arch", [
        
        [128, 128],
        [128, 64, 32],
        [256, 128],
        [128, 128, 64],
        [256, 128, 64, 32],
        [256, 256, 64],
        [512, 256, 128],
        [512, 256, 64]

        ])
    elif 'jpetstore' in dataset_name:

        trial_config: JpetstoreConfig = JpetstoreConfig()
        
        print('DQN config suggested')
        trial_config.environment.action_masking = True

        trial_config.dqn_agent.learning_rate = trial.suggest_loguniform("lr", 1e-5, 1e-3)
        trial_config.dqn_agent.batch_size = trial.suggest_categorical("batch_size", [256, 512, 1024, 2048])
        trial_config.dqn_agent.train_freq =  trial.suggest_categorical("train_freq", [(n_classes), (n_classes*2), (n_classes*3)]) # therefore expressed as episodes
        trial_config.dqn_agent.tau = trial.suggest_uniform("tau", 0.9, 1)
        trial_config.dqn_agent.exploration_final_eps = trial.suggest_uniform("exploration_final_eps", 0.01, 0.15)
        trial_config.dqn_agent.exploration_fraction = trial.suggest_uniform("exploration_fraction", 0.5, 0.8)
        trial_config.dqn_agent.buffer_size = trial.suggest_categorical("buffer_size", [2500, 5000, 7500])
        trial_config.dqn_agent.prioritized_replay_alpha = trial.suggest_uniform("prioritized_alpha", 0.4, 1.0)
        trial_config.dqn_agent.prioritized_replay_beta = trial.suggest_uniform("prioritized_beta", 0.4, 1.6)
        trial_config.dqn_agent.learning_starts = trial_config.dqn_agent.buffer_size
        trial_config.dqn_agent.grad_clip = trial.suggest_categorical("max_grad_norm", [1, 2, 5])
        trial_config.dqn_agent.prioritized_replay_alpha = trial.suggest_uniform("alpha", 0.4, 1.0)
        trial_config.dqn_agent.prioritized_replay_beta = trial.suggest_uniform("beta", 0.55, 1.6)

        trial_config.dqn_agent.architecture = trial.suggest_categorical("net_arch", [

        [128, 128],
        [128, 64, 32],
        [128, 128, 64],
        [256, 128, 64, 32],
        [256, 256, 64],
        [512, 256, 128],
        [512, 256, 64]

        ])
    elif 'daytrader' in dataset_name:

        trial_config: DaytraderConfig = DaytraderConfig()
        
        print('DQN config suggested')
        trial_config.environment.action_masking = True

        trial_config.dqn_agent.learning_rate = trial.suggest_loguniform("lr", 1e-5, 1e-3)
        trial_config.dqn_agent.batch_size = trial.suggest_categorical("batch_size", [256, 512, 1024, 2048])
        trial_config.dqn_agent.train_freq =  trial.suggest_categorical("train_freq", [(n_classes), (n_classes*2), (n_classes*3)]) # therefore expressed as episodes
        trial_config.dqn_agent.tau = trial.suggest_uniform("tau", 0.9, 1)
        trial_config.dqn_agent.exploration_final_eps = trial.suggest_uniform("exploration_final_eps", 0.01, 0.15)
        trial_config.dqn_agent.exploration_fraction = trial.suggest_uniform("exploration_fraction", 0.5, 0.8)
        trial_config.dqn_agent.buffer_size = trial.suggest_categorical("buffer_size", [2500, 5000, 7500])
        trial_config.dqn_agent.prioritized_replay_alpha = trial.suggest_uniform("prioritized_alpha", 0.4, 1.0)
        trial_config.dqn_agent.prioritized_replay_beta = trial.suggest_uniform("prioritized_beta", 0.4, 1.6)
        trial_config.dqn_agent.learning_starts = trial_config.dqn_agent.buffer_size
        trial_config.dqn_agent.grad_clip = trial.suggest_categorical("max_grad_norm", [1, 2, 5])
        trial_config.dqn_agent.prioritized_replay_alpha = trial.suggest_uniform("alpha", 0.4, 1.0)
        trial_config.dqn_agent.prioritized_replay_beta = trial.suggest_uniform("beta", 0.55, 1.6)

        trial_config.dqn_agent.architecture = trial.suggest_categorical("net_arch", [
        
        [512, 256, 64],
        [512, 256, 128],
        [512, 256, 128, 64],
        [1024, 512, 128, 64],
        [1024, 512, 256, 64],
        [1024, 512, 256],
        [1024, 512, 256, 64], 
        [1024, 512, 256, 128],
        [1024, 1024, 512, 256],
        [2048, 1024, 512, 256],

        ])
    
    elif 'roller' in dataset_name:

        trial_config: RollerConfig = RollerConfig()
        
        print('DQN config suggested')
        trial_config.environment.action_masking = True

        trial_config.dqn_agent.learning_rate = trial.suggest_loguniform("lr", 1e-6, 1e-4)
        trial_config.dqn_agent.batch_size = trial.suggest_categorical("batch_size", [2048, 4096])
        trial_config.dqn_agent.target_update_interval =  trial.suggest_categorical("target_update_freq", [n_classes*2, n_classes*3]) # therefore expressed as episodes
        trial_config.dqn_agent.tau = trial.suggest_uniform("tau", 0.8, 1)
        trial_config.dqn_agent.exploration_final_eps = trial.suggest_uniform("exploration_final_eps", 0.01, 0.15)
        trial_config.dqn_agent.exploration_fraction = trial.suggest_uniform("exploration_fraction", 0.4, 0.8)
        trial_config.dqn_agent.buffer_size = trial.suggest_categorical("buffer_size", [5000, 7500, 10000])
        trial_config.dqn_agent.learning_starts = trial.suggest_categorical("learning_starts", [10000])
        trial_config.dqn_agent.grad_clip = trial.suggest_categorical("max_grad_norm", [5.0])
        trial_config.dqn_agent.prioritized_replay_alpha = trial.suggest_uniform("alpha", 0.4, 0.7)
        trial_config.dqn_agent.prioritized_replay_beta = trial.suggest_uniform("beta", 1.0, 1.4)
        trial_config.dqn_agent.architecture = trial.suggest_categorical("net_arch", [

        # Moderate capacity
        [1024, 512, 256, 128],
        [2048, 1024, 512, 256],
        
        # High capacity (if you have complex state representations)
        [2048, 1024, 512, 128],
        [1024, 1024, 512, 256],
        
        # Bottleneck approaches (compress then expand slightly)
        [1024, 256, 512, 256],

        ])

    # Add another search space here if needed

    return trial_config

def get_training_config(dataset_name):

     
    training_configs = {
        'plants': PlantsConfig(),
        'jpetstore': JpetstoreConfig(),
        'daytrader': DaytraderConfig(),
        'roller': RollerConfig(),
        # Add another config class here if needed
    }

    if dataset_name not in training_configs:
        raise ValueError(f"Unknown dataset name: {dataset_name}")

    config = training_configs.get(dataset_name)

    return config