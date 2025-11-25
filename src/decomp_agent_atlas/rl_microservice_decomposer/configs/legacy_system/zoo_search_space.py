from rl_microservice_decomposer.configs.legacy_system.config_legacy_system_dqn import LegacyConfig

def get_config_legacy_system(trial, agent_type = 'dqn', n_classes=None):
    
    trial_config: LegacyConfig = LegacyConfig()

    print('DQN config suggested')
    trial_config.environment.action_masking = True

    trial_config.dqn_agent.learning_rate = trial.suggest_loguniform("lr", 1e-6, 5e-5)
    trial_config.dqn_agent.batch_size = trial.suggest_categorical("batch_size", [2048, 4096])
    trial_config.dqn_agent.exploration_final_eps = trial.suggest_uniform("exploration_final_eps", 0.05, 0.1)
    trial_config.dqn_agent.exploration_fraction = trial.suggest_uniform("exploration_fraction", 0.75, 0.8)
    trial_config.dqn_agent.buffer_size = trial.suggest_categorical("buffer_size", [n_classes*100])
    trial_config.dqn_agent.grad_clip = trial.suggest_categorical("max_grad_norm", [50.0])
    trial_config.dqn_agent.prioritized_replay_alpha = trial.suggest_uniform("alpha", 0.4, 0.9)
    trial_config.dqn_agent.prioritized_replay_beta = trial.suggest_uniform("beta", 0.9, 1.6)
    trial_config.dqn_agent.train_freq = trial.suggest_categorical("train_freq", [(1, "episode"), (2, "episode")])

    # Calculate target update relative to training frequency
    episodes_per_target_update = trial.suggest_categorical("target_update_episodes", [2, 4])
    trial_config.dqn_agent.target_update_interval = episodes_per_target_update * n_classes

    trial_config.dqn_agent.gradient_steps = trial.suggest_categorical("gradient_steps", [2, 4, 8])

    trial_config.dqn_agent.architecture = trial.suggest_categorical("net_arch", [


    [8192, 4096, 2048, 1024],
    [16384, 8192, 4096, 2048, 1024],
    ])

    return trial_config
