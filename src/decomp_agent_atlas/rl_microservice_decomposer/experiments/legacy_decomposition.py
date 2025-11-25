import os
import optuna
import json

import numpy as np
from datetime import datetime
from pathlib import Path

from rl_microservice_decomposer.agent.dqn_agent import DQNDecompositionAgent
from rl_microservice_decomposer.environment.legacy_system_env import SequentialEnvLegacySystem
from rl_microservice_decomposer.configs.legacy_system.zoo_search_space import get_config_legacy_system
from src.decomp_agent_atlas.rl_microservice_decomposer.configs.legacy_system.config_legacy_system_dqn import LegacyConfig

def save_results(training_results, config, start_time, early_stopped, agent_type, opt_metric, dataset_name):
    # Make the dir if it does not exist
    result_dir = f'C:\Git\atlas_decomposition_agent\data\decompositions\performed_in_thesis\legacy_system\\'
    os.makedirs(os.path.dirname(result_dir), exist_ok=True)
    # Convert the result structure
    def safe_convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, (datetime)):
            return str(obj)
        elif hasattr(obj, 'item'):  # numpy scalars
            return obj.item()
        else:
            return str(obj)
    
    final_results = {
        'dataset': dataset_name,
        'date': f'{datetime.now().strftime('%Y%m%d_%H%M%S')}',
        'early_stopped': early_stopped,
        'agent_type': agent_type,
        'agent_config': config.dqn_agent.to_dict() if agent_type.upper() == 'DQN' else config.ppo_agent.to_dict(),
        'env_config': config.environment.to_dict(),
        'training_config': config.training.to_dict(),
        'opt_metric': opt_metric
    } | training_results

    result_path = result_dir + f'/zoo_hpo_{agent_type}_{start_time}.json'

    with open(result_path, 'w') as f:
        json.dump(final_results, f, indent=2, default=safe_convert)

def optimize_hyperparameters(adjacency_matrix, agent_type, n_trials=100, dataset_name=None):
    def objective(trial, adj_matrix):
       
        start_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        print(f"Getting config with agent type: {agent_type}")
        
        config = LegacyConfig()

        print(f"Buffer Size: {config.dqn_agent.buffer_size}")
        def env_factory():
            return SequentialEnvLegacySystem(adjacency_matrix=adj_matrix, config=config, dataset_name=dataset_name)
        
        # Create a vectorized environment
        # Train agent
        agent = DQNDecompositionAgent(adjacency_matrix=adj_matrix, env_factory=env_factory, config=config, embedded_input=True)

        training_results = agent.train(config.training.total_timesteps)

        opt_metric = (training_results['best_decomposition_quality'] + 
                        training_results['mean_decomposition_quality_last_N'] - 
                        0.25 * training_results['std_decomposition_quality'])

        trial.report(opt_metric, config.training.total_timesteps)

        save_results(training_results, config, start_time, early_stopped = False,
                        agent_type=agent_type,
                        opt_metric=opt_metric,
                        dataset_name=dataset_name
                    )

        agent.close()
        
        return opt_metric
    
    # Create study with pruning
    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=1)
    )
    
    study.optimize(lambda trial: objective(trial, adjacency_matrix), n_trials=n_trials)
    
    print(f"Best trial: {study.best_trial.value}")
    print(f"Best params: {study.best_params}")
    
    return study

# Run optimization
if __name__ == "__main__":

    agent_type = 'dqn'
    dataset_names = ['legacy_system_shuffled']
    dataset_paths = [
        r'',
    ]

    for dataset_name, dataset_path in zip(dataset_names, dataset_paths):
        print(f"Processing dataset: {dataset_name}")
        
        input_dir=Path('')
        data = np.load(dataset_path, allow_pickle=True)
        adjacency_matrix = data['adjacency_matrix']

        study = optimize_hyperparameters(adjacency_matrix, agent_type=agent_type, n_trials=10, dataset_name=dataset_name)   