import os
import optuna
import json

import numpy as np
from datetime import datetime
from pathlib import Path

from rl_microservice_decomposer.agent.dqn_agent import DQNDecompositionAgent
from rl_microservice_decomposer.environment.open_source_env import SequentialEnvOS
from rl_microservice_decomposer.configs.config_provider import get_tuning_config
from rl_microservice_decomposer.configs.config_provider import get_training_config
from rl_microservice_decomposer.utils.matrix_loader import load_adjacency_matrix

def save_results(training_results, config, start_time, early_stopped, agent_type, opt_metric, dataset_name):
    
    # Make the dir if it does not exist
    result_dir = f'C:\\Git\\atlas_decomposition_agent\\data\\decompositions\\new_decompositions\\{dataset_name}\\'
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
        'agent_config': config.dqn_agent.to_dict(),
        'env_config': config.environment.to_dict(),
        'training_config': config.training.to_dict(),
        'opt_metric': opt_metric
    } | training_results

    result_path = result_dir + f'/decomposition_{agent_type}_{start_time}.json'

    with open(result_path, 'w') as f:
        json.dump(final_results, f, indent=2, default=safe_convert)

def optimize_hyperparameters(adjacency_matrix, agent_type, n_trials=100, dataset_name=None):
    
    def objective(trial, adj_matrix):

        start_time = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Get trial-specific config
        config = get_tuning_config(trial, n_classes=adjacency_matrix.shape[0], dataset_name=dataset_name)

        def env_factory():
            return SequentialEnvOS(adjacency_matrix=adj_matrix, config=config)
        
        # Create agent
        agent = DQNDecompositionAgent(adjacency_matrix=adj_matrix, env_factory=env_factory, config=config, embedded_input=config.environment.observation_components.get('embed_decomposition', False))

        # Train agent
        training_results = agent.train(config.training.total_timesteps)

        # Compute optimization metric - this can be adjusted as needed
        opt_metric = (training_results['best_decomposition_quality'] + 
                        training_results['mean_decomposition_quality_last_N'] - 
                        0.25 * training_results['std_decomposition_quality'])

        trial.report(opt_metric, config.training.total_timesteps)

        save_results(training_results, config, start_time, early_stopped = False,
                        agent_type=agent_type,
                        opt_metric=opt_metric,
                        dataset_name=dataset_name)

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

def perform_singular_decomposition(adjacency_matrix, agent_type, dataset_name):
    
    start_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    config = get_training_config(dataset_name)

    def env_factory():
            return SequentialEnvOS(adjacency_matrix=adjacency_matrix, config=config)
    
    agent = DQNDecompositionAgent(adjacency_matrix=adjacency_matrix, env_factory=env_factory, config=config, embedded_input=config.environment.observation_components.get('embed_decomposition', False))

    training_results = agent.train(config.training.total_timesteps)

    save_results(training_results, config, start_time, early_stopped = False,
                        agent_type=agent_type,
                        opt_metric=None,
                        dataset_name=dataset_name)

    agent.close()

    print(f"Completed singular decomposition for dataset: {dataset_name}")

# Run optimization
if __name__ == "__main__":

    """
    Main function to run hyperparameter optimization or singular decomposition

    Configure the dataset paths and names below. Set 'perform_hyperparameter_tuning' to True
    to run hyperparameter optimization, or False to perform a singular decomposition with default settings.
    """
    
    perform_hyperparameter_tuning = False
    dataset_names = ['plants', 'jpetstore', 'daytrader', 'roller']
    dataset_paths = [
       r'C:\Git\atlas_decomposition_agent\data\application_data\plants\call_graph_plants.csv',
       r'C:\Git\atlas_decomposition_agent\data\application_data\jpetstore\call_graph_jpetstore.csv',
       r'C:\Git\atlas_decomposition_agent\data\application_data\daytrader\call_graph_daytrader.csv',
       r'C:\Git\atlas_decomposition_agent\data\application_data\roller\class_interactions.parquet'
   ]

    for dataset_name, dataset_path in zip(dataset_names, dataset_paths):
        
        print(f"Processing dataset: {dataset_name}")
        
        adjacency_matrix = load_adjacency_matrix(dataset_path, normalization='binary')
        
        # Making sure no self-loops included
        np.fill_diagonal(adjacency_matrix, 0)
        
        print(f"Number of classes: {adjacency_matrix.shape[0]}")

        if perform_hyperparameter_tuning:
            study = optimize_hyperparameters(adjacency_matrix, agent_type='dqn', n_trials=100, dataset_name=dataset_name)   

        else:
            perform_singular_decomposition(adjacency_matrix, agent_type='dqn', dataset_name=dataset_name)
 

