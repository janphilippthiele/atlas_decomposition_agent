from stable_baselines3.common.callbacks import BaseCallback
from typing import Optional, Dict, Any
import numpy as np

class CustomCallback(BaseCallback):
        
    def __init__(self, verbose: int = 1):
        super().__init__(verbose)
        self.episode_results = []
        self.decomposition_qualities = []
        self.best_reward = float('-inf')
        self.best_decomposition_quality = float('-inf')
        self.mean_decomposition_quality_last_N = float('-inf')
        self.best_mean_decomposition_quality_N = float('-inf')
        self.N = 10
        self.current_episode = 0

    def _on_step(self) -> bool:
        """
        Called after each environment step.
        """

        # Get info from all environments
        infos = self.locals.get('infos', [])
        
        # Process each environment's info
        for info in infos:
            if info and info.get('episode_complete', False):
                self._log_episode_end(info)
        
        return True
    
    def _log_episode_end(self, info: Dict[str, Any]):      
        # Extract metrics
        
        episode_reward = info.get('episode_reward', 0)
        decomposition = info.get('decomposition', [])
        episode_count = info.get('episode_count', None)
        decomposition_quality = info.get('decomposition_quality', None)

         # Store episode data
        episode_data = {
            'episode_id': episode_count,
            'episode_reward': episode_reward,
            'decomposition_quality': decomposition_quality,
            'state_metrics': info.get('state_metrics', None),
            'decomposition': decomposition,
        }

        self.episode_results.append(episode_data)
        self.decomposition_qualities.append(decomposition_quality)
        
        # Update best values
        if episode_reward > self.best_reward:
            self.best_reward = episode_reward

        if decomposition_quality > self.best_decomposition_quality:
            self.best_decomposition_quality = decomposition_quality

        # Calculate mean decomposition quality of last 25 episodes (modularity for open source and Legacy with cohesion and conductance)
        if len(self.decomposition_qualities) >= self.N:
            self.mean_decomposition_quality_last_N = np.mean(self.decomposition_qualities[-self.N:])
            if self.mean_decomposition_quality_last_N > self.best_mean_decomposition_quality_N:
                self.best_mean_decomposition_quality_N = self.mean_decomposition_quality_last_N
        self.current_episode += 1
        # Simple logging
        if self.verbose > 0:
            if self.current_episode % 20 == 0:
                print(f'mean last N Decomposition Qualities: {np.mean(self.decomposition_qualities[-self.N:])}')
                print(f"Episode {self.current_episode}: Reward={episode_reward:.3f}, "
                      f"Decomposition Quality={decomposition_quality:.3f}, Best={self.best_decomposition_quality:.3f}")

    def get_training_metrics(self):
        training_metrics = {
            'best_decomposition_quality': self.best_decomposition_quality,
            'mean_decomposition_quality_last_N': self.mean_decomposition_quality_last_N,
            'std_decomposition_quality': np.std(self.decomposition_qualities),
            'opt_metric': self.mean_decomposition_quality_last_N - 0.2 * np.std(self.decomposition_qualities),
            'decomposition_qualities': self.decomposition_qualities,
            'best_mean_decomposition_quality_N': self.best_mean_decomposition_quality_N,
        }
        return self.episode_results.copy(), training_metrics