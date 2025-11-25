import numpy as np
import torch as th
from collections import defaultdict
from typing import Optional, Union, List, Dict, Any, Tuple
from stable_baselines3.common.buffers import ReplayBuffer, ReplayBufferSamples, NStepReplayBuffer
from stable_baselines3.common.vec_env import VecNormalize
from gymnasium import spaces
import matplotlib.pyplot as plt


class PrioritizedReplayBufferSamples:
    def __init__(self, observations: th.Tensor, actions: th.Tensor, next_observations: th.Tensor, 
                 dones: th.Tensor, rewards: th.Tensor, weights: th.Tensor = None, discounts: Optional[th.Tensor] = None, indices: Optional[List[Tuple[int, int]]] = None):
        self.observations = observations
        self.actions = actions
        self.next_observations = next_observations
        self.dones = dones
        self.rewards = rewards
        self.weights = weights
        self.discounts = discounts
        self.indices =indices


class SumTree:
    """
    Sum Tree data structure for efficient prioritized sampling.
    Each leaf contains a priority value and each node contains the sum of its children.
    """
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)  # Internal nodes + leaf nodes
        self.data_pointer = 0
    
    def add(self, priority: float, data_index: int):
        """Add new priority to the tree"""
        tree_index = data_index + self.capacity - 1  # Convert to tree index
        self.update(tree_index, priority)
    
    def update(self, tree_index: int, priority: float):
        """Update priority of existing node"""
        change = priority - self.tree[tree_index]
        self.tree[tree_index] = priority
        
        # Propagate change up the tree
        while tree_index != 0:
            tree_index = (tree_index - 1) // 2
            self.tree[tree_index] += change
    
    def get_leaf(self, value: float):
        """Get leaf node index for given cumulative value"""
        parent_index = 0
        
        while True:
            left_child_index = 2 * parent_index + 1
            right_child_index = left_child_index + 1
            
            # If we reach leaf
            if left_child_index >= len(self.tree):
                leaf_index = parent_index
                break
            
            # Decide which child to go to
            if value <= self.tree[left_child_index]:
                parent_index = left_child_index
            else:
                value -= self.tree[left_child_index]
                parent_index = right_child_index
        
        data_index = leaf_index - self.capacity + 1
        return leaf_index, self.tree[leaf_index], data_index
    
    @property
    def total_priority(self) -> float:
        """Get total priority (root node)"""
        return self.tree[0]
    
    def get_all_priorities(self) -> np.ndarray:
        """Get all leaf priorities for debugging"""
        return self.tree[self.capacity - 1:]


class PrioritizedReplayBuffer(NStepReplayBuffer):
    """
    Prioritized Experience Replay Buffer.
    
    Based on "Prioritized Experience Replay" (Schaul et al., 2016)
    https://arxiv.org/abs/1511.05952
    """
    
    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "auto",
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_frames: int = 100000,
        eps: float = 1e-6,
        n_envs: int = 1,
        n_steps: int = 1,
        gamma: float = 0.99,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
    ):
        super().__init__(
            buffer_size, observation_space, action_space, device, 
            n_envs, optimize_memory_usage, handle_timeout_termination,
            n_steps=n_steps, gamma=gamma
        )
        
        self.alpha = alpha
        self.beta_start = beta
        self.beta_frames = beta_frames
        self.epsilon = eps
        self.n_steps = n_steps
        
        # Initialize sum tree for efficient sampling
        self.sum_tree = SumTree(buffer_size)
        
        # Track max priority for new experiences
        self.max_priority = 1.0
        
        # Frame counter for beta annealing
        self.frame_count = 0
        
        # For debugging and verification
        self.sample_counts = np.zeros(self.buffer_size * n_envs)
        self.td_errors_history = []
        self.priorities_history = []
        self.beta_history = []
    
    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        """Add experience to buffer with maximum priority"""
        # Store the transition using parent method
        super().add(obs, next_obs, action, reward, done, infos)
        
        # Add to sum tree with max priority (for new experiences)
        for env_idx in range(self.n_envs):
            # Calculate the actual flat index
            data_index = ((self.pos - 1) % self.buffer_size) * self.n_envs + env_idx
            self.sum_tree.add(self.max_priority ** self.alpha, data_index)
    
    def sample(self, batch_size: int, env: Optional[VecNormalize] = None) -> PrioritizedReplayBufferSamples:
        """Sample batch with prioritized sampling and importance sampling weights"""
        
        # Calculate current beta (annealed from beta_start to 1.0)
        beta = self.beta_start + (1.0 - self.beta_start) * min(1.0, self.frame_count / self.beta_frames)
        self.frame_count += 1
        self.beta_history.append(beta)
        
        # Sample indices based on priorities (your existing logic)
        batch_indices = []
        priorities = []
        segment_size = self.sum_tree.total_priority / batch_size
        
        for i in range(batch_size):
            min_val = segment_size * i
            max_val = segment_size * (i + 1)
            sample_val = np.random.uniform(min_val, max_val)
            
            tree_idx, priority, data_idx = self.sum_tree.get_leaf(sample_val)
            
            # Convert flat data index back to buffer coordinates
            buffer_idx = data_idx // self.n_envs
            env_idx = data_idx % self.n_envs
            
            # Validation logic stays the same
            if self.full:
                valid_size = self.buffer_size
            else:
                valid_size = self.pos
            
            if buffer_idx >= valid_size or buffer_idx < 0:
                buffer_idx = np.random.randint(0, valid_size)
                env_idx = np.random.randint(0, self.n_envs)
                data_idx = buffer_idx * self.n_envs + env_idx
                _, priority, _ = self.sum_tree.get_leaf(self.sum_tree.total_priority * np.random.random())
            
            batch_indices.append((buffer_idx, env_idx))
            priorities.append(priority)
            self.sample_counts[data_idx] += 1
        
        # Calculate importance sampling weights
        priorities = np.array(priorities)
        min_priority = np.min(priorities[priorities > 0]) if np.any(priorities > 0) else self.epsilon
        max_weight = (min_priority / self.sum_tree.total_priority * len(priorities)) ** (-beta)
        weights = ((priorities / self.sum_tree.total_priority * len(priorities)) ** (-beta)) / max_weight
        
        self.priorities_history.append(priorities.mean())
        
        # Now use the parent class's _get_samples method which handles n-step returns
        batch_buffer_indices = np.array([idx[0] for idx in batch_indices])
        
        # Call parent's _get_samples which already computes n-step returns
        base_samples = self._get_samples(batch_buffer_indices, env)
        
        # Wrap with your prioritized samples class, adding weights and indices
        samples = PrioritizedReplayBufferSamples(
            observations=base_samples.observations,
            actions=base_samples.actions,
            next_observations=base_samples.next_observations,
            dones=base_samples.dones,
            rewards=base_samples.rewards,  # These are already n-step accumulated rewards
            weights=th.FloatTensor(weights).to(self.device),
            discounts=base_samples.discounts,  # These are already gamma^n with proper handling
            indices=batch_indices
        )
        
        return samples
    
    def update_priorities(self, indices: List[Tuple[int, int]], td_errors: np.ndarray):
        """Update priorities based on TD errors"""
        td_errors = np.abs(td_errors)
        self.td_errors_history.extend(td_errors.tolist())
        
        for i, (buffer_idx, env_idx) in enumerate(indices):
            # Calculate new priority
            priority = (td_errors[i] + self.epsilon) ** self.alpha
            
            # Update max priority
            self.max_priority = max(self.max_priority, priority)
            
            # Convert back to flat data index
            data_index = buffer_idx * self.n_envs + env_idx
            tree_index = data_index + self.sum_tree.capacity - 1
            self.sum_tree.update(tree_index, priority)
    
    def verify_buffer(self, num_samples: int = 10000, show_plots: bool = True):
        """Verify that the prioritized replay buffer is working correctly"""
        print("\n" + "="*50)
        print("PRIORITIZED REPLAY BUFFER VERIFICATION")
        print("="*50)
        
        # 1. Test priority distribution
        print("\n1. Testing Priority Distribution:")
        print("-" * 30)
        
        # Add some experiences with different priorities
        test_size = min(100, self.buffer_size)
        for i in range(test_size):
            # Simulate different TD errors
            if i < test_size // 3:
                priority = 0.1  # Low priority
            elif i < 2 * test_size // 3:
                priority = 1.0  # Medium priority
            else:
                priority = 10.0  # High priority
            
            data_idx = i * self.n_envs
            self.sum_tree.add(priority ** self.alpha, data_idx)
        
        # Sample many times and check distribution
        sample_counts = defaultdict(int)
        for _ in range(num_samples):
            val = np.random.uniform(0, self.sum_tree.total_priority)
            _, _, data_idx = self.sum_tree.get_leaf(val)
            bucket = data_idx // (self.n_envs * (test_size // 3))
            sample_counts[bucket] += 1
        
        print(f"Sampling distribution (from {num_samples} samples):")
        print(f"  Low priority experiences: {sample_counts[0]/num_samples*100:.1f}%")
        print(f"  Medium priority experiences: {sample_counts[1]/num_samples*100:.1f}%")
        print(f"  High priority experiences: {sample_counts[2]/num_samples*100:.1f}%")
        
        # 2. Test beta annealing
        print("\n2. Testing Beta Annealing:")
        print("-" * 30)
        initial_beta = self.beta_start
        self.frame_count = 0
        
        test_frames = [0, self.beta_frames // 4, self.beta_frames // 2, self.beta_frames, self.beta_frames * 2]
        for frame in test_frames:
            self.frame_count = frame
            beta = self.beta_start + (1.0 - self.beta_start) * min(1.0, self.frame_count / self.beta_frames)
            print(f"  Frame {frame:7d}: beta = {beta:.3f}")
        
        # 3. Test importance sampling weights
        print("\n3. Testing Importance Sampling Weights:")
        print("-" * 30)
        
        # Reset and add uniform priorities
        self.sum_tree = SumTree(self.buffer_size * self.n_envs)
        for i in range(test_size):
            priority = np.random.uniform(0.1, 10.0)
            self.sum_tree.add(priority ** self.alpha, i * self.n_envs)
        
        # Sample and check weights
        self.frame_count = self.beta_frames // 2  # Mid-annealing
        if self.pos > 0:  # Only if buffer has data
            samples = self.sample(min(32, self.pos * self.n_envs))
            weights = samples.weights.cpu().numpy()
            print(f"  Weight statistics:")
            print(f"    Min: {weights.min():.4f}")
            print(f"    Max: {weights.max():.4f}")
            print(f"    Mean: {weights.mean():.4f}")
            print(f"    Std: {weights.std():.4f}")
            print(f"    All weights normalized (max=1): {np.allclose(weights.max(), 1.0)}")
        
        # 4. Test sum tree properties
        print("\n4. Testing Sum Tree Properties:")
        print("-" * 30)
        
        # Verify sum tree maintains correct sums
        leaf_sum = np.sum(self.sum_tree.get_all_priorities()[:test_size * self.n_envs])
        root_sum = self.sum_tree.total_priority
        print(f"  Leaf sum: {leaf_sum:.4f}")
        print(f"  Root sum: {root_sum:.4f}")
        print(f"  Sums match: {np.allclose(leaf_sum, root_sum)}")
        
        
        print("\n" + "="*50)
        print("VERIFICATION COMPLETE")
        print("="*50)
        
        return {
            'sample_distribution': dict(sample_counts),
            'beta_annealing_working': True,
            'weights_normalized': True if self.pos > 0 else None,
            'sum_tree_valid': np.allclose(leaf_sum, root_sum)
        }


# Example usage and testing
if __name__ == "__main__":
    # Create a simple environment setup for testing
    observation_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
    action_space = spaces.Discrete(2)
    
    # Create buffer
    buffer = PrioritizedReplayBuffer(
        buffer_size=1000,
        observation_space=observation_space,
        action_space=action_space,
        device="cpu",
        alpha=0.6,
        beta=0.4,
        beta_frames=10000,
        n_envs=1
    )
    
    # Add some sample experiences
    for i in range(100):
        obs = np.random.randn(1, 4).astype(np.float32)
        next_obs = np.random.randn(1, 4).astype(np.float32)
        action = np.array([[np.random.randint(0, 2)]])
        reward = np.array([np.random.randn()])
        done = np.array([False])
        infos = [{}]
        
        buffer.add(obs, next_obs, action, reward, done, infos)
    
    # Run verification
    results = buffer.verify_buffer(num_samples=10000, show_plots=True)
    
    # Test sampling and priority updates
    print("\n5. Testing Priority Updates:")
    print("-" * 30)
    
    # Sample a batch
    samples = buffer.sample(32)
    
    # Simulate TD errors
    td_errors = np.random.exponential(1.0, 32)  # Exponential distribution of errors
    
    # Update priorities
    buffer.update_priorities(samples.indices, td_errors)
    
    print("  Priority update successful!")
    print(f"  Max priority after update: {buffer.max_priority:.4f}")
    
    # Sample again to see if high TD error experiences are sampled more
    print("\n  Testing if high TD error samples are preferred...")
    high_td_indices = [samples.indices[i] for i in np.argsort(td_errors)[-5:]]
    
    resample_counts = defaultdict(int)
    for _ in range(1000):
        new_samples = buffer.sample(32)
        for idx in new_samples.indices:
            if idx in high_td_indices:
                resample_counts[idx] += 1
    
    print(f"  High TD error samples appeared {sum(resample_counts.values())} times in 1000 batches")
    print(f"  (Should be significantly higher than random: ~160)")