import torch as th
import torch.nn as nn
import numpy as np

from stable_baselines3.dqn.policies import QNetwork
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces


class EmbeddingQNetwork(QNetwork):
    """
    Custom Q-Network with embedding layer for mixed input types.
    Handles MS IDs (categorical) and binary dependency matrix features.
    """
    
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        features_extractor: BaseFeaturesExtractor,
        features_extractor_kwargs: dict = None,
        normalize_images: bool = True,
        n_classes: int = 500,
        n_services: int = 20,
        embedding_dim: int = 8,
        net_arch: list = None,
        activation_fn: nn.Module = nn.ReLU,
        device: th.device = "auto",
    ):
        # Set up the features extractor to handle our custom input processing
        if features_extractor_kwargs is None:
            features_extractor_kwargs = {}
        features_extractor_kwargs.update({
            'n_classes': n_classes,
            'n_services': n_services, 
            'embedding_dim': embedding_dim
        })
        
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            features_extractor=features_extractor,
            features_extractor_kwargs=features_extractor_kwargs,
            normalize_images=normalize_images,
            net_arch=net_arch,
            activation_fn=activation_fn,
            device=device,
        )


class MixedInputFeaturesExtractor(BaseFeaturesExtractor):
    """
    Custom features extractor that handles embedding for MS IDs and 
    passes through binary dependency matrix features.
    """
    
    def __init__(
        self, 
        observation_space: spaces.Space,
        n_classes: int = None,
        max_n_services: int = 20,
        embedding_dim: int = 2,
        observation_components: dict = None
    ):
        self.n_classes = n_classes
        self.max_n_services = max_n_services
        self.embedding_dim = embedding_dim
        self.observation_components = observation_components or {}

        # Calculate feature dimensions
        # Embedded MS IDs: n_classes * embedding_dim
        # Binary features (row + col): 2 * n_classes
        features_dim = n_classes * embedding_dim + 2 * n_classes
        
        super().__init__(observation_space, features_dim)
        
        # Embedding layer for MS IDs (vocab size = n_services + 1 for padding/unknown)
        self.ms_embedding = nn.Embedding(
            num_embeddings=max_n_services + 1,  # +1 for unknown/padding
            embedding_dim=embedding_dim,
            padding_idx=0  # Use 0 for unknown/padding
        )
        
        # Initialize embeddings
        nn.init.xavier_uniform_(self.ms_embedding.weight)
    
    def forward(self, observations: th.Tensor) -> th.Tensor:
        """
        Process the mixed input:
        Input shape: [batch_size, n_classes * 3]
        - First n_classes elements: MS IDs (integers 1-20)
        - Next n_classes elements: Row from dependency matrix (binary)
        - Last n_classes elements: Column from dependency matrix (binary)
        """
        batch_size = observations.shape[0]
        
        # Split the input into three parts
        ms_ids = observations[:, :self.n_classes].long()  # Convert to long for embedding
        dependency_row = observations[:, self.n_classes:2*self.n_classes]
        dependency_col = observations[:, 2*self.n_classes:3*self.n_classes]
        if self.observation_components.get("service_qualities"):
            service_qualities = observations[:, 3*self.n_classes:]
        
        # Clamp MS IDs to valid range (handle any out-of-bounds values)
        ms_ids = th.clamp(ms_ids, 0, self.max_n_services)
        
        # Apply embedding to MS IDs
        # Shape: [batch_size, n_classes, embedding_dim]
        embedded_ms_ids = self.ms_embedding(ms_ids)
        
        # Flatten embedded MS IDs: [batch_size, n_classes * embedding_dim]
        embedded_ms_ids = embedded_ms_ids.view(batch_size, -1)
        
        # Concatenate all features
        # Final shape: [batch_size, n_classes * embedding_dim + 2 * n_classes]
        features = th.cat([embedded_ms_ids, dependency_row, dependency_col], dim=1)
        if self.observation_components.get("service_qualities"):
            features = th.cat([features, service_qualities], dim=1)

        return features