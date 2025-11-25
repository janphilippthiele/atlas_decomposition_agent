import numpy as np
from scipy.sparse import csr_matrix
from rl_microservice_decomposer.utils.matrix_loader import load_adjacency_matrix
import time

class SparseVectorizedMetricsCalculator:
    
    def __init__(self, adjacency_matrix: np.ndarray, max_microservices: int):
        """
        Initialize MetricsCalculator using sparse matrices for efficient computation.
        
        Args:
            adjacency_matrix: The adjacency matrix representing class interactions
            max_microservices: Maximum number of microservices (not including unassigned/monolith)
        """
        # Convert dense adjacency matrix to sparse CSR format for efficient operations
        self.adjacency_matrix = csr_matrix(adjacency_matrix, dtype=np.float32)
        self.n_classes = adjacency_matrix.shape[0]
        self.max_microservices = max_microservices
        
        # Convert to binary sparse matrix for modularity calculation
        self.binary_adjacency = (self.adjacency_matrix > 0).astype(np.float32)

    def calculate_state_metrics(self, state: np.ndarray) -> dict:
        # Get all unique services actually used in this state (including 0 if present)
        unique_services = np.unique(state)
        
        # Create one-hot encoding for actually used services (keep as dense for now)
        assignment_matrix = self.create_service_assignment_matrix_sparse(state, include_monolith=False)

        density_term, coupling_term, densities, service_couplings = self._calculate_cohesion_and_coupling(assignment_matrix)
        
        # conductance = self._calculate_conductance(microservices_onehot)

        decomposition_quality = density_term - coupling_term

        return dict(
            decomposition_quality=decomposition_quality,
            density=density_term,
            coupling=coupling_term,
            densities=densities,
            couplings=service_couplings
        )
    
    def create_service_assignment_matrix_sparse(self, state, include_monolith=False):
        """Create sparse service assignment matrix and active services mask."""
        
        assigned_obj = state[state > 0]
        n_obj = len(state)
        # Standard matrix with service 0 included
        row_indices = np.arange(n_obj)
        col_indices = state
        data = np.ones(n_obj, dtype=np.int32)

        S_sparse = csr_matrix((data, (row_indices, col_indices)), 
                                shape=(n_obj, np.max(state) + 1), dtype=np.int32)

        return S_sparse

    def _calculate_cohesion_and_coupling(self, S: csr_matrix) -> tuple:
        """
        Vectorized calculation of the SM formula:
        SM = (1/K)Σ(u_k/N_k²) - (1/K(K-1))Σ(σ_{k₁,k₂}/(N_{k₁}N_{k₂}))
        
        Modified to handle directed graphs by counting edges in both directions.
        Only considers assigned services (excludes service 0).
        
        Args:
            S: Sparse service assignment matrix [n_objects, n_services]
        Returns:
            Tuple of (density_term, coupling_term, densities, individual_couplings)
        """
        
        # Compute interaction matrix: σ values for all service pairs
        interaction_matrix = S.T @ self.adjacency_matrix @ S
        
        # Convert to dense for easier manipulation
        interaction_dense = interaction_matrix.toarray()
        
        # Service sizes: N_k
        service_sizes = np.asarray(S.sum(axis=0)).flatten()  # [K]
        
        # Exclude service 0 (unassigned objects) from all calculations
        interaction_dense = interaction_dense[1:, 1:]  # Remove first row and column
        service_sizes = service_sizes[1:]  # Remove service 0
        
        K = np.sum(service_sizes > 0)
        
        # FIRST TERM: (1/K)Σ(u_k/N_k²)
        internal_edges = interaction_dense.diagonal()  # u_k
        
        # Avoid division by zero
        valid_services = service_sizes > 0
        densities = np.zeros(len(service_sizes))
        densities[valid_services] = internal_edges[valid_services] / (service_sizes[valid_services] ** 2)

        valid_service_sizes = service_sizes[valid_services]
        weights = valid_service_sizes / np.sum(valid_service_sizes)  # Weight by service size

        density_term = np.average(densities[valid_services], weights=weights) if K > 0 else 0.0
        
        # SECOND TERM: Modified for directed graphs
        service_sizes_col = service_sizes.reshape(-1, 1)  # [K, 1]
        service_sizes_row = service_sizes.reshape(1, -1)  # [1, K]
        normalization_matrix = service_sizes_col * service_sizes_row  # [K, K]
        
        # Avoid division by zero
        normalization_matrix = np.where(normalization_matrix == 0, 1, normalization_matrix)
        
        # Normalize the entire interaction matrix
        normalized_interactions = interaction_dense / normalization_matrix
        
        # Sum the entire matrix to get all directed edges between all service pairs
        total_sum = normalized_interactions.sum()
        
        # Subtract the diagonal because we only want coupling between different services
        coupling_sum = total_sum - normalized_interactions.diagonal().sum()

        # Individual couplings
        if K > 1:
            row_sums = normalized_interactions.sum(axis=1) - normalized_interactions.diagonal()
            col_sums = normalized_interactions.sum(axis=0) - normalized_interactions.diagonal()
            individual_couplings = (row_sums + col_sums) / (2 * (K - 1))
        else:
            individual_couplings = np.zeros(len(service_sizes))

        # Divide by K(K-1)
        num_ordered_pairs = K * (K - 1)
        coupling_term = coupling_sum / num_ordered_pairs if num_ordered_pairs > 0 else 0
        
        return density_term, coupling_term, densities, individual_couplings

    def _calculate_conductance(self, microservices_onehot: np.ndarray) -> float:
        """
        Calculate conductance using sparse matrix operations.
        
        Args:
            microservices_onehot: One-hot encoded microservice assignments
        Returns:
            Average conductance across all services
        """
        # Compute interaction matrix between services using sparse operations
        # interaction_matrix[i,j] = total edges from service i to service j
        temp = self.binary_adjacency.dot(microservices_onehot)  # Sparse @ dense
        interaction_matrix = microservices_onehot.T.dot(temp)  # Dense @ dense
        
        # Compute service volumes (total out-degree for each service)
        # Using sparse matrix sum is more efficient
        node_degrees = np.asarray(self.binary_adjacency.sum(axis=1)).flatten()  # [n_classes]
        service_volumes = microservices_onehot.T @ node_degrees  # [n_services]
        
        # Compute cuts (outgoing edges from each service to all other services)
        service_cuts = interaction_matrix.sum(axis=1) - interaction_matrix.diagonal()
        
        # Total volume of entire graph
        total_volume = node_degrees.sum()
        
        # Compute complement volumes for each service
        complement_volumes = total_volume - service_volumes
        
        # Compute conductance: cut(S, S̄) / min(vol(S), vol(S̄))
        min_volumes = np.minimum(service_volumes, complement_volumes)
        
        # Avoid division by zero
        min_volumes = np.where(min_volumes > 0, min_volumes, 1.0)
        
        # Calculate conductance for each service
        conductances = service_cuts / min_volumes
        
        # Return average conductance across all services
        return np.mean(conductances)
    
    def call_graph_sparsity(self):
        """Calculate sparsity of the adjacency matrix."""
        # For sparse matrices, use nnz (number of non-zero elements)
        total_elements = self.n_classes * self.n_classes
        non_zero_elements = self.adjacency_matrix.nnz
        sparsity = 1 - (non_zero_elements / total_elements)
        return sparsity


if __name__ == "__main__":
    adj_matrix = load_adjacency_matrix(
        r"C:\Git\atlas_decomposition_agent\data\application_data\roller\class_interactions.parquet", 
        'binary'
    )

    np.random.seed(42)
    # Randomly reorder rows
    adj_matrix = np.random.permutation(adj_matrix)

    n_classes = adj_matrix.shape[0]

    random_decomposition = np.random.randint(0, 5, size=n_classes)
    sparse_metrics_calculator = SparseVectorizedMetricsCalculator(adj_matrix, 0)

    start_time = time.time()
    reward_dict = sparse_metrics_calculator.calculate_state_metrics(random_decomposition)
    sparse_time = time.time() - start_time

    print(f"Sparse Vectorized reward: {reward_dict['decomposition_quality']:.9f}, Time Taken: {sparse_time:.4f}")
    print(f"Cohesion: {reward_dict['density']:.9f}, Coupling: {reward_dict['coupling']:.9f}")
    print(f"Sparsity: {sparse_metrics_calculator.call_graph_sparsity():.9f}")