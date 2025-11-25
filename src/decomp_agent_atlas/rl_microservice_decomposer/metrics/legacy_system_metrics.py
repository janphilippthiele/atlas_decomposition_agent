import numpy as np
from scipy.sparse import csr_matrix
from rl_microservice_decomposer.utils.matrix_loader import load_legacy_system_help_matrices

class LegacySystemMetricsCalculator:

    def __init__(self, adjacency_matrix: np.ndarray=None, dataset_name: np.ndarray = 'legacy_system'):

        help_matrices = load_legacy_system_help_matrices(dataset_name)

        self.adjacency_matrix = adjacency_matrix
        if adjacency_matrix is None:
            self.adjacency_matrix = self._extract_if_wrapped(help_matrices.get("adjacency_matrix", None))

        self.R = self._extract_if_wrapped(help_matrices.get("read_adjacency_matrix_sparse"))
        self.W = self._extract_if_wrapped(help_matrices.get("write_adjacency_matrix_sparse"))
        self.EP = self._extract_if_wrapped(help_matrices.get("execution_paths_adjacency_matrix_sparse"))
        self.C = self._extract_if_wrapped(help_matrices.get("call_adjacency_matrix_sparse"))

        self.can_read_mask = help_matrices.get("can_read_mask", None)
        self.can_write_mask = help_matrices.get("can_write_mask", None)
        self.readable_mask = help_matrices.get("readable_mask", None)
        self.writable_mask = help_matrices.get("writable_mask", None)

        self.hub_objects_vec = self._extract_if_wrapped(help_matrices.get("hub_objects_vector", None))
        self.hub_degrees_vec = self._extract_if_wrapped(help_matrices.get("hub_degrees_vector", None))

        # Pre-calculations
        self.RW = self.R + self.W
        self.total_hub_objects = np.sum(self.hub_degrees_vec)
        self.n_obj = self.adjacency_matrix.shape[0]

        self.cached_metric = None
        self.previous_state = None
        self.cached_S = None # Cached service assignment matrix
        self.cached_service_data = {} # Store per-service intermediate calculations

    def _extract_if_wrapped(self, value):
        # Helper function to extract npz properly
        """Extract object from 0-d numpy array if wrapped."""
        if isinstance(value, np.ndarray) and value.shape == () and value.dtype == object:
            return value.item()
        return value
    
    def calculate_state_metrics(self, state: np.ndarray, include_monolith: bool = False) -> dict:
        
        # This is the metrics for the end of episode reward
        service_assignment = self.create_service_assignment_matrix_sparse(state, include_monolith=include_monolith)

        # Array of each services read write density
        read_write_densities = self.calculate_read_write_density(service_assignment)

        # Array of each services call density
        call_densities = self.calculate_call_density(service_assignment)

        # Array of each services conductance
        conductances = self.calculate_conductances(service_assignment)

        # Array of each services hub concentration
        hub_concentrations = self.calculate_hub_concentration(service_assignment)

        # execution_path_isolation = self.calculate_execution_path_isolation(service_assignment, active_services)
        
        ned_penalty = self.calculate_ned(service_assignment)

        service_qualities = 1000*(0.7 * read_write_densities + 0.3 * call_densities) - 0.01 * conductances - 0.05 * hub_concentrations
        
        # weights = service_sizes / np.sum(service_sizes)
        service_quality = np.average(service_qualities)
        decomposition_quality = service_quality - 0.05 * ned_penalty 

        episode_reward_metrics = {
            "decomposition_quality": decomposition_quality,
            "read_write_densities": read_write_densities,
            "call_densities": call_densities,
            "conductances": conductances,
            "hub_concentrations": hub_concentrations,
            "read_write_density": np.average(read_write_densities),
            "call_density": np.average(call_densities),
            "conductance": np.average(conductances),
            "hub_concentration": np.average(hub_concentrations),
            "ned": ned_penalty
        }

        return episode_reward_metrics

    def create_service_assignment_matrix_sparse(self, state, include_monolith=False):
        """Create sparse service assignment matrix and active services mask."""

        n_services = np.max(state)
        # Exclude service 0 - only include assigned objects
        assigned_mask = state > 0
        row_indices = np.where(assigned_mask)[0]
        col_indices = state[assigned_mask] - 1  # Shift indices down (1->0, 2->1, etc.)
        data = np.ones(len(row_indices), dtype=np.int32)

        S_sparse = csr_matrix((data, (row_indices, col_indices)), 
                            shape=(self.n_obj, n_services), dtype=np.int32)
        
        return S_sparse
    
    def calculate_read_write_density(self, S):
        """
        Correct density calculation using masks properly.
        
        Args:
            S: Service assignment matrix (sparse)
            R, W: Actual operations (sparse)
            can_read_mask: 1D array where 1 = object can perform reads
            readable_mask: 1D array where 1 = object can be read from
            can_write_mask: 1D array where 1 = object can perform writes
            writable_mask: 1D array where 1 = object can be written to
        """
        # Actual operations (sparse)
        actual_interaction = S.T @ self.RW @ S
        actual_per_service = np.array(actual_interaction.diagonal()).flatten()
        
        # Count readers and readables per service
        readers_per_service = np.asarray(S.T @ self.can_read_mask).flatten()
        readable_per_service = np.asarray(S.T @ self.readable_mask).flatten()

        # Count writers and writables per service
        writers_per_service = np.asarray(S.T @ self.can_write_mask).flatten()
        writable_per_service = np.asarray(S.T @ self.writable_mask).flatten()

        # Possible operations within each service
        possible_reads = readers_per_service * readable_per_service
        possible_writes = writers_per_service * writable_per_service
        possible_per_service = possible_reads + possible_writes
        
        # Density calculation
        rw_density = np.divide(actual_per_service, possible_per_service,
                            where=possible_per_service > 0,
                            out=np.zeros_like(actual_per_service, dtype=float))
        
        return rw_density
    
    def calculate_call_density(self, S):
        """
        Args:
            S: Service assignment matrix (sparse)
            C: Actual operations (sparse)
            can_write_mask: 1D array where 1 = object can perform writes
        """
        # Actual operations (sparse)
        actual_interaction = S.T @ self.C @ S
        actual_per_service = np.array(actual_interaction.diagonal()).flatten()
        
        # Assuming right now that objects that can write can perform call as well
        callers_per_service = np.asarray(S.T @ self.can_write_mask).flatten()

        # Possible calls excluding self-calls
        possible_calls = callers_per_service ** 2 - callers_per_service
        
        # Density calculation
        call_density = np.divide(actual_per_service, possible_calls,
                            where=possible_calls > 0,
                            out=np.zeros_like(actual_per_service, dtype=float))

        return call_density

    def calculate_conductances(self, S):
        """
        Calculate conductance using sparse matrix operations.
        Args:
            microservices_onehot: One-hot encoded microservice assignments
        Returns:
            Average conductance across all services
        """
        # Compute interaction matrix between services using sparse operations
        # interaction_matrix[i,j] = total edges from service i to service j
        interactions_matrix = S.T @ self.adjacency_matrix @ S
        
        # Compute service volumes (total out-degree for each service)
        # Using sparse matrix sum is more efficient
        node_degrees = np.asarray(self.adjacency_matrix.sum(axis=1)).flatten()  # [n_classes]
        service_volumes = S.T @ node_degrees  # [n_services]

        # Compute cuts (outgoing edges from each service to all other services)
        service_cuts = np.asarray(interactions_matrix.sum(axis=1)).flatten() - interactions_matrix.diagonal()
        
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
        return conductances
    
    def calculate_ned(self, S, min_size=10, max_size=150):
        # Service sizes: n_k
        service_sizes = np.asarray(S.sum(axis=0)).flatten() 
        
        # Total number of objects: |V|
        K =  np.sum(service_sizes > 0)
        
        # Mask for non-extreme services: size within [min_size, max_size]
        not_extreme_mask = (service_sizes >= min_size) & (service_sizes <= max_size)
        
        # Sum of objects in non-extreme services
        services_non_extreme = len(service_sizes[not_extreme_mask])
        
        # NED formula
        ned = services_non_extreme / K

        return 1-ned

    def calculate_hub_concentration(self, S):
        hubs_in_service = S.T @ self.hub_objects_vec  # Shape: [n_services] or [n_services, 1]
        hubs_per_service = np.asarray(hubs_in_service).flatten()
        
        if self.total_hub_objects == 0:
            return np.zeros(len(hubs_per_service))
        
        hub_concentration = hubs_per_service / self.total_hub_objects
        return hub_concentration