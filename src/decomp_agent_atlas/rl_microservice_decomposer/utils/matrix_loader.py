import pandas as pd
import numpy as np
import os
from pathlib import Path

def load_adjacency_matrix(file_path, normalization='binary'):

    if '.csv' in file_path:
        call_graph_df = pd.read_csv(file_path, header=None)
    elif '.parquet' in file_path:
        call_graph_df = pd.read_parquet(file_path)
    else:
        raise ValueError("Unsupported file format. Please provide a .csv or .parquet file.")

    adj_matrix = call_graph_df.to_numpy(dtype=int)
    
    if normalization.lower() == 'binary':
        "Change all entries to ones if they are not zero"
        adj_matrix[adj_matrix != 0] = 1
    elif normalization.lower() == 'symmetrical':
        "Perform symmetric normalization A_n = D^(−1/2)*A*D^(-1/2) with degree matrix D"
        adj_matrix = adj_matrix.astype(float)
        A_tilde=np.eye(adj_matrix.shape[0])+adj_matrix
        degrees = np.sum(A_tilde, axis=1)
        degrees[degrees == 0] = 1  # Handle isolated nodes
        D_inv = np.diag(np.power(degrees, -0.5))  
        adj_matrix = D_inv @ A_tilde @ D_inv
    elif normalization.lower() == 'row':
        "Normalize per each row"
        adj_norm = adj_matrix.astype(float)
        row_sums = adj_norm.sum(axis=1)        
        row_sums[row_sums == 0] = 1
        adj_matrix = adj_norm / row_sums[:, np.newaxis]
    else:
        "No normalization applied"
        adj_matrix = adj_matrix.astype(float)

    return adj_matrix

def load_legacy_system_help_matrices(input_dir=None, dataset_name = 'legacy_system'):
    """
    Load matrices and metadata for training.
    Returns:
        Tuple of (matrices_dict, metadata_dict)
    """
    input_path = r'C:\Git\atlas_decomposition_agent\data\application_data\legacy_system'
    
    # Load numpy arrays
    if dataset_name == 'legacy_system':
        data = np.load(input_path + '', allow_pickle=True)
    elif dataset_name == 'legacy_system_shuffled':
        data = np.load(input_path + '', allow_pickle=True)
    else:
        raise ValueError('Unkown dataset name for legacy_system training.')

    help_matrices = {key: data[key] for key in data.files}

    return help_matrices

