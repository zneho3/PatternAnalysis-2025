
""""
File: dataset.py
Description: Dataset classes for loading and preprocessing graph data using PyTorch Geometric.
Normally this would be done using the npz files provided, but here we use the csv files directly since we dont have the npz files.
This will also mean that the data is not the simplified 128 feature version but the full sparse feature version.
Possibly can change this later but I think AWS being down might be causing issues with npz file host.
"""
import pandas as pd
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import to_networkx



class FacebookData:
    """
    PyTorch Geometric Dataset class for loading Facebook graph data from CSV files.
    """
    def __init__(self, file_path_prefix='/content/drive/MyDrive/COMP3710/A3/facebook/musae_facebook'):
        self.data = (file_path_prefix)
        self.train_mask, self.val_mask, self.test_mask = split_data(data, train_ratio=0.8, val_ratio=0.1)
        #split the data into train, val, test here 
        
        
    def load_data(file_path_prefix='/content/drive/MyDrive/COMP3710/A3/facebook/musae_facebook'):
        """
        Load graph data from CSV files and create a PyTorch Geometric Data object. Maybe can update to use npz files later.
        
        file_path_prefix (str): Prefix path to the folder containign the csv files
        Returns:
            data (Data): PyTorch Geometric Data object containing node features, edge indices, and labels.
        """
        
        
        # Load node features (sparse)
        node_df = pd.read_csv(f'{file_path_prefix}/musae_facebook_features.csv')

        # Get all unique nodes and features
        node_ids = node_df['node_id'].unique()
        feature_ids = node_df['feature_id'].unique()

        # Map node IDs to row indices
        node_map = {nid: i for i, nid in enumerate(node_ids)}
        feature_map = {fid: i for i, fid in enumerate(feature_ids)}

        # Create zero matrix
        x = torch.zeros((len(node_ids), len(feature_ids)), dtype=torch.float)

        # Fill in features
        for _, row in node_df.iterrows():
            node_idx = node_map[row['node_id']]
            feat_idx = feature_map[row['feature_id']]
            x[node_idx, feat_idx] = 1.0

        # Load edges
        edge_df = pd.read_csv(f'{file_path_prefix}/musae_facebook_edges.csv')

        # Filter edge_df to keep only edges where both nodes are in node_map
        edge_df_filtered = edge_df[edge_df['id_1'].isin(node_map) & edge_df['id_2'].isin(node_map)]

        # Map node IDs to indices for edge_index using the filtered edge_df
        edge_index = torch.tensor([[node_map[i] for i in edge_df_filtered['id_1']],
                                [node_map[i] for i in edge_df_filtered['id_2']]], dtype=torch.long)


        # Load labels
        label_df = pd.read_csv(f'{file_path_prefix}/musae_facebook_target.csv')
        # Only keep labels for nodes that exist in node_map
        label_df = label_df[label_df['id'].isin(node_map)]
        # Map node IDs to indices
        y = torch.zeros(len(node_ids), dtype=torch.long) - 1  # -1 for unlabeled
        le = LabelEncoder()
        label_df['y'] = le.fit_transform(label_df['page_type'])
        for _, row in label_df.iterrows():
            y[node_map[row['id']]] = row['y']

        # Create PyG data object
        data = Data(x=x, edge_index=edge_index, y=y)
        return data
    
    def split_data(self, data, train_ratio=0.8, val_ratio=0.1):
        """
        Split the data into training, validation, and test sets.
        
        Args:
            data (Data): PyTorch Geometric Data object.
            train_ratio (float): Proportion of data to use for training.
            val_ratio (float): Proportion of data to use for validation.
            
        Returns:
            train_mask, val_mask, test_mask (torch.BoolTensor): Boolean masks for train, val, and test sets.
        """
        num_nodes = data.num_nodes
        indices = torch.randperm(num_nodes)

        train_end = int(train_ratio * num_nodes)
        val_end = int((train_ratio + val_ratio) * num_nodes)

        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)

        train_mask[indices[:train_end]] = True
        val_mask[indices[train_end:val_end]] = True
        test_mask[indices[val_end:]] = True

        return train_mask, val_mask, test_mask
    
    