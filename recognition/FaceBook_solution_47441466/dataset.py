
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
from torch_geometric.utils import to_undirected



class FacebookData:
    """
    PyTorch Geometric Dataset class for loading Facebook graph data from CSV files.
    """
    def __init__(self, file_path_prefix='/content/drive/MyDrive/COMP3710/A3/facebook'):
        # store the path; actual Data object created by load_data()
        self.file_path_prefix = file_path_prefix
        self.data = None
        self.train_mask = None
        self.val_mask = None
        self.test_mask = None
        
        
    def load_data(self, file_path_prefix=None):
        """
        Load graph data from CSV files and create a PyTorch Geometric Data object. Maybe can update to use npz files later.
        Microsoft Copilot generated most of this function as I was couldnt find any references online for loading from csv files,
        since normally the data is in npz format.
        file_path_prefix (str): Prefix path to the folder containign the csv files
        Returns:
            data (Data): PyTorch Geometric Data object containing node features, edge indices, and labels.
        """
        
        
        # allow override, otherwise use stored prefix
        if file_path_prefix is None:
            file_path_prefix = self.file_path_prefix

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
        # Load edges and map to indices
        edge_df = pd.read_csv(f'{file_path_prefix}/musae_facebook_edges.csv')
        edge_df_filtered = edge_df[edge_df['id_1'].isin(node_map) & edge_df['id_2'].isin(node_map)]
        src = [node_map[i] for i in edge_df_filtered['id_1']]
        dst = [node_map[i] for i in edge_df_filtered['id_2']]
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        # make undirected
        edge_index = to_undirected(edge_index)

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

        # create train/val/test masks (only among labeled nodes) and attach to data
        self.train_mask, self.val_mask, self.test_mask = self.split_data(data, train_ratio=0.8, val_ratio=0.1)
        data.train_mask = self.train_mask
        data.val_mask = self.val_mask
        data.test_mask = self.test_mask

        self.data = data
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

        # Only split among labeled nodes (y >= 0)
        if not hasattr(data, 'y'):
            raise ValueError('Data object must contain labels in data.y to create masks')

        labeled = (data.y >= 0).nonzero(as_tuple=True)[0]
        if labeled.numel() == 0:
            # no labels available; return empty masks
            train_mask = torch.zeros(num_nodes, dtype=torch.bool)
            val_mask = torch.zeros(num_nodes, dtype=torch.bool)
            test_mask = torch.zeros(num_nodes, dtype=torch.bool)
            return train_mask, val_mask, test_mask

        perm = labeled[torch.randperm(labeled.size(0))]
        n = perm.size(0)
        train_end = int(train_ratio * n)
        val_end = int((train_ratio + val_ratio) * n)

        train_idx = perm[:train_end]
        val_idx = perm[train_end:val_end]
        test_idx = perm[val_end:]

        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)

        train_mask[train_idx] = True
        val_mask[val_idx] = True
        test_mask[test_idx] = True

        return train_mask, val_mask, test_mask
    
    