"""
Author: Zachariah Nehow
Student Number: S4744146
Description: Module defining the Graph Neural Network (GNN) model architecture using PyTorch Geometric.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GCNConv
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


#Reference: https://www.datacamp.com/tutorial/comprehensive-introduction-graph-neural-networks-gnns-tutorial
class GCN(nn.Module):
    """
    Graph Convolutional Network (GCN) model for node classification.
    Two graph convolutional layers with ReLU activation in between.
    """
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        #change later if we need things like normalization, dropout, etc.
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.relu = nn.ReLU()

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.conv2(x, edge_index)
        return x


