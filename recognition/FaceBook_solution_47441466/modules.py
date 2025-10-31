#Components of the model architecture
# pip install torch torchvision torchaudio torch-geometric
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GCNConv
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


#used this resource to understand GCNs better, as wel as code structure/example
# https://www.geeksforgeeks.org/deep-learning/graph-convolutional-networks-gcns-architectural-insights-and-applications/

#basic GCN model for now, can flesh out later
class GCN(nn.Module):
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


