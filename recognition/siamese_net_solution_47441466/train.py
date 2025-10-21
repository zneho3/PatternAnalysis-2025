import torch
from dataset import FacebookData
from modules import GCN

#Training, validation, testing and saving is done here
#imports from dataset.py and modules.py
#make sure to plot losses and metrics during training
# Check if there is a prebuilt loss function for graph tasks in PyTorch Geometric
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

#import dataset and model
dataset = FacebookData()
model = GCN(in_channels=dataset.data.num_node_features, hidden_channels=64, out_channels=32).to(device)

#train below