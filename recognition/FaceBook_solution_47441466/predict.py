# show usage of the trained model. 
# print out results and provide visualizations 
import torch
import random
import numpy as np
from dataset import FacebookData
from modules import GCN
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


class Predictor:
    """
    Class to load a trained GCN model and make predictions on the Facebook graph data. Made a class for easier usage and encapsulation 
    and also to use in Google Colab later.
    """
    def __init__(self, model_path='best_model.pth', data_path_prefix='/content/drive/MyDrive/COMP3710/A3/facebook/musae_facebook'):
        
        self.data = FacebookData(data_path_prefix).load_data()
        self.data.train_mask, self.data.val_mask, self.data.test_mask = FacebookData(data_path_prefix).train_mask, \
            FacebookData(data_path_prefix).val_mask, \
            FacebookData(data_path_prefix).test_mask
        
        
        #model setup
        in_channels = self.data.num_features
        out_channels = int(self.data.y.max().item())  # Number of classes
        hidden_channels = 16
        dropout_rate = 0.5  # will i need this
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = GCN(in_channels=in_channels, hidden_channels=hidden_channels, out_channels=out_channels).to(device)
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device))
        self.model.eval()
        self.data = self.data.to(device)
        
    def predict(self):
        with torch.no_grad():
            out = self.model(self.data.x, self.data.edge_index)
            pred = out.argmax(dim=1)
            true_labels = self.data.y[self.data.test_mask].cpu()
        
        return pred[self.data.test_mask].cpu(), true_labels
        
    
    def get_accuracy(self):
        with torch.no_grad():
            out = self.model(self.data.x, self.data.edge_index)
            pred = out.argmax(dim=1)
            correct = (pred[self.data.test_mask] == self.data.y[self.data.test_mask]).sum()
            acc = int(correct) / int(self.data.test_mask.sum())
            print(f'Accuracy: {acc:.4f}')
            return acc

    def visualize(self, graph, color):
        """Visualise a 2D TSNE plot of the embeddings h with colors color.
        """
        z = TSNE(n_components=2).fit_transform(graph.detach().cpu().numpy())

        plt.figure(figsize=(10,10))
        plt.xticks([])
        plt.yticks([])

        plt.scatter(z[:, 0], z[:, 1], s=70, c=color, cmap="Set2")
        plt.show()



