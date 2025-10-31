"""
Author: Zachariah Nehow
Student Number: S4744146
Description: Trainer class for training, predicting, and visualizing a GCN model on the Facebook dataset.
Includes early stopping based on validation loss.
"""

import torch
from modules import GCN
import matplotlib.pyplot as plt
import random
import numpy as np
from sklearn.manifold import TSNE
from dataset import FacebookData
from config import SEED, NUM_EPOCHS, LEARNING_RATE, MODEL_PATH, DATA_PATH_PREFIX, LOSS_BUFFER, TRAIN_RATIO, VAL_RATIO, TEST_RATIO

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Set seed for reproducibility 
if SEED is None:
    SEED = random.randint(0, 10000)
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Trainer:
    """
    Class to handle training, prediction, and visualization of the GCN model on the Facebook dataset.
    Includes early stopping based on validation loss.
    All filepaths are set in config.py
    This can be set to prediction mode to load a pre-trained model for inference.
    Heavy encapsulation to allow easy use in both scripts and notebooks
    """
    def __init__(self, model_path=MODEL_PATH, data_path_prefix=DATA_PATH_PREFIX, predict_mode=False):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # load dataset (load_data will attach train/val/test masks)
        dataset = FacebookData(data_path_prefix)
        data = dataset.load_data().to(self.device)
        self.data = data

        # infer input / output sizes
        in_channels = int(self.data.x.size(1))
        out_channels = int(self.data.y[self.data.y >= 0].max().item()) + 1 if (self.data.y >= 0).any() else 0

        self.model = GCN(in_channels=in_channels, hidden_channels=16, out_channels=out_channels).to(self.device)
        # load weights only if a path was provided
        if predict_mode and model_path is not None:
            state = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state)

        self.num_epochs = NUM_EPOCHS
        self.learning_rate = LEARNING_RATE

    def train(self):
        """
        Train the model over NUM_EPOCHS. Checks the validation loss after each epoch and 
        if no improvement, stop training early. Saves the model to MODEL_PATH specified in config.py.
        """
        model = self.model
        data = self.data.to(self.device)
        train_mask, val_mask = data.train_mask, data.val_mask
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        criterion = torch.nn.CrossEntropyLoss()
        train_losses = []
        val_losses = []
        
        best_val_loss = float('inf')
        stagnant_epochs = 0
        
        
        for epoch in range(self.num_epochs):
            model.train()
            optimizer.zero_grad()
            out = model(data.x, data.edge_index)
            loss = criterion(out[train_mask], data.y[train_mask])
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

            model.eval()
            with torch.no_grad():
                val_out = model(data.x, data.edge_index)
                val_loss = criterion(val_out[val_mask], data.y[val_mask])
                val_losses.append(val_loss.item())
                
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                stagnant_epochs = 0
            else:
                stagnant_epochs += 1
                
            if stagnant_epochs >= LOSS_BUFFER:
                print(f'Early stopping at epoch {epoch+1} due to no improvement in validation loss for {LOSS_BUFFER} epochs.')
                torch.save(model.state_dict(), MODEL_PATH)
                print(f"Saved new best model to {MODEL_PATH}.")
                break
            if (epoch+1) % 10 == 0:
                print(f'Epoch {epoch+1}, Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}')

            #Save model at last epoch if not early stopped
            if epoch == self.num_epochs - 1:
                torch.save(model.state_dict(), MODEL_PATH)
                print(f"Saved final model to {MODEL_PATH}.")
                
        #Plot training and validation losses        
        plt.plot(range(len(train_losses)), train_losses, label='Train Loss')
        plt.plot(range(len(val_losses)), val_losses, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title(f'Training and Validation Loss over {epoch+1} epochs, \n Data split - {TRAIN_RATIO*100:.0f}:{VAL_RATIO*100:.0f}:{TEST_RATIO*100:.0f}')
        plt.legend()
        plt.savefig('loss_plot.png')

    def get_model(self):
        """
        Getter function to return the model 
        """
        return self.model
    
    def get_data(self):
        """
        Getter function to return the data
        """
        return self.data
    
    
    
    #Reference: https://www.datacamp.com/tutorial/comprehensive-introduction-graph-neural-networks-gnns-tutorial
    def visualise(self, output, labels):
        """
        output: embeddings from the model output
        labels: true labels for coloring
        
        Visualise node embeddings using t-SNE. Can be used for both training and prediction outputs.
        """
        z = TSNE(n_components=2).fit_transform(output.detach().cpu().numpy())
        plt.figure(figsize=(10, 10))
        plt.xticks([])
        plt.yticks([])
        plt.scatter(z[:, 0], z[:, 1], s=70, c=labels, cmap="Set2")
        plt.colorbar()
        plt.title("t-SNE plot of Node Embeddings")
        plt.xlabel("tSNE dimension 1")
        plt.ylabel("tSNE dimension 2")
        plt.savefig('tSNE.png')
        plt.legend()
        plt.show()
    
    def print_data_shapes(self):
        """
        Debugging function. Can most likely remove in final version
        """
        # model = self.model
        data = self.data
        test_mask = data.test_mask
        print("data.x:", data.x.shape)
        print("data.y:", data.y.shape)
        print("test_mask:", test_mask.shape)
        
    def predict(self):
        """
        Predict on the test set using the loaded model.
        Returns predictions, true labels, and embeddings for visualization.
        """
        model = self.model
        data = self.data
        test_mask = data.test_mask
        # print("data.x:", data.x.shape)
        # print("data.y:", data.y.shape)
        # print("test_mask:", test_mask.shape)

        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index)
            filtered_out = out[test_mask]
            labels = data.y[test_mask].cpu()
            pred = filtered_out.argmax(dim=1).cpu()
            correct = (pred == labels).sum().item()
            accuracy = correct / len(labels)
            print(f'Test Accuracy: {accuracy:.4f}')
            return (pred, labels, filtered_out)


if __name__ == "__main__":
    
    # All training is encapsulated in the Trainer class
    trainer = Trainer(data_path_prefix=DATA_PATH_PREFIX)
    trainer.train()
