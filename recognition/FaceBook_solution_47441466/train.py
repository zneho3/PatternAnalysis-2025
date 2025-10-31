import torch
from dataset import FacebookData
from modules import GCN
import matplotlib.pyplot as plt
from config import TRAIN_RATIO, VAL_RATIO, TEST_RATIO, SEED, NUM_EPOCHS, LEARNING_RATE
from sklearn.manifold import TSNE
#set seed for reproducibility
# torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")


class Trainer:
    def __init__(self, model_path=None, data_path_prefix='/content/drive/MyDrive/COMP3710/A3/facebook'):
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
        if model_path:
            state = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state)

        self.num_epochs = NUM_EPOCHS
        self.learning_rate = LEARNING_RATE

    def train(self):
        #train below
        model = self.model
        data = self.data.to(self.device)
        train_mask, val_mask, test_mask = data.train_mask, data.val_mask, data.test_mask
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        criterion = torch.nn.CrossEntropyLoss()
        train_losses = []
        val_losses = []

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

            if (epoch+1) % 10 == 0:
                print(f'Epoch {epoch+1}, Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}')
        # Save the trained model
        torch.save(model.state_dict(), 'gcn_facebook_model.pth')
        # Plot training and validation losses
        plt.plot(range(self.num_epochs), train_losses, label='Train Loss')
        plt.plot(range(self.num_epochs), val_losses, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('loss_plot.png')


        #do this in predict.py later        
        # Evaluate on test set
        # model.eval()
        # with torch.no_grad():
        #     test_out = model(data.x, data.edge_index)
        #     pred = test_out[test_mask].argmax(dim=1)
        #     correct = (pred == data.y[test_mask]).sum().item()
        #     accuracy = correct / test_mask.sum().item()
        #     print(f'Test Accuracy: {accuracy:.4f}')

    def get_model(self):
        return self.model
    
    def get_data(self):
        return self.data
    

    def visualise(self, output, labels):
        """
        model_out: output from model (embeddings)
        labels: true labels for coloring
        """
        z = TSNE(n_components=2).fit_transform(output.detach().cpu().numpy())
        plt.figure(figsize=(10, 10))
        plt.xticks([])
        plt.yticks([])
        plt.scatter(z[:, 0], z[:, 1], s=70, c=labels, cmap="Set2")
        plt.colorbar()
        plt.title("t-SNE of Node Embeddings")
        plt.xlabel("tSNE dimension 1")
        plt.ylabel("tSNE dimension 2")
        plt.savefig('tSNE.png')
        plt.legend()
        plt.show()
    
    def print_data_shapes(self):
        # model = self.model
        data = self.data
        test_mask = data.test_mask
        print("data.x:", data.x.shape)
        print("data.y:", data.y.shape)
        print("test_mask:", test_mask.shape)
        
    def predict(self):
        model = self.model
        data = self.data
        test_mask = data.test_mask
        print("data.x:", data.x.shape)
        print("data.y:", data.y.shape)
        print("test_mask:", test_mask.shape)

        model.eval()
        with torch.no_grad():
            # print("Starting forward pass")
            out = model(data.x, data.edge_index)
            # print("Forward pass complete")
            filtered_out = out[test_mask]
            labels = data.y[test_mask].cpu()
            # print(f"Labels extracted: {len(labels)}")
            pred = filtered_out.argmax(dim=1).cpu()
            # print(f"Predictions extracted: {len(pred)}")
            correct = (pred == labels).sum().item()
            accuracy = correct / len(labels)
            print(f'Test Accuracy: {accuracy:.4f}')
            # print(f"Pred and Labels: {pred}, {labels}")
            # print("Returning results...")
            # Return the embeddings (filtered_out) for visualization, along with pred and labels
            return (pred, labels, filtered_out)