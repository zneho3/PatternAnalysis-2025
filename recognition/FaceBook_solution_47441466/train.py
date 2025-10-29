import torch
from dataset import FacebookData
from modules import GCN
import matplotlib.pyplot as plt
from config import TRAIN_RATIO, VAL_RATIO, TEST_RATIO, SEED

#set seed for reproducibility
# torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class Trainer:
    def __init__(self, model_path='best_model.pth', data_path_prefix='/content/drive/MyDrive/COMP3710/A3/facebook/musae_facebook'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.data = FacebookData(data_path_prefix).load_data().to(self.device)
        self.data.train_mask, self.data.val_mask, self.data.test_mask = FacebookData(data_path_prefix).train_mask, \
            FacebookData(data_path_prefix).val_mask, \
            FacebookData(data_path_prefix).test_mask
        self.model = GCN(in_channels=self.data.num_features, hidden_channels=16, out_channels=self.data.y.max().item() + 1).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    def train(self):
        #train below
        model = self.model
        data = self.data.to(device)
        train_mask, val_mask, test_mask = data.split_data(data, TRAIN_RATIO, VAL_RATIO)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = torch.nn.CrossEntropyLoss()
        num_epochs = 100
        train_losses = []
        val_losses = []

        for epoch in range(num_epochs):
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
        plt.plot(range(num_epochs), train_losses, label='Train Loss')
        plt.plot(range(num_epochs), val_losses, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('loss_plot.png')
        
        # Evaluate on test set
        model.eval()
        with torch.no_grad():
            test_out = model(data.x, data.edge_index)
            pred = test_out[test_mask].argmax(dim=1)
            correct = (pred == data.y[test_mask]).sum().item()
            accuracy = correct / test_mask.sum().item()
            print(f'Test Accuracy: {accuracy:.4f}')

    def get_model(self):
        return self.model
    
    def get_data(self):
        return self.data
