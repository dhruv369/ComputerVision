import torch
import tqdm
import torch.nn.functional as F
from torch import optim 
from mnist_datasets import MNISTDataset
from mnist_model import MNISTModel
import matplotlib.pyplot as plt
import json
import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter


# Custom EarlyStopping class from scratch
class EarlyStopping:
    def __init__(self, model, patience=5, delta=0.001, path='checkpoint.pt', verbose=True):
        self.model = model
        self.patience = patience
        self.delta = delta
        self.path = path
        self.verbose = verbose
        self.best_loss = None
        self.no_improvement_count = 0
      
    def check_early_stop(self, val_loss):
        if self.best_loss is None or val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.no_improvement_count = 0
            # Save checkpoint if improvement observed
            torch.save(self.model.state_dict(), self.path)
            if self.verbose:
                print(f"Model improved; checkpoint saved at loss {val_loss:.4f}")
        else:
            self.no_improvement_count += 1
            if self.no_improvement_count >= self.patience:
                print("Early stopping triggered.")
                return True  # Signal to stop training
        return False

# Trainer class to handle the training loop
class Trainer:
    def __init__(self, model, device, train_loader, val_loader, optimizer, model_name="mnist_model"):
        self.model = model
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer  
        self.model_name = model_name
        self.writer = SummaryWriter("runs/mnist_experiment")
        self.checkpoint_path = f'mnist/model_weights/{model_name}_checkpoint.pth'
        self.metrics_path = f'mnist/model_weights/{model_name}_metrics.json'
        os.makedirs(os.path.dirname(self.checkpoint_path), exist_ok=True)
        self.early_stopping = EarlyStopping(self.model, patience=5, delta=0.001, path=self.checkpoint_path, verbose=True)
        self.train_losses = []  # List to store training losses for each epoch
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []


    def calculate_accuracy(self):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in self.train_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
            val_accuracies = correct / total
            self.val_accuracies.append(val_accuracies)
        return val_accuracies

    def calculate_loss(self):
        val_loss = 0
        self.model.eval()
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                val_loss += F.cross_entropy(output, target).item()
            avg_val_loss = val_loss / len(self.val_loader)
            self.val_losses.append(avg_val_loss)
        return avg_val_loss

    def train(self, epochs):
        
        # Set the model to training mode
        self.model.train()
        for epoch in range(1, epochs + 1):  # Train for the specified number of epochs
            train_loss = 0
            total = 0
            # Loop over the training data in batches
            for batch_idx, (data, target) in enumerate(tqdm.tqdm(self.train_loader)):
                data, target = data.to(self.device), target.to(self.device)
                # Zero the gradients, perform a forward pass, compute the loss, backpropagate, and update the model parameters
                self.optimizer.zero_grad()
                # Forward pass: compute the model output for the current batch of data
                output = self.model(data)
                total += target.size(0)
                # Compute the loss using cross-entropy
                loss = F.cross_entropy(output, target)
                train_loss += loss.item()
                
                # Backpropagate the loss and update the model parameters
                loss.backward()
                self.optimizer.step()

            # Calculate average training loss
            avg_train_loss = train_loss / len(self.train_loader)
            self.train_losses.append(avg_train_loss)

            # Calculate validation loss and accuracy
            avg_val_loss = self.calculate_loss()
            val_accuracy = self.calculate_accuracy()
            self.writer.add_scalar("Loss/train", avg_train_loss, epoch)
            print(f"Epoch {epoch}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
            
            # Check early stopping
            early_stop = self.early_stopping.check_early_stop(avg_val_loss)
            if early_stop:
                print(f"Early stopping at epoch {epoch}")
                break

        # Save metrics
        metrics = {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "val_accuracies": self.val_accuracies,
            "epochs": list(range(1, len(self.train_losses) + 1))
        }
        with open(self.metrics_path, 'w') as f:
            json.dump(metrics, f)
        self.writer.close()

    def plot_training_loss(self):
        # Plot the training loss over epochs
        plt.figure(figsize=(10, 5))
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss Over Epochs')
        plt.legend()
        plt.savefig(f'mnist/model_weights/{self.model_name}_loss_plot.png')
        plt.close()

    def plot_accuracy(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.val_accuracies, label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Validation Accuracy Over Epochs')
        plt.legend()
        plt.savefig(f'mnist/model_weights/{self.model_name}_acc_plot.png')
        plt.close()

def main():
    # Set device to GPU if available, otherwise use CPU
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    # Set the number of epochs for training
    epochs = 5
    learn_rate = 0.001
    model_name = "mnist_model"  # Can be changed for different runs

    # Load the MNIST dataset
    train_dataset = MNISTDataset(train=True)
    test_dataset = MNISTDataset(train=False)
    train_loader = train_dataset.train_dataloader(batch_size=64, shuffle=True)
    val_loader = test_dataset.test_dataloader(batch_size=64, shuffle=False)  # Using test as validation

    # Initialize the model and move it to the appropriate device
    model = MNISTModel().to(device)

    # Set up the optimizer 
    optimizer = optim.Adam(model.parameters(), lr=learn_rate)

    # Create a Trainer instance and start training for a specified number of epochs
    trainer = Trainer(model, device, train_loader, val_loader, optimizer, model_name)
    trainer.train(epochs)
    # Metrics and plots are saved automatically


if __name__ == "__main__":  
    main()
