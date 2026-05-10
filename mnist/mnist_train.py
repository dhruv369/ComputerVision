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
    def __init__(self, model, device, train_loader, val_loader, optimizer, model_name="mnist_model", run_name="default"):
        self.model = model
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer  
        self.model_name = model_name
        self.run_name = run_name+datetime.now().strftime('%Y%m%d_%H%M%S')
        self.base_dir = "runs/experiments"
        self.run_dir = os.path.join(self.base_dir, self.run_name)
        self.ckpt_dir = os.path.join(self.run_dir, f'mnist/model_weights/{model_name}_checkpoints')
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.log_dir = os.path.join(self.run_dir, f'mnist/model_weights/{model_name}_logs')
        self.writer = SummaryWriter(self.run_dir)

        self.checkpoint_path = os.path.join(self.ckpt_dir, f'{model_name}_checkpoint.pth')
        self.metrics_path = os.path.join(self.log_dir, f'{model_name}_metrics.json')

        self.early_stopping = EarlyStopping(self.model, patience=5, delta=0.001, path=self.checkpoint_path, verbose=True)

        self.metrics = {
            "epochs": [],
            "train_losses": [],
            "train_accuracies": [],
            "val_losses": [],
            "val_accuracies": [],
            "learning_rate": optimizer.param_groups[0]['lr'],
            "gradient_norms": [],
            "epoch_durations": [],
        }

    def calculate_training_accuracy(self, correct, total):
        train_accuracy = correct / total if total > 0 else 0
        self.metrics["train_accuracies"].append(train_accuracy)
        return train_accuracy
    
    def calculate_training_loss(self, train_loss):
        train_loss = train_loss / len(self.train_loader) if len(self.train_loader) > 0 else 0
        self.metrics["train_losses"].append(train_loss)
        return train_loss

    def calculate_validation_accuracy_loss(self):
        self.model.eval()
        correct = 0
        total = 0
        val_loss = 0
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                total += target.size(0)
                # Forward pass: compute the model output for the current batch of validation data
                output = self.model(data)
                pred = output.argmax(dim=1, keepdim=True)
                # Calculate the number of correct predictions for the batch and accumulate it
                correct += pred.eq(target.view_as(pred)).sum().item()
                # Calculate validation loss for the batch and accumulate it
                val_loss += F.cross_entropy(output, target).item()
            # Calculate average validation loss and store it
            validation_loss = val_loss / len(self.val_loader)
            self.metrics["val_losses"].append(validation_loss)    
            # Calculate validation accuracy and store it
            validation_accuracies = correct / total
            self.metrics["val_accuracies"].append(validation_accuracies)
        return validation_accuracies, validation_loss

    def train(self, epochs):
        
        # Set the model to training mode
        self.model.train()
        for epoch in range(1, epochs + 1):  # Train for the specified number of epochs
            train_current_loss = 0
            total = 0
            correct = 0
            # Loop over the training data in batches
            for batch_idx, (data, target) in enumerate(tqdm.tqdm(self.train_loader)):
                data, target = data.to(self.device), target.to(self.device)
                total += target.size(0)
                # Zero the gradients, perform a forward pass, compute the loss, backpropagate, and update the model parameters
                self.optimizer.zero_grad()
                # Forward pass: compute the model output for the current batch of data
                output = self.model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                
                # Compute the loss using cross-entropy
                loss = F.cross_entropy(output, target)
                train_current_loss += loss.item()
                
                # Backpropagate the loss and update the model parameters
                loss.backward()
                self.optimizer.step()

                if batch_idx % 100 == 0:
                    print(f"Epoch {epoch} [{batch_idx * len(data)}/{len(self.train_loader.dataset)}] Loss: {loss.item():.4f}")

            self.metrics["epochs"].append(epoch)
            #self.metrics["learning_rate"].append(self.optimizer.param_groups[0]['lr'])

            # Calculate average training accuracy and loss for the epoch and store it
            self.calculate_training_accuracy(correct, total)
            self.calculate_training_loss(train_current_loss)

            # Calculate validation loss and accuracy
            self.calculate_validation_accuracy_loss()

            self.writer.add_scalar("Accuracy/train", self.metrics["train_accuracies"][-1], epoch)
            self.writer.add_scalar("Loss/train", self.metrics["train_losses"][-1], epoch)
            self.writer.add_scalar("Loss/validation", self.metrics["val_losses"][-1], epoch)
            self.writer.add_scalar("Accuracy/validation", self.metrics["val_accuracies"][-1], epoch)

            print(f"Epoch {epoch}, Train_acc: {self.metrics['train_accuracies'][-1]:.4f}, Train Loss: {self.metrics['train_losses'][-1]:.4f}, Val Loss: {self.metrics['val_losses'][-1]:.4f}, Val Acc: {self.metrics['val_accuracies'][-1]:.4f}")

            # Check early stopping
            early_stop = self.early_stopping.check_early_stop(self.metrics["val_losses"][-1])
            if early_stop:
                print(f"Early stopping at epoch {epoch}")
                break

        # Save metrics to JSON file and close TensorBoard writer
        with open(self.metrics_path, 'w') as f:
            json.dump(self.metrics, f)
        self.writer.close()

    def plot_training_loss(self):
        # Plot the training loss over epochs
        plt.figure(figsize=(10, 5))
        plt.plot(self.metrics["train_losses"], label='Training Loss')
        plt.plot(self.metrics["val_losses"], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss Over Epochs')
        plt.legend()
        plt.savefig(f'mnist/model_weights/{self.model_name}_loss_plot.png')
        plt.close()

    def plot_accuracy(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.metrics["val_accuracies"], label='Validation Accuracy')
        plt.plot(self.metrics["train_accuracies"], label='Training Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Validation Accuracy Over Epochs')
        plt.legend()
        plt.savefig(f'mnist/model_weights/{self.model_name}_acc_plot.png')
        plt.close()

def set_hyperparameters():
    # Define hyperparameters for training
    hyperparameters = {
        "learning_rate": 0.001,
        "batch_size": 64,
        "epochs": 5,
        "optimizer": "Adam",
        "model_name": "mnist_model",
        "run_name": "cnn_adam_lr1e-3"
    }
    return hyperparameters

def main():
    # Set device to GPU if available, otherwise use CPU
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
   
    # Set hyperparameters
    hyperparameters = set_hyperparameters()

    # Load the MNIST dataset
    train_dataset = MNISTDataset(train=True)
    test_dataset = MNISTDataset(train=False)
    train_loader = train_dataset.train_dataloader(batch_size=64, shuffle=True)
    val_loader = test_dataset.test_dataloader(batch_size=64, shuffle=False)  # Using test as validation

    # Initialize the model and move it to the appropriate device
    model = MNISTModel().to(device)

    # Set up the optimizer 
    optimizer = optim.Adam(model.parameters(), lr=hyperparameters["learning_rate"])

    # Create a Trainer instance and start training for a specified number of epochs
    trainer = Trainer(model, device, train_loader, val_loader, optimizer, hyperparameters["model_name"], hyperparameters["run_name"])
    trainer.train(hyperparameters["epochs"])
    # Metrics and plots are saved automatically


if __name__ == "__main__":  
    main()
