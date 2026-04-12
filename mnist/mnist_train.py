import torch
import tqdm
import torch.nn.functional as F
from torch import optim 
from mnist_datasets import MNISTDataset
from mnist_model import MNISTModel
import tensorboard as tb
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt


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
    def __init__(self, model, device, train_loader, optimizer):
        self.model = model
        self.device = device
        self.train_loader = train_loader
        self.optimizer = optimizer  
        self.checkpoint_path = 'mnist/model_weights/mnist_model_checkpoint.pth'
        self.early_stopping = EarlyStopping(self.model, patience=5000, delta=0.001, path=self.checkpoint_path, verbose=True)
        self.train_losses = []  # List to store training losses for each epoch

    def train(self, epochs):
        
        # Set the model to training mode
        self.model.train()
        for epoch in range(1, epochs + 1):  # Train for the specified number of epochs
            train_loss = 0
            # Loop over the training data in batches
            for batch_idx, (data, target) in enumerate(tqdm.tqdm(self.train_loader)):
                data, target = data.to(self.device), target.to(self.device)
                # Zero the gradients, perform a forward pass, compute the loss, backpropagate, and update the model parameters
                self.optimizer.zero_grad()
                # Forward pass: compute the model output for the current batch of data
                output = self.model(data)
                # Compute the loss using cross-entropy
                loss = F.cross_entropy(output, target)
                train_loss += loss.item()
                
                # Backpropagate the loss and update the model parameters
                loss.backward()
                self.optimizer.step()
                loss_value = loss.item()

                # save the best model
                early_stop = self.early_stopping.check_early_stop(loss_value)
                if early_stop:
                    print(f"Early stopping at epoch {epoch}, batch {batch_idx}")
                    return  # Exit the training loop if early stopping is triggered
                
            # Calculate and print the average training loss for the epoch
            avg_train_loss = train_loss / len(self.train_loader)
            print(f"Epoch {epoch}, Average Training Loss: {avg_train_loss:.4f}")
            self.train_losses.append(avg_train_loss) 

    def plot_training_loss(self):
        # Plot the training loss over epochs
        plt.figure(figsize=(10, 5))
        plt.plot(self.train_losses, label='Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss Over Epochs')
        plt.legend()
        plt.show()

            
def main():
    # Set device to GPU if available, otherwise use CPU
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    # Set the number of epochs for training
    epochs = 5
    learn_rate = 0.001


    # Load the MNIST dataset
    train_dataset = MNISTDataset(train=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

    # Initialize the model and move it to the appropriate device
    model = MNISTModel().to(device)

    # Set up the optimizer 
    optimizer = optim.Adam(model.parameters(), lr=learn_rate)

    # Create a Trainer instance and start training for a specified number of epochs
    trainer = Trainer(model, device, train_loader, optimizer)
    trainer.train(epochs)
    # Plot the training loss over epochs
    trainer.plot_training_loss()



if __name__ == "__main__":  
    main()
