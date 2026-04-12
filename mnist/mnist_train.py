import torch
import tqdm
import torch.nn.functional as F
from torch import optim 
from mnist_datasets import MNISTDataset
from mnist_model import MNISTModel

# Trainer class to handle the training loop
class Trainer:
    def __init__(self, model, device, train_loader, optimizer):
        self.model = model
        self.device = device
        self.train_loader = train_loader
        self.optimizer = optimizer  

    def train(self, epoch):
        # Set the model to training mode
        self.model.train()
        # Loop over the training data in batches
        for batch_idx, (data, target) in enumerate(tqdm.tqdm(self.train_loader)):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            self.optimizer.step()
            if batch_idx % 100 == 0:
                print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(self.train_loader.dataset)} ({100. * batch_idx / len(self.train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

def main():
    # Set device to GPU if available, otherwise use CPU
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    epochs = 1

    # Load the MNIST dataset
    train_dataset = MNISTDataset(train=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

    # Initialize the model and move it to the appropriate device
    model = MNISTModel().to(device)

    # Set up the optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Create a Trainer instance and start training for a specified number of epochs
    trainer = Trainer(model, device, train_loader, optimizer)
    for epoch in range(1, epochs + 1):  # Train for the specified number of epochs
        trainer.train(epoch)

if __name__ == "__main__":  
    main()
