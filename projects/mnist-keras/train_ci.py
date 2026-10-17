import torch
import torch.nn.functional as F
from torch import optim
from mnist_datasets import MNISTDataset
from mnist_model import MNISTModel
import os
from datetime import datetime

def train_model():
    # Set device
    device = torch.device("cpu")  # Use CPU for CI/CD to avoid GPU issues

    # Load dataset
    train_dataset = MNISTDataset(train=True)
    train_loader = train_dataset.train_dataloader(batch_size=64, shuffle=True)

    # Initialize model
    model = MNISTModel().to(device)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train for 1 epoch
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

    # Create timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save model with timestamp
    model_path = f"mnist/model_weights/mnist_model_{timestamp}.pth"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), model_path)

    print(f"Model trained and saved to {model_path}")
    return model_path

if __name__ == "__main__":
    train_model()