# MNIST Dataset and DataLoader implementation using PyTorch
# Author: Dhruv Vyas
# Date: 2024-06-01

import random
import torch.utils.data 
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# Define the MNIST dataset class
class MNISTDataset(torch.utils.data.Dataset):
    def __init__(self, train=True):
        self.train = train
        self.img_shape = None

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.dataset = datasets.MNIST(root='./data', train=train, download=True, transform=self.transform)
        self.dataset_summary(display_sample=False)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        return image, label
    
    def image_shape_display(self):
        if self.img_shape is None:
            image, _ = self.dataset[0]
            self.img_shape = image.shape
        return self.img_shape

    def dataset_summary(self, display_sample=True):
        if self.train:
            print("Training Dataset Summary:")
        else:
            print("Test Dataset Summary:")
        #print(f"type of dataset: {type(self.dataset)}")
        print(f"Number of samples: {len(self.dataset)}")
        print(f"Image shape: {self.image_shape_display()}")
        # Display a sample image from the training dataset
        if display_sample and self.train:
            self.display_image(random.randint(0, len(self.dataset)-1))

    def loader_summary(self, loader):
        if self.train:
            print("Training DataLoader Summary:")
        else:
            print("Test DataLoader Summary:")
        print(f"Number of batches: {len(loader)}")
        for index, (data, target) in enumerate(loader):
            print(f'Batch {index+1}: data shape {data.shape}, target shape {target.shape}')
            break  # Just check the first batch


    def display_image(self, idx):
        image, label = self.dataset[idx]
        plt.imshow(image.squeeze(), cmap='gray')
        plt.title(f'Label: {label}')
        plt.show()  


    def train_dataloader(self, batch_size=64, shuffle=True):
        print(f"Creating train dataloader with batch size {batch_size} and shuffle={shuffle}")
        train_loader = torch.utils.data.DataLoader(self.dataset, batch_size=batch_size, shuffle=shuffle)
        self.loader_summary(train_loader)
        return train_loader

    def test_dataloader(self, batch_size=64, shuffle=False):
        print(f"Creating test dataloader with batch size {batch_size} and shuffle={shuffle}")
        test_loader = torch.utils.data.DataLoader(self.dataset, batch_size=batch_size, shuffle=shuffle)
        self.loader_summary(test_loader)
        return test_loader

if __name__ == "__main__":
    train_dataset = MNISTDataset(train=True)
    test_dataset = MNISTDataset(train=False)

    train_loader = train_dataset.train_dataloader(batch_size=64, shuffle=True)
    test_loader = test_dataset.test_dataloader(batch_size=64, shuffle=False)
    
    for index, (data, target) in enumerate(train_loader):
        print(f'Batch {index+1}: data shape {data.shape}, target shape {target.shape}')
        print( target.size(0))
        break  # Just check the first batch



