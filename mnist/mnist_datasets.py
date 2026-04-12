import torch.utils.data 
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


# Define the MNIST dataset class
class MNISTDataset(torch.utils.data.Dataset):
    def __init__(self, train=True):
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.train_datasets = datasets.MNIST(root='./data', train=train, download=True, transform=self.transform)
        self.test_datasets = datasets.MNIST(root='./data', train=False, download=True, transform=self.transform)


    def __len__(self):
        return len(self.train_datasets)

    def __getitem__(self, idx):
        image, label = self.train_datasets[idx]
        return image, label
    
    def image_shape(self):
        image, _ = self.train_datasets[0]
        return image.shape

    def display_image(self, idx):
        image, label = self.train_datasets[idx]
        plt.imshow(image.squeeze(), cmap='gray')
        plt.title(f'Label: {label}')
        plt.show()  


    def train_dataloader(self, batch_size=64, shuffle=True):
        return torch.utils.data.DataLoader(self.train_datasets, batch_size=batch_size, shuffle=shuffle)    

    def test_dataloader(self, batch_size=64, shuffle=False):
        return torch.utils.data.DataLoader(self.test_datasets, batch_size=batch_size, shuffle=shuffle)        

if __name__ == "__main__":
    train_dataset = MNISTDataset(train=True)
    test_dataset = MNISTDataset(train=False)

    train_loader = train_dataset.train_dataloader(batch_size=64, shuffle=True)
    test_loader = test_dataset.test_dataloader(batch_size=64, shuffle=False)

    # Print the number of samples in the training and test datasets
    print(f'Training samples: {len(train_dataset)}')
    print(f'Test samples: {len(test_dataset)}')
    # Display a sample image from the training dataset
    train_dataset.display_image(0)
    image_shape = train_dataset.image_shape()
    print(f'Image shape: {image_shape}')
    



