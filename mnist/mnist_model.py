import torch 
import torch.nn as nn
import torch.nn.functional as F
import torchsummary
from mnist_datasets import MNISTDataset
from torchsummary import summary

# Define the MNIST model class
class MNISTModel(nn.Module):
    def __init__(self):
        super().__init__()
        # input size is 28x28x1
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        # input size is 26x26x32 after conv1, then max pooling reduces it to 13x13x32
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        # input size is 11x11x64 after conv2, then max pooling reduces it to 5x5x64
        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        # output size is 128, then fully connected layer reduces it to 10 for the 10 classes
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 64 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)
        return x    
    
    def model_summary(self):
        print("Model Summary:")
        summary(self, input_size=(1, 28, 28), verbose=1)
    

mnistModel =  MNISTModel()
mnistModel.model_summary()
