import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

# Define the MNIST model class
class MNISTModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Conv layers
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3)
        # Fc layers
        self.fc1 = nn.Linear(64 * 11 * 11, 32)
        self.fc2 = nn.Linear(32, 10)
        self.relu = nn.ReLU

    def forward(self, x):
        x = F.relu(self.conv1(x)) # [B,16,26,26]
        x = F.relu(self.conv2(x)) # [B,32,24,24]
        x = F.relu(self.conv3(x)) # [B,64,22,22]
        x = F.max_pool2d(x, 2)    # [B,64,11,11]
        x  = torch.flatten(x,1)     # [B, 7744 ] 
        x = F.relu(self.fc1(x))     
        x = self.fc2(x)           # [B, 10 ]
        return x    
    
    def model_summary(self):
        print("Model Summary:")
        summary(self, input_size=(1, 28, 28), verbose=1)
    

mnistModel =  MNISTModel()
mnistModel.model_summary()
