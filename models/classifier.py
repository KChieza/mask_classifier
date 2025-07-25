import torch
import torch.nn as nn
import torch.nn.functional as F

class FishMaskClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super(FishMaskClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)  # Input is 1-channel (grayscale)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Add pooling layer
        self.fc1 = nn.Linear(64 * 16 * 16, 128)  # Assuming input size is 128x128
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        # Conv + ReLU + Pool
        x = self.pool(F.relu(self.conv1(x)))  # shape: [batch, 16, 64, 64]
        x = self.pool(F.relu(self.conv2(x)))  # shape: [batch, 32, 32, 32]
        x = self.pool(F.relu(self.conv3(x)))  # shape: [batch, 64, 16, 16]
        
        # Flatten
        x = x.view(-1, 64 * 16 * 16)  # flatten for FC layer
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # no softmax here — use CrossEntropyLoss
        
        return x