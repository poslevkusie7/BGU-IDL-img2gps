import torch
import torch.nn as nn
from torchvision import models

class CampusGPSModel(nn.Module):
    def __init__(self, freeze_backbone=True):
        super(CampusGPSModel, self).__init__()
        
        # 1. Load Pre-trained ResNet18
        # weights='DEFAULT' is the modern way to load ImageNet weights
        self.backbone = models.resnet18(weights='DEFAULT')
        
        # 2. Freeze the backbone (Optional but recommended for small datasets)
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # 3. Replace the Head
        # The original fc layer takes 512 inputs and outputs 1000 classes.
        # We replace it to output 2 values: Latitude and Longitude.
        num_features = self.backbone.fc.in_features
        
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_features, 128),  # Intermediate layer for better learning
            nn.ReLU(),
            nn.Dropout(0.2),               # Dropout helps prevent overfitting on small data
            nn.Linear(128, 2)              # Final Output: [Lat, Lon]
        )

    def forward(self, x):
        return self.backbone(x)