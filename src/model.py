import torch
import torch.nn as nn
from torchvision import models

class CampusGPSModel(nn.Module):
    def __init__(self, freeze_backbone=True):
        super(CampusGPSModel, self).__init__()
        
        # 1. Upgrade to ResNet50
        self.backbone = models.resnet50(weights='DEFAULT')
        
        # 2. Freeze the backbone initially
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # 3. Head 
        num_features = self.backbone.fc.in_features
        
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512), # Helps stabilize training
            nn.ReLU(),
            nn.Dropout(0.3),     # 30% of neurons switched off per batch
            
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(128, 2),    # Final Output: [Lat, Lon]
            nn.Tanh()
        )

    def forward(self, x):
        return self.backbone(x)