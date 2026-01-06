# model.py
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights


class CampusGPSModel(nn.Module):
    def __init__(self, freeze_backbone: bool = True):
        super().__init__()

        # Backbone
        self.backbone = models.resnet50(weights=ResNet50_Weights.DEFAULT)

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # Replace FC head
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),

            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),

            nn.Linear(128, 2),  # [lat_norm, lon_norm] (NO tanh)
        )

    def forward(self, x):
        return self.backbone(x)