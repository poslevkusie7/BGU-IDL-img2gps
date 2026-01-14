import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class GeM(nn.Module):
    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        x = x.clamp(min=self.eps).pow(self.p)
        x = F.avg_pool2d(x, (x.size(-2), x.size(-1)))
        return x.pow(1.0 / self.p)

class EmbedNet(nn.Module):
    def __init__(self, emb_dim=512):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.pool = GeM(p=3.0)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, emb_dim),
            nn.BatchNorm1d(emb_dim),
        )

    def forward(self, x):
        f = self.backbone(x)
        p = self.pool(f)
        e = self.head(p)
        return F.normalize(e, p=2, dim=1)

class RefineHead(nn.Module):
    def __init__(self, emb_dim=512):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim + 2, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 2)
        )

    def forward(self, emb, xy0):
        x = torch.cat([emb, xy0], dim=1)
        return self.mlp(x)