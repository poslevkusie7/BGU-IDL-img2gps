import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# 1. The GeM Layer Definition
class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        # p is a learnable parameter, initialized to 3
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return F.avg_pool2d(x.clamp(min=self.eps).pow(self.p), (x.size(-2), x.size(-1))).pow(1./self.p)

# 2. The Main Model
class MultiTaskResNet(nn.Module):
    def __init__(self, num_sectors=5):
        super(MultiTaskResNet, self).__init__()
        
        # --- BACKBONE ---
        weights = models.ResNet50_Weights.DEFAULT
        resnet = models.resnet50(weights=weights)
        
        # We keep the convolutional layers (children[:-2]) 
        # dropping the original 'avgpool' and 'fc' layers.
        self.features = nn.Sequential(*list(resnet.children())[:-2])
        
        # Add our custom GeM pooling
        self.gem_pool = GeM(p=3)
        
        # Feature dimension for ResNet50 is 2048
        self.feature_dim = 2048
        
        # --- HEAD 1: SECTOR CLASSIFICATION ---
        self.cls_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_sectors)
        )
        
        # --- HEAD 2: METRIC EMBEDDING ---
        self.emb_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 512)
            # L2 Norm is applied in forward()
        )

    def forward(self, x):
        # 1. Extract Spatial Features
        x = self.features(x)
        
        # 2. GeM Pooling
        x = self.gem_pool(x)
        
        # 3. Head 1 (Classification)
        cls_logits = self.cls_head(x)
        
        # 4. Head 2 (Embedding)
        emb_feat = self.emb_head(x)
        # Force embedding onto the hypersphere (L2 Norm)
        embedding = F.normalize(emb_feat, p=2, dim=1)
        
        return cls_logits, embedding