import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class GeM(nn.Module):
    """Generalized Mean Pooling - crucial for place recognition"""
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return F.avg_pool2d(x.clamp(min=self.eps).pow(self.p), 
                           (x.size(-2), x.size(-1))).pow(1./self.p)

class CampusLocalizationModel(nn.Module):
    """
    Optimized for small campus with repetitive building views.
    Uses strong metric learning instead of GPS regression.
    """
    def __init__(self, embedding_dim=512, num_sectors=3):
        super(CampusLocalizationModel, self).__init__()
        
        # Backbone - ResNet50
        weights = models.ResNet50_Weights.DEFAULT
        resnet = models.resnet50(weights=weights)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        
        # GeM pooling (better than avg pooling for landmarks)
        self.gem_pool = GeM(p=3)
        
        # Feature dimension
        backbone_dim = 2048
        
        # Embedding head - this is the MAIN task
        self.embedding_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(backbone_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
        )
        
        # Sector classifier - auxiliary task
        self.sector_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(backbone_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_sectors)
        )
        
        # GPS refinement - predicts OFFSET from retrieved location
        self.gps_refinement = nn.Sequential(
            nn.Flatten(),
            nn.Linear(backbone_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2)  # (x, y) offset in meters
        )

    def forward(self, x):
        # Extract features
        features = self.backbone(x)
        pooled = self.gem_pool(features)
        
        # Main embedding (L2 normalized)
        embedding = self.embedding_head(pooled)
        embedding = F.normalize(embedding, p=2, dim=1)
        
        # Auxiliary outputs
        sector_logits = self.sector_head(pooled)
        gps_offset = self.gps_refinement(pooled)
        
        return embedding, sector_logits, gps_offset
    
    def extract_embedding(self, x):
        """For inference: just get the embedding"""
        features = self.backbone(x)
        pooled = self.gem_pool(features)
        embedding = self.embedding_head(pooled)
        return F.normalize(embedding, p=2, dim=1)