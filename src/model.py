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

class MultiTaskResNet(nn.Module):
    def __init__(self, num_sectors: int):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        # conv backbone output: [B, 2048, H, W]
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.gem_pool = GeM(p=3.0)

        backbone_dim = 2048
        emb_dim = 512

        self.sector_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(backbone_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_sectors),
        )

        self.embedding_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(backbone_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, emb_dim),
            nn.BatchNorm1d(emb_dim),
        )

    def forward(self, x):
        feats = self.backbone(x)          # [B,2048,H,W]
        pooled = self.gem_pool(feats)     # [B,2048,1,1]

        cls_logits = self.sector_head(pooled)
        emb = self.embedding_head(pooled)
        emb = F.normalize(emb, p=2, dim=1)

        return cls_logits, emb

    @torch.no_grad()
    def extract_embedding(self, x):
        self.eval()
        feats = self.backbone(x)
        pooled = self.gem_pool(feats)
        emb = self.embedding_head(pooled)
        return F.normalize(emb, p=2, dim=1)