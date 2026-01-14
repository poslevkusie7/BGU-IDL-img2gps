import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

try:
    import timm
except ImportError:
    timm = None

# 1. The GeM Layer Definition
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


class DinoV2CoordRegressor(nn.Module):
    def __init__(
        self,
        pretrained=True,
        model_name="vit_base_patch14_dinov2.lvd142m",
        drop_rate=0.1,
    ):
        super().__init__()
        if timm is None:
            raise ImportError("timm is required for DINOv2 models. Install timm or use a different backbone.")
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )
        feat_dim = getattr(self.backbone, "num_features", 768)
        self.regressor = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Dropout(drop_rate),
            nn.Linear(feat_dim, 256),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(256, 2),
        )

    def forward(self, x):
        feats = self.backbone(x)
        if isinstance(feats, (tuple, list)):
            feats = feats[0]
        coords = self.regressor(feats)
        return coords

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = True


class SwinRegionClassifier(nn.Module):
    def __init__(self, num_classes=3, pretrained=True, drop_rate=0.1):
        super().__init__()
        weights = models.Swin_B_Weights.DEFAULT if pretrained else None
        self.backbone = models.swin_base_patch4_window7_224(weights=weights)
        in_features = self.backbone.head.in_features
        self.backbone.head = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x):
        return self.backbone(x)

    def freeze_backbone(self):
        for name, p in self.backbone.named_parameters():
            if not name.startswith("head."):
                p.requires_grad = False

    def unfreeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = True
