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
