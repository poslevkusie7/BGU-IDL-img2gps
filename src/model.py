import torch.nn as nn

try:
    import timm
except ImportError:
    timm = None


class DinoV2CoordRegressor(nn.Module):
    """DINOv2/Vision Transformer backbone for coordinate regression."""

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
            global_pool="token",
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


class SharedDinoMultiTask(nn.Module):
    """Shared DINOv2/ViT backbone with separate heads for classification and regression."""

    def __init__(
        self,
        num_classes,
        coord_model_name="vit_base_patch14_dinov2.lvd142m",
        pretrained=True,
        drop_rate=0.1,
    ):
        super().__init__()
        if timm is None:
            raise ImportError("timm is required for DINOv2 models. Install timm or use a different backbone.")
        self.backbone = timm.create_model(
            coord_model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="token",
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
        self.classifier = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Dropout(drop_rate),
            nn.Linear(feat_dim, num_classes),
        )

    def forward(self, x):
        feats = self.backbone(x)
        if isinstance(feats, (tuple, list)):
            feats = feats[0]
        coords = self.regressor(feats)
        logits = self.classifier(feats)
        return logits, coords

    def freeze_backbones(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbones(self):
        for p in self.backbone.parameters():
            p.requires_grad = True
