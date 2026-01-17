import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

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


class SwinRegionClassifier(nn.Module):
    """Swin-B classifier for sector labels."""

    def __init__(self, num_classes=3, pretrained=True, drop_rate=0.1):
        super().__init__()
        self.use_timm_head = False
        self.input_size = 224
        if hasattr(models, "swin_base_patch4_window7_224"):
            weights = models.Swin_B_Weights.DEFAULT if pretrained else None
            self.backbone = models.swin_base_patch4_window7_224(weights=weights)
            in_features = self.backbone.head.in_features
            self.backbone.head = nn.Sequential(
                nn.Dropout(drop_rate),
                nn.Linear(in_features, num_classes),
            )
        elif timm is not None:
            # Fallback for older torchvision; use timm implementation.
            self.backbone = timm.create_model(
                "swin_base_patch4_window7_224",
                pretrained=pretrained,
                num_classes=num_classes,
                drop_rate=drop_rate,
            )
            self.use_timm_head = True
            try:
                self.input_size = timm.data.resolve_model_data_config(self.backbone)["input_size"][1]
            except Exception:
                self.input_size = 224
        else:
            raise ImportError(
                "Swin transformer not available. Upgrade torchvision to 0.15+ or install timm."
            )

    def forward(self, x):
        return self.backbone(x)

    def freeze_backbone(self):
        if self.use_timm_head:
            for name, p in self.backbone.named_parameters():
                if not name.startswith("head"):
                    p.requires_grad = False
        else:
            for name, p in self.backbone.named_parameters():
                if not name.startswith("head."):
                    p.requires_grad = False

    def unfreeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = True


class MultiTaskModel(nn.Module):
    """
    Two-head model: DINOv2 regression head + Swin classification head.
    Backbones are independent so each can specialize.
    """

    def __init__(
        self,
        num_classes,
        coord_model_name="vit_base_patch14_dinov2.lvd142m",
        pretrained=True,
        drop_rate=0.1,
    ):
        super().__init__()
        self.regressor = DinoV2CoordRegressor(
            pretrained=pretrained,
            model_name=coord_model_name,
            drop_rate=drop_rate,
        )
        self.classifier = SwinRegionClassifier(num_classes=num_classes, pretrained=pretrained, drop_rate=drop_rate)
        self.cls_input_size = getattr(self.classifier, "input_size", 224)

    def forward(self, x):
        coords = self.regressor(x)
        if x.shape[-1] != self.cls_input_size:
            cls_inp = F.interpolate(x, size=(self.cls_input_size, self.cls_input_size), mode="bilinear", align_corners=False)
        else:
            cls_inp = x
        logits = self.classifier(cls_inp)
        return logits, coords

    def freeze_backbones(self):
        self.regressor.freeze_backbone()
        self.classifier.freeze_backbone()

    def unfreeze_backbones(self):
        self.regressor.unfreeze_backbone()
        self.classifier.unfreeze_backbone()
