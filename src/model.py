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

class MDNResNet(nn.Module):
    """
    Outputs a mixture of K diagonal Gaussians in 2D:
      pi_logits: [B,K]
      mu:       [B,K,2]
      log_sigma:[B,K,2]
    """
    def __init__(self, K=5, hidden=1024):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.pool = GeM(p=3.0)

        self.trunk = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        self.pi = nn.Linear(hidden, K)
        self.mu = nn.Linear(hidden, K * 2)
        self.log_sigma = nn.Linear(hidden, K * 2)

        self.K = K

    def forward(self, x):
        f = self.backbone(x)
        p = self.pool(f)
        h = self.trunk(p)

        pi_logits = self.pi(h)                         # [B,K]
        mu = self.mu(h).view(-1, self.K, 2)            # [B,K,2]
        log_sigma = self.log_sigma(h).view(-1, self.K, 2)  # [B,K,2]
        # stabilize sigma
        log_sigma = torch.clamp(log_sigma, -7.0, 7.0)

        return pi_logits, mu, log_sigma