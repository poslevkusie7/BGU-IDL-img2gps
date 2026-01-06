import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import pandas as pd
from torch.amp import GradScaler, autocast

from .dataset import CampusGPSDataset
from .model import CampusGPSModel


CSV_PATH = "data/metadata.csv"
IMG_DIR = "data/images"
BATCH_SIZE = 16
EPOCHS = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
TRAIN_FRAC = 0.8

os.makedirs("checkpoints", exist_ok=True)


def haversine_distance(pred, target, lat_mean, lat_std, lon_mean, lon_std):
    pred = pred.astype(np.float64)
    target = target.astype(np.float64)

    p_lat = (pred[:, 0] * lat_std) + lat_mean
    p_lon = (pred[:, 1] * lon_std) + lon_mean
    t_lat = (target[:, 0] * lat_std) + lat_mean
    t_lon = (target[:, 1] * lon_std) + lon_mean

    R = 6371000.0
    phi1, phi2 = np.radians(p_lat), np.radians(t_lat)
    dphi = np.radians(t_lat - p_lat)
    dlambda = np.radians(t_lon - p_lon)

    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    c = 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return float(np.mean(c))


def make_split_indices(csv_path: str, train_frac: float = 0.8, seed: int = 42):
    df = pd.read_csv(csv_path).reset_index(drop=True)
    n = len(df)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)

    train_size = int(train_frac * n)
    train_idx = perm[:train_size].tolist()
    val_idx = perm[train_size:].tolist()
    return train_idx, val_idx


def train():
    best_error = float("inf")

    # Transforms: PIL ops first, then ToTensor, then Normalize
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # Leak-free split
    train_idx, val_idx = make_split_indices(CSV_PATH, TRAIN_FRAC, SEED)

    # Train dataset computes stats ONLY from train split
    train_dataset = CampusGPSDataset(
        CSV_PATH, IMG_DIR, transform=transform,
        indices=train_idx, compute_stats=True,
        filename_col="filename", lat_col="latitude", lon_col="longitude"
    )
    train_stats = train_dataset.get_stats()

    # Val dataset reuses train stats (no leakage)
    val_dataset = CampusGPSDataset(
        CSV_PATH, IMG_DIR, transform=transform,
        indices=val_idx, stats=train_stats,
        filename_col="filename", lat_col="latitude", lon_col="longitude"
    )

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=4, pin_memory=True, persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=True, persistent_workers=True
    )

    # Model + loss
    model = CampusGPSModel(freeze_backbone=False).to(DEVICE)
    criterion = nn.HuberLoss(delta=1.0)

    optimizer = optim.Adam([
        {"params": model.backbone.conv1.parameters(), "lr": 1e-6},
        {"params": model.backbone.layer1.parameters(), "lr": 1e-6},
        {"params": model.backbone.layer2.parameters(), "lr": 1e-6},
        {"params": model.backbone.layer3.parameters(), "lr": 1e-5},
        {"params": model.backbone.layer4.parameters(), "lr": 1e-5},
        {"params": model.backbone.fc.parameters(), "lr": 5e-4},
    ])

    scaler = GradScaler("cuda", enabled=(DEVICE.type == "cuda"))
    print(f"Starting training on {DEVICE}...")
    pbar = tqdm(range(EPOCHS), desc="Training Progress")

    for epoch in pbar:
        model.train()
        train_loss = 0.0

        for images, targets in train_loader:
            images = images.to(DEVICE, non_blocking=True)
            targets = targets.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast("cuda", enabled=(DEVICE.type == "cuda")):
                outputs = model(images)
                loss = criterion(outputs, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0.0
        meter_errors = []

        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(DEVICE, non_blocking=True)
                targets = targets.to(DEVICE, non_blocking=True)

                with autocast("cuda", enabled=(DEVICE.type == "cuda")):
                    outputs = model(images)
                    val_loss += criterion(outputs, targets).item()

                m_err = haversine_distance(
                    outputs.cpu().numpy(),
                    targets.cpu().numpy(),
                    train_stats["lat_mean"], train_stats["lat_std"],
                    train_stats["lon_mean"], train_stats["lon_std"],
                )
                meter_errors.append(m_err)

        avg_train_loss = train_loss / max(1, len(train_loader))
        avg_val_loss = val_loss / max(1, len(val_loader))
        avg_meter_error = float(np.mean(meter_errors)) if meter_errors else float("inf")

        pbar.set_postfix({
            "Epoch": epoch + 1,
            "T-Loss": f"{avg_train_loss:.2e}",
            "V-Loss": f"{avg_val_loss:.2e}",
            "Err(m)": f"{avg_meter_error:.2f}",
        })

        if avg_meter_error < best_error:
            best_error = avg_meter_error
            torch.save({
                "model_state_dict": model.state_dict(),
                "error_m": avg_meter_error,
                "stats": train_stats,
            }, "checkpoints/best_model.pth")

    print(f"Training complete. Best mean error: {best_error:.2f} m")


if __name__ == "__main__":
    train()