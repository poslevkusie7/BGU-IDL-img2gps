import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn as nn

from .dataset import CampusGPSDataset
from .model import CampusGPSModel


CKPT_PATH = "checkpoints/best_model.pth"
TEST_CSV = "data/test/test_metadata.csv"
TEST_IMG_DIR = "data/test/test_images"
BATCH_SIZE = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def main():
    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    stats = ckpt["stats"]

    # IMPORTANT: no random augmentations at test time
    test_transform = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    test_dataset = CampusGPSDataset(
        csv_file=TEST_CSV,
        img_dir=TEST_IMG_DIR,
        transform=test_transform,
        stats=stats,                # reuse TRAIN stats from checkpoint
        compute_stats=False,
        filename_col="filename",
        lat_col="latitude",
        lon_col="longitude",
    )

    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=4, pin_memory=True)

    model = CampusGPSModel(freeze_backbone=False).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    criterion = nn.HuberLoss(delta=1.0)

    losses = []
    meter_errors = []

    with torch.no_grad():
        for images, targets in test_loader:
            images = images.to(DEVICE)
            targets = targets.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, targets).item()
            losses.append(loss)

            err_m = haversine_distance(
                outputs.cpu().numpy(),
                targets.cpu().numpy(),
                stats["lat_mean"], stats["lat_std"],
                stats["lon_mean"], stats["lon_std"],
            )
            meter_errors.append(err_m)

    print(f"[TEST] Loss: {np.mean(losses):.4f}")
    print(f"[TEST] Mean error (m): {np.mean(meter_errors):.2f}")


if __name__ == "__main__":
    main()