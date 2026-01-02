import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import numpy as np
from tqdm import tqdm
import os

# Import your custom classes
from .dataset import CampusGPSDataset
from .model import CampusGPSModel

# --- Configuration & Hyperparameters ---
CSV_PATH = "data/metadata.csv"
IMG_DIR = "data/images"
BATCH_SIZE = 16
LEARNING_RATE = 0.0005
EPOCHS = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs("checkpoints", exist_ok=True)


def haversine_distance(pred, target, lat_mean, lat_std, lon_mean, lon_std):
    """
    Calculates the distance in meters between two GPS points.
    This helps you understand the 'real world' error of your model.
    """
    pred = pred.astype(np.float64)
    target = target.astype(np.float64)
    # 1. Un-normalize back to degrees
    p_lat = (pred[:, 0] * lat_std) + lat_mean
    p_lon = (pred[:, 1] * lon_std) + lon_mean
    t_lat = (target[:, 0] * lat_std) + lat_mean
    t_lon = (target[:, 1] * lon_std) + lon_mean

    # 2. Haversine Formula
    R = 6371000  # Earth radius in meters
    phi1, phi2 = np.radians(p_lat), np.radians(t_lat)
    dphi = np.radians(t_lat - p_lat)
    dlambda = np.radians(t_lon - p_lon)

    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
    res = 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return np.mean(res)

def train():
    best_error = float('inf')

    # 1. Image Preprocessing
    transform = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        # transforms.RandomHorizontalFlip(p=0.5), - Cpmmented out, since it may make model lost it 
        transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)),
        # transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 2. Data Loading
    full_dataset = CampusGPSDataset(CSV_PATH, IMG_DIR, transform=transform, filename_col="image", 
                                    lat_col="latitude", lon_col="longitude", compute_stats=True)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_set, val_set = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

    # 3. Model, Loss, Optimizer
    model = CampusGPSModel(freeze_backbone=False).to(DEVICE)
    criterion = nn.HuberLoss(delta=1.0) 
    optimizer = optim.Adam([
        {'params': model.backbone.conv1.parameters(), 'lr': 1e-6},
        {'params': model.backbone.layer1.parameters(), 'lr': 1e-6},
        {'params': model.backbone.layer2.parameters(), 'lr': 1e-6},
        {'params': model.backbone.layer3.parameters(), 'lr': 1e-5},
        {'params': model.backbone.layer4.parameters(), 'lr': 1e-5},
        {'params': model.backbone.fc.parameters(), 'lr': 5e-4}
    ])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    # optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)

    # 4. Training Loop
    print(f"Starting training on {DEVICE}...")
    pbar = tqdm(range(EPOCHS), desc="Training Progress")    
    for epoch in pbar:
        model.train()
        train_loss = 0.0
        
        for images, targets in train_loader:
            images, targets = images.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # 5. Validation Loop
        model.eval()
        val_loss = 0.0
        meter_errors = []
        
        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = images.to(DEVICE), targets.to(DEVICE)
                outputs = model(images)
                val_loss += criterion(outputs, targets).item()
                
                m_err = haversine_distance(
                    outputs.cpu().numpy(), targets.cpu().numpy(),
                    full_dataset.lat_mean, full_dataset.lat_std,
                    full_dataset.lon_mean, full_dataset.lon_std
                )
                meter_errors.append(m_err)
                
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        avg_meter_error = np.mean(meter_errors)
        scheduler.step(avg_meter_error)

        pbar.set_postfix({
                "Epoch": epoch + 1,
                "T-Loss": f"{avg_train_loss:.2e}",
                "V-Loss": f"{avg_val_loss:.2e}",
                "Err(m)": f"{avg_meter_error:.2f}"
        })
        
        # Save best model
        if avg_meter_error < best_error:
            best_error = avg_meter_error
            torch.save({
                'model_state_dict': model.state_dict(),
                'error': avg_meter_error,
                'stats': full_dataset.get_stats()
            }, "checkpoints/best_model.pth")

    print(f"Training Complete. Model saved. With the best error: {best_error}")

if __name__ == "__main__":
    train()