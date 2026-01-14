import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import pandas as pd
import os
import utm  
import numpy as np

class LocalizationDataset(Dataset):
    def __init__(self, dataframe, img_dir, transform=None):
        """
        Args:
            dataframe (pd.DataFrame): Columns ['image_id', 'sector_label', 'lat', 'lon']
            img_dir (str): Path to image folder.
            transform (callable, optional): PyTorch transforms.
        """
        self.img_dir = img_dir
        self.transform = transform
        
        # --- PRE-PROCESSING GPS TO METERS ---
        lats = dataframe['lat'].values
        lons = dataframe['lon'].values
        
        # utm.from_latlon returns (zone_number, zone_letter) which we dont need
        eastings, northings, _, _ = utm.from_latlon(lats, lons)
        
        self.mean_easting = np.mean(eastings)
        self.mean_northing = np.mean(northings)
        
        # Store centered coordinates
        self.gps_x = eastings - self.mean_easting
        self.gps_y = northings - self.mean_northing
        
        # Store other data
        self.image_ids = dataframe['image_id'].values
        self.labels = dataframe['sector_label'].values

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        # 1. Load Image
        img_name = self.image_ids[idx]
        img_path = os.path.join(self.img_dir, img_name)
        
        try:
            image = Image.open(img_path).convert('RGB')
        except (OSError, FileNotFoundError):
            image = Image.new('RGB', (224, 224)) 

        if self.transform:
            image = self.transform(image)
            
        # 2. Get sector (Head 1)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        
        # 3. Get GPS (Head 2)
        gps_target = torch.tensor([self.gps_x[idx], self.gps_y[idx]], dtype=torch.float32)
        
        return image, label, gps_target

    def get_reference_origin(self):
        """Helper to recover original coordinates if needed"""
        return self.mean_easting, self.mean_northing
    
# --- 2. Augmentation / Transform Logic ---
# --- 2. Augmentation / Transform Logic ---
def get_transforms(mode="train", img_size=224):
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    if mode == "train":
        # Keep this mild; retrieval systems hate heavy stochastic transforms
        return transforms.Compose([
            transforms.Resize(256),
            # optional: add mild crop if you want
            # transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
            transforms.CenterCrop(img_size),  # stable by default
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    # val / test / db: deterministic
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


# --- 3. The Helper Function to Create Loaders ---
def get_dataloader(
    df,
    img_dir,
    batch_size=64,
    mode="train",          # "train" | "val" | "test" | "db"
    num_workers=4,
    pin_memory=True,
):
    """
    mode semantics:
      - "train": shuffled + drop_last + train transforms
      - "val"/"test": no shuffle + no drop_last + val transforms
      - "db": same as val (for embedding bank / retrieval database)
    """
    if mode not in {"train", "val", "test", "db"}:
        raise ValueError(f"mode must be one of train/val/test/db, got: {mode}")

    tfm_mode = "train" if mode == "train" else "val"
    transform = get_transforms(mode=tfm_mode)

    dataset = LocalizationDataset(df, img_dir, transform=transform)

    shuffle = (mode == "train")
    drop_last = (mode == "train")

    kwargs = dict(
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    # Only valid when num_workers > 0
    if num_workers > 0:
        kwargs.update(dict(persistent_workers=True, prefetch_factor=4))

    return DataLoader(dataset, **kwargs)