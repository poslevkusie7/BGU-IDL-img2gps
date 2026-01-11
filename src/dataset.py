import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
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
    
from torchvision import transforms

def get_transforms(mode='train', img_size=224):
    # Standard ImageNet normalization (Required for pre-trained ResNet)
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    if mode == 'train':
        return transforms.Compose([
            # 1. Geometric Augmentations (Position/Scale)
            # Randomly resize and crop to simulate different distances/zooms
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
            
            # 2. Photometric Augmentations (Lighting/Weather)
            # Simulates different times of day, shadows, and camera exposure
            transforms.ColorJitter(
                brightness=0.2, 
                contrast=0.2, 
                saturation=0.2, 
                hue=0.05
            ),
            
            # 3. Regularization
            # Occasionally turns image grayscale (helps focus on structure, not just color)
            transforms.RandomGrayscale(p=0.1),
            
            # Final formatting
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        
    else: # 'val' or 'test'
        return transforms.Compose([
            # Deterministic resize
            transforms.Resize(256),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])