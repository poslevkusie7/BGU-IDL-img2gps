import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class CampusGPSDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            img_dir (string): Directory with all the images.
            transform (callable, optional): Transform to apply on images.
        """
        self.gps_frame = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

        # Calculate statistics for normalization (centering the data)
        self.lat_mean = self.gps_frame['latitude'].mean()
        self.lat_std = self.gps_frame['latitude'].std()
        self.lon_mean = self.gps_frame['longitude'].mean()
        self.lon_std = self.gps_frame['longitude'].std()
        
        print(f"Dataset Loaded: {len(self.gps_frame)} images.")
        print(f"Center Lat: {self.lat_mean:.5f}, Center Lon: {self.lon_mean:.5f}")

    def __len__(self):
        return len(self.gps_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get image path
        img_name = os.path.join(self.img_dir, self.gps_frame.iloc[idx]['filename'])
        
        try:
            image = Image.open(img_name).convert('RGB')
        except FileNotFoundError:
            # Handle missing files gracefully
            print(f"Warning: File {img_name} not found. Returning black image.")
            image = Image.new('RGB', (224, 224))

        # Get Raw GPS
        lat = self.gps_frame.iloc[idx]['latitude']
        lon = self.gps_frame.iloc[idx]['longitude']

        # Normalize GPS (Z-score normalization)
        # (Value - Mean) / Std Dev
        lat_norm = (lat - self.lat_mean) / self.lat_std
        lon_norm = (lon - self.lon_mean) / self.lon_std
        
        # Create tensor for targets [Lat, Lon]
        target = torch.tensor([lat_norm, lon_norm], dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, target