# dataset.py
import os
from typing import Optional, Dict, Sequence

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class CampusGPSDataset(Dataset):
    """
    CSV columns expected:
      - filename
      - region (optional, will be kept but not required)
      - latitude
      - longitude

    Targets returned are z-score normalized using train-only stats.
    """

    def __init__(
        self,
        csv_file: str,
        img_dir: str,
        transform=None,
        stats: Optional[Dict[str, float]] = None,
        compute_stats: bool = False,
        indices: Optional[Sequence[int]] = None,
        return_raw: bool = False,
        filename_col: str = "filename",
        lat_col: str = "latitude",
        lon_col: str = "longitude",
        region_col: str = "region",
    ):
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.return_raw = return_raw

        self.filename_col = filename_col
        self.lat_col = lat_col
        self.lon_col = lon_col
        self.region_col = region_col

        # Validate required columns
        required = [filename_col, lat_col, lon_col]
        missing = [c for c in required if c not in self.df.columns]
        if missing:
            raise ValueError(f"CSV missing columns {missing}. Found: {list(self.df.columns)}")

        # Clean types
        self.df[lat_col] = pd.to_numeric(self.df[lat_col], errors="coerce")
        self.df[lon_col] = pd.to_numeric(self.df[lon_col], errors="coerce")

        # Drop invalid GPS rows
        before = len(self.df)
        self.df = self.df.dropna(subset=[lat_col, lon_col]).reset_index(drop=True)
        after = len(self.df)
        if after < before:
            print(f"[Dataset] Dropped {before - after} rows with invalid lat/lon.")

        # Optional subsetting (crucial for leak-free stats)
        if indices is not None:
            self.df = self.df.iloc[list(indices)].reset_index(drop=True)

        # Stats handling
        if stats is not None:
            self.lat_mean = float(stats["lat_mean"])
            self.lat_std = float(stats["lat_std"])
            self.lon_mean = float(stats["lon_mean"])
            self.lon_std = float(stats["lon_std"])
        else:
            if not compute_stats:
                raise ValueError(
                    "stats is None but compute_stats=False. "
                    "For val/test, pass train stats. For train, set compute_stats=True."
                )
            self.lat_mean = float(self.df[lat_col].mean())
            self.lat_std = float(self.df[lat_col].std())
            self.lon_mean = float(self.df[lon_col].mean())
            self.lon_std = float(self.df[lon_col].std())

        # Guard against zero std
        eps = 1e-8
        if abs(self.lat_std) <= eps:
            self.lat_std = 1.0
        if abs(self.lon_std) <= eps:
            self.lon_std = 1.0

        print(f"[Dataset] Loaded {len(self.df)} samples from {csv_file}")
        print(f"[Dataset] GPS center: lat={self.lat_mean:.6f}, lon={self.lon_mean:.6f}")

    def __len__(self):
        return len(self.df)

    def _resolve_image_path(self, filename: str) -> str:
        filename = str(filename)
        if os.path.isabs(filename) and os.path.exists(filename):
            return filename
        if os.path.exists(filename):
            return filename
        return os.path.join(self.img_dir, filename)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self._resolve_image_path(row[self.filename_col])

        try:
            image = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            print(f"[Dataset] Warning: missing file: {img_path} (idx={idx}). Using black image.")
            image = Image.new("RGB", (224, 224))

        lat = float(row[self.lat_col])
        lon = float(row[self.lon_col])

        # z-score normalize
        lat_norm = (lat - self.lat_mean) / self.lat_std
        lon_norm = (lon - self.lon_mean) / self.lon_std
        target = torch.tensor([lat_norm, lon_norm], dtype=torch.float32)

        if self.transform is not None:
            image = self.transform(image)

        if self.return_raw:
            raw = torch.tensor([lat, lon], dtype=torch.float32)
            return image, target, raw

        return image, target

    def get_stats(self) -> Dict[str, float]:
        return {
            "lat_mean": self.lat_mean,
            "lat_std": self.lat_std,
            "lon_mean": self.lon_mean,
            "lon_std": self.lon_std,
        }