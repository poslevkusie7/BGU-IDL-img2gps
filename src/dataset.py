import os
from typing import Optional, Dict, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


EARTH_R = 6378137.0  # meters (WGS84-ish)


class CampusGPSDataset(Dataset):
    """
    Dataset for Image -> GPS regression, using a LOCAL METERS coordinate system.

    CSV columns expected:
      - filename
      - latitude
      - longitude
      - region (optional)

    Target:
      - returns normalized (x, y) in meters relative to an origin (lat0, lon0)
      - stats MUST be computed on train only, then reused for val/test (no leakage)
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

        # -----------------------------
        # Stats handling (meters system)
        # -----------------------------
        if stats is not None:
            # Use train stats (for val/test)
            self.lat0 = float(stats["lat0"])
            self.lon0 = float(stats["lon0"])
            self.x_mean = float(stats["x_mean"])
            self.x_std = float(stats["x_std"])
            self.y_mean = float(stats["y_mean"])
            self.y_std = float(stats["y_std"])
            stats_source = "provided (train stats)"
        else:
            # Compute stats (train only)
            if not compute_stats:
                raise ValueError(
                    "stats is None but compute_stats=False. "
                    "For val/test, pass stats from train. For train, set compute_stats=True."
                )

            # Origin is the mean lat/lon of THIS SPLIT (train split)
            self.lat0 = float(self.df[lat_col].mean())
            self.lon0 = float(self.df[lon_col].mean())

            # Convert all points to meters relative to origin, then compute mean/std
            x, y = self.latlon_to_xy_m(
                self.df[lat_col].values,
                self.df[lon_col].values,
                self.lat0,
                self.lon0
            )

            self.x_mean = float(np.mean(x))
            self.y_mean = float(np.mean(y))
            self.x_std = float(np.std(x))
            self.y_std = float(np.std(y))
            stats_source = "computed (this split)"

        # Guard against zero std
        eps = 1e-8
        if abs(self.x_std) <= eps:
            self.x_std = 1.0
        if abs(self.y_std) <= eps:
            self.y_std = 1.0

        print(f"[Dataset] Loaded {len(self.df)} samples from {csv_file}")
        print(f"[Dataset] Origin ({stats_source}): lat0={self.lat0:.6f}, lon0={self.lon0:.6f}")
        print(f"[Dataset] XY std (m): x_std={self.x_std:.3f}, y_std={self.y_std:.3f}")

    def __len__(self) -> int:
        return len(self.df)

    def _resolve_image_path(self, filename: str) -> str:
        filename = str(filename)
        if os.path.isabs(filename) and os.path.exists(filename):
            return filename
        if os.path.exists(filename):
            return filename
        return os.path.join(self.img_dir, filename)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = self._resolve_image_path(row[self.filename_col])

        try:
            image = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            print(f"[Dataset] Warning: missing file: {img_path} (idx={idx}). Using black image.")
            image = Image.new("RGB", (224, 224))

        lat = float(row[self.lat_col])
        lon = float(row[self.lon_col])

        # Convert to local meters
        x_m, y_m = self.latlon_to_xy_m(lat, lon, self.lat0, self.lon0)

        # Normalize
        x_norm = (x_m - self.x_mean) / self.x_std
        y_norm = (y_m - self.y_mean) / self.y_std

        target = torch.tensor([x_norm, y_norm], dtype=torch.float32)

        if self.transform is not None:
            image = self.transform(image)

        if self.return_raw:
            raw_latlon = torch.tensor([lat, lon], dtype=torch.float32)
            raw_xy = torch.tensor([float(x_m), float(y_m)], dtype=torch.float32)
            return image, target, raw_latlon, raw_xy

        return image, target

    def get_stats(self) -> Dict[str, float]:
        return {
            "lat0": self.lat0,
            "lon0": self.lon0,
            "x_mean": self.x_mean,
            "x_std": self.x_std,
            "y_mean": self.y_mean,
            "y_std": self.y_std,
        }

    @staticmethod
    def latlon_to_xy_m(lat, lon, lat0: float, lon0: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Equirectangular approximation (good for small areas like a campus).
        Returns x,y in meters relative to origin (lat0, lon0).

        x: Easting meters
        y: Northing meters
        """
        lat = np.asarray(lat, dtype=np.float64)
        lon = np.asarray(lon, dtype=np.float64)

        dlat = np.radians(lat - lat0)
        dlon = np.radians(lon - lon0)

        x = EARTH_R * dlon * np.cos(np.radians(lat0))
        y = EARTH_R * dlat
        return x, y

    @staticmethod
    def xy_m_to_latlon(x, y, lat0: float, lon0: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Inverse of latlon_to_xy_m.
        """
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        dlat = y / EARTH_R
        dlon = x / (EARTH_R * np.cos(np.radians(lat0)))

        lat = lat0 + np.degrees(dlat)
        lon = lon0 + np.degrees(dlon)
        return lat, lon