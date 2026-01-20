import os

import numpy as np
from PIL import Image, ImageOps
import torch
from torch.utils.data import Dataset

try:
    import utm
except ImportError:  # Optional when using lat/lon directly.
    utm = None


def _coords_from_df(dataframe, coord_mode):
    """Convert lat/lon to either UTM meters or keep raw lat/lon."""
    lats = dataframe["lat"].values
    lons = dataframe["lon"].values

    if coord_mode == "utm":
        if utm is None:
            raise ImportError("utm is required for coord_mode='utm'. Install it or use coord_mode='latlon'.")
        eastings, northings, _, _ = utm.from_latlon(lats, lons)
        coords = np.stack([eastings, northings], axis=1)
    elif coord_mode == "latlon":
        coords = np.stack([lats, lons], axis=1)
    else:
        raise ValueError(f"Unsupported coord_mode: {coord_mode}")

    return coords


def compute_coord_stats(dataframe, coord_mode="latlon"):
    """Compute mean/std for coordinate normalization."""
    coords = _coords_from_df(dataframe, coord_mode=coord_mode)
    mean = coords.mean(axis=0)
    std = coords.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return {"mean": mean, "std": std, "mode": coord_mode}


class LocalizationDataset(Dataset):
    """
    Returns: image tensor, sector label (long), normalized coords (float32).
    """

    def __init__(
        self,
        dataframe,
        img_dir,
        transform=None,
        coord_mode="utm",
        coord_norm="center",
        coord_stats=None,
    ):
        self.img_dir = img_dir
        self.transform = transform

        coords = _coords_from_df(dataframe, coord_mode=coord_mode)

        if coord_stats is None:
            coord_stats = compute_coord_stats(dataframe, coord_mode=coord_mode)
        self.coord_stats = coord_stats

        if coord_norm == "standard":
            coords = (coords - coord_stats["mean"]) / coord_stats["std"]
        elif coord_norm == "center":
            coords = coords - coord_stats["mean"]
        elif coord_norm == "none":
            pass
        else:
            raise ValueError(f"Unsupported coord_norm: {coord_norm}")

        self.coords = coords.astype(np.float32)

        self.image_ids = dataframe["image_id"].values
        self.labels = dataframe["sector_label"].values

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_name = self.image_ids[idx]
        img_path = os.path.join(self.img_dir, img_name)


        image = Image.open(img_path)
        image = ImageOps.exif_transpose(image)
        image = image.convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(self.labels[idx], dtype=torch.long)
        gps_target = torch.tensor(self.coords[idx], dtype=torch.float32)
        return image, label, gps_target

    def get_coord_stats(self):
        return self.coord_stats
