import os

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

try:
    import utm
except ImportError:  # Optional when using lat/lon directly.
    utm = None

def _coords_from_df(dataframe, coord_mode):
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
    coords = _coords_from_df(dataframe, coord_mode=coord_mode)
    mean = coords.mean(axis=0)
    std = coords.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return {"mean": mean, "std": std, "mode": coord_mode}


class LocalizationDataset(Dataset):
    def __init__(
        self,
        dataframe,
        img_dir,
        transform=None,
        coord_mode="utm",
        coord_norm="center",
        coord_stats=None,
    ):
        """
        Args:
            dataframe (pd.DataFrame): Columns ['image_id', 'sector_label', 'lat', 'lon']
            img_dir (str): Path to image folder.
            transform (callable, optional): PyTorch transforms.
            coord_mode (str): 'utm' or 'latlon'.
            coord_norm (str): 'center', 'standard', or 'none'.
            coord_stats (dict, optional): {'mean': ..., 'std': ...} from train data.
        """
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
        gps_target = torch.tensor(self.coords[idx], dtype=torch.float32)
        
        return image, label, gps_target

    def get_reference_origin(self):
        """Helper to recover original coordinates if needed."""
        return self.coord_stats["mean"]

    def get_coord_stats(self):
        return self.coord_stats
    
# --- 2. Augmentation / Transform Logic ---
def get_transforms(mode='train', img_size=224):
    """
    Returns the correct transformation pipeline based on the mode.
    Input images are 512x512, Model expects 224x224.
    """
    # Standard ImageNet normalization
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    if mode == 'train':
        return transforms.Compose([
            # 1. Geometric: Random Crop & Resize
            # Takes a random piece of the 512 image and resizes it to 224
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0), ratio=(0.9, 1.1)), # the only under big question, but should be good
            
            # 2. Geometric: Horizontal Flip (Caution: flips left/right orientation) $ how will gps react ?
            transforms.RandomHorizontalFlip(p=0.5),
            
            # 3. Photometric: Color Jitter
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
            
            # 4. Regularization: Grayscale
            transforms.RandomGrayscale(p=0.1),
            
            # 5. Regularization: Blur
            transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 5)),
            
            # 6. Formatting
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        
    else: # 'val' or 'test'
        return transforms.Compose([
            # Squashes 512x512 -> 224x224 (No cropping) maybe but why?
            transforms.Resize((img_size, img_size)), 
            
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

# --- 3. The Helper Function to Create Loaders ---
def get_dataloader(
    df,
    img_dir,
    batch_size=32,
    mode="train",
    num_workers=4,
    pin_memory=True,
    coord_mode="utm",
    coord_norm="center",
    coord_stats=None,
):
    transform = get_transforms(mode=mode)
    dataset = LocalizationDataset(
        df,
        img_dir,
        transform=transform,
        coord_mode=coord_mode,
        coord_norm=coord_norm,
        coord_stats=coord_stats,
    )
    should_shuffle = (mode == 'train')

    kwargs = dict(batch_size=batch_size, shuffle=should_shuffle, num_workers=num_workers, pin_memory=pin_memory, drop_last=(mode == 'train'))

    # Only valid when num_workers > 0
    if num_workers > 0:
        kwargs.update(dict(persistent_workers=True, prefetch_factor=4))

    return DataLoader(dataset, **kwargs)
