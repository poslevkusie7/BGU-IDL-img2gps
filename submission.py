import os
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torchvision import transforms

from src.model import DinoV2CoordRegressor, SharedDinoMultiTask


_MODEL = None
_PAYLOAD = None
_DEVICE = None
_TASK = None
_TRANSFORMS = None


def _build_transforms(img_size=518):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def _resolve_checkpoint():
    candidates = [
        os.environ.get("IMG2GPS_CHECKPOINT"),
        "runs/coords_latlon/best.pt",
        "runs/coords_best.pt",
        "runs/multitask_best.pt",
        "checkpoints/best.pt",
    ]
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate)
        if path.exists():
            return path
    raise FileNotFoundError(
        "No checkpoint found. Set IMG2GPS_CHECKPOINT or place weights in "
        "runs/coords_latlon/best.pt."
    )


def _load_checkpoint(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _denormalize_coords(coords, coord_stats, coord_norm):
    coords = coords.float()
    mean = torch.tensor(coord_stats["mean"], device=coords.device, dtype=torch.float32)
    std = torch.tensor(coord_stats["std"], device=coords.device, dtype=torch.float32)
    if coord_norm == "standard":
        return coords * std + mean
    if coord_norm == "center":
        return coords + mean
    return coords


def _init_model():
    global _MODEL, _PAYLOAD, _DEVICE, _TASK, _TRANSFORMS
    if _MODEL is not None:
        return

    device_pref = os.environ.get("IMG2GPS_DEVICE", "auto").lower()
    if device_pref == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_pref)

    checkpoint = _resolve_checkpoint()
    payload = _load_checkpoint(checkpoint, device)

    if not isinstance(payload, dict) or "model_state" not in payload:
        raise ValueError(f"Checkpoint {checkpoint} is missing a 'model_state' dict.")

    task = payload.get("task", "coords")
    if task == "multitask":
        num_classes = payload.get("num_classes")
        if num_classes is None:
            raise ValueError("Multitask checkpoint missing 'num_classes'.")
        model = SharedDinoMultiTask(
            num_classes=int(num_classes),
            coord_model_name=payload.get("dinov2_name", "vit_base_patch14_dinov2.lvd142m"),
            pretrained=False,
        )
    else:
        model = DinoV2CoordRegressor(
            pretrained=False,
            model_name=payload.get("dinov2_name", "vit_base_patch14_dinov2.lvd142m"),
        )

    model.load_state_dict(payload["model_state"], strict=True)
    model.to(device)
    model.eval()

    img_size = int(os.environ.get("IMG2GPS_IMG_SIZE", "518"))
    _MODEL = model
    _PAYLOAD = payload
    _DEVICE = device
    _TASK = task
    _TRANSFORMS = _build_transforms(img_size)


def predict_gps(image: np.ndarray) -> np.ndarray:
    """
    Predict GPS latitude and longitude from a single RGB image.
    Input: numpy.ndarray of shape (H, W, 3), dtype=uint8, RGB order.
    Output: numpy.ndarray of shape (2,), dtype=float32 -> [lat, lon].
    """
    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a numpy.ndarray")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must have shape (H, W, 3)")
    if image.dtype != np.uint8:
        raise TypeError("image dtype must be uint8")

    _init_model()

    pil_image = Image.fromarray(image, mode="RGB")
    tensor = _TRANSFORMS(pil_image).unsqueeze(0).to(_DEVICE)

    with torch.no_grad():
        if _TASK == "multitask":
            _, coords = _MODEL(tensor)
        else:
            coords = _MODEL(tensor)

    coord_norm = _PAYLOAD.get("coord_norm", "none")
    coord_stats = _PAYLOAD.get("coord_stats")
    if coord_norm != "none" and coord_stats is None:
        raise ValueError("Checkpoint missing coord_stats required to denormalize coords.")
    if coord_stats is None:
        coord_stats = {"mean": [0.0, 0.0], "std": [1.0, 1.0]}

    coords = _denormalize_coords(coords, coord_stats, coord_norm).squeeze(0)
    coord_mode = _PAYLOAD.get("coord_mode", "latlon")
    if coord_mode != "latlon":
        raise ValueError("predict_gps expects a latlon checkpoint. Retrain with --coord-mode latlon.")

    return coords.cpu().numpy().astype(np.float32)


__all__ = ["predict_gps"]
