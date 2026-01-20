import argparse
from pathlib import Path

from PIL import Image
import torch
from torchvision import transforms

from .model import DinoV2CoordRegressor, SharedDinoMultiTask


def build_transforms(img_size=518):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def denormalize_coords(coords, coord_stats, coord_norm):
    coords = coords.float()
    mean = torch.tensor(coord_stats["mean"], device=coords.device, dtype=torch.float32)
    std = torch.tensor(coord_stats["std"], device=coords.device, dtype=torch.float32)
    if coord_norm == "standard":
        return coords * std + mean
    if coord_norm == "center":
        return coords + mean
    return coords


def load_checkpoint(path, device):
    payload = torch.load(path, map_location=device)
    if not isinstance(payload, dict) or "model_state" not in payload:
        raise ValueError(f"Checkpoint {path} is missing a 'model_state' dict.")
    return payload


def build_model(payload, dinov2_name, device):
    task = payload.get("task", "coords")
    if task == "multitask":
        num_classes = payload.get("num_classes")
        if num_classes is None:
            raise ValueError("Multitask checkpoint missing 'num_classes'.")
        model = SharedDinoMultiTask(
            num_classes=int(num_classes),
            coord_model_name=dinov2_name,
            pretrained=False,
        )
    else:
        model = DinoV2CoordRegressor(
            pretrained=False,
            model_name=dinov2_name,
        )
    model.load_state_dict(payload["model_state"], strict=True)
    model.to(device)
    model.eval()
    return model, task


def main():
    parser = argparse.ArgumentParser(description="Run inference on a single image.")
    parser.add_argument("--image", required=True, help="Path to an input image.")
    parser.add_argument(
        "--checkpoint",
        default="runs/multitask_best.pt",
        help="Path to a .pt checkpoint (default: runs/multitask_best.pt).",
    )
    parser.add_argument(
        "--dinov2-name",
        default="vit_base_patch14_dinov2.lvd142m",
        help="Backbone name used during training.",
    )
    parser.add_argument("--img-size", type=int, default=518, help="Resize input to this size.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    payload = load_checkpoint(ckpt_path, device)
    model, task = build_model(payload, args.dinov2_name, device)

    image = Image.open(args.image).convert("RGB")
    tensor = build_transforms(args.img_size)(image).unsqueeze(0).to(device)

    with torch.no_grad():
        if task == "multitask":
            logits, coords = model(tensor)
            pred_class = int(logits.argmax(dim=1).item())
        else:
            coords = model(tensor)
            pred_class = None

    coord_norm = payload.get("coord_norm", "none")
    coord_stats = payload.get("coord_stats")
    if coord_norm != "none" and coord_stats is None:
        raise ValueError("Checkpoint missing coord_stats required to denormalize coords.")
    if coord_stats is None:
        coord_stats = {"mean": [0.0, 0.0], "std": [1.0, 1.0]}
    coords = denormalize_coords(coords, coord_stats, coord_norm).cpu().numpy()[0]

    coord_mode = payload.get("coord_mode", "latlon")
    if coord_mode == "latlon":
        print(f"Predicted lat/lon: {coords[0]:.6f}, {coords[1]:.6f}")
    else:
        print(f"Predicted UTM (easting, northing): {coords[0]:.3f}, {coords[1]:.3f}")
    if pred_class is not None:
        print(f"Predicted class: {pred_class}")


if __name__ == "__main__":
    main()
