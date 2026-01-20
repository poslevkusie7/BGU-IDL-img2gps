import argparse
import csv
import math
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


def haversine_m(lat1, lon1, lat2, lon2):
    radius = 6371000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    c = 2 * math.asin(min(1.0, math.sqrt(a)))
    return radius * c


def resolve_id_column(fieldnames, preferred):
    if not fieldnames:
        raise ValueError("Ground-truth CSV must include a header row.")
    field_map = {name.lower(): name for name in fieldnames}
    if preferred:
        key = preferred.lower()
        if key in field_map:
            return field_map[key]
        raise ValueError(f"Column '{preferred}' not found in ground-truth CSV.")
    for candidate in ("image_id", "filename", "file", "image", "name"):
        if candidate in field_map:
            return field_map[candidate]
    raise ValueError("Ground-truth CSV must include an image id column (e.g. image_id or filename).")


def load_ground_truth(csv_path, image_name, id_column=None):
    with open(csv_path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        id_col = resolve_id_column(reader.fieldnames, id_column)
        for row in reader:
            raw_id = row.get(id_col, "")
            if raw_id == image_name:
                return row
            if Path(raw_id).name == image_name:
                return row
            if Path(raw_id).stem == Path(image_name).stem:
                return row
    raise ValueError(f"Image id '{image_name}' not found in {csv_path} (column: {id_col}).")


def extract_latlon(row):
    if "lat" in row and "lon" in row:
        return float(row["lat"]), float(row["lon"])
    raise ValueError("Ground-truth CSV must include 'lat' and 'lon' columns.")


def main():
    parser = argparse.ArgumentParser(description="Run inference on a single image.")
    parser.add_argument("--image", help="Path to an input image.")
    parser.add_argument("--image-dir", help="Path to a folder of images to run inference on.")
    parser.add_argument(
        "--checkpoint",
        default="runs/coords_best.pt",
        help="Path to a .pt checkpoint (default: runs/multitask_best.pt).",
    )
    parser.add_argument(
        "--dinov2-name",
        default="vit_base_patch14_dinov2.lvd142m",
        help="Backbone name used during training.",
    )
    parser.add_argument("--img-size", type=int, default=518, help="Resize input to this size.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument(
        "--gt-csv",
        "--gt",
        dest="gt_csv",
        help="Optional CSV with ground-truth lat/lon and image id (e.g. image_id or filename).",
    )
    parser.add_argument(
        "--gt-image-id",
        help="Override image_id lookup in the ground-truth CSV (default: image basename).",
    )
    parser.add_argument(
        "--gt-id-col",
        help="Optional column name for image id in the ground-truth CSV (e.g. filename).",
    )
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

    if not args.image and not args.image_dir:
        parser.error("Either --image or --image-dir is required.")

    if args.image_dir:
        image_paths = sorted(
            p for p in Path(args.image_dir).iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
        )
        if not image_paths:
            raise FileNotFoundError(f"No images found in {args.image_dir}")
    else:
        image_paths = [Path(args.image)]

    coord_norm = payload.get("coord_norm", "none")
    coord_stats = payload.get("coord_stats")
    if coord_norm != "none" and coord_stats is None:
        raise ValueError("Checkpoint missing coord_stats required to denormalize coords.")
    if coord_stats is None:
        coord_stats = {"mean": [0.0, 0.0], "std": [1.0, 1.0]}

    coord_mode = payload.get("coord_mode", "latlon")

    for image_path in image_paths:
        image = Image.open(image_path).convert("RGB")
        tensor = build_transforms(args.img_size)(image).unsqueeze(0).to(device)

        with torch.no_grad():
            if task == "multitask":
                logits, coords = model(tensor)
                pred_class = int(logits.argmax(dim=1).item())
            else:
                coords = model(tensor)
                pred_class = None

        coords = denormalize_coords(coords, coord_stats, coord_norm).cpu().numpy()[0]

        if coord_mode == "latlon":
            pred_text = f"Predicted lat/lon: {coords[0]:.6f}, {coords[1]:.6f}"
        else:
            pred_text = f"Predicted UTM (easting, northing): {coords[0]:.3f}, {coords[1]:.3f}"
        print(f"{image_path.name} -> {pred_text}")
        if pred_class is not None:
            print(f"{image_path.name} -> Predicted class: {pred_class}")

        if args.gt_csv:
            image_id = args.gt_image_id or image_path.name
            row = load_ground_truth(args.gt_csv, image_id, id_column=args.gt_id_col)
            gt_lat, gt_lon = extract_latlon(row)
            if coord_mode == "latlon":
                dist_m = haversine_m(coords[0], coords[1], gt_lat, gt_lon)
                print(f"{image_path.name} -> GT lat/lon: {gt_lat:.6f}, {gt_lon:.6f}")
            else:
                try:
                    import utm
                except ImportError as exc:
                    raise ImportError("utm is required to compare UTM predictions with lat/lon ground truth.") from exc
                gt_e, gt_n, _, _ = utm.from_latlon(gt_lat, gt_lon)
                dist_m = math.hypot(coords[0] - gt_e, coords[1] - gt_n)
                print(f"{image_path.name} -> GT lat/lon: {gt_lat:.6f}, {gt_lon:.6f}")
                print(f"{image_path.name} -> GT UTM (easting, northing): {gt_e:.3f}, {gt_n:.3f}")
            print(f"{image_path.name} -> Distance to GT (m): {dist_m:.3f}")


if __name__ == "__main__":
    main()
