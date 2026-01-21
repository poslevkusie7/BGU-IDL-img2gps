import argparse
import csv
import math
import statistics
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


def load_image_list(csv_path, id_column=None):
    with open(csv_path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        id_col = resolve_id_column(reader.fieldnames, id_column)
        image_ids = []
        for row in reader:
            value = (row.get(id_col) or "").strip()
            if value:
                image_ids.append(value)
    if not image_ids:
        raise ValueError(f"No image ids found in {csv_path} (column: {id_col}).")
    return image_ids


def resolve_image_path(image_dir, image_id):
    raw_path = Path(image_id)
    if raw_path.is_absolute():
        return raw_path
    if image_dir:
        return Path(image_dir) / image_id
    return raw_path


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
        "--image-csv",
        help="Optional CSV listing image ids to run (defaults to --gt-csv when provided).",
    )
    parser.add_argument(
        "--image-id-col",
        help="Optional column name for image id in the image CSV.",
    )
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
    parser.add_argument(
        "--use-gt-defaults",
        action="store_true",
        help="Use data/processed_images and data/gt.csv when image-dir/gt-csv are not provided.",
    )
    args = parser.parse_args()

    if args.use_gt_defaults:
        if not args.image_dir:
            args.image_dir = "data/processed_images"
        if not args.gt_csv:
            args.gt_csv = "data/gt.csv"

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
    if not args.gt_csv:
        parser.error("--gt-csv is required to compute dataset metrics.")

    if args.image_dir:
        image_csv = args.image_csv or args.gt_csv
        if image_csv:
            image_ids = load_image_list(image_csv, id_column=args.image_id_col or args.gt_id_col)
            image_paths = [resolve_image_path(args.image_dir, image_id) for image_id in image_ids]
        else:
            image_paths = sorted(
                p for p in Path(args.image_dir).iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
            )
        if not image_paths:
            raise FileNotFoundError(f"No images found in {args.image_dir}")
    else:
        image_paths = [Path(args.image)]

    distances = []
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

        coord_norm = payload.get("coord_norm", "none")
        coord_stats = payload.get("coord_stats")
        if coord_norm != "none" and coord_stats is None:
            raise ValueError("Checkpoint missing coord_stats required to denormalize coords.")
        if coord_stats is None:
            coord_stats = {"mean": [0.0, 0.0], "std": [1.0, 1.0]}
        coords = denormalize_coords(coords, coord_stats, coord_norm).cpu().numpy()[0]

        coord_mode = payload.get("coord_mode", "latlon")
        image_id = args.gt_image_id or image_path.name
        row = load_ground_truth(args.gt_csv, image_id, id_column=args.gt_id_col)
        gt_lat, gt_lon = extract_latlon(row)
        if coord_mode == "latlon":
            dist_m = haversine_m(coords[0], coords[1], gt_lat, gt_lon)
        else:
            try:
                import utm
            except ImportError as exc:
                raise ImportError("utm is required to compare UTM predictions with lat/lon ground truth.") from exc
            gt_e, gt_n, _, _ = utm.from_latlon(gt_lat, gt_lon)
            dist_m = math.hypot(coords[0] - gt_e, coords[1] - gt_n)
        distances.append(float(dist_m))

    if distances:
        count = len(distances)
        mean_dist = sum(distances) / count
        median_dist = statistics.median(distances)
        rmse_dist = math.sqrt(sum(d * d for d in distances) / count)
        p10 = sum(d <= 10.0 for d in distances) / count
        p25 = sum(d <= 25.0 for d in distances) / count
        print(
            "metrics: "
            f"val_dist_m={mean_dist:.2f} "
            f"median_dist_m={median_dist:.2f} "
            f"rmse_dist_m={rmse_dist:.2f} "
            f"p10m={p10:.3f} "
            f"p25m={p25:.3f}"
        )


if __name__ == "__main__":
    main()
