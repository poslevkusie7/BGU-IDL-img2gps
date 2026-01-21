import argparse
import csv
import math
import os
import random
import shutil
from pathlib import Path


def _read_rows(csv_path):
    with open(csv_path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if not rows:
        raise ValueError(f"No rows found in {csv_path}")
    return reader.fieldnames, rows


def _grid_group(lat, lon, grid_meters):
    lat_m = lat * 111000.0
    lon_m = lon * 111000.0 * math.cos(math.radians(lat))
    lat_bin = int(math.floor(lat_m / grid_meters))
    lon_bin = int(math.floor(lon_m / grid_meters))
    return f"grid:{lat_bin}:{lon_bin}"


def _group_rows(fieldnames, rows, group_col, grid_meters):
    groups = {}
    group_col = None if (group_col or "").lower() == "none" else group_col
    use_col = group_col and group_col in fieldnames

    for idx, row in enumerate(rows):
        group_id = None
        if use_col:
            raw = row.get(group_col, "").strip()
            if raw and raw != "-1":
                group_id = f"{group_col}:{raw}"
        if group_id is None:
            if "lat" in row and "lon" in row:
                lat = float(row["lat"])
                lon = float(row["lon"])
                group_id = _grid_group(lat, lon, grid_meters)
            else:
                group_id = f"row:{idx}"

        groups.setdefault(group_id, []).append(idx)
    return groups


def _pick_groups(groups, target_count):
    selected = []
    remaining = []
    count = 0
    for group_id, idxs in groups:
        if count < target_count:
            selected.append((group_id, idxs))
            count += len(idxs)
        else:
            remaining.append((group_id, idxs))
    return selected, remaining, count


def _flatten(groups):
    indices = []
    for _, idxs in groups:
        indices.extend(idxs)
    return indices


def split_dataset(rows, groups, seed, test_frac, val_frac, val_from_train):
    total = len(rows)
    rng = random.Random(seed)
    items = list(groups.items())
    rng.shuffle(items)

    test_target = int(round(total * test_frac))
    test_groups, remaining, test_count = _pick_groups(items, test_target)

    if val_from_train:
        remaining_total = total - test_count
        val_target = int(round(remaining_total * val_frac))
    else:
        val_target = int(round(total * val_frac))

    val_groups, train_groups, val_count = _pick_groups(remaining, val_target)

    train_idx = _flatten(train_groups)
    val_idx = _flatten(val_groups)
    test_idx = _flatten(test_groups)

    return (
        [rows[i] for i in train_idx],
        [rows[i] for i in val_idx],
        [rows[i] for i in test_idx],
    )


def _write_csv(path, fieldnames, rows):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _copy_images(rows, images_dir, output_images_dir, split_name):
    src_root = Path(images_dir)
    dst_root = Path(output_images_dir) / split_name
    dst_root.mkdir(parents=True, exist_ok=True)
    missing = 0
    for row in rows:
        image_id = row.get("image_id", "")
        if not image_id:
            continue
        src_path = src_root / image_id
        if not src_path.exists():
            missing += 1
            continue
        shutil.copy2(src_path, dst_root / src_path.name)
    return missing


def main():
    parser = argparse.ArgumentParser(
        description="Split metadata CSV into train/val/test with optional grouping."
    )
    parser.add_argument("--input-csv", default="data/metadata2.csv")
    parser.add_argument("--output-dir", default="data/splits")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-frac", type=float, default=0.2)
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument(
        "--val-from-total",
        action="store_true",
        help="Interpret val-frac as a fraction of total (default: fraction of train split).",
    )
    parser.add_argument(
        "--group-col",
        default="cluster",
        help="Column used to group similar images (set to 'none' to disable).",
    )
    parser.add_argument(
        "--grid-meters",
        type=float,
        default=5.0,
        help="Grid size (meters) for spatial grouping when group-col is missing.",
    )
    parser.add_argument("--images-dir", default=None)
    parser.add_argument(
        "--copy-images",
        action="store_true",
        help="Copy images into train/val/test folders (disabled by default).",
    )
    parser.add_argument("--output-images-dir", default="data/images_split")
    args = parser.parse_args()

    fieldnames, rows = _read_rows(args.input_csv)
    if args.group_col and args.group_col.lower() != "none" and args.group_col not in fieldnames:
        print(f"Group column '{args.group_col}' not found; falling back to spatial grid.")
    groups = _group_rows(fieldnames, rows, args.group_col, args.grid_meters)

    train_rows, val_rows, test_rows = split_dataset(
        rows,
        groups,
        args.seed,
        args.test_frac,
        args.val_frac,
        val_from_train=not args.val_from_total,
    )

    out_dir = args.output_dir
    train_path = os.path.join(out_dir, "train.csv")
    val_path = os.path.join(out_dir, "val.csv")
    test_path = os.path.join(out_dir, "test.csv")
    _write_csv(train_path, fieldnames, train_rows)
    _write_csv(val_path, fieldnames, val_rows)
    _write_csv(test_path, fieldnames, test_rows)

    print(f"Train rows: {len(train_rows)}")
    print(f"Val rows: {len(val_rows)}")
    print(f"Test rows: {len(test_rows)}")
    print(f"Wrote: {train_path}")
    print(f"Wrote: {val_path}")
    print(f"Wrote: {test_path}")

    if args.copy_images:
        if not args.images_dir:
            raise ValueError("--images-dir is required when --copy-images is set.")
        missing_train = _copy_images(train_rows, args.images_dir, args.output_images_dir, "train")
        missing_val = _copy_images(val_rows, args.images_dir, args.output_images_dir, "val")
        missing_test = _copy_images(test_rows, args.images_dir, args.output_images_dir, "test")
        missing = missing_train + missing_val + missing_test
        print(f"Copied images to {args.output_images_dir} (missing: {missing})")


if __name__ == "__main__":
    main()
