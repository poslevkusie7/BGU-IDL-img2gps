import os
import csv
from PIL import Image, ImageOps
from PIL.ExifTags import GPSTAGS
from PIL.TiffImagePlugin import IFDRational
from tqdm import tqdm
import pillow_heif

pillow_heif.register_heif_opener()

# ================= CONFIG =================

ROOT_FOLDER       = "data_set"        # input images: data_set/<region>/<image>
RESIZED_FOLDER    = "data/images"     # final images
CSV_FILE          = "data/metadata.csv"     # CSV output
TARGET_WIDTH      = 518               # final width (3:4 → height = 690)

# CSV Columns
FILE_NAME = "image_id"
REGION    = "sector_label"
LAT       = "lat"
LON       = "lon"

# =========================================


def get_gps(img_path):
    """Extract GPS coordinates from original image EXIF"""
    try:
        with Image.open(img_path) as img:
            exif = img.getexif()
            if not exif:
                return None, None

            gps_ifd = exif.get_ifd(34853)
            if not gps_ifd:
                return None, None

            gps = {GPSTAGS.get(k, k): v for k, v in gps_ifd.items()}

            if not all(k in gps for k in (
                "GPSLatitude", "GPSLatitudeRef",
                "GPSLongitude", "GPSLongitudeRef"
            )):
                return None, None

            def frac(x):
                if isinstance(x, IFDRational):
                    return float(x)
                if isinstance(x, tuple):
                    return x[0] / x[1]
                return float(x)

            def dms_to_decimal(dms, ref):
                deg = frac(dms[0])
                min_ = frac(dms[1]) / 60
                sec = frac(dms[2]) / 3600
                val = deg + min_ + sec
                return -val if ref in ("S", "W") else val

            lat = dms_to_decimal(gps["GPSLatitude"], gps["GPSLatitudeRef"])
            lon = dms_to_decimal(gps["GPSLongitude"], gps["GPSLongitudeRef"])
            return lat, lon

    except Exception:
        return None, None


def crop_to_3_4_vertical(img):
    """Center-crop to strict 3:4 vertical (W:H = 3:4)"""
    w, h = img.size
    target_ratio = 4 / 3        # H / W
    current_ratio = h / w

    if current_ratio > target_ratio:
        new_h = int(w * target_ratio)
        top = (h - new_h) // 2
        img = img.crop((0, top, w, top + new_h))
    elif current_ratio < target_ratio:
        new_w = int(h / target_ratio)
        left = (w - new_w) // 2
        img = img.crop((left, 0, left + new_w, h))

    return img


def resize_to_width(img, target_width):
    w, h = img.size
    scale = target_width / w
    return img.resize(
        (target_width, int(h * scale)),
        Image.Resampling.LANCZOS
    )


def process_images():
    os.makedirs(RESIZED_FOLDER, exist_ok=True)
    os.makedirs(os.path.dirname(CSV_FILE), exist_ok=True)

    rows = []

    regions = sorted(
        d for d in os.listdir(ROOT_FOLDER)
        if d.isdigit() and os.path.isdir(os.path.join(ROOT_FOLDER, d))
    )

    for region in tqdm(regions, desc="Regions"):
        region_path = os.path.join(ROOT_FOLDER, region)

        files = sorted(
            f for f in os.listdir(region_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".heic"))
        )

        for fname in tqdm(files, desc=f"Region {region}", leave=False):
            src = os.path.join(region_path, fname)
            base = os.path.splitext(fname)[0]
            out_name = f"region_{region}_{base}.jpg"
            dst = os.path.join(RESIZED_FOLDER, out_name)

            try:
                with Image.open(src) as img:
                    exif = img.getexif()

                    # 1) Fix orientation ONCE
                    img = ImageOps.exif_transpose(img)
                    img = img.convert("RGB")

                    # Remove EXIF orientation to avoid double rotation
                    if 274 in exif:   # Orientation tag
                        del exif[274]

                    # 2) Force 3:4 vertical crop
                    img = crop_to_3_4_vertical(img)

                    # 3) Resize → 518 × 690
                    img = resize_to_width(img, TARGET_WIDTH)

                    img.save(dst, exif=exif, quality=92, optimize=True)

                # GPS from original image
                lat, lon = get_gps(src)
                if lat is not None and lon is not None:
                    rows.append([
                        out_name,
                        region,
                        f"{lat:.8f}",
                        f"{lon:.8f}"
                    ])

            except Exception as e:
                print(f"Failed {src}: {e}")

    # Write CSV
    with open(CSV_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([FILE_NAME, REGION, LAT, LON])
        writer.writerows(rows)


def main():
    print("Processing images: orientation → 3:4 crop → resize → GPS → CSV")
    process_images()
    print("DONE")
    print(f"Images → {RESIZED_FOLDER}")
    print(f"CSV    → {CSV_FILE}")


if __name__ == "__main__":
    main()
