import os
import csv
import shutil
from PIL import Image
from PIL.ExifTags import GPSTAGS
from PIL.TiffImagePlugin import IFDRational

import pillow_heif
pillow_heif.register_heif_opener()

# ================= CONFIG =================

ROOT_FOLDER      = "."
TEMP_FOLDER      = "_tmp_flat"
RESIZED_FOLDER   = "resized"
CSV_FILE         = "labels.csv"
TARGET_SHORT_SIDE = 518

# CSV Columns
FILE_NAME = "image_id"
REGION = "sector_label"
LAT = "lat"
LON = "lon"

# =========================================


def get_gps(img_path):
    try:
        with Image.open(img_path) as img:
            exif = img.getexif()
            if not exif:
                return None, None

            gps_ifd = exif.get_ifd(34853)
            if not gps_ifd:
                return None, None

            gps = {}
            for k, v in gps_ifd.items():
                gps[GPSTAGS.get(k, k)] = v

            if not all(x in gps for x in (
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


def extract_region(filename):
    # expects region_<i>_....
    parts = filename.split("_", 2)
    if len(parts) >= 2 and parts[0] == "region":
        return parts[1]
    return None


def resize_image(img, target):
    w, h = img.size
    scale = target / min(w, h)
    return img.resize((int(w * scale), int(h * scale)),
                      Image.Resampling.LANCZOS)


def stage1_flatten():
    os.makedirs(TEMP_FOLDER, exist_ok=True)

    for region in sorted(os.listdir(ROOT_FOLDER)):
        region_path = os.path.join(ROOT_FOLDER, region)
        if not region.isdigit() or not os.path.isdir(region_path):
            continue

        for fname in os.listdir(region_path):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png", ".heic")):
                continue

            src = os.path.join(region_path, fname)
            base = os.path.splitext(fname)[0]
            out_name = f"region_{region}_{base}.jpg"
            dst = os.path.join(TEMP_FOLDER, out_name)

            with Image.open(src) as img:
                exif = img.getexif()
                img = img.convert("RGB")
                img.save(dst, exif=exif, quality=95)


def stage2_process():
    os.makedirs(RESIZED_FOLDER, exist_ok=True)
    rows = []

    for fname in sorted(os.listdir(TEMP_FOLDER)):
        src = os.path.join(TEMP_FOLDER, fname)

        region = extract_region(fname)
        if region is None:
            continue

        lat, lon = get_gps(src)

        with Image.open(src) as img:
            exif = img.getexif()
            resized = resize_image(img, TARGET_SHORT_SIDE)
            resized.save(
                os.path.join(RESIZED_FOLDER, fname),
                exif=exif,
                quality=92,
                optimize=True
            )
            
        if lat is not None and lon is not None:
            rows.append([
                fname,
                region,
                f"{lat:.8f}",
                f"{lon:.8f}"
            ])

    with open(CSV_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([FILE_NAME, REGION, LAT, LON])
        writer.writerows(rows)


def main():
    print("Stage 1: flatten & rename")
    stage1_flatten()

    print("Stage 2: resize + GPS + CSV")
    stage2_process()

    print("Cleanup")
    shutil.rmtree(TEMP_FOLDER)

    print("DONE")
    print(f"Resized images → {RESIZED_FOLDER}")
    print(f"CSV labels     → {CSV_FILE}")


if __name__ == "__main__":
    main()
