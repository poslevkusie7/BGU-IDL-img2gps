import argparse

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform


def describe_dataset(csv_path):
    df = pd.read_csv(csv_path)
    if {"lat", "lon", "sector_label"} - set(df.columns):
        raise KeyError("CSV must contain 'lat', 'lon', and 'sector_label' columns.")

    print("=" * 60)
    print("DATASET STATISTICS")
    print("=" * 60)
    print(f"\nTotal images: {len(df)}")
    print(f"Unique sectors: {df['sector_label'].nunique()}")
    print(f"\nImages per sector:")
    print(df["sector_label"].value_counts().sort_index())

    lats = df["lat"].values
    lons = df["lon"].values
    lat_range_m = (lats.max() - lats.min()) * 111000  # 1 degree lat ≈ 111km
    lon_range_m = (lons.max() - lons.min()) * 111000 * np.cos(np.radians(lats.mean()))

    print(f"\n" + "=" * 60)
    print("GEOGRAPHIC COVERAGE")
    print("=" * 60)
    print(f"Latitude range: {lats.min():.6f} to {lats.max():.6f}")
    print(f"Longitude range: {lons.min():.6f} to {lons.max():.6f}")
    print(f"Coverage area: {lat_range_m/1000:.2f} km × {lon_range_m/1000:.2f} km")
    print(f"Total area: ~{(lat_range_m * lon_range_m) / 1e6:.2f} km²")

    coords_rad = np.radians(np.column_stack([lats, lons]))
    if len(df) > 1000:
        sample_idx = np.random.choice(len(df), 1000, replace=False)
        sample_coords = coords_rad[sample_idx]
    else:
        sample_coords = coords_rad

    distances = pdist(sample_coords, metric="euclidean") * 6371000  # Earth radius
    print(f"\n" + "=" * 60)
    print("DISTANCE BETWEEN IMAGES")
    print("=" * 60)
    print(f"Minimum distance: {distances.min():.1f} m")
    print(f"Average distance: {distances.mean():.1f} m")
    print(f"Median distance: {np.median(distances):.1f} m")
    print(f"Maximum distance: {distances.max():.1f} m")

    dist_matrix = squareform(distances)
    np.fill_diagonal(dist_matrix, np.inf)
    nearest_neighbor_dists = dist_matrix.min(axis=1)

    print(f"\nNearest neighbor distances:")
    print(f"  Min: {nearest_neighbor_dists.min():.1f} m")
    print(f"  Average: {nearest_neighbor_dists.mean():.1f} m")
    print(f"  Median: {np.median(nearest_neighbor_dists):.1f} m")
    print(f"  Max: {nearest_neighbor_dists.max():.1f} m")

    print(f"\n" + "=" * 60)
    print("DIAGNOSIS")
    print("=" * 60)

    density = len(df) / ((lat_range_m * lon_range_m) / 1e6)
    print(f"Image density: {density:.1f} images/km²")

    if len(df) < 1000:
        print("⚠️  WARNING: Dataset very small (<1000 images)")
        print("   → GPS localization will be extremely difficult")

    if density < 10:
        print("⚠️  WARNING: Low image density")
        print("   → Not enough images per area for the model to learn")

    if nearest_neighbor_dists.mean() > 100:
        print("⚠️  WARNING: Images are very sparse")
        print(f"   → Average spacing: {nearest_neighbor_dists.mean():.0f}m")
        print("   → Model can't learn fine-grained localization")

    if lat_range_m > 10000 or lon_range_m > 10000:
        print("⚠️  WARNING: Very large geographic area")
        print("   → Consider training separate models per region")

    print(f"\n" + "=" * 60)
    print("RECOMMENDATIONS")
    print("=" * 60)

    if len(df) < 5000 and (lat_range_m > 5000 or lon_range_m > 5000):
        print("1. Dataset may be too small for the geographic area")
        print("   → Collect more images OR focus on a smaller region")

    if nearest_neighbor_dists.mean() > 50:
        print("2. Images are sparse; expected error bounded by spacing")
        print(f"   → Best possible error: ~{nearest_neighbor_dists.mean()/2:.0f}m")

    print("3. Suggested hyperparameters (for triplet-style objectives):")
    print(f"   pos_thresh = {min(15, nearest_neighbor_dists.mean()/3):.1f}")
    print(f"   neg_thresh = {max(100, nearest_neighbor_dists.mean()*2):.1f}")
    print(f"   lambda_gps = {10 if density > 50 else 20}")


def main():
    parser = argparse.ArgumentParser(description="Quick dataset stats and heuristics.")
    parser.add_argument("--csv-path", default="dataset_root/metadata.csv")
    args = parser.parse_args()
    describe_dataset(args.csv_path)


if __name__ == "__main__":
    main()
