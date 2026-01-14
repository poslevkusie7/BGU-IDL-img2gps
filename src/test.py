import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform

# Load your data
df = pd.read_csv("data/metadata.csv")

print("="*60)
print("DATASET STATISTICS")
print("="*60)

# Basic stats
print(f"\nTotal images: {len(df)}")
print(f"Unique sectors: {df['sector_label'].nunique()}")
print(f"\nImages per sector:")
print(df['sector_label'].value_counts().sort_index())

# GPS spread analysis
lats = df['lat'].values
lons = df['lon'].values

# Convert to meters (approximate)
lat_range_m = (lats.max() - lats.min()) * 111000  # 1 degree lat ≈ 111km
lon_range_m = (lons.max() - lons.min()) * 111000 * np.cos(np.radians(lats.mean()))

print(f"\n" + "="*60)
print("GEOGRAPHIC COVERAGE")
print("="*60)
print(f"Latitude range: {lats.min():.6f} to {lats.max():.6f}")
print(f"Longitude range: {lons.min():.6f} to {lons.max():.6f}")
print(f"Coverage area: {lat_range_m/1000:.2f} km × {lon_range_m/1000:.2f} km")
print(f"Total area: ~{(lat_range_m * lon_range_m) / 1e6:.2f} km²")

# Distance analysis
coords = np.column_stack([lats, lons])
coords_rad = np.radians(coords)

# Haversine distances (sample if too large)
if len(df) > 1000:
    sample_idx = np.random.choice(len(df), 1000, replace=False)
    sample_coords = coords_rad[sample_idx]
else:
    sample_coords = coords_rad

# Compute pairwise distances
from scipy.spatial.distance import pdist
distances = pdist(sample_coords, metric='euclidean') * 6371000  # Earth radius in meters

print(f"\n" + "="*60)
print("DISTANCE BETWEEN IMAGES")
print("="*60)
print(f"Minimum distance: {distances.min():.1f} m")
print(f"Average distance: {distances.mean():.1f} m")
print(f"Median distance: {np.median(distances):.1f} m")
print(f"Maximum distance: {distances.max():.1f} m")

# Nearest neighbor analysis
dist_matrix = squareform(distances)
np.fill_diagonal(dist_matrix, np.inf)
nearest_neighbor_dists = dist_matrix.min(axis=1)

print(f"\nNearest neighbor distances:")
print(f"  Min: {nearest_neighbor_dists.min():.1f} m")
print(f"  Average: {nearest_neighbor_dists.mean():.1f} m")
print(f"  Median: {nearest_neighbor_dists.median():.1f} m")
print(f"  Max: {nearest_neighbor_dists.max():.1f} m")

# Diagnosis
print(f"\n" + "="*60)
print("DIAGNOSIS")
print("="*60)

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

# Recommendations
print(f"\n" + "="*60)
print("RECOMMENDATIONS")
print("="*60)

if len(df) < 5000 and (lat_range_m > 5000 or lon_range_m > 5000):
    print("1. Your dataset is too small for the geographic area")
    print("   → Collect more images OR focus on smaller region")
    
if nearest_neighbor_dists.mean() > 50:
    print("2. Images are too sparse for precise localization")
    print("   → Expected accuracy will be limited by image spacing")
    print(f"   → Best possible error: ~{nearest_neighbor_dists.mean()/2:.0f}m")

print("3. Try these hyperparameters:")
print(f"   pos_thresh = {min(15, nearest_neighbor_dists.mean()/3):.1f}")
print(f"   neg_thresh = {max(100, nearest_neighbor_dists.mean()*2):.1f}")
print(f"   lambda_gps = {10 if density > 50 else 20}")