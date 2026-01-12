import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def load_points(csv_path):
    df = pd.read_csv(csv_path)
    cols = df.columns

    if "lat" in cols and "lon" in cols:
        lat = df["lat"]
        lon = df["lon"]
    elif "latitude" in cols and "longitude" in cols:
        lat = df["latitude"]
        lon = df["longitude"]
    else:
        raise KeyError("CSV must contain lat/lon or latitude/longitude columns.")

    labels = None
    if "sector_label" in cols:
        labels = df["sector_label"]
    elif "region" in cols:
        labels = df["region"]

    return lat, lon, labels


def plot_points(lat, lon, labels=None, out_path=None, show=False, title=None):
    plt.figure(figsize=(7, 7))

    if labels is None:
        plt.scatter(lon, lat, s=6, alpha=0.7)
    else:
        # Factorize labels for stable colors.
        codes, uniques = pd.factorize(labels)
        sc = plt.scatter(lon, lat, c=codes, s=8, alpha=0.75, cmap="tab20")
        cbar = plt.colorbar(sc, shrink=0.9)
        cbar.set_label("sector")

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    if title:
        plt.title(title)
    plt.tight_layout()

    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=200)
    if show:
        plt.show()
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot dataset GPS points.")
    parser.add_argument("--csv", default="data/metadata.csv", help="Path to CSV file.")
    parser.add_argument("--out", default="plots/points.png", help="Output image path.")
    parser.add_argument("--show", action="store_true", help="Show the plot window.")
    parser.add_argument("--title", default="Dataset GPS points", help="Plot title.")
    args = parser.parse_args()

    lat, lon, labels = load_points(args.csv)
    plot_points(lat, lon, labels=labels, out_path=args.out, show=args.show, title=args.title)


if __name__ == "__main__":
    main()
