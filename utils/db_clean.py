import argparse

import folium
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN


def plot_results(df, output_html):
    """
    Plot DBSCAN results on a Folium map.
    Valid points (cluster != -1) are green, outliers (-1) are red.
    """
    if len(df) == 0:
        print("No data to plot.")
        return

    center_lat = df["lat"].mean()
    center_lon = df["lon"].mean()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=17)

    for index, row in df.iterrows():
        if row["cluster"] == -1:
            color = "red"
            popup_text = f"Outlier (Point {index})"
        else:
            color = "green"
            popup_text = f"Cluster {row['cluster']} (Point {index})"

        folium.Marker(
            location=[row["lat"], row["lon"]],
            popup=popup_text,
            icon=folium.Icon(color=color, icon="info-sign"),
        ).add_to(m)

    m.save(output_html)
    print(f"Map saved to {output_html}")


def clean_with_dbscan(input_csv, output_csv, output_html, eps_meters=10.0, min_samples=5):
    df = pd.read_csv(input_csv)
    if {"lat", "lon"} - set(df.columns):
        raise KeyError("CSV must contain 'lat' and 'lon' columns.")

    coords_rad = np.radians(df[["lat", "lon"]].values)
    kms_per_radian = 6371.0088
    eps_rad = eps_meters / 1000.0 / kms_per_radian

    db = DBSCAN(
        eps=eps_rad,
        min_samples=min_samples,
        metric="haversine",
        algorithm="ball_tree",
    ).fit(coords_rad)

    df["cluster"] = db.labels_
    clean_df = df[df["cluster"] != -1]
    outliers_df = df[df["cluster"] == -1]

    print(f"Total points: {len(df)}")
    print(f"Valid points: {len(clean_df)}")
    print(f"Outliers found: {len(outliers_df)}")

    clean_df.to_csv(output_csv, index=False)
    print(f"Saved clean data to {output_csv}")

    plot_results(df, output_html)
    return clean_df, outliers_df


def main():
    parser = argparse.ArgumentParser(description="DBSCAN-based GPS outlier removal.")
    parser.add_argument("--input-csv", default="data/metadata.csv")
    parser.add_argument("--output-csv", default="data/metadata1.csv")
    parser.add_argument("--output-html", default="clustered_map.html")
    parser.add_argument("--eps-meters", type=float, default=10.0, help="Neighborhood radius in meters.")
    parser.add_argument(
        "--min-samples",
        type=int,
        default=5,
        help="Minimum points to form a cluster (noise labeled -1).",
    )
    args = parser.parse_args()
    clean_with_dbscan(args.input_csv, args.output_csv, args.output_html, args.eps_meters, args.min_samples)


if __name__ == "__main__":
    main()
