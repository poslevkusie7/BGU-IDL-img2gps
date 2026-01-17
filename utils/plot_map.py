import argparse

import folium
import pandas as pd


def plot_with_folium(csv_path, output_html, zoom=13):
    """Render all lat/lon points from CSV to a Folium map."""
    df = pd.read_csv(csv_path)
    if {"lat", "lon"} - set(df.columns):
        raise KeyError("CSV must contain 'lat' and 'lon' columns.")

    center_lat = df["lat"].mean()
    center_lon = df["lon"].mean()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=zoom)

    for index, row in df.iterrows():
        folium.Marker(
            location=[row["lat"], row["lon"]],
            popup=f"Point {index}",
            icon=folium.Icon(color="red", icon="info-sign"),
        ).add_to(m)

    m.save(output_html)
    print(f"Map saved to {output_html}")


def main():
    parser = argparse.ArgumentParser(description="Plot lat/lon CSV onto an interactive map.")
    parser.add_argument("--csv-path", default="data/metadata1.csv")
    parser.add_argument("--output-html", default="map.html")
    parser.add_argument("--zoom", type=int, default=13)
    args = parser.parse_args()
    plot_with_folium(args.csv_path, args.output_html, zoom=args.zoom)


if __name__ == "__main__":
    main()
