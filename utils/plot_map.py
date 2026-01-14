import pandas as pd
import folium

def plot_with_folium(csv_path, output_html):
    # 1. Read the CSV
    df = pd.read_csv(csv_path)

    # 2. Create a map centered on the average coordinates
    center_lat = df['lat'].mean()
    center_lon = df['lon'].mean()
    
    m = folium.Map(location=[center_lat, center_lon], zoom_start=13)

    # 3. Add a marker for each point
    for index, row in df.iterrows():
        folium.Marker(
            location=[row['lat'], row['lon']],
            popup=f"Point {index}", # Text when you click the marker
            icon=folium.Icon(color="red", icon="info-sign")
        ).add_to(m)

    # 4. Save the map
    m.save(output_html)
    print(f"Map saved to {output_html}")

# Usage
plot_with_folium('data/metadata.csv', 'my_osm_map.html')