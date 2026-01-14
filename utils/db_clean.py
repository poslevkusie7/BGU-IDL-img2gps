import pandas as pd
import numpy as np
import folium
from sklearn.cluster import DBSCAN

# --- 1. Define the Plotting Function ---
def plot_results(df, output_html):
    """
    Plots the dataframe on a Folium map.
    Valid points (Cluster != -1) are Green.
    Outliers (Cluster == -1) are Red.
    """
    if len(df) == 0:
        print("No data to plot.")
        return

    center_lat = df['lat'].mean()
    center_lon = df['lon'].mean()
    
    m = folium.Map(location=[center_lat, center_lon], zoom_start=17) # Zoomed in for buildings

    for index, row in df.iterrows():
        # Determine color based on DBSCAN result
        # Cluster -1 is noise (outlier)
        if row['cluster'] == -1:
            color = "red"
            popup_text = f"Outlier (Point {index})"
        else:
            color = "green"
            popup_text = f"Cluster {row['cluster']} (Point {index})"

        folium.Marker(
            location=[row['lat'], row['lon']],
            popup=popup_text,
            icon=folium.Icon(color=color, icon="info-sign")
        ).add_to(m)

    m.save(output_html)
    print(f"Map saved to {output_html}")

# --- 2. Main Processing Script ---

# Load your data
# Make sure your CSV has 'lat' and 'lon' columns
df = pd.read_csv('data/metadata.csv') 

# Convert Lat/Lon to Radians for Haversine metric
coords = np.radians(df[['lat', 'lon']].values)

# --- DBSCAN Configuration ---
kms_per_radian = 6371.0088
epsilon_dist_meters = 10  # 15 meters max distance
eps_rad = epsilon_dist_meters / 1000 / kms_per_radian

# Note: I changed min_samples from 50 to 5. 
# 50 is very strict; it means a building needs 50+ points to be valid.
# 5 is safer for smaller clusters.
min_samples = 50

# Run DBSCAN
db = DBSCAN(eps=eps_rad, min_samples=min_samples, metric='haversine', algorithm='ball_tree').fit(coords)

# Assign labels to the DataFrame
df['cluster'] = db.labels_

# --- 3. Filter and Plot ---

# Separate the data if you want to save clean files
clean_df = df[df['cluster'] != -1]
outliers_df = df[df['cluster'] == -1]

print(f"Total points: {len(df)}")
print(f"Valid points: {len(clean_df)}")
print(f"Outliers found: {len(outliers_df)}")

clean_df.to_csv('data/metadata1.csv', index=False)
print("Saved clean data to data/metadata1.csv")

# Plot the graph with colors
plot_results(df, 'clustered_map.html')