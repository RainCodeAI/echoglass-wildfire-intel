import pandas as pd
import folium
import os
from folium.plugins import MarkerCluster

# Load data
base_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(base_dir, '..', 'data', 'raw', 'NFDB_point_20240613.txt')
df = pd.read_csv(file_path, encoding='latin1')

# Drop missing coordinates or size, sample for performance
sample_df = df[['LATITUDE', 'LONGITUDE', 'SIZE_HA', 'CAUSE']].dropna().sample(1000, random_state=42)

# Init the map
m = folium.Map(location=[56.1304, -106.3468], zoom_start=4, tiles='CartoDB dark_matter')
marker_cluster = MarkerCluster().add_to(m)

# Function to assign color based on size
def get_color(size):
    if size < 10:
        return 'green'
    elif size < 1000:
        return 'orange'
    else:
        return 'red'

# Add circle markers with hover info and dynamic styling
for _, row in sample_df.iterrows():
    folium.CircleMarker(
        location=[row['LATITUDE'], row['LONGITUDE']],
        radius=4 if row['SIZE_HA'] < 10 else 6 if row['SIZE_HA'] < 1000 else 8,
        color=get_color(row['SIZE_HA']),
        fill=True,
        fill_opacity=0.7,
        popup=f"Size: {row['SIZE_HA']} ha<br>Cause: {row['CAUSE']}"
    ).add_to(marker_cluster)

# Save map
output_path = os.path.join(base_dir, '..', 'outputs', 'wildfire_map_upgraded.html')
m.save(output_path)
print("✅ Upgraded wildfire map saved to /outputs/wildfire_map_upgraded.html")
