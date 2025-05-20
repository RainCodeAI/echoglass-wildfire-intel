import pandas as pd
import folium
from folium.plugins import MarkerCluster
import os

# === CONFIG ===
YEAR_TO_VIEW = 2023  # 🔁 Change this for different years!
file_path = "../data/raw/NFDB_point_20240613.txt"
output_file = f"outputs/wildfire_map_{YEAR_TO_VIEW}.html"

# === Color logic based on size ===
def get_color(size):
    if size < 10:
        return 'green'
    elif size < 1000:
        return 'orange'
    else:
        return 'red'

# === Load and filter ===
df = pd.read_csv(file_path, encoding='latin1')
df_filtered = df[df['YEAR'] == YEAR_TO_VIEW].dropna(subset=['LATITUDE', 'LONGITUDE'])

# === Create map ===
m = folium.Map(location=[56.1304, -106.3468], zoom_start=4, tiles='CartoDB dark_matter')
marker_cluster = MarkerCluster().add_to(m)

for _, row in df_filtered.iterrows():
    color = get_color(row['SIZE_HA'])
    tooltip_text = f"Size: {row['SIZE_HA']} ha, Cause: {row['CAUSE']}"

    folium.CircleMarker(
        location=[row['LATITUDE'], row['LONGITUDE']],
        radius=4 if row['SIZE_HA'] < 10 else 6 if row['SIZE_HA'] < 1000 else 8,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.7,
        tooltip=tooltip_text,
        popup=f"Size: {row['SIZE_HA']} ha<br>Cause: {row['CAUSE']}"
    ).add_to(marker_cluster)

# Add title
title_html = f"""
    <h3 align="center" style="font-size:20px"><b>Wildfires in {YEAR_TO_VIEW}</b></h3>
"""
m.get_root().html.add_child(folium.Element(title_html))

# Save
m.save(output_file)
print(f"Map saved to: {output_file}")
