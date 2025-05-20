import pandas as pd
import folium
from folium.plugins import MarkerCluster
import os

# Load data
base_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(base_dir, '..', 'data', 'raw', 'NFDB_point_20240613.txt')
df = pd.read_csv(file_path, encoding='latin1')

# --- Filter Settings ---
# Choose province/territory code(s): e.g., ['BC'], ['ON', 'QC']
selected_provinces = ['ON', 'QC']

# Filter dataset
df_filtered = df[df['SRC_AGENCY'].isin(selected_provinces)]
df_filtered = df_filtered[['LATITUDE', 'LONGITUDE', 'SIZE_HA', 'CAUSE']].dropna().sample(1000, random_state=42)

# Init map
m = folium.Map(location=[56.1304, -106.3468], zoom_start=4, tiles='CartoDB dark_matter')

title_html = f'''
     <div style="position: fixed; 
                 top: 10px; left: 50%; transform: translateX(-50%);
                 z-index: 9999; font-size: 22px; color: white; 
                 background-color: rgba(0,0,0,0.5); 
                 padding: 5px 20px; border-radius: 8px;">
         Wildfires in {' + '.join(selected_provinces)}
     </div>
'''
m.get_root().html.add_child(folium.Element(title_html))


marker_cluster = MarkerCluster().add_to(m)


# Color logic
def get_color(size):
    if size < 10:
        return 'green'
    elif size < 1000:
        return 'orange'
    else:
        return 'red'

# Add filtered markers
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
        popup=f"Size: {row['SIZE_HA']} ha<br>Cause: {row['CAUSE']}",
        tooltip=tooltip_text
    ).add_to(marker_cluster)


# Save
province_str = "_".join(selected_provinces)
output_path = os.path.join(base_dir, '..', 'outputs', f'wildfire_map_{province_str}.html')
m.save(output_path)
print(f"✅ Filtered map saved to /outputs/wildfire_map_{province_str}.html")
