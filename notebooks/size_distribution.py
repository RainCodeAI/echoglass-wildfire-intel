import pandas as pd
import matplotlib.pyplot as plt
import os

# Load data
base_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(base_dir, '..', 'data', 'raw', 'NFDB_point_20240613.txt')
df = pd.read_csv(file_path, encoding='latin1')

# Remove missing or invalid fire sizes
df = df[df['SIZE_HA'].notna() & (df['SIZE_HA'] > 0)]

# Categorize fire sizes
def categorize_size(size):
    if size < 10:
        return 'Small (<10 ha)'
    elif size < 1000:
        return 'Medium (10–1000 ha)'
    else:
        return 'Large (>1000 ha)'

df['SIZE_CATEGORY'] = df['SIZE_HA'].apply(categorize_size)

# Count how many in each category
category_counts = df['SIZE_CATEGORY'].value_counts().reindex(['Small (<10 ha)', 'Medium (10–1000 ha)', 'Large (>1000 ha)'])

# Plot
plt.figure(figsize=(8, 5))
category_counts.plot(kind='bar', color='darkred', edgecolor='black')

plt.title("Wildfires by Size Category", fontsize=14)
plt.ylabel("Number of Fires")
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.6)

# Save + show
output_path = os.path.join(base_dir, '..', 'outputs', 'fire_size_categories.png')
plt.tight_layout()
plt.savefig(output_path)
plt.show()
print("✅ Fire size category chart saved to /outputs/fire_size_categories.png")
