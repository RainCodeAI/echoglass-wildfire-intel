import pandas as pd
import matplotlib.pyplot as plt
import os

# Load data
base_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(base_dir, '..', 'data', 'raw', 'NFDB_point_20240613.txt')
df = pd.read_csv(file_path, encoding='latin1')

# Clean + filter data
df = df[['SRC_AGENCY', 'YEAR']].dropna()
df = df[df['YEAR'].between(1950, 2025)]  # reasonable date range

# Group by year + agency
yearly_counts = df.groupby(['YEAR', 'SRC_AGENCY']).size().reset_index(name='FIRE_COUNT')

# Pivot so each province is its own column
pivot_df = yearly_counts.pivot(index='YEAR', columns='SRC_AGENCY', values='FIRE_COUNT').fillna(0)

# Plot
plt.figure(figsize=(14, 8))
pivot_df.plot(ax=plt.gca(), lw=2)

plt.title("Wildfires Per Year by Province/Territory (NFDB)", fontsize=16)
plt.xlabel("Year", fontsize=12)
plt.ylabel("Number of Fires", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(title='Province / Agency', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

# Save output
output_path = os.path.join(base_dir, '..', 'outputs', 'fires_by_province_over_time.png')
plt.savefig(output_path)
plt.show()

print("✅ Wildfire trendline by province saved to /outputs/fires_by_province_over_time.png")
