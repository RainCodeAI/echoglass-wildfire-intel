import pandas as pd
import os
import matplotlib.pyplot as plt

# 🔁 Load data like before
base_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(base_dir, '..', 'data', 'raw', 'NFDB_point_20240613.txt')
df = pd.read_csv(file_path, encoding='latin1')

# 📊 Group by year and count fires
yearly_counts = df['YEAR'].value_counts().sort_index()

# 🔥 Plotting
plt.figure(figsize=(14, 6))
yearly_counts.plot(kind='bar', color='firebrick')

plt.title('Wildfires Per Year in Canada (NFDB)', fontsize=16)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Number of Fires', fontsize=12)
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig(os.path.join(base_dir, '..', 'outputs', 'wildfires_per_year.png'))
print("✅ Chart saved successfully.")
plt.show()


