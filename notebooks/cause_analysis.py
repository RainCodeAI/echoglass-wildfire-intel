import pandas as pd
import matplotlib.pyplot as plt
import os

# Build file path
base_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(base_dir, '..', 'data', 'raw', 'NFDB_point_20240613.txt')

# Load data
df = pd.read_csv(file_path, encoding='latin1')

# Count causes
cause_counts = df['CAUSE'].value_counts()

# Bar chart
plt.figure(figsize=(8, 5))
cause_counts.plot(kind='bar', color='orange', edgecolor='black')

plt.title("Wildfire Causes in Canada (NFDB)", fontsize=14)
plt.xlabel("Cause", fontsize=12)
plt.ylabel("Number of Fires", fontsize=12)
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.6)

# Save + show
output_path = os.path.join(base_dir, '..', 'outputs', 'cause_breakdown.png')
plt.tight_layout()
plt.savefig(output_path)
plt.show()
print("✅ Cause breakdown chart saved to /outputs/cause_breakdown.png")
