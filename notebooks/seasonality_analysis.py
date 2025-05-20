import pandas as pd
import matplotlib.pyplot as plt
import os

# Load the data
base_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(base_dir, '..', 'data', 'raw', 'NFDB_point_20240613.txt')

df = pd.read_csv(file_path, encoding='latin1')

# Group by MONTH and count
monthly_counts = df['MONTH'].value_counts().sort_index()

# Month number to label conversion
month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

# Plot it
plt.figure(figsize=(10, 5))
monthly_counts.plot(kind='bar', color='green', edgecolor='black')

plt.title('Wildfires by Month (Canada NFDB)', fontsize=14)
plt.xlabel('Month', fontsize=12)
plt.ylabel('Number of Fires', fontsize=12)
plt.xticks(ticks=range(12), labels=month_labels, rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.6)

# Save & show
output_path = os.path.join(base_dir, '..', 'outputs', 'seasonality_chart.png')
plt.tight_layout()
plt.savefig(output_path)
plt.show()
print("✅ Seasonality chart saved to /outputs/seasonality_chart.png")
