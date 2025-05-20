import pandas as pd
import matplotlib.pyplot as plt
import os

# Load the data
base_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(base_dir, '..', 'data', 'raw', 'NFDB_point_20240613.txt')
df = pd.read_csv(file_path, encoding='latin1')

# 🔁 Province code to full name map
province_map = {
    'BC': 'British Columbia',
    'AB': 'Alberta',
    'SK': 'Saskatchewan',
    'MB': 'Manitoba',
    'ON': 'Ontario',
    'QC': 'Quebec',
    'NB': 'New Brunswick',
    'NS': 'Nova Scotia',
    'NL': 'Newfoundland & Labrador',
    'PE': 'Prince Edward Island',
    'YT': 'Yukon',
    'NT': 'Northwest Territories',
    'NU': 'Nunavut',
    'PC': 'Parks Canada'
}

# Count fires by reporting agency (often province-based)
agency_counts = df['SRC_AGENCY'].map(province_map).value_counts().sort_values(ascending=True)

# Optional: replace agency codes with full province names (if desired later)

# Plot
plt.figure(figsize=(10, 8))
agency_counts.plot(kind='barh', color='skyblue', edgecolor='black')

plt.title("Wildfires Reported by Province/Territory (SRC_AGENCY)", fontsize=14)
plt.xlabel("Number of Fires")
plt.ylabel("Province / Agency Code")
plt.grid(axis='x', linestyle='--', alpha=0.6)

# Save + show
output_path = os.path.join(base_dir, '..', 'outputs', 'fires_by_province.png')
plt.tight_layout()
plt.savefig(output_path)
plt.show()

print("✅ Fires by province chart saved to /outputs/fires_by_province.png")
