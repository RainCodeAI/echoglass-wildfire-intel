import pandas as pd

# Load your dataset (make sure this path matches your structure)
file_path = "../data/raw/NFDB_point_20240613.txt"
df = pd.read_csv(file_path, encoding="latin1")

# See all unique years and how many fires in each
print("Year counts in the dataset:")
print(df['YEAR'].value_counts().sort_index())

# Specifically, how many in 2015?
fires_2015 = (df['YEAR'] == 2015).sum()
print(f"\nNumber of fires in 2015: {fires_2015}")
