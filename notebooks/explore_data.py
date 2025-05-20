import pandas as pd
import os

# Dynamically build the path to the data file
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # goes up to project root
file_path = os.path.join(base_dir, 'data', 'raw', 'NFDB_point_20240613.txt')

# Load the data
df = pd.read_csv(file_path, encoding='latin1')

# Basic info
print("Shape of data:", df.shape)
print("\nColumn names:\n", df.columns)
print("\nFirst 5 rows:")
print(df.head())

# Check for missing values
print("\nMissing values per column:")
print(df.isnull().sum())
