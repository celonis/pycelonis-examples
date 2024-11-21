import pandas as pd

# Load the CSV file
csv_file_path = ""
data = pd.read_csv(csv_file_path)

# Convert to Parquet
parquet_file_path = ""
data.to_parquet(parquet_file_path, index=False)

print("Conversion to Parquet completed.")