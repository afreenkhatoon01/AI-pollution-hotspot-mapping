
import pandas as pd
from pathlib import Path

# Folder paths
clean_folder = Path("data/cleaned")
output_file = clean_folder / "cpcb_january_to_september_2026.csv"

# Find the 9 cleaned monthly CSV files
files = sorted(
    clean_folder.glob("cpcb_*_2026_cleaned.csv")
)

if len(files) != 9:
    print(f"Warning: Expected 9 monthly files, found {len(files)}")

if not files:
    raise FileNotFoundError("No cleaned monthly CSV files found.")

dataframes = []

for file in files:
    df = pd.read_csv(file)

    required_columns = [
        "DateTime", "Station", "City", "Agency", "AQI"
    ]

    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        print(f"Skipping {file.name}: missing {missing_columns}")
        continue

    df = df[required_columns].copy()
    df["DateTime"] = pd.to_datetime(
        df["DateTime"], errors="coerce"
    )
    df["AQI"] = pd.to_numeric(df["AQI"], errors="coerce")

    print(f"{file.name}: {len(df)} rows")
    dataframes.append(df)

if not dataframes:
    raise ValueError("No valid monthly datasets were found.")

# Combine all months
combined = pd.concat(dataframes, ignore_index=True)

# Remove invalid rows and duplicates
combined = combined.dropna(
    subset=["DateTime", "Station", "AQI"]
)
combined = combined[combined["AQI"].between(0, 500)]
combined = combined.drop_duplicates(
    subset=["DateTime", "Station"]
)
combined = combined.sort_values("DateTime").reset_index(drop=True)

# Save combined dataset
combined.to_csv(output_file, index=False)

# Quality report
print("\n--- COMBINED DATA REPORT ---")
print("Total records:", len(combined))
print("Date range:", combined["DateTime"].min(), "to",
      combined["DateTime"].max())
print("\nMissing values:")
print(combined.isnull().sum())
print("\nRecords by month:")
print(combined.groupby(combined["DateTime"].dt.month).size())
print("\nSaved file:", output_file)
