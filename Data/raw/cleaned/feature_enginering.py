
import pandas as pd
from pathlib import Path

# Input and output paths
input_file = Path(
    "data/cleaned/cpcb_january_to_september_2026.csv"
)

output_file = Path(
    "data/cleaned/cpcb_aqi_features_2026.csv"
)

# 1. Load dataset
df = pd.read_csv(input_file)

# 2. Convert DateTime and AQI to correct types
df["DateTime"] = pd.to_datetime(
    df["DateTime"], errors="coerce"
)
df["AQI"] = pd.to_numeric(
    df["AQI"], errors="coerce"
)

df = df.dropna(subset=["DateTime", "AQI"])

# 3. Sort records chronologically
df = df.sort_values(
    ["Station", "DateTime"]
).reset_index(drop=True)

# 4. Create time-based features
df["hour"] = df["DateTime"].dt.hour
df["day"] = df["DateTime"].dt.day
df["day_of_week"] = df["DateTime"].dt.dayofweek
df["month"] = df["DateTime"].dt.month

# 5. Create previous-hour AQI features
grouped_aqi = df.groupby("Station")["AQI"]

df["aqi_previous_hour"] = grouped_aqi.shift(1)
df["aqi_2_hours_ago"] = grouped_aqi.shift(2)
df["aqi_3_hours_ago"] = grouped_aqi.shift(3)
df["aqi_24_hours_ago"] = grouped_aqi.shift(24)

# 6. Save features for inspection
output_file.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(output_file, index=False)

print("Feature engineering completed!")
print("Total records:", len(df))
print("\nNew columns:")
print(df.columns.tolist())
print("\nFirst 10 records:")
print(df.head(10).to_string(index=False))
print("\nSaved to:", output_file)
