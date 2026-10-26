
import pandas as pd
from pathlib import Path

# 1. Set project folder
BASE_DIR = Path(r"C:\Users\Dell\Desktop\ai-air-pollution-monitoring")

# 2. Define input and output paths
input_file = BASE_DIR / "reports" / "aqi_predictions.csv"
output_file = BASE_DIR / "reports" / "aqi_risk_classification.csv"

# 3. Check input file
print("Input file:", input_file)

if not input_file.exists():
    raise FileNotFoundError(
        f"Prediction file not found: {input_file}\n"
        "Run evaluate_model.py first."
    )

# 4. Load predictions
df = pd.read_csv(input_file)

print("Dataset loaded:", df.shape)

# 5. Validate required columns
required_columns = ["Actual_AQI", "Predicted_AQI"]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing columns: {missing_columns}\n"
        f"Available columns: {df.columns.tolist()}"
    )

# 6. Define CPCB AQI categories
def classify_aqi(aqi):
    if pd.isna(aqi):
        return "Unknown"
    elif aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Satisfactory"
    elif aqi <= 200:
        return "Moderate"
    elif aqi <= 300:
        return "Poor"
    elif aqi <= 400:
        return "Very Poor"
    else:
        return "Severe"

# 7. Classify actual and predicted AQI
df["Actual_Category"] = df["Actual_AQI"].apply(classify_aqi)
df["Predicted_Category"] = df["Predicted_AQI"].apply(classify_aqi)

# 8. Save classified results
output_file.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(output_file, index=False)

# 9. Display results
print("\nAQI classification completed successfully!")
print("Output file:", output_file)
print("Total records:", len(df))

print("\nPredicted AQI categories:")
print(df["Predicted_Category"].value_counts())

print("\nActual AQI categories:")
print(df["Actual_Category"].value_counts())

print("\nFirst 5 records:")
print(
    df[
        [
            "Actual_AQI",
            "Predicted_AQI",
            "Actual_Category",
            "Predicted_Category"
        ]
    ].head()
)



