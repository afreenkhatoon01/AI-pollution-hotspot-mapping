
import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Project root
BASE_DIR = Path(r"C:\Users\Dell\Desktop\ai-air-pollution-monitoring")

# File paths
input_file = BASE_DIR / "Data" / "cleaned" / "cpcb_aqi_features_2026.csv"
model_file = BASE_DIR / "models" / "aqi_random_forest.pkl"
reports_dir = BASE_DIR / "reports"
reports_dir.mkdir(parents=True, exist_ok=True)

# Load data and trained model
df = pd.read_csv(input_file, parse_dates=["DateTime"])
df = df.sort_values("DateTime").reset_index(drop=True)

model = joblib.load(model_file)

features = [
    "hour",
    "day_of_week",
    "month",
    "aqi_previous_hour",
    "aqi_2_hours_ago",
    "aqi_3_hours_ago"
]

target = "AQI"

# Check required columns
missing = [col for col in features + [target] if col not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

# Match training preprocessing
model_df = df.dropna(subset=features + [target]).copy()

# Chronological 80/20 split, same as training
split_index = int(len(model_df) * 0.8)
test = model_df.iloc[split_index:].copy()

X_test = test[features]
y_test = test[target]

# Predict
predictions = model.predict(X_test)

print("Testing records:", len(test))
print(f"MAE: {mean_absolute_error(y_test, predictions):.2f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, predictions)):.2f}")
print(f"R2 Score: {r2_score(y_test, predictions):.3f}")

# Save predictions
test["Predicted_AQI"] = predictions
test["Prediction_Error"] = test[target] - test["Predicted_AQI"]

output_file = reports_dir / "aqi_predictions.csv"
test.to_csv(output_file, index=False)

print("\nPredictions saved successfully!")
print("Location:", output_file)

