
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

input_file = BASE_DIR / "reports" / "aqi_predictions.csv"
output_file = BASE_DIR / "reports" / "aqi_prediction_plot.png"

if not input_file.exists():
    raise FileNotFoundError(
        f"{input_file} not found. Run evaluate_model.py first."
    )

df = pd.read_csv(input_file)

actual = df["AQI"]
predicted = df["Predicted_AQI"]

print("Test records:", len(df))
print(f"MAE: {mean_absolute_error(actual, predicted):.2f}")
print(f"RMSE: {np.sqrt(mean_squared_error(actual, predicted)):.2f}")
print(f"R2 Score: {r2_score(actual, predicted):.3f}")

if "DateTime" in df.columns:
    df["DateTime"] = pd.to_datetime(df["DateTime"], errors="coerce")
    df = df.sort_values("DateTime")

plt.figure(figsize=(12, 5))
plt.plot(df["AQI"].to_numpy(), label="AQI")
plt.plot(df["Predicted_AQI"].to_numpy(), label="Predicted AQI")
plt.title("Actual vs Predicted AQI — Test Set")
plt.xlabel("Test record (chronological order)")
plt.ylabel("AQI")
plt.legend()
plt.tight_layout()
plt.savefig(output_file, dpi=150)
plt.close()

print("Graph saved:", output_file)
