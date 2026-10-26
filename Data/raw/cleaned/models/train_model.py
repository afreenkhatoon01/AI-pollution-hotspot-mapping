
import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Load feature-engineered data
input_file = Path("data/cleaned/cpcb_aqi_features_2026.csv")

df = pd.read_csv(input_file, parse_dates=["DateTime"])
df = df.sort_values("DateTime").reset_index(drop=True)

# 2. Select input features and target
features = [
    "hour",
    "day_of_week",
    "month",
    "aqi_previous_hour",
    "aqi_2_hours_ago",
    "aqi_3_hours_ago"
]

target = "AQI"

# 3. Remove rows with missing feature values
model_df = df.dropna(subset=features + [target]).copy()

# 4. Split chronologically: 80% training, 20% testing
split_index = int(len(model_df) * 0.8)

train = model_df.iloc[:split_index]
test = model_df.iloc[split_index:]

X_train = train[features]
y_train = train[target]

X_test = test[features]
y_test = test[target]

# 5. Train the model
model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    min_samples_leaf=3,
    n_jobs=-1
)

model.fit(X_train, y_train)

# 6. Evaluate the model
predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

print("Training records:", len(train))
print("Testing records:", len(test))
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R2 Score: {r2:.3f}")

# 7. Save the trained model
model_folder = Path("models")
model_folder.mkdir(exist_ok=True)

joblib.dump(model, model_folder / "aqi_random_forest.pkl")

print("\nModel saved successfully!")
print("Location: models/aqi_random_forest.pkl")
