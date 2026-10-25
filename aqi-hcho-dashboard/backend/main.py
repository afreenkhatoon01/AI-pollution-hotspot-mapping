from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from io import StringIO

app = FastAPI(title="AQI & HCHO Prediction API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


rng = np.random.default_rng(42)
n = 700
demo = pd.DataFrame({
    "pm25": rng.uniform(5, 180, n),
    "pm10": rng.uniform(10, 250, n),
    "no2": rng.uniform(5, 100, n),
    "temperature": rng.uniform(10, 42, n),
    "humidity": rng.uniform(15, 95, n),
    "hcho": rng.uniform(0.001, 0.04, n),
})
demo["aqi"] = np.clip(
    15 + 0.55 * demo.pm25 + 0.12 * demo.pm10 + 0.18 * demo.no2
    + 0.2 * (demo.temperature - 25) - 0.04 * (demo.humidity - 50)
    + rng.normal(0, 8, n), 0, 500
)
FEATURES = ["pm25", "pm10", "no2", "temperature", "humidity", "hcho"]
X_train, X_test, y_train, y_test = train_test_split(
    demo[FEATURES], demo["aqi"], test_size=0.2, random_state=42
)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
mae = mean_absolute_error(y_test, model.predict(X_test))

class PredictionInput(BaseModel):
    pm25: float = Field(ge=0, le=1000)
    pm10: float = Field(ge=0, le=2000)
    no2: float = Field(ge=0, le=2000)
    temperature: float = Field(ge=-50, le=70)
    humidity: float = Field(ge=0, le=100)
    hcho: float = Field(ge=0, le=10)

def aqi_category(aqi: float) -> str:
    if aqi <= 50: return "Good"
    if aqi <= 100: return "Moderate"
    if aqi <= 150: return "Unhealthy for sensitive groups"
    if aqi <= 200: return "Unhealthy"
    if aqi <= 300: return "Very unhealthy"
    return "Hazardous"

def hcho_risk(value: float) -> str:
  
    if value < 0.01: return "Low (demo band)"
    if value < 0.03: return "Medium (demo band)"
    return "High (demo band)"

@app.get("/")
def root():
    return {"message": "AQI & HCHO API is running", "docs": "/docs"}

@app.get("/api/summary")
def summary():
    return {
        "current_aqi": 78,
        "hcho_ppm": 0.012,
        "hotspots": 4,
        "model_mae_demo": round(float(mae), 2),
        "note": "Illustrative demo values; not live environmental readings."
    }

@app.get("/api/forecast")
def forecast():
  
    hours = ["Now", "+4h", "+8h", "+12h", "+16h", "+20h", "+24h"]
    values = [78, 74, 81, 89, 94, 86, 79]
    return {"forecast": [{"time": t, "aqi": v} for t, v in zip(hours, values)],
            "note": "Illustrative demo forecast."}

@app.get("/api/hotspots")
def hotspots():
    return {
        "hotspots": [
            {"id": 1, "location": "Zone A", "hcho": 0.034, "risk": "High (demo band)"},
            {"id": 2, "location": "Zone B", "hcho": 0.022, "risk": "Medium (demo band)"},
            {"id": 3, "location": "Zone C", "hcho": 0.008, "risk": "Low (demo band)"},
            {"id": 4, "location": "Zone D", "hcho": 0.031, "risk": "High (demo band)"},
        ],
        "note": "Fictional locations and illustrative risk bands."
    }

@app.post("/api/predict")
def predict(payload: PredictionInput):
    row = pd.DataFrame([payload.model_dump()])[FEATURES]
    predicted = float(model.predict(row)[0])
    return {
        "predicted_aqi": round(predicted, 1),
        "category": aqi_category(predicted),
        "hcho_risk": hcho_risk(payload.hcho),
        "note": "Demo model trained on synthetic data. Do not use for health decisions."
    }

@app.post("/api/upload-csv")
async def upload_csv(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a .csv file.")
    raw = await file.read()
    try:
        df = pd.read_csv(StringIO(raw.decode("utf-8-sig")))
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read CSV. Check its format.")
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing columns: {', '.join(missing)}")
    if df.empty:
        raise HTTPException(status_code=400, detail="CSV has no data rows.")
    X = df[FEATURES].apply(pd.to_numeric, errors="coerce")
    if X.isna().any().any():
        raise HTTPException(status_code=400, detail="Feature columns must contain numeric values.")
    df["predicted_aqi"] = np.round(model.predict(X), 1)
    return {
        "rows": len(df),
        "preview": df.head(10).to_dict(orient="records"),
        "note": "Predictions use the demo model trained on synthetic data."
    }
