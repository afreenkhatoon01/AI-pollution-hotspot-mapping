
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


# Project paths
BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "aqi_random_forest.pkl"
CSV_PATH = BASE_DIR / "reports" / "aqi_predictions.csv"

HCHO_CSV_PATH = (
    BASE_DIR
    / "Data"
    / "cleaned"
    / "anand_vihar_hcho_2025-04-26.csv"
)

FEATURES = [
    "hour",
    "day_of_week",
    "month",
    "aqi_previous_hour",
    "aqi_2_hours_ago",
    "aqi_3_hours_ago",
]


# FastAPI application
app = FastAPI(
    title="AQI & HCHO Monitoring API",
    version="2.1",
    description="AQI prediction and satellite-derived HCHO observation API.",
)


# Frontend CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Load the existing trained model. Do not retrain it here.
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

model = joblib.load(MODEL_PATH)


# Check expected feature order when available
if hasattr(model, "feature_names_in_"):
    if list(model.feature_names_in_) != FEATURES:
        raise ValueError(
            "Model features do not match API FEATURES. "
            f"Expected: {FEATURES}; "
            f"found: {list(model.feature_names_in_)}"
        )


# Load historical AQI data
def load_data():
    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"AQI CSV not found: {CSV_PATH}"
        )

    df = pd.read_csv(CSV_PATH)

    required = ["DateTime", "AQI", *FEATURES]

    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            f"Missing columns in AQI CSV: {missing}"
        )

    df["DateTime"] = pd.to_datetime(
        df["DateTime"], errors="coerce"
    )

    for col in required:
        if col != "DateTime":
            df[col] = pd.to_numeric(
                df[col], errors="coerce"
            )

    df = df.dropna(subset=required)

    return (
        df.sort_values("DateTime")
        .reset_index(drop=True)
    )


def get_data():
    try:
        return load_data()
    except (
        FileNotFoundError,
        ValueError,
        pd.errors.ParserError,
    ) as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc


# Indian AQI category labels
def aqi_category(value):
    if value <= 50:
        return "Good"
    if value <= 100:
        return "Satisfactory"
    if value <= 200:
        return "Moderate"
    if value <= 300:
        return "Poor"
    if value <= 400:
        return "Very Poor"
    return "Severe"


# Convert records to JSON-safe Python objects
def json_safe_records(df):
    cleaned = df.replace([np.inf, -np.inf], np.nan)
    return json.loads(
        cleaned.to_json(
            orient="records",
            date_format="iso",
        )
    )


# Prediction request validation
class PredictionInput(BaseModel):
    hour: int = Field(ge=0, le=23)
    day_of_week: int = Field(ge=0, le=6)
    month: int = Field(ge=1, le=12)
    aqi_previous_hour: float = Field(ge=0, le=500)
    aqi_2_hours_ago: float = Field(ge=0, le=500)
    aqi_3_hours_ago: float = Field(ge=0, le=500)


# Basic routes
@app.get("/")
def root():
    return {
        "message": "AQI & HCHO Monitoring API is running",
        "docs": "/docs",
        "health": "/api/health",
    }


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "model_loaded": True,
        "model_features": FEATURES,
        "hcho_csv_exists": HCHO_CSV_PATH.exists(),
    }


# AQI summary
@app.get("/api/summary")
def summary():
    df = get_data()

    if df.empty:
        raise HTTPException(
            status_code=404,
            detail="No valid AQI records found.",
        )

    latest = df.iloc[-1]

    predicted_aqi = None
    if (
        "Predicted_AQI" in df.columns
        and pd.notna(latest["Predicted_AQI"])
        and np.isfinite(float(latest["Predicted_AQI"]))
    ):
        predicted_aqi = round(
            float(latest["Predicted_AQI"]), 1
        )

    return {
        "station": str(latest.get("Station", "Anand Vihar")),
        "city": str(latest.get("City", "Delhi")),
        "last_record_time": latest["DateTime"].isoformat(),
        "current_aqi": round(float(latest["AQI"]), 1),
        "predicted_aqi": predicted_aqi,
        "category": aqi_category(float(latest["AQI"])),
        "records": int(len(df)),
        "note": (
            "Latest available historical dataset record; "
            "not a live sensor reading."
        ),
    }


# AQI historical records
@app.get("/api/aqi")
def aqi_records(limit: int = 100):
    if not 1 <= limit <= 1000:
        raise HTTPException(
            status_code=400,
            detail="limit must be between 1 and 1000.",
        )

    df = get_data()
    result = df.tail(limit).copy()

    result["DateTime"] = result["DateTime"].astype(str)

    return {
        "count": len(result),
        "data": json_safe_records(result),
    }


# Predict AQI from supplied temporal features
@app.post("/api/predict")
def predict(payload: PredictionInput):
    row = pd.DataFrame(
        [[getattr(payload, feature) for feature in FEATURES]],
        columns=FEATURES,
    )

    value = float(model.predict(row)[0])
    value = float(np.clip(value, 0, 500))

    return {
        "predicted_aqi": round(value, 1),
        "category": aqi_category(value),
        "model": "Trained Random Forest",
        "note": (
            "Prediction based on temporal AQI features. "
            "It is not a live sensor measurement."
        ),
    }


# Recursive 24-hour AQI forecast
@app.get("/api/forecast")
def forecast():
    df = get_data()

    if df.empty:
        raise HTTPException(
            status_code=404,
            detail="No valid AQI records available for forecasting.",
        )

    latest = df.iloc[-1]
    timestamp = latest["DateTime"]

    previous = float(latest["AQI"])
    two_hours_ago = float(latest["aqi_previous_hour"])
    three_hours_ago = float(latest["aqi_2_hours_ago"])

    results = []

    for step in range(1, 25):
        timestamp = timestamp + pd.Timedelta(hours=1)

        row = pd.DataFrame([{
            "hour": timestamp.hour,
            "day_of_week": timestamp.dayofweek,
            "month": timestamp.month,
            "aqi_previous_hour": previous,
            "aqi_2_hours_ago": two_hours_ago,
            "aqi_3_hours_ago": three_hours_ago,
        }])[FEATURES]

        value = float(
            np.clip(model.predict(row)[0], 0, 500)
        )

        results.append({
            "time": timestamp.isoformat(),
            "hour_ahead": step,
            "aqi": round(value, 1),
            "category": aqi_category(value),
        })

        three_hours_ago = two_hours_ago
        two_hours_ago = previous
        previous = value

    return {
        "forecast": results,
        "based_on": latest["DateTime"].isoformat(),
        "note": (
            "Recursive model estimates based on historical AQI "
            "features. Weather and live pollutant data are not included."
        ),
    }


# Latest satellite-derived HCHO observation
@app.get("/api/hcho")
def get_hcho():
    if not HCHO_CSV_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail=f"HCHO CSV not found: {HCHO_CSV_PATH}",
        )

    try:
        df = pd.read_csv(HCHO_CSV_PATH)
    except (OSError, pd.errors.ParserError) as exc:
        raise HTTPException(
            status_code=500,
            detail="Unable to read the HCHO CSV file.",
        ) from exc

    if df.empty:
        raise HTTPException(
            status_code=404,
            detail="HCHO CSV contains no observations.",
        )

    required = [
        "date",
        "location",
        "latitude",
        "longitude",
        "hcho_column_mol_m2",
        "units",
        "source",
        "measurement_type",
    ]

    missing = [col for col in required if col not in df.columns]
    if missing:
        raise HTTPException(
            status_code=500,
            detail=f"Missing HCHO CSV columns: {missing}",
        )

    # This file currently contains a single daily observation.
    record = json_safe_records(df.head(1))[0]

    record["measurement_description"] = (
        "Satellite-derived atmospheric total column density; "
        "not ground-level concentration."
    )

    return record


# Hotspots are not calculated from one observation
@app.get("/api/hotspots")
def hotspots():
    return {
        "hotspots": [],
        "status": "insufficient_spatial_data",
        "note": (
            "A single HCHO observation cannot identify hotspots. "
            "Multiple spatially distributed observations and a "
            "defined hotspot method are required."
        ),
    }


# Upload a CSV and generate AQI predictions
@app.post("/api/upload-csv")
async def upload_csv(file: UploadFile = File(...)):
    if (
        not file.filename
        or not file.filename.lower().endswith(".csv")
    ):
        raise HTTPException(
            status_code=400,
            detail="Please upload a CSV file.",
        )

    try:
        df = pd.read_csv(file.file)
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail="Could not read the CSV file.",
        ) from exc
    finally:
        await file.close()

    missing = [col for col in FEATURES if col not in df.columns]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Missing required features: {missing}",
        )

    X = df[FEATURES].apply(
        pd.to_numeric, errors="coerce"
    )

    if X.empty or X.isna().any().any():
        raise HTTPException(
            status_code=400,
            detail=(
                "CSV must contain rows with valid numeric "
                "feature values."
            ),
        )

    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise HTTPException(
            status_code=400,
            detail="CSV feature values must be finite numbers.",
        )

    values = np.clip(model.predict(X), 0, 500)

    df["predicted_aqi"] = np.round(values, 1)
    df["predicted_category"] = [
        aqi_category(value) for value in values
    ]

    return {
        "rows": len(df),
        "preview": json_safe_records(df.head(10)),
        "note": (
            "Predictions generated using the saved Random Forest model."
        ),
    }

