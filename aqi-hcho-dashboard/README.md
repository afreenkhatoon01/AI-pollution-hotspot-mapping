# AI-Based Surface AQI Prediction & HCHO Hotspot Detection

A beginner-friendly React + FastAPI dashboard starter.

## Requirements
- Python 3.10+ (3.11 recommended)
- Node.js 18+ (20 LTS recommended)

## Folder structure
- `backend/main.py`: FastAPI endpoints and demo Random Forest model
- `frontend/src/App.jsx`: React dashboard
- `frontend/src/style.css`: dashboard styling
- `sample_data.csv`: sample CSV format

## 1. Start the Python backend
Open a terminal in the project folder:

```bash
cd backend
python -m venv .venv
```

Windows PowerShell:
```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn main:app --reload
```

If PowerShell blocks activation, use Command Prompt and run:
```bat
.venv\Scripts\activate.bat
```

Backend docs: http://127.0.0.1:8000/docs

## 2. Start the React frontend
Open a SECOND terminal:

```bash
cd frontend
npm install
npm run dev
```

Open the URL printed by Vite, usually http://localhost:5173

## 3. Test CSV upload
Upload `sample_data.csv`. Required numeric columns:
`pm25, pm10, no2, temperature, humidity, hcho`

## Important scientific limitation
The starter model is trained on synthetic demo data. Summary values, forecast points, hotspot locations and HCHO risk bands are placeholders. This is only a UI/API prototype, not a validated environmental model. For a credible project, replace synthetic data with sourced, quality-controlled sensor/satellite observations, document units and averaging windows, train/test by time or location to avoid leakage, compare baselines, and validate HCHO thresholds with authoritative guidance. The AQI calculation standard depends on the country and pollutant data; the example is not an official AQI calculation.
