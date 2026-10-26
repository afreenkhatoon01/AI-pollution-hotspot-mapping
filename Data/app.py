
import streamlit as st
import pandas as pd
from pathlib import Path

BASE_DIR = Path(r"C:\Users\Dell\Desktop\ai-air-pollution-monitoring")

data_file = BASE_DIR / "reports" / "aqi_risk_classification.csv"

print("Looking for:", data_file)
print("File exists:", data_file.exists())

st.set_page_config(
    page_title="AI-Based AQI Monitoring",
    page_icon="🌍",
    layout="wide"
)

st.title("🌍 AI-Based Surface AQI Prediction")
st.caption("Anand Vihar, Delhi — station-level historical data and test-set predictions")

if not data_file.exists():
    st.error(f"Classification file not found: {data_file}")
    st.info("Run evaluate_model.py and classify_aqi.py first.")
    st.stop()

df = pd.read_csv(data_file)

required = [
    "DateTime",
    "Actual_AQI",
    "Predicted_AQI",
    "Actual_Category",
    "Predicted_Category"
]

missing = [col for col in required if col not in df.columns]

if missing:
    st.error(f"Missing columns: {missing}")
    st.write("Available columns:", df.columns.tolist())
    st.stop()

df["DateTime"] = pd.to_datetime(df["DateTime"], errors="coerce")
df = df.sort_values("DateTime")

# Summary metrics
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Test Records", f"{len(df):,}")

with col2:
    mae = (df["Actual_AQI"] - df["Predicted_AQI"]).abs().mean()
    st.metric("Mean Absolute Error", f"{mae:.2f}")

with col3:
    st.metric(
        "Mean Actual AQI",
        f"{df['Actual_AQI'].mean():.1f}"
    )

st.subheader("Actual vs Predicted AQI")

chart_df = df.set_index("DateTime")[
    ["Actual_AQI", "Predicted_AQI"]
]
st.line_chart(chart_df, width="stretch")

st.subheader("Predicted AQI Categories")
category_counts = (
    df["Predicted_Category"]
    .value_counts()
    .reindex(
        ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"],
        fill_value=0
    )
)
st.bar_chart(category_counts, width="stretch")

st.subheader("Prediction Results")
st.dataframe(df, width="stretch", hide_index=True)

st.download_button(
    label="Download Classification CSV",
    data=df.to_csv(index=False).encode("utf-8"),
    file_name="aqi_risk_classification.csv",
    mime="text/csv"
)

