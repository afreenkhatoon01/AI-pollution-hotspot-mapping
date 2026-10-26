
import pandas as pd
from pathlib import Path

raw_folder = Path("data/raw")
clean_folder = Path("data/cleaned")
clean_folder.mkdir(parents=True, exist_ok=True)

month_numbers = {
    "january": 1, "february": 2, "march": 3,
    "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9
}

files = list(raw_folder.glob("*.xlsx"))

if not files:
    raise FileNotFoundError(
        "data/raw folder mein Excel files nahi mili."
    )

for file in files:
    month_name = next(
        (m for m in month_numbers if m in file.stem.lower()),
        None
    )

    if month_name is None:
        print(f"Skipped (month name not detected): {file.name}")
        continue

    month = month_numbers[month_name]
    print(f"\nCleaning {file.name}...")

    df = pd.read_excel(file)
    df.columns = df.columns.astype(str).str.strip()

    if "Date" not in df.columns:
        print("Skipped: Date column not found.")
        continue

    hour_columns = [
        col for col in df.columns
        if str(col).strip() in
        [f"{h:02d}:00" for h in range(24)]
    ]

    if not hour_columns:
        print("Skipped: Hourly columns not found.")
        continue

    data = df.melt(
        id_vars=["Date"],
        value_vars=hour_columns,
        var_name="Hour",
        value_name="AQI"
    )

    data["Date"] = pd.to_numeric(data["Date"], errors="coerce")
    data["AQI"] = pd.to_numeric(data["AQI"], errors="coerce")
    data = data.dropna(subset=["Date", "AQI"])

    # Use 2026 and the month detected from the filename
    data = data[data["Date"].between(1, 31)].copy()

    data["DateTime"] = pd.to_datetime(
        {
            "year": 2026,
            "month": month,
            "day": data["Date"].astype(int)
        },
        errors="coerce"
    ) + pd.to_timedelta(
        data["Hour"].str[:2].astype(int),
        unit="h"
    )

    data = data.dropna(subset=["DateTime"])
    data = data[data["AQI"].between(0, 500)]

    data["Station"] = "Anand Vihar"
    data["City"] = "Delhi"
    data["Agency"] = "DPCC"

    data = data[
        ["DateTime", "Station", "City", "Agency", "AQI"]
    ]
    data = data.drop_duplicates(
        subset=["DateTime", "Station"]
    )
    data = data.sort_values("DateTime")

    output = clean_folder / f"cpcb_{month_name}_2026_cleaned.csv"
    data.to_csv(output, index=False)

    print(f"Saved: {output}")
    print(f"Records: {len(data)}")

print("\nMonthly cleaning process finished.")
