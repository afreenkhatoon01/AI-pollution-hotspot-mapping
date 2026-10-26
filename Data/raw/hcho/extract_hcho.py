
import rasterio
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[3]

tif_path = BASE_DIR / "Data" / "raw" / "hcho" / "hcho_2025-04-26.tif"
output_path = BASE_DIR / "Data" / "cleaned" / "anand_vihar_hcho_2025-04-26.csv"

latitude = 28.65
longitude = 77.316
observation_date = "2025-04-26"

with rasterio.open(tif_path) as src:
    row, col = src.index(longitude, latitude)
    raw_value = float(src.read(1)[row, col])

    nodata = src.nodata
    scale = src.scales[0] if src.scales else 1.0
    offset = src.offsets[0] if src.offsets else 0.0

    if pd.isna(raw_value) or (nodata is not None and raw_value == nodata):
        raise ValueError("HCHO pixel is missing or marked as NoData.")

    hcho_column = raw_value * scale + offset

    record = {
        "date": observation_date,
        "location": "Anand Vihar, Delhi",
        "latitude": latitude,
        "longitude": longitude,
        "raw_pixel_value": raw_value,
        "scale_factor": scale,
        "hcho_column_mol_m2": hcho_column,
        "units": "mol/m2",
        "source": "DLR Sentinel-5P TROPOMI OFFL L3 HCHO",
        "spatial_resolution": "5.5 x 3.5 km2",
        "measurement_type": "satellite atmospheric total column"
    }

output_path.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame([record]).to_csv(output_path, index=False)

print("CSV saved:", output_path)
print(pd.DataFrame([record]).to_string(index=False))

