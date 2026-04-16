import pandas as pd
import numpy as np
from src.feature_engineering import build_segment_features

# =====================================================
# 1. LOAD DATA
# =====================================================

ais = pd.read_csv("data/raw/ais2024.csv")
ais["BaseDateTime"] = pd.to_datetime(ais["BaseDateTime"])
ais = ais.sort_values(["MMSI", "BaseDateTime"])
ais["date"] = ais["BaseDateTime"].dt.date

gfw = pd.read_csv("data/raw/gfw.csv")

gfw = gfw.rename(columns={
    "mmsi": "MMSI",
    "cell_ll_lat": "lat_bin",
    "cell_ll_lon": "lon_bin"
})

gfw["date"] = pd.to_datetime(gfw["date"]).dt.date
gfw["MMSI"] = gfw["MMSI"].astype(int)

# =====================================================
# 2. SEGMENT AIS
# =====================================================

ais["segment_id"] = (
    ais.groupby("MMSI")["BaseDateTime"]
    .diff()
    .gt(pd.Timedelta("60min"))
    .cumsum()
)

# =====================================================
# 3. FEATURE ENGINEERING
# =====================================================

segments = build_segment_features(ais)

# =====================================================
# 4. GRID MATCHING
# =====================================================

segments["lat_bin"] = np.floor(segments["mean_lat"] * 10) / 10
segments["lon_bin"] = np.floor(segments["mean_lon"] * 10) / 10

segments["MMSI"] = segments["MMSI"].astype(int)

# =====================================================
# 5. MERGE WEAK LABELS
# =====================================================

segments = segments.merge(
    gfw[["MMSI", "date", "lat_bin", "lon_bin", "fishing_hours"]],
    on=["MMSI", "date", "lat_bin", "lon_bin"],
    how="left"
)

segments["fishing_hours"] = segments["fishing_hours"].fillna(0)
segments["label"] = (segments["fishing_hours"] > 0).astype(int)

# =====================================================
# 6. SAVE DATASET
# =====================================================

segments.to_csv("data/processed/segments_with_labels.csv", index=False)

print("Dataset rebuilt successfully.")
print("Class distribution:")
print(segments["label"].value_counts())