import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import GroupShuffleSplit

# =====================================================
# 1. LOAD DATA
# =====================================================

# --- AIS ---
ais = pd.read_csv("data/raw/AIS_2024_11_14.csv")

ais["BaseDateTime"] = pd.to_datetime(ais["BaseDateTime"])
ais = ais.sort_values(["MMSI", "BaseDateTime"])
ais["date"] = ais["BaseDateTime"].dt.date

# --- GFW ---
gfw = pd.read_csv("data/raw/mmsi-daily-csvs-10-v3-2024-11-14.csv")

gfw = gfw.rename(columns={
    "mmsi": "MMSI",
    "cell_ll_lat": "lat_bin",
    "cell_ll_lon": "lon_bin"
})

gfw["date"] = pd.to_datetime(gfw["date"]).dt.date
gfw["MMSI"] = gfw["MMSI"].astype(int)
gfw["lat_bin"] = gfw["lat_bin"].astype(float)
gfw["lon_bin"] = gfw["lon_bin"].astype(float)

# =====================================================
# 2. SEGMENT AIS (60 min gap rule)
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

def turning_rate(cog_series):
    cog_series = cog_series.dropna().values
    if len(cog_series) < 2:
        return 0
    diff = np.diff(cog_series)
    diff = (diff + 180) % 360 - 180
    return np.mean(np.abs(diff))

segments = ais.groupby(["MMSI", "segment_id"]).agg({
    "SOG": ["mean", "std"],
    "COG": turning_rate,
    "LAT": "mean",
    "LON": "mean",
    "date": "first"
}).reset_index()

segments.columns = [
    "MMSI", "segment_id",
    "mean_speed",
    "speed_std",
    "turning_rate",
    "mean_lat",
    "mean_lon",
    "date"
]

# =====================================================
# 4. GRID TO 0.1° (CRITICAL: use floor for negatives)
# =====================================================

segments["lat_bin"] = np.floor(segments["mean_lat"] * 10) / 10
segments["lon_bin"] = np.floor(segments["mean_lon"] * 10) / 10

segments["MMSI"] = segments["MMSI"].astype(int)
segments["lat_bin"] = segments["lat_bin"].astype(float)
segments["lon_bin"] = segments["lon_bin"].astype(float)

# =====================================================
# 5. MERGE WITH GFW (WEAK LABELS)
# =====================================================

segments = segments.merge(
    gfw[["MMSI", "date", "lat_bin", "lon_bin", "fishing_hours"]],
    on=["MMSI", "date", "lat_bin", "lon_bin"],
    how="left"
)

segments["fishing_hours"] = segments["fishing_hours"].fillna(0)
segments["label"] = (segments["fishing_hours"] > 0).astype(int)

print("Class distribution:")
print(segments["label"].value_counts())

# =====================================================
# 6. VESSEL-LEVEL SPLIT
# =====================================================

X = segments[["mean_speed", "speed_std", "turning_rate"]]
y = segments["label"]
groups = segments["MMSI"]

gss = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups=groups))

train = segments.iloc[train_idx].copy()
test = segments.iloc[test_idx].copy()

# =====================================================
# 7. BASELINE MODEL
# =====================================================

def baseline_rule(row):
    if 2 <= row["mean_speed"] <= 5 and row["turning_rate"] > 10:
        return 1
    return 0

train["baseline_pred"] = train.apply(baseline_rule, axis=1)
test["baseline_pred"] = test.apply(baseline_rule, axis=1)

# =====================================================
# 8. EVALUATION
# =====================================================

y_true = test["label"]
y_pred = test["baseline_pred"]

print("\nBaseline performance (vessel-level split):")
print("Precision:", precision_score(y_true, y_pred, zero_division=0))
print("Recall:", recall_score(y_true, y_pred, zero_division=0))
print("F1 score:", f1_score(y_true, y_pred, zero_division=0))

# =====================================================
# 9. OPTIONAL: SPEED-ONLY THRESHOLD SEARCH
# =====================================================

best_f1 = 0
best_threshold = None

for low in np.linspace(0, 6, 30):
    for high in np.linspace(6, 12, 30):
        temp_pred = (
            (train["mean_speed"] >= low) &
            (train["mean_speed"] <= high)
        ).astype(int)

        f1 = f1_score(train["label"], temp_pred, zero_division=0)

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = (low, high)

print("\nBest speed-only threshold (train):", best_threshold)
print("Best train F1:", best_f1)
