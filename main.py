from src.data_loading import load_ais2026, load_ais2024, load_gfw, load_barentswatch
from src.segmentation import segment_trajectories
from src.feature_engineering import compute_features
from src.weak_labels import assign_weak_labels
from src.strong_labels import label_ais_points, aggregate_segments
from src.models import train_gb
from src.evaluation import evaluate

from sklearn.preprocessing import StandardScaler
from datetime import timedelta

# -----------------------
# CONFIG
# -----------------------
FEATURES = ["mean_speed", "speed_std", "turning_rate"]

# -----------------------
# LOAD DATA
# -----------------------
# Train (weak)
ais24 = load_ais2024("data/raw/ais2024.csv")

# Test (strong)
ais26 = load_ais2026("data/raw/ais2026.csv")

gfw = load_gfw("data/raw/gfw.csv")
bw = load_barentswatch("data/raw/barentswatch.csv")

# -----------------------
# FILTER BW TO AIS WINDOW (14-day constraint)
# -----------------------
ais_max_time = bw["setupDateTime"].max()
ais_min_time = ais_max_time - timedelta(days=14)

bw = bw[
    (bw["setupDateTime"] >= ais_min_time) &
    (bw["setupDateTime"] <= ais_max_time)
]

assert len(bw) > 0, "No BW data in AIS-accessible window"

# -----------------------
# SANITY CHECKS
# -----------------------
assert ais24["timestamp"].notna().all()
assert ais26["timestamp"].notna().all()
assert bw["setupDateTime"].notna().all()

print("AIS24 range:", ais24["timestamp"].min(), ais24["timestamp"].max())
print("AIS26 range:", ais26["timestamp"].min(), ais26["timestamp"].max())
print("BW range:", bw["setupDateTime"].min(), bw["setupDateTime"].max())

# -----------------------
# SPLIT VESSELS (LEAKAGE CONTROL)
# -----------------------
test_vessels = bw["mmsi"].unique()

# -----------------------
# TRAIN PIPELINE (WEAK)
# -----------------------
ais24 = segment_trajectories(ais24)
features_train = compute_features(ais24)

weak_df = assign_weak_labels(features_train, gfw)

# remove test vessels
weak_df = weak_df[~weak_df["mmsi"].isin(test_vessels)]
assert len(weak_df) > 0, "Empty training set"

X_train = weak_df[FEATURES]
y_train = weak_df["weak_label"]

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

model = train_gb(X_train, y_train)

# -----------------------
# TEST PIPELINE (STRONG)
# -----------------------
# keep only vessels that exist in BW
ais26 = ais26[ais26["mmsi"].isin(test_vessels)]
assert len(ais26) > 0, "No AIS data for BW vessels"

# segment + features
ais26 = segment_trajectories(ais26)
features_test = compute_features(ais26)

# strong labeling
ais26 = label_ais_points(ais26, bw)

print("Fishing points in AIS26:", ais26["fishing"].sum())
assert ais26["fishing"].sum() > 0, "No fishing points found"

# aggregate to segments
strong_segments = aggregate_segments(ais26)

print("Strong segments:", len(strong_segments))
print("Class balance:\n", strong_segments["strong_label"].value_counts())

assert len(strong_segments) > 0, "No strong segments created"
assert strong_segments["strong_label"].nunique() == 2, "Test set must contain both classes"

# -----------------------
# MERGE FEATURES + LABELS
# -----------------------
test_df = features_test.merge(
    strong_segments,
    on=["mmsi", "segment_id"],
    how="inner"
)

assert len(test_df) > 0, "Empty test set after merge"

# -----------------------
# TEST PREP
# -----------------------
X_test = test_df[FEATURES]
y_test = test_df["strong_label"]

X_test = scaler.transform(X_test)

# -----------------------
# PREDICTION
# -----------------------
y_prob = model.predict_proba(X_test)[:, 1]
y_pred = (y_prob >= 0.5).astype(int)

# -----------------------
# EVALUATION
# -----------------------
results = evaluate(y_test, y_pred, y_prob)

print(results)