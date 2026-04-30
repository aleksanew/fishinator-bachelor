"""
main.py – Fishing Vessel Detection: CNN / ForwardRNN / BiRNN on AIS sequences
==============================================================================

Pipeline
--------
1.  Load AIS 2024 + GFW (weak labels)
2.  Compute all candidate point-level features from {MMSI, timestamp, LAT, LON, SOG, COG}
3.  Assign weak labels per vessel via GFW grid matching
4.  Feature selection: rank candidate point-level features by mutual information
    with the weak label (broadcast to pings).  The selected subset is the ONLY
    input the models ever receive — this is where selection has real effect.
5.  Split vessels into train / test (no leakage: same vessel never spans both)
6.  Build fixed-length sequence windows using selected features only
7.  Normalise (StandardScaler fit on train only)
8.  Train CNN1D / ForwardRNN / BiRNN
9.  Evaluate and print a comparison table

Usage
-----
    python main.py
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.features import (
    compute_point_features,
    select_features,
    build_sequences,
    fit_scaler,
    apply_scaler,
    CANDIDATE_FEATURES,
    SEQ_LEN,
)
from src.weak_labels import assign_weak_labels, label_summary
from src.models import build_model
from src.training import train_model, predict, evaluate

# =============================================================================
# CONFIG
# =============================================================================
AIS_PATH  = "data/raw/ais2024.csv"
GFW_PATH  = "data/raw/gfw.csv"

SEQ_LENGTH     = SEQ_LEN   # 60 time-steps per window
TOP_K          = 9         # how many point-level features to keep
TEST_SIZE      = 0.20      # fraction of vessels held out
RANDOM_SEED    = 42
EPOCHS         = 30
BATCH_SIZE     = 64
LEARNING_RATE  = 1e-3
DEVICE = "cuda"

MODELS_TO_RUN  = ["cnn", "forward_rnn", "bi_rnn"]


# =============================================================================
# 1.  LOAD DATA
# =============================================================================
print("=" * 62)
print("1. Loading data")
print("=" * 62)

ais = pd.read_csv(AIS_PATH)
ais["BaseDateTime"] = pd.to_datetime(ais["BaseDateTime"])
ais = ais.sort_values(["MMSI", "BaseDateTime"]).reset_index(drop=True)
ais = ais[["MMSI", "BaseDateTime", "LAT", "LON", "SOG", "COG"]].dropna(subset=["MMSI","BaseDateTime","LAT","LON"])

print(f"  AIS rows:        {len(ais):>10,}")
print(f"  Unique vessels:  {ais['MMSI'].nunique():>10,}")
print(f"  Date range:      {ais['BaseDateTime'].min()}  →  {ais['BaseDateTime'].max()}")

gfw = pd.read_csv(GFW_PATH)
print(f"\n  GFW rows:        {len(gfw):>10,}")
print(f"  Fishing rows:    {(gfw['fishing_hours'] > 0).sum():>10,}")


# =============================================================================
# 2.  POINT-LEVEL FEATURE COMPUTATION
# =============================================================================
print("\n" + "=" * 62)
print("2. Computing point-level features")
print("=" * 62)
print(f"  Candidate features ({len(CANDIDATE_FEATURES)}): {CANDIDATE_FEATURES}")
ais = compute_point_features(ais)


# =============================================================================
# 3.  WEAK LABEL ASSIGNMENT
# =============================================================================
print("\n" + "=" * 62)
print("3. Assigning weak labels via GFW grid matching")
print("=" * 62)
vessel_labels = assign_weak_labels(ais, gfw)
label_summary(vessel_labels)


# =============================================================================
# 4.  FEATURE SELECTION  (point-level MI, determines model input)
# =============================================================================
print("\n" + "=" * 62)
print("4. Feature selection (point-level mutual information)")
print("=" * 62)
print(f"  Selecting top {TOP_K} of {len(CANDIDATE_FEATURES)} candidate features.")
print("  Label is broadcast from vessel → each of its pings.")
print("  Only selected features will be passed to the sequence models.\n")

selected_features = select_features(ais, vessel_labels, top_k=TOP_K, verbose=True)


# =============================================================================
# 5.  TRAIN / TEST VESSEL SPLIT  (vessel-level → no window leakage)
# =============================================================================
print("=" * 62)
print("5. Train / test vessel split")
print("=" * 62)

all_mmsi   = vessel_labels.index.values
all_labels = vessel_labels.values

mmsi_train, mmsi_test = train_test_split(
    all_mmsi,
    test_size    = TEST_SIZE,
    stratify     = all_labels,
    random_state = RANDOM_SEED,
)

print(f"  Train vessels: {len(mmsi_train):>6}  |  Test vessels: {len(mmsi_test):>6}")

ais_train = ais[ais["MMSI"].isin(mmsi_train)]
ais_test  = ais[ais["MMSI"].isin(mmsi_test)]


# =============================================================================
# 6.  BUILD SEQUENCES using ONLY the selected features
# =============================================================================
print("\n" + "=" * 62)
print("6. Building sequence windows")
print("=" * 62)
print(f"  Using features: {selected_features}")
print(f"  Sequence length: {SEQ_LENGTH} time-steps")

seq_train, mmsi_train_seq = build_sequences(ais_train, feature_cols=selected_features, seq_len=SEQ_LENGTH)
seq_test,  mmsi_test_seq  = build_sequences(ais_test,  feature_cols=selected_features, seq_len=SEQ_LENGTH)

def map_labels(mmsi_array, label_series):
    return np.array([label_series.get(m, 0) for m in mmsi_array])

y_train = map_labels(mmsi_train_seq, vessel_labels)
y_test  = map_labels(mmsi_test_seq,  vessel_labels)

N_FEATURES = seq_train.shape[2]
print(f"\n  Train windows: {len(seq_train):>7,}  "
      f"(fishing={y_train.sum():,} | non-fishing={(y_train==0).sum():,})")
print(f"  Test  windows: {len(seq_test):>7,}  "
      f"(fishing={y_test.sum():,} | non-fishing={(y_test==0).sum():,})")
print(f"  Input shape per window: ({SEQ_LENGTH}, {N_FEATURES})")


# =============================================================================
# 7.  NORMALISE
# =============================================================================
print("\n" + "=" * 62)
print("7. Normalising sequences (StandardScaler fit on train only)")
print("=" * 62)
scaler, seq_train_s = fit_scaler(seq_train)
seq_test_s          = apply_scaler(scaler, seq_test)
print("  Done.")


# =============================================================================
# 8.  TRAIN & EVALUATE
# =============================================================================
all_results = []

for model_name in MODELS_TO_RUN:
    print("\n" + "=" * 62)
    print(f"8. Model: {model_name.upper()}")
    print("=" * 62)

    model = build_model(model_name, n_features=N_FEATURES, seq_len=SEQ_LENGTH)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    train_model(
        model,
        X_train    = seq_train_s,
        y_train    = y_train,
        epochs     = EPOCHS,
        batch_size = BATCH_SIZE,
        lr         = LEARNING_RATE,
        device     = DEVICE,
        verbose    = True,
    )

    y_prob, y_pred = predict(model, seq_test_s, device=DEVICE)
    results = evaluate(y_test, y_pred, y_prob, model_name=model_name)
    all_results.append(results)


# =============================================================================
# 9.  SUMMARY TABLE
# =============================================================================
print("\n" + "=" * 62)
print("SUMMARY")
print(f"  Features used ({N_FEATURES}): {selected_features}")
print("=" * 62)
print(f"{'Model':<15} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>8} {'AUC':>8}")
print("-" * 62)
for r in all_results:
    print(
        f"{r['model']:<15} "
        f"{r['accuracy']:>9.4f} "
        f"{r['precision']:>10.4f} "
        f"{r['recall']:>8.4f} "
        f"{r['f1']:>8.4f} "
        f"{r['roc_auc']:>8.4f}"
    )
print("=" * 62)
