"""
main.py
=======
Unified fishing detection pipeline.

Three experiments (Teodor's structure):
  A) Train STRONG → Validate STRONG → Test STRONG   (upper bound with real labels)
  B) Train STRONG → Validate STRONG → Test WEAK     (do Norwegian patterns generalize?)
  C) Train WEAK   → Validate WEAK   → Test WEAK     (can weak labels substitute?)

Experiments A and B share the same trained models — the strong-trained models
are evaluated on both strong and weak test sets, directly answering whether
fishing behavior is universal or region-specific.

Classical models (LogReg, RF, GB) use segment-level feature vectors.
Neural networks (CNN1D, ForwardRNN, BiRNN) use raw AIS point sequences.

Usage:
    python main.py

Data paths (edit CONFIG below):
    AIS 2024        data/raw/ais2024.csv           (US waters, weak label set)
    GFW             data/raw/gfw.csv               (weak label source)
    AIS historic    data/raw/bw_ais_historic.csv   (Norwegian waters, strong label set)
    BarentsWatch    data/raw/barentswatch.csv       (strong label source)
"""

import os
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.preprocessing import StandardScaler

from src.data_loading import load_ais2024, load_ais_historic, load_gfw, load_barentswatch
from src.segmentation import segment_trajectories
from src.feature_engineering import compute_features, SEGMENT_FEATURES
from src.features_seq import (
    compute_point_features, select_features,
    build_sequences, fit_scaler, apply_scaler, SEQ_LEN,
)
from src.weak_labels import assign_weak_labels, assign_weak_labels_vessel_level
from src.strong_labels import (
    label_ais_points, aggregate_segments,
    assign_strong_labels_vessel_level,
)
from src.models_classical import CLASSICAL_MODELS, fit_classical, baseline_predict
from src.models_nn import build_nn_model, train_nn, predict_nn, NN_MODELS
from src.evaluation import (
    evaluate_model, plot_pr_curves, plot_threshold_curves,
    print_summary_table,
)

# =============================================================================
# CONFIG
# =============================================================================

AIS2024_PATH  = "data/raw/ais2024.csv"
GFW_PATH      = "data/raw/gfw.csv"
AIS_HIST_PATH = "data/raw/bw_ais_historic.csv"
BW_PATH       = "data/raw/barentswatch.csv"

RESULTS_DIR  = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

RANDOM_STATE     = 42
N_FOLDS          = 5
TOP_K_FEATS      = 8
NN_EPOCHS        = 30
NN_BATCH         = 64
NN_LR            = 1e-3
DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"
TEST_VESSEL_FRAC = 0.20

print(f"Device:            {DEVICE}")
print(f"Results directory: {RESULTS_DIR}\n")

# =============================================================================
# LOAD DATA
# =============================================================================

print("=" * 65)
print("LOADING DATA")
print("=" * 65)

print("\n[Weak data — US waters]")
ais_weak = load_ais2024(AIS2024_PATH)
gfw      = load_gfw(GFW_PATH)
print(f"  AIS2024 rows:    {len(ais_weak):,}")
print(f"  AIS2024 vessels: {ais_weak['mmsi'].nunique():,}")

print("\n[Strong data — Norwegian waters]")
ais_strong = load_ais_historic(AIS_HIST_PATH)
bw         = load_barentswatch(BW_PATH)
print(f"  AIS historic rows:    {len(ais_strong):,}")
print(f"  AIS historic vessels: {ais_strong['mmsi'].nunique():,}")

# =============================================================================
# SEGMENTATION & SEGMENT-LEVEL FEATURES
# =============================================================================

print("\n" + "=" * 65)
print("SEGMENTATION & FEATURE ENGINEERING")
print("=" * 65)

print("\n[Weak AIS]")
ais_weak_seg = segment_trajectories(ais_weak)
feats_weak   = compute_features(ais_weak_seg)
print(f"  Segments: {len(feats_weak):,}")

print("\n[Strong AIS]")
ais_strong_seg = segment_trajectories(ais_strong)
feats_strong   = compute_features(ais_strong_seg)
print(f"  Segments: {len(feats_strong):,}")

# =============================================================================
# LABEL CONSTRUCTION
# =============================================================================

print("\n" + "=" * 65)
print("LABEL CONSTRUCTION")
print("=" * 65)

print("\n[Weak labels]")
labeled_weak = assign_weak_labels(feats_weak, gfw)
labeled_weak = labeled_weak.dropna(subset=SEGMENT_FEATURES + ["weak_label"])
print(f"  Segments:      {len(labeled_weak):,}")
print(f"  Positive rate: {labeled_weak['weak_label'].mean():.3f}")

print("\n[Strong labels]")
print("  Labeling AIS points (MMSI + time window + spatial radius)...")
ais_strong_labeled = label_ais_points(ais_strong_seg, bw)
print("  Aggregating to segments...")
strong_seg_labels  = aggregate_segments(ais_strong_labeled)

labeled_strong = feats_strong.merge(
    strong_seg_labels[["mmsi", "segment_id", "strong_label", "fishing_ratio"]],
    on=["mmsi", "segment_id"],
    how="inner",
).dropna(subset=SEGMENT_FEATURES)
print(f"  Segments:      {len(labeled_strong):,}")
print(f"  Positive rate: {labeled_strong['strong_label'].mean():.3f}")

print("\n[Vessel-level labels for NN models]")
vessel_labels_weak   = assign_weak_labels_vessel_level(ais_weak, gfw)
vessel_labels_strong = assign_strong_labels_vessel_level(ais_strong, bw)
print(f"  Weak   — fishing: {(vessel_labels_weak==1).sum()}  non-fishing: {(vessel_labels_weak==0).sum()}")
print(f"  Strong — fishing: {(vessel_labels_strong==1).sum()}  non-fishing: {(vessel_labels_strong==0).sum()}")

# =============================================================================
# VESSEL SPLITS
# =============================================================================

print("\n" + "=" * 65)
print("VESSEL SPLITS")
print("=" * 65)

bw_mmsis           = set(bw["mmsi"].unique())
labeled_weak_clean = labeled_weak[~labeled_weak["mmsi"].isin(bw_mmsis)].copy()
print(f"\n[Weak — after removing BW vessels]")
print(f"  Segments: {len(labeled_weak_clean):,} | vessels: {labeled_weak_clean['mmsi'].nunique():,}")

strong_vessels = labeled_strong["mmsi"].unique()
strong_train_v, strong_test_v = train_test_split(
    strong_vessels, test_size=TEST_VESSEL_FRAC, random_state=RANDOM_STATE,
)
labeled_strong_train = labeled_strong[labeled_strong["mmsi"].isin(strong_train_v)].copy()
labeled_strong_test  = labeled_strong[labeled_strong["mmsi"].isin(strong_test_v)].copy()

print(f"\n[Strong train/test split]")
print(f"  Train: {len(labeled_strong_train):,} segs | {len(strong_train_v)} vessels")
print(f"  Test:  {len(labeled_strong_test):,} segs  | {len(strong_test_v)} vessels")
assert labeled_strong_test["strong_label"].nunique() == 2, \
    "Test set must contain both classes — increase TEST_VESSEL_FRAC"

# =============================================================================
# POINT-LEVEL FEATURES & SELECTION
# =============================================================================

print("\n" + "=" * 65)
print("POINT-LEVEL FEATURES & FEATURE SELECTION (NN models)")
print("=" * 65)

print("\nComputing point features for weak AIS...")
ais_weak_pts       = compute_point_features(ais_weak)
ais_weak_pts_clean = ais_weak_pts[~ais_weak_pts["mmsi"].isin(bw_mmsis)]
vl_weak_clean      = vessel_labels_weak[~vessel_labels_weak.index.isin(bw_mmsis)]

print("Selecting features via mutual information (weak data)...")
selected_feats = select_features(ais_weak_pts_clean, vl_weak_clean, top_k=TOP_K_FEATS)
print(f"Selected: {selected_feats}")

print("\nComputing point features for strong AIS...")
ais_strong_pts       = compute_point_features(ais_strong)
ais_strong_train_pts = ais_strong_pts[ais_strong_pts["mmsi"].isin(strong_train_v)]
ais_strong_test_pts  = ais_strong_pts[ais_strong_pts["mmsi"].isin(strong_test_v)]
vl_strong_train      = vessel_labels_strong[vessel_labels_strong.index.isin(strong_train_v)]
vl_strong_test       = vessel_labels_strong[vessel_labels_strong.index.isin(strong_test_v)]

# =============================================================================
# HELPERS
# =============================================================================

def run_classical(X_train, y_train, X_test, y_test):
    scaler  = StandardScaler().fit(X_train)
    results = []
    for name, factory in CLASSICAL_MODELS.items():
        model  = factory()
        Xtr    = scaler.transform(X_train) if name == "logreg" else X_train
        Xte    = scaler.transform(X_test)  if name == "logreg" else X_test
        fit_classical(model, Xtr, y_train)
        y_prob = model.predict_proba(Xte)[:, 1]
        results.append(evaluate_model(y_test, y_prob, name))
    X_test_df = pd.DataFrame(X_test, columns=SEGMENT_FEATURES)
    results.append(evaluate_model(y_test, baseline_predict(X_test_df).astype(float), "baseline"))
    return results, scaler


def run_nn(ais_train_pts, vl_train, ais_test_pts, vl_test, seq_scaler=None):
    print("    Building sequences...")
    seq_train, mmsi_tr = build_sequences(ais_train_pts, selected_feats)
    seq_test,  mmsi_te = build_sequences(ais_test_pts,  selected_feats)
    y_tr = np.array([vl_train.get(m, 0) for m in mmsi_tr])
    y_te = np.array([vl_test.get(m,  0) for m in mmsi_te])
    if seq_scaler is None:
        seq_scaler, seq_tr_s = fit_scaler(seq_train)
    else:
        seq_tr_s = apply_scaler(seq_scaler, seq_train)
    seq_te_s   = apply_scaler(seq_scaler, seq_test)
    n_features = seq_tr_s.shape[2]
    results    = []
    for name in NN_MODELS:
        print(f"    Training {name}...")
        model = build_nn_model(name, n_features=n_features, seq_len=SEQ_LEN)
        train_nn(model, seq_tr_s, y_tr, epochs=NN_EPOCHS, batch_size=NN_BATCH,
                 lr=NN_LR, device=DEVICE, verbose=True)
        y_prob = predict_nn(model, seq_te_s, device=DEVICE)
        results.append(evaluate_model(y_te, y_prob, name))
    return results, seq_scaler


def run_classical_cv(X_all, y_all, groups):
    gkf       = GroupKFold(n_splits=N_FOLDS)
    oof_probs = {name: np.zeros(len(y_all)) for name in CLASSICAL_MODELS}
    oof_y     = np.zeros(len(y_all))
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X_all, y_all, groups=groups)):
        X_tr, X_va = X_all[tr_idx], X_all[va_idx]
        y_tr, y_va = y_all[tr_idx], y_all[va_idx]
        oof_y[va_idx] = y_va
        sc = StandardScaler().fit(X_tr)
        print(f"  Fold {fold+1}/{N_FOLDS}  train={len(tr_idx)} val={len(va_idx)} pos_rate={y_va.mean():.3f}")
        for name, factory in CLASSICAL_MODELS.items():
            model = factory()
            Xtr   = sc.transform(X_tr) if name == "logreg" else X_tr
            Xva   = sc.transform(X_va) if name == "logreg" else X_va
            fit_classical(model, Xtr, y_tr)
            oof_probs[name][va_idx] = model.predict_proba(Xva)[:, 1]
    return oof_probs, oof_y


def run_nn_cv(ais_pts_all, vessel_labels_all, all_vessels):
    gkf        = GroupKFold(n_splits=N_FOLDS)
    oof_probs  = {name: {} for name in NN_MODELS}
    oof_labels = {}
    for fold, (tr_idx, va_idx) in enumerate(
        gkf.split(all_vessels, np.zeros(len(all_vessels)), groups=all_vessels)
    ):
        fold_train_v = set(all_vessels[tr_idx])
        fold_val_v   = set(all_vessels[va_idx])
        ais_tr = ais_pts_all[ais_pts_all["mmsi"].isin(fold_train_v)]
        ais_va = ais_pts_all[ais_pts_all["mmsi"].isin(fold_val_v)]
        seq_tr, mmsi_tr = build_sequences(ais_tr, selected_feats)
        seq_va, mmsi_va = build_sequences(ais_va, selected_feats)
        y_tr_nn = np.array([vessel_labels_all.get(m, 0) for m in mmsi_tr])
        sc_seq, seq_tr_s = fit_scaler(seq_tr)
        seq_va_s = apply_scaler(sc_seq, seq_va)
        n_feats  = seq_tr_s.shape[2]
        print(f"  Fold {fold+1}/{N_FOLDS}  train_v={len(fold_train_v)} val_v={len(fold_val_v)}")
        for name in NN_MODELS:
            print(f"    {name}...")
            model = build_nn_model(name, n_features=n_feats, seq_len=SEQ_LEN)
            train_nn(model, seq_tr_s, y_tr_nn, epochs=NN_EPOCHS, batch_size=NN_BATCH,
                     lr=NN_LR, device=DEVICE, verbose=False)
            probs_va = predict_nn(model, seq_va_s, device=DEVICE)
            for m, p in zip(mmsi_va, probs_va):
                oof_probs[name].setdefault(m, []).append(p)
        for m in fold_val_v:
            oof_labels[m] = vessel_labels_all.get(m, 0)
    all_v  = sorted(oof_labels.keys())
    y_oof  = np.array([oof_labels[m] for m in all_v])
    p_flat = {name: np.array([np.mean(oof_probs[name].get(m, [0.0])) for m in all_v])
              for name in NN_MODELS}
    return p_flat, y_oof


def save_plots(results, exp_label, title):
    plot_pr_curves(results, f"PR Curve — {title}",
                   os.path.join(RESULTS_DIR, f"{exp_label}_pr_curve.png"))
    plot_threshold_curves(results, f"Threshold — {title}",
                          os.path.join(RESULTS_DIR, f"{exp_label}_threshold_curve.png"))


def results_to_df(results, experiment):
    return pd.DataFrame([{
        "experiment": experiment,
        "model":      r["model"],
        "ap":         round(r["ap"], 4),
        "f1":         round(r["f1"], 4),
        "precision":  round(r["precision"], 4),
        "recall":     round(r["recall"], 4),
        "threshold":  round(r["threshold"], 3),
    } for r in results])

# =============================================================================
# EXPERIMENT A: Train STRONG → Test STRONG  (upper bound)
# =============================================================================

print("\n" + "=" * 65)
print("EXPERIMENT A: Train STRONG → Test STRONG")
print("Upper bound — real labels, same domain (Norwegian waters)")
print("=" * 65)

X_train_A = labeled_strong_train[SEGMENT_FEATURES].values
y_train_A = labeled_strong_train["strong_label"].values
X_test_A  = labeled_strong_test[SEGMENT_FEATURES].values
y_test_A  = labeled_strong_test["strong_label"].values

print("\n[Classical models]")
results_A_classical, scaler_A = run_classical(X_train_A, y_train_A, X_test_A, y_test_A)

print("\n[NN models]")
results_A_nn, seq_scaler_A = run_nn(
    ais_strong_train_pts, vl_strong_train,
    ais_strong_test_pts,  vl_strong_test,
)

results_A = results_A_classical + results_A_nn
print_summary_table(results_A, "EXPERIMENT A: Strong → Strong")
save_plots(results_A, "A", "Train Strong / Test Strong")

# =============================================================================
# EXPERIMENT B: Train STRONG → Test WEAK
# Key question: do Norwegian patterns generalize to global/US vessels?
# Uses the same strong-trained models from A — no retraining for classical.
# =============================================================================

print("\n" + "=" * 65)
print("EXPERIMENT B: Train STRONG → Test WEAK")
print("Transfer — do Norwegian fishing patterns generalize globally?")
print("If B ≈ A: behavior is universal. If B << A: region-specific.")
print("=" * 65)

X_test_B = labeled_weak_clean[SEGMENT_FEATURES].values
y_test_B = labeled_weak_clean["weak_label"].values

print("\n[Classical models — retrained on strong, tested on weak]")
results_B_classical = []
for name, factory in CLASSICAL_MODELS.items():
    model  = factory()
    Xtr    = scaler_A.transform(X_train_A) if name == "logreg" else X_train_A
    Xte    = scaler_A.transform(X_test_B)  if name == "logreg" else X_test_B
    fit_classical(model, Xtr, y_train_A)
    y_prob = model.predict_proba(Xte)[:, 1]
    results_B_classical.append(evaluate_model(y_test_B, y_prob, name))
X_test_B_df = pd.DataFrame(X_test_B, columns=SEGMENT_FEATURES)
results_B_classical.append(evaluate_model(y_test_B, baseline_predict(X_test_B_df).astype(float), "baseline"))

print("\n[NN models — retrained on strong, tested on weak]")
# Build weak test sequences using the scaler fitted on strong train sequences
seq_test_B,  mmsi_te_B = build_sequences(ais_weak_pts_clean, selected_feats)
y_test_B_nn = np.array([vl_weak_clean.get(m, 0) for m in mmsi_te_B])
seq_test_B_s = apply_scaler(seq_scaler_A, seq_test_B)

seq_train_B, mmsi_tr_B = build_sequences(ais_strong_train_pts, selected_feats)
y_train_B_nn = np.array([vl_strong_train.get(m, 0) for m in mmsi_tr_B])
_, seq_train_B_s = fit_scaler(seq_train_B)
n_feats_B = seq_train_B_s.shape[2]

results_B_nn = []
for name in NN_MODELS:
    print(f"  Training {name} (strong) → testing on weak...")
    model = build_nn_model(name, n_features=n_feats_B, seq_len=SEQ_LEN)
    train_nn(model, seq_train_B_s, y_train_B_nn, epochs=NN_EPOCHS,
             batch_size=NN_BATCH, lr=NN_LR, device=DEVICE, verbose=True)
    y_prob = predict_nn(model, seq_test_B_s, device=DEVICE)
    results_B_nn.append(evaluate_model(y_test_B_nn, y_prob, name))

results_B = results_B_classical + results_B_nn
print_summary_table(results_B, "EXPERIMENT B: Strong → Weak")
save_plots(results_B, "B", "Train Strong / Test Weak")

# =============================================================================
# EXPERIMENT C: Train WEAK → Validate WEAK → Test WEAK  (GroupKFold CV)
# Can cheap GFW-derived labels produce a working classifier?
# =============================================================================

print("\n" + "=" * 65)
print("EXPERIMENT C: Train WEAK → Test WEAK (GroupKFold CV)")
print("Weak-only pipeline — baseline weak supervision performance")
print("=" * 65)

X_C      = labeled_weak_clean[SEGMENT_FEATURES].values
y_C      = labeled_weak_clean["weak_label"].values
groups_C = labeled_weak_clean["mmsi"].values

print(f"\n[Classical models — {N_FOLDS}-fold GroupKFold CV]")
oof_probs_C, oof_y_C = run_classical_cv(X_C, y_C, groups_C)

results_C_classical = []
for name in CLASSICAL_MODELS:
    results_C_classical.append(evaluate_model(oof_y_C, oof_probs_C[name], name))
X_C_df = pd.DataFrame(X_C, columns=SEGMENT_FEATURES)
results_C_classical.append(evaluate_model(oof_y_C, baseline_predict(X_C_df).astype(float), "baseline"))

print(f"\n[NN models — {N_FOLDS}-fold GroupKFold CV]")
vessels_C = labeled_weak_clean["mmsi"].unique()
oof_probs_C_nn, oof_y_C_nn = run_nn_cv(ais_weak_pts_clean, vl_weak_clean, vessels_C)

results_C_nn = []
for name in NN_MODELS:
    results_C_nn.append(evaluate_model(oof_y_C_nn, oof_probs_C_nn[name], name))

results_C = results_C_classical + results_C_nn
print_summary_table(results_C, "EXPERIMENT C: Weak → Weak (OOF CV)")
save_plots(results_C, "C", "Train Weak / Test Weak (OOF CV)")

# =============================================================================
# FINAL SUMMARY
# =============================================================================

print("\n" + "=" * 65)
print("FINAL SUMMARY — ALL EXPERIMENTS")
print("=" * 65)

summary = pd.concat([
    results_to_df(results_A, "A: strong→strong"),
    results_to_df(results_B, "B: strong→weak"),
    results_to_df(results_C, "C: weak→weak"),
], ignore_index=True)

pivot = summary.pivot_table(index="model", columns="experiment", values="ap").round(4)
print("\nAverage Precision (AP) by model and experiment:")
print(pivot.to_string())

csv_path = os.path.join(RESULTS_DIR, "all_results.csv")
summary.to_csv(csv_path, index=False)
print(f"\nFull results saved to: {csv_path}")
print("\nAll experiments complete.")