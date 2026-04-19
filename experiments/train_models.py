from src.data_loading import load_ais2024, load_gfw
from src.segmentation import segment_trajectories
from src.feature_engineering import compute_features
from src.weak_labels import assign_weak_labels

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    average_precision_score, precision_recall_curve
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# ----------------------
# CONFIG
# ----------------------
FEATURES = [
    # Speed features
    "mean_speed",
    "speed_std",
    "min_speed",
    "max_speed",
    "low_speed_fraction",   # fraction of points below 3 knots

    # Heading / turning features
    "turning_rate",
    "cog_std",              # std of course-over-ground within segment

    # Trajectory shape
    "sinuosity",            # total path length / straight-line displacement
]

RANDOM_STATE = 42
THRESHOLDS = np.linspace(0.0, 1.0, 50)
N_FOLDS = 5

RESULTS_DIR = "../results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ----------------------
# LOAD + PREP DATA
# ----------------------
ais = load_ais2024("../data/raw/ais2024.csv")
gfw = load_gfw("../data/raw/gfw.csv")

ais = segment_trajectories(ais)
features = compute_features(ais)

df = assign_weak_labels(features, gfw)

# enforce types + clean
df["mmsi"] = df["mmsi"].astype(int)
df = df.dropna(subset=FEATURES + ["weak_label"])

print(f"Total segments: {len(df)}")
print(f"Positive rate: {df['weak_label'].mean():.3f}")
print(f"Unique vessels: {df['mmsi'].nunique()}")

# ----------------------
# CLASS IMBALANCE RATIO
# ----------------------
n_neg = (df["weak_label"] == 0).sum()
n_pos = (df["weak_label"] == 1).sum()
pos_weight = n_neg / n_pos
print(f"Imbalance ratio (neg/pos): {pos_weight:.1f}")

# ----------------------
# MODEL DEFINITIONS
# ----------------------
def make_logreg():
    return LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        C=0.1,                  # regularisation - tune if needed
        solver="lbfgs"
    )

def make_rf():
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_leaf=5,
        class_weight="balanced_subsample",  # resamples per tree - better for RF
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

def make_gb():
    return GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,          # stochastic GB - reduces overfitting
        min_samples_leaf=10,
        random_state=RANDOM_STATE
    )

def baseline_predict(X):
    """Rule-based baseline: slow + turning."""
    return (
        (X["mean_speed"] < 5) &
        (X["turning_rate"] > 10)
    ).astype(int)

# ----------------------
# THRESHOLD SWEEP
# ----------------------
def evaluate_thresholds(y_true, y_prob, model_name):
    rows = []
    for t in THRESHOLDS:
        y_pred = (y_prob >= t).astype(int)
        rows.append({
            "model": model_name,
            "threshold": t,
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall":    recall_score(y_true, y_pred, zero_division=0),
            "f1":        f1_score(y_true, y_pred, zero_division=0),
        })
    return pd.DataFrame(rows)

def best_threshold(df_model):
    idx = df_model["f1"].idxmax()
    return df_model.loc[idx]

# ----------------------
# CROSS-VALIDATION LOOP
# ----------------------
vessels = df["mmsi"].values
X_all = df[FEATURES].values
y_all = df["weak_label"].values

gkf = GroupKFold(n_splits=N_FOLDS)

model_configs = {
    "logreg": (make_logreg, True),   # (factory, needs_scaling)
    "rf":     (make_rf,     False),
    "gb":     (make_gb,     False),
}

# Collect out-of-fold predictions for each model
oof_probs  = {name: np.zeros(len(df)) for name in model_configs}
oof_labels = np.zeros(len(df))

print(f"\nRunning {N_FOLDS}-fold group cross-validation...")

for fold, (train_idx, val_idx) in enumerate(gkf.split(X_all, y_all, groups=vessels)):
    X_tr, X_val = X_all[train_idx], X_all[val_idx]
    y_tr, y_val = y_all[train_idx], y_all[val_idx]

    oof_labels[val_idx] = y_val

    scaler = StandardScaler()
    X_tr_scaled  = scaler.fit_transform(X_tr)
    X_val_scaled = scaler.transform(X_val)

    for name, (factory, needs_scale) in model_configs.items():
        model = factory()
        Xtr = X_tr_scaled if needs_scale else X_tr
        Xvl = X_val_scaled if needs_scale else X_val

        if name == "gb":
            weights = compute_sample_weight("balanced", y_tr)
            model.fit(Xtr, y_tr, sample_weight=weights)
        else:
            model.fit(Xtr, y_tr)

        oof_probs[name][val_idx] = model.predict_proba(Xvl)[:, 1]

    pos_rate = y_val.mean()
    print(f"  Fold {fold+1}: val_size={len(val_idx)}, pos_rate={pos_rate:.3f}")

# ----------------------
# OOF METRICS + THRESHOLD SWEEP
# ----------------------
print("\n--- Out-of-fold Average Precision ---")
results_all = []

for name, y_prob in oof_probs.items():
    ap = average_precision_score(oof_labels, y_prob)
    print(f"  {name}: AP = {ap:.4f}")

    df_thresh = evaluate_thresholds(oof_labels, y_prob, name)
    results_all.append(df_thresh)

# Baseline (no threshold sweep)
X_df = df[FEATURES]
y_pred_base = baseline_predict(X_df)
base_row = pd.DataFrame([{
    "model":     "baseline",
    "threshold": None,
    "precision": precision_score(oof_labels, y_pred_base, zero_division=0),
    "recall":    recall_score(oof_labels, y_pred_base, zero_division=0),
    "f1":        f1_score(oof_labels, y_pred_base, zero_division=0),
}])
results_all.append(base_row)

results_df = pd.concat(results_all, ignore_index=True)
results_df.to_csv(os.path.join(RESULTS_DIR, "threshold_sweep.csv"), index=False)

# ----------------------
# PRINT BEST THRESHOLDS
# ----------------------
print("\n--- Best threshold by F1 (OOF) ---")
for name in model_configs:
    sub = results_df[results_df["model"] == name]
    best = best_threshold(sub)
    print(f"\n{name}:")
    print(f"  threshold={best['threshold']:.3f}  "
          f"P={best['precision']:.3f}  "
          f"R={best['recall']:.3f}  "
          f"F1={best['f1']:.3f}")

# ----------------------
# PLOT 1: PROPER PR CURVE
# (uses sklearn PR curve, not threshold sweep zigzag)
# ----------------------
def plot_pr_curve(oof_probs, y_true, results_df):
    plt.figure(figsize=(8, 6))

    colors = {"logreg": "tab:blue", "rf": "tab:orange", "gb": "tab:green"}

    for name, y_prob in oof_probs.items():
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)
        plt.plot(recall, precision,
                 label=f"{name} (AP={ap:.3f})",
                 color=colors[name],
                 linewidth=2)

    # Baseline as a single point
    base = results_df[results_df["model"] == "baseline"].iloc[0]
    plt.scatter(base["recall"], base["precision"],
                marker="X", s=120, zorder=5,
                color="red", label=f"baseline")

    # No-skill line
    pos_rate = y_true.mean()
    plt.axhline(y=pos_rate, color="gray", linestyle="--",
                linewidth=1, label=f"no-skill ({pos_rate:.3f})")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve (OOF)")
    plt.legend(loc="upper right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "pr_curve.png"), dpi=150)
    plt.close()
    print("Saved pr_curve.png")

# ----------------------
# PLOT 2: THRESHOLD CURVES
# ----------------------
def plot_threshold_metrics(results_df):
    plt.figure(figsize=(10, 6))

    colors = {"logreg": "tab:blue", "rf": "tab:orange", "gb": "tab:green"}

    for name in model_configs:
        sub = results_df[results_df["model"] == name]
        c = colors[name]
        plt.plot(sub["threshold"], sub["precision"],
                 color=c, label=f"{name}-precision")
        plt.plot(sub["threshold"], sub["recall"],
                 color=c, linestyle="--", label=f"{name}-recall", alpha=0.7)

    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title("Threshold vs Precision / Recall (OOF)")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "threshold_curve.png"), dpi=150)
    plt.close()
    print("Saved threshold_curve.png")

# ----------------------
# PLOT 3: AP PER FOLD (sanity check for variance)
# ----------------------
def plot_ap_per_fold(X_all, y_all, vessels):
    fold_aps = {name: [] for name in model_configs}

    for fold, (train_idx, val_idx) in enumerate(
            gkf.split(X_all, y_all, groups=vessels)):

        X_tr, X_val = X_all[train_idx], X_all[val_idx]
        y_tr, y_val = y_all[train_idx], y_all[val_idx]

        scaler = StandardScaler()
        X_tr_s  = scaler.fit_transform(X_tr)
        X_val_s = scaler.transform(X_val)

        for name, (factory, needs_scale) in model_configs.items():
            model = factory()
            Xtr = X_tr_s if needs_scale else X_tr
            Xvl = X_val_s if needs_scale else X_val

            if name == "gb":
                weights = compute_sample_weight("balanced", y_tr)
                model.fit(Xtr, y_tr, sample_weight=weights)
            else:
                model.fit(Xtr, y_tr)

            prob = model.predict_proba(Xvl)[:, 1]
            fold_aps[name].append(average_precision_score(y_val, prob))

    fig, ax = plt.subplots(figsize=(7, 4))
    names = list(fold_aps.keys())
    data  = [fold_aps[n] for n in names]

    ax.boxplot(data, labels=names, patch_artist=True)
    ax.set_ylabel("Average Precision")
    ax.set_title("AP per Fold (GroupKFold)")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "ap_per_fold.png"), dpi=150)
    plt.close()
    print("Saved ap_per_fold.png")

# ----------------------
# RUN PLOTS
# ----------------------
plot_pr_curve(oof_probs, oof_labels, results_df)
plot_threshold_metrics(results_df)
plot_ap_per_fold(X_all, y_all, vessels)

print("\nDone.")