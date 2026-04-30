"""
Feature engineering for fishing vessel detection.

Design principles
-----------------
* Only the 6 universally guaranteed AIS fields are used as inputs:
      MMSI, BaseDateTime, LAT, LON, SOG, COG
  No static vessel metadata (VesselType, Length, etc.) is required,
  making the pipeline applicable to any vessel regardless of registry completeness.

* All candidate features are point-level (one value per AIS ping).
  Feature selection operates on these point-level features and directly
  controls which columns are passed into the sequence models.

* Feature selection uses mutual information between each point-level feature
  and the vessel's weak label (broadcast to every ping of that vessel).
  This is the correct level: the model is trained on sequences of pings,
  so we want to know which per-ping signals are most informative.
"""

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Constants / sentinels
# ---------------------------------------------------------------------------
COG_UNAVAILABLE = 360.0   # AIS standard sentinel value
SOG_UNAVAILABLE = 102.3   # AIS standard sentinel value
STOPPED_KN      = 0.5     # knots; below this ≈ stationary
FISHING_MAX_KN  = 6.0     # knots; heuristic upper bound for active fishing


# ---------------------------------------------------------------------------
# 1.  Candidate point-level features
#     ALL derived solely from {MMSI, BaseDateTime, LAT, LON, SOG, COG}
# ---------------------------------------------------------------------------

#: Full set of candidate feature names.  Feature selection picks a subset.
CANDIDATE_FEATURES = [
    "sog",             # Speed Over Ground (knots)
    "turn_rate",       # ΔCOG / Δt  (deg/s)  – encodes manoeuvring
    "acceleration",    # ΔSOG / Δt  (kn/s)   – encodes speed changes
    "is_stopped",      # 1 if SOG ≤ 0.5 kn   – drifting / anchored
    "is_fishing_speed",# 1 if 0.5 < SOG ≤ 6 kn – typical trawling/jigging range
    "cog_sin",         # sin(COG)             – circular heading encoding
    "cog_cos",         # cos(COG)             – circular heading encoding
    "dist_m",          # metres travelled since previous ping
    "hour_sin",        # sin(2π × hour/24)    – time-of-day fishing pattern
    "hour_cos",        # cos(2π × hour/24)
]


# ---------------------------------------------------------------------------
# 2.  Compute all candidate features
# ---------------------------------------------------------------------------

def compute_point_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all candidate point-level features to the dataframe.

    Input must have columns: MMSI, BaseDateTime (datetime64), LAT, LON, SOG, COG.
    Returns df with additional feature columns; original columns are preserved.
    """
    df = df.copy()
    df = df.sort_values(["MMSI", "BaseDateTime"]).reset_index(drop=True)

    # Replace AIS sentinel values with NaN so they don't pollute derived features
    df["SOG"] = df["SOG"].replace(SOG_UNAVAILABLE, np.nan)
    df["COG"] = df["COG"].replace(COG_UNAVAILABLE, np.nan)

    # Time delta between consecutive pings for the same vessel (seconds)
    dt_sec = (
        df.groupby("MMSI")["BaseDateTime"]
          .diff()
          .dt.total_seconds()
          .fillna(0)
          .clip(lower=0)
    )

    # SOG (raw, NaN -> 0)
    df["sog"] = df["SOG"].fillna(0.0)

    # Turn rate: ΔCOG / Δt  (degrees per second, wrapped)
    dcog = df.groupby("MMSI")["COG"].diff()
    dcog = ((dcog + 180) % 360) - 180          # wrap to [-180, 180]
    df["turn_rate"] = (dcog / dt_sec.replace(0, np.nan)).fillna(0.0)

    # Acceleration: ΔSOG / Δt  (knots per second)
    dsog = df.groupby("MMSI")["SOG"].diff()
    df["acceleration"] = (dsog / dt_sec.replace(0, np.nan)).fillna(0.0)

    # Speed regime binary flags
    df["is_stopped"]       = (df["sog"] <= STOPPED_KN).astype(np.float32)
    df["is_fishing_speed"] = (
        (df["sog"] > STOPPED_KN) & (df["sog"] <= FISHING_MAX_KN)
    ).astype(np.float32)

    # Heading: encode COG as (sin, cos) to avoid 0/360 wrap discontinuity
    cog_rad      = np.deg2rad(df["COG"].fillna(0.0))
    df["cog_sin"] = np.sin(cog_rad).astype(np.float32)
    df["cog_cos"] = np.cos(cog_rad).astype(np.float32)

    # Distance to previous ping (metres, approximate great-circle)
    lat1 = np.deg2rad(df.groupby("MMSI")["LAT"].shift(1).fillna(df["LAT"]))
    lat2 = np.deg2rad(df["LAT"])
    lon1 = np.deg2rad(df.groupby("MMSI")["LON"].shift(1).fillna(df["LON"]))
    lon2 = np.deg2rad(df["LON"])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    df["dist_m"] = (6_371_000 * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))).fillna(0.0)

    # Time-of-day (circular encoding)
    hour = df["BaseDateTime"].dt.hour + df["BaseDateTime"].dt.minute / 60.0
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24).astype(np.float32)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24).astype(np.float32)

    return df



# ---------------------------------------------------------------------------
# 3.  Feature selection  (point-level MI → controls model input)
# ---------------------------------------------------------------------------

def select_features(
    df: pd.DataFrame,
    vessel_labels: pd.Series,
    top_k: int = 8,
    verbose: bool = True,
) -> list:
    """
    Rank candidate point-level features by mutual information with the vessel
    weak label and return the top-k feature names.

    The label is broadcast from vessel level to every ping (all pings of a
    fishing vessel get label=1).  This ensures feature selection operates at
    the same granularity as model training.

    Parameters
    ----------
    df            : AIS dataframe after compute_point_features()
    vessel_labels : pd.Series  index=MMSI, values={0,1}
    top_k         : number of features to keep
    verbose       : print MI scores

    Returns
    -------
    List of selected feature column names (length = top_k).
    These names are the only columns passed to build_sequences() and
    therefore the only features the models ever see.
    """
    # Broadcast vessel label to every ping
    ping_labels = df["MMSI"].map(vessel_labels)
    valid_mask  = ping_labels.notna()

    X = df.loc[valid_mask, CANDIDATE_FEATURES].fillna(0).astype(np.float32)
    y = ping_labels[valid_mask].astype(int)

    mi_scores = mutual_info_classif(X, y, random_state=42, n_neighbors=5)
    ranked    = sorted(zip(CANDIDATE_FEATURES, mi_scores), key=lambda x: x[1], reverse=True)

    if verbose:
        print("\n[Feature Selection] Point-level Mutual Information with weak label")
        print(f"  (computed on {valid_mask.sum():,} pings from {df['MMSI'].nunique()} vessels)")
        print(f"  {'Feature':<22s}  {'MI score':>10s}  {'Selected':>8s}")
        print("  " + "-" * 46)
        for i, (name, score) in enumerate(ranked):
            tag = "  ✓" if i < top_k else ""
            print(f"  {name:<22s}  {score:>10.5f}  {tag}")

    selected = [name for name, _ in ranked[:top_k]]
    dropped  = [name for name, _ in ranked[top_k:]]

    if verbose:
        print(f"\n  → Keeping top {top_k}: {selected}")
        if dropped:
            print(f"  → Dropped {len(dropped)}: {dropped}")
        print()

    return selected


# ---------------------------------------------------------------------------
# 4.  Sequence windowing  (uses ONLY the selected features)
# ---------------------------------------------------------------------------

SEQ_LEN = 60   # default fixed-length window


def build_sequences(
    df: pd.DataFrame,
    feature_cols: list,
    seq_len: int = SEQ_LEN,
) -> tuple:
    """
    Slide a fixed-length window over each vessel's track.

    Only `feature_cols` (the output of select_features()) are included.
    This is the single point of control: the selected features here are
    exactly what every model receives — nothing more.

    Returns
    -------
    sequences : np.ndarray  (N, seq_len, len(feature_cols))  float32
    mmsi_ids  : np.ndarray  (N,)   MMSI for each window
    """
    all_seqs, all_mmsi = [], []

    MIN_PINGS = 30 # require at least half a sequence window worth of real data

    ping_counts = df.groupby("MMSI").size()
    valid_mmsi   = ping_counts[ping_counts >= MIN_PINGS].index
    df = df[df["MMSI"].isin(valid_mmsi)]

    print(f"Vessels after min-ping filter: {df['MMSI'].nunique()} "
      f"(dropped {(ping_counts < MIN_PINGS).sum()} with < {MIN_PINGS} pings)")

    for mmsi, group in df.groupby("MMSI"):
        group = group.sort_values("BaseDateTime")
        vals  = group[feature_cols].fillna(0).values.astype(np.float32)
        n     = len(vals)

        if n < seq_len:
            # Pre-pad short tracks with zeros
            pad  = np.zeros((seq_len - n, len(feature_cols)), dtype=np.float32)
            vals = np.vstack([pad, vals])
            all_seqs.append(vals[np.newaxis])
            all_mmsi.append(mmsi)
        else:
            # 50% overlap stride
            stride = max(1, seq_len // 2)
            for start in range(0, n - seq_len + 1, stride):
                all_seqs.append(vals[start : start + seq_len][np.newaxis])
                all_mmsi.append(mmsi)

    sequences = np.concatenate(all_seqs, axis=0)
    mmsi_ids  = np.array(all_mmsi)
    return sequences, mmsi_ids


# ---------------------------------------------------------------------------
# 5.  Normalisation
# ---------------------------------------------------------------------------

def fit_scaler(sequences: np.ndarray):
    """Fit StandardScaler on training sequences. Returns (scaler, scaled_seqs)."""
    N, T, F = sequences.shape
    flat    = sequences.reshape(-1, F)
    scaler  = StandardScaler()
    scaled  = scaler.fit_transform(flat).reshape(N, T, F).astype(np.float32)
    return scaler, scaled


def apply_scaler(scaler: StandardScaler, sequences: np.ndarray) -> np.ndarray:
    N, T, F = sequences.shape
    flat    = sequences.reshape(-1, F)
    return scaler.transform(flat).reshape(N, T, F).astype(np.float32)
