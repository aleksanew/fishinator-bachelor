"""
features_seq.py

"""

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SOG_UNAVAILABLE  = 102.3
COG_UNAVAILABLE  = 360.0
STOPPED_KN       = 0.5
FISHING_MAX_KN   = 6.0
SEQ_LEN          = 60

CANDIDATE_FEATURES = [
    "sog",
    "turn_rate",
    "acceleration",
    "is_stopped",
    "is_fishing_speed",
    "cog_sin",
    "cog_cos",
    "dist_m",
    "hour_sin",
    "hour_cos",
]


# ---------------------------------------------------------------------------
# 1. Point-level feature computation
# ---------------------------------------------------------------------------

def compute_point_features(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy().sort_values(["mmsi", "timestamp"]).reset_index(drop=True)

    # Sentinel replacement
    df["speed"]  = df["speed"].replace(SOG_UNAVAILABLE, np.nan)
    df["course"] = df["course"].replace(COG_UNAVAILABLE, np.nan)

    dt_sec = (
        df.groupby("mmsi")["timestamp"]
          .diff()
          .dt.total_seconds()
          .fillna(0)
          .clip(lower=0)
    )

    df["sog"] = df["speed"].fillna(0.0)

    # Turn rate (deg/s), wrapped
    dcog = df.groupby("mmsi")["course"].diff()
    dcog = ((dcog + 180) % 360) - 180
    df["turn_rate"] = (dcog / dt_sec.replace(0, np.nan)).fillna(0.0)

    # Acceleration (kn/s)
    dsog = df.groupby("mmsi")["speed"].diff()
    df["acceleration"] = (dsog / dt_sec.replace(0, np.nan)).fillna(0.0)

    # Speed flags
    df["is_stopped"]       = (df["sog"] <= STOPPED_KN).astype(np.float32)
    df["is_fishing_speed"] = (
        (df["sog"] >= STOPPED_KN) & (df["sog"] <= FISHING_MAX_KN)
    ).astype(np.float32)

    # Heading (sin/cos encoding)
    cog_rad       = np.deg2rad(df["course"].fillna(0.0))
    df["cog_sin"] = np.sin(cog_rad).astype(np.float32)
    df["cog_cos"] = np.cos(cog_rad).astype(np.float32)

    # Distance to previous ping (metres, haversine)
    lat1 = np.deg2rad(df.groupby("mmsi")["lat"].shift(1).fillna(df["lat"]))
    lat2 = np.deg2rad(df["lat"])
    lon1 = np.deg2rad(df.groupby("mmsi")["lon"].shift(1).fillna(df["lon"]))
    lon2 = np.deg2rad(df["lon"])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = (np.sin(dlat / 2)**2
         + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2)
    df["dist_m"] = (6_371_000 * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))).fillna(0.0)

    # Time-of-day (circular)
    hour = df["timestamp"].dt.hour + df["timestamp"].dt.minute / 60.0
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24).astype(np.float32)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24).astype(np.float32)

    return df


# ---------------------------------------------------------------------------
# 2. Feature selection via mutual information
# ---------------------------------------------------------------------------

def select_features(
    df: pd.DataFrame,
    vessel_labels: pd.Series,
    top_k: int = 8,
    verbose: bool = True,
) -> list:

    ping_labels = df["mmsi"].map(vessel_labels)
    valid       = ping_labels.notna()

    X = df.loc[valid, CANDIDATE_FEATURES].fillna(0).astype(np.float32)
    y = ping_labels[valid].astype(int)

    mi = mutual_info_classif(X, y, random_state=42, n_neighbors=5)
    ranked = sorted(zip(CANDIDATE_FEATURES, mi), key=lambda x: x[1], reverse=True)

    if verbose:
        print("\n[Feature Selection] Mutual Information scores:")
        for i, (name, score) in enumerate(ranked):
            tag = " ✓" if i < top_k else ""
            print(f"  {name:<22s}  {score:.5f}{tag}")

    return [name for name, _ in ranked[:top_k]]


# ---------------------------------------------------------------------------
# 3. Sequence windowing
# ---------------------------------------------------------------------------

def build_sequences(
    df: pd.DataFrame,
    feature_cols: list,
    seq_len: int = SEQ_LEN,
    min_pings: int = 30,
) -> tuple:

    ping_counts = df.groupby("mmsi").size()
    valid_mmsi  = ping_counts[ping_counts >= min_pings].index
    df = df[df["mmsi"].isin(valid_mmsi)].copy()

    all_seqs, all_mmsi = [], []
    stride = max(1, seq_len // 2)   # 50% overlap

    for mmsi, grp in df.groupby("mmsi"):
        vals = (
            grp.sort_values("timestamp")[feature_cols]
               .fillna(0)
               .values
               .astype(np.float32)
        )
        n = len(vals)

        if n < seq_len:
            pad  = np.zeros((seq_len - n, len(feature_cols)), dtype=np.float32)
            vals = np.vstack([pad, vals])
            all_seqs.append(vals[np.newaxis])
            all_mmsi.append(mmsi)
        else:
            for start in range(0, n - seq_len + 1, stride):
                all_seqs.append(vals[start:start + seq_len][np.newaxis])
                all_mmsi.append(mmsi)

    sequences = np.concatenate(all_seqs, axis=0)
    mmsi_ids  = np.array(all_mmsi)
    return sequences, mmsi_ids


# ---------------------------------------------------------------------------
# 4. Normalisation
# ---------------------------------------------------------------------------

def fit_scaler(sequences: np.ndarray):
    N, T, F = sequences.shape
    flat    = sequences.reshape(-1, F)
    scaler  = StandardScaler()
    scaled  = scaler.fit_transform(flat).reshape(N, T, F).astype(np.float32)
    return scaler, scaled


def apply_scaler(scaler: StandardScaler, sequences: np.ndarray) -> np.ndarray:
    N, T, F = sequences.shape
    return scaler.transform(sequences.reshape(-1, F)).reshape(N, T, F).astype(np.float32)
