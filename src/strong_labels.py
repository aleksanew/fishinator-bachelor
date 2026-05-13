"""
strong_labels.py
Label AIS points and segments using BarentsWatch gear deployment data.

Labeling logic — MMSI match + temporal match are always required.
Spatial match is applied only when gear coordinates are available,
using a generous radius since vessels often move away from gear while it soaks.

Two labeling strategies are supported:
  - "spatial": vessel must be within MAX_DIST_KM of gear (strict)
  - "temporal": any point within the deployment window counts (lenient)
The default is "temporal" because with only 100 overlapping vessels,
the spatial filter reduces fishing segments to ~133 which is too few.
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MAX_DIST_KM       = 50.0   # increased from 10km — vessels move away from gear
MIN_FISHING_RATIO = 0.3    # lowered from 0.5 — easier to qualify as fishing segment
STRATEGY          = "temporal"  # "temporal" or "spatial"


# ---------------------------------------------------------------------------
# Haversine distance (vectorised)
# ---------------------------------------------------------------------------

def _haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


# ---------------------------------------------------------------------------
# Point-level labeling
# ---------------------------------------------------------------------------

def label_ais_points(
    ais_df: pd.DataFrame,
    bw_df: pd.DataFrame,
    max_dist_km: float = MAX_DIST_KM,
    strategy: str = STRATEGY,
) -> pd.DataFrame:
    """
    Label each AIS point as fishing (1) or not (0).

    strategy="temporal": label all points within the gear deployment window.
        Rationale: if a vessel registered gear, it was fishing during that period.
        Simple and maximises the number of labeled positive points.

    strategy="spatial": additionally require the vessel to be within
        max_dist_km of the gear coordinates.
        More precise but misses vessels that moved away while gear soaks.
    """
    ais_df = ais_df.copy()
    ais_df["fishing"] = 0

    bw_mmsis = set(bw_df["mmsi"].unique())
    overlap  = set(ais_df["mmsi"].unique()) & bw_mmsis
    print(f"  Vessels in AIS:  {ais_df['mmsi'].nunique()}")
    print(f"  Vessels in BW:   {len(bw_mmsis)}")
    print(f"  Overlap:         {len(overlap)}")
    print(f"  Labeling strategy: {strategy}  max_dist_km={max_dist_km}")

    for mmsi in overlap:
        ais_vessel = ais_df[ais_df["mmsi"] == mmsi].index
        bw_vessel  = bw_df[bw_df["mmsi"] == mmsi]

        for _, gear in bw_vessel.iterrows():
            # Temporal filter — always applied
            if pd.isna(gear["removed_dt"]):
                time_mask = ais_df.loc[ais_vessel, "timestamp"] >= gear["setup_dt"]
            else:
                time_mask = (
                    (ais_df.loc[ais_vessel, "timestamp"] >= gear["setup_dt"]) &
                    (ais_df.loc[ais_vessel, "timestamp"] <= gear["removed_dt"])
                )

            candidate_idx = ais_vessel[time_mask]
            if len(candidate_idx) == 0:
                continue

            if strategy == "temporal":
                # Time match is sufficient — vessel registered gear in this window
                ais_df.loc[candidate_idx, "fishing"] = 1

            elif strategy == "spatial":
                if pd.isna(gear["gear_lat"]) or pd.isna(gear["gear_lon"]):
                    # No coordinates — fall back to temporal
                    ais_df.loc[candidate_idx, "fishing"] = 1
                    continue
                dist = _haversine_km(
                    ais_df.loc[candidate_idx, "lat"].values,
                    ais_df.loc[candidate_idx, "lon"].values,
                    gear["gear_lat"],
                    gear["gear_lon"],
                )
                ais_df.loc[candidate_idx[dist <= max_dist_km], "fishing"] = 1

    n_fishing = ais_df["fishing"].sum()
    n_total   = len(ais_df)
    print(f"  Fishing points:  {n_fishing:,} / {n_total:,} ({100*n_fishing/n_total:.2f}%)")
    return ais_df


# ---------------------------------------------------------------------------
# Segment-level aggregation
# ---------------------------------------------------------------------------

def aggregate_segments(
    ais_df: pd.DataFrame,
    min_fishing_ratio: float = MIN_FISHING_RATIO,
) -> pd.DataFrame:
    """
    Aggregate point-level fishing labels to segment level.

    Labels:
      1 (fishing)     — >= min_fishing_ratio of points are labeled fishing
      0 (non-fishing) — 0 points labeled fishing
      -1 (ambiguous)  — between 0 and min_fishing_ratio, excluded
    """
    agg = (
        ais_df.groupby(["mmsi", "segment_id"])["fishing"]
              .mean()
              .reset_index()
              .rename(columns={"fishing": "fishing_ratio"})
    )

    agg["strong_label"] = -1
    agg.loc[agg["fishing_ratio"] == 0.0,               "strong_label"] = 0
    agg.loc[agg["fishing_ratio"] >= min_fishing_ratio, "strong_label"] = 1

    n_ambiguous = (agg["strong_label"] == -1).sum()
    agg = agg[agg["strong_label"].isin([0, 1])].copy()

    print(f"  Segments total:     {len(agg) + n_ambiguous:,}")
    print(f"  Segments ambiguous: {n_ambiguous:,} (excluded)")
    print(f"  Segments kept:      {len(agg):,}")
    print(f"  Fishing segments:   {(agg['strong_label']==1).sum():,}")
    print(f"  Non-fishing segs:   {(agg['strong_label']==0).sum():,}")

    return agg


# ---------------------------------------------------------------------------
# Vessel-level strong labels (for NN sequence models)
# ---------------------------------------------------------------------------

def assign_strong_labels_vessel_level(
    ais_df: pd.DataFrame,
    bw_df: pd.DataFrame,
    max_dist_km: float = MAX_DIST_KM,
    strategy: str = STRATEGY,
) -> pd.Series:
    """
    Assign strong labels at vessel level for NN sequence models.
    A vessel is labeled 1 if ANY of its AIS points pass the labeling filter.
    """
    labeled = label_ais_points(ais_df, bw_df,
                               max_dist_km=max_dist_km, strategy=strategy)
    return (
        labeled.groupby("mmsi")["fishing"]
               .max()
               .rename("strong_label")
    )