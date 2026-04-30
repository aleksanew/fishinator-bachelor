"""
Weak label assignment using Global Fishing Watch (GFW) data.

Strategy:
  - For each vessel, bin its positions into 0.1° lat/lon grid cells (matching GFW resolution)
  - If any of the vessel's grid cells on that date have fishing_hours > 0, label = 1
  - Vessels with no GFW record at all are labelled 0 (non-fishing)
"""

import numpy as np
import pandas as pd


def assign_weak_labels(ais: pd.DataFrame, gfw: pd.DataFrame) -> pd.Series:
    """
    Parameters
    ----------
    ais : pd.DataFrame
        AIS data with columns [MMSI, BaseDateTime (datetime), LAT, LON, ...]
    gfw : pd.DataFrame
        GFW data with columns [mmsi, date (date), cell_ll_lat, cell_ll_lon, fishing_hours]

    Returns
    -------
    labels : pd.Series  index = MMSI,  values = {0, 1}
    """
    # ---- Prepare GFW fishing lookup ----------------------------------------
    gfw = gfw.copy()
    gfw["date"] = pd.to_datetime(gfw["date"]).dt.date
    gfw_fishing = gfw[gfw["fishing_hours"] > 0].copy()

    # Unique (mmsi, date, lat_bin, lon_bin) tuples with fishing
    gfw_fishing["lat_bin"] = np.floor(gfw_fishing["cell_ll_lat"] * 10) / 10
    gfw_fishing["lon_bin"] = np.floor(gfw_fishing["cell_ll_lon"] * 10) / 10
    gfw_set = set(
        zip(
            gfw_fishing["mmsi"].astype(int),
            gfw_fishing["date"],
            gfw_fishing["lat_bin"],
            gfw_fishing["lon_bin"],
        )
    )

    # GFW vessels seen at all (regardless of fishing_hours)
    gfw_known_mmsi = set(gfw["mmsi"].astype(int).unique())

    # ---- Match AIS → GFW ---------------------------------------------------
    ais = ais.copy()
    ais["date"] = ais["BaseDateTime"].dt.date
    ais["lat_bin"] = np.floor(ais["LAT"] * 10) / 10
    ais["lon_bin"] = np.floor(ais["LON"] * 10) / 10

    def vessel_label(group):
        mmsi = int(group.name)
        if mmsi not in gfw_known_mmsi:
            return 0   # not in GFW → non-fishing
        for _, row in group.iterrows():
            key = (mmsi, row["date"], row["lat_bin"], row["lon_bin"])
            if key in gfw_set:
                return 1
        return 0   # in GFW but no fishing cell match → non-fishing

    labels = (
        ais.groupby("MMSI")
           .apply(vessel_label)
           .rename("weak_label")
    )
    return labels


def label_summary(labels: pd.Series) -> None:
    vc = labels.value_counts()
    total = len(labels)
    print("\n[Weak Labels]")
    print(f"  Fishing     (1): {vc.get(1, 0):>6d}  ({100*vc.get(1,0)/total:.1f}%)")
    print(f"  Non-fishing (0): {vc.get(0, 0):>6d}  ({100*vc.get(0,0)/total:.1f}%)")
    print(f"  Total           : {total}")
