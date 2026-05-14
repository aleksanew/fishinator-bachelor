"""
weak_labels.py
"""

import numpy as np
import pandas as pd


def assign_weak_labels(segments_df: pd.DataFrame, gfw_df: pd.DataFrame) -> pd.DataFrame:
    segments_df = segments_df.copy()
    segments_df["lat_bin"] = np.floor(segments_df["lat"] * 10) / 10
    segments_df["lon_bin"] = np.floor(segments_df["lon"] * 10) / 10

    gfw_positive = gfw_df[gfw_df["fishing_hours"] > 0][
        ["lat_bin", "lon_bin"]
    ].drop_duplicates()
    gfw_positive["weak_label"] = 1

    merged = segments_df.merge(gfw_positive, on=["lat_bin", "lon_bin"], how="left")
    merged["weak_label"] = merged["weak_label"].fillna(0).astype(int)

    return merged


def assign_weak_labels_vessel_level(
    ais_df: pd.DataFrame,
    gfw_df: pd.DataFrame,
) -> pd.Series:

    ais = ais_df.copy()
    ais["lat_bin"] = np.floor(ais["lat"] * 10) / 10
    ais["lon_bin"] = np.floor(ais["lon"] * 10) / 10

    gfw_positive = set(
        zip(
            gfw_df.loc[gfw_df["fishing_hours"] > 0, "lat_bin"],
            gfw_df.loc[gfw_df["fishing_hours"] > 0, "lon_bin"],
        )
    )

    def vessel_label(grp):
        for _, row in grp.iterrows():
            if (row["lat_bin"], row["lon_bin"]) in gfw_positive:
                return 1
        return 0

    labels = ais.groupby("mmsi").apply(vessel_label).rename("weak_label")
    return labels
