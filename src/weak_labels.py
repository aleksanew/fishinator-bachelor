"""
weak_labels.py
Assign fishing labels using Global Fishing Watch grid data.
Works at segment level (for classical ML) and vessel level (for NNs).
"""

import numpy as np
import pandas as pd


def assign_weak_labels(segments_df: pd.DataFrame, gfw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign weak labels to trajectory segments using GFW 0.1° grid cells.

    A segment is labeled 1 (fishing) if its mean position falls in a GFW
    grid cell with fishing_hours > 0.

    Parameters
    ----------
    segments_df : output of compute_features() — must have lat, lon columns
    gfw_df      : output of load_gfw()

    Returns
    -------
    segments_df with added column 'weak_label' (0 or 1)
    """
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
    """
    Assign weak labels at vessel level for NN sequence models.
    A vessel is labeled 1 if ANY of its pings fall in a GFW fishing cell.

    Returns pd.Series indexed by mmsi with values {0, 1}.
    """
    ais = ais_df.copy()
    ais["lat_bin"] = np.floor(ais["lat"] * 10) / 10
    ais["lon_bin"] = np.floor(ais["lon"] * 10) / 10

    gfw_positive = set(
        zip(
            np.floor(gfw_df.loc[gfw_df["fishing_hours"] > 0, "lat_bin"] * 10) / 10,
            np.floor(gfw_df.loc[gfw_df["fishing_hours"] > 0, "lon_bin"] * 10) / 10,
        )
    )

    def vessel_label(grp):
        for _, row in grp.iterrows():
            if (row["lat_bin"], row["lon_bin"]) in gfw_positive:
                return 1
        return 0

    labels = ais.groupby("mmsi").apply(vessel_label).rename("weak_label")
    return labels