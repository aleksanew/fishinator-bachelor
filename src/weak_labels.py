import pandas as pd

import numpy as np

def assign_weak_labels(segments_df, gfw_df):
    segments_df["lat_bin"] = np.floor(segments_df["lat"] * 10) / 10
    segments_df["lon_bin"] = np.floor(segments_df["lon"] * 10) / 10

    merged = segments_df.merge(
        gfw_df[["lat_bin", "lon_bin", "fishing_hours"]],
        on=["lat_bin", "lon_bin"],
        how="left"
    )

    merged["fishing_hours"] = merged["fishing_hours"].fillna(0)
    merged["weak_label"] = (merged["fishing_hours"] > 0).astype(int)

    return merged