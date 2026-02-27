import pandas as pd
import numpy as np

def compute_turning_rate(cog_series):
    cog_series = cog_series.dropna().values
    if len(cog_series) < 2:
        return np.nan
    diff = np.diff(cog_series)
    diff = (diff + 180) % 360 - 180
    return np.mean(np.abs(diff))


def build_segment_features(ais):
    segments = ais.groupby(["MMSI", "segment_id"]).agg({
        "SOG": ["mean", "std"],
        "COG": compute_turning_rate,
        "LAT": "mean",
        "LON": "mean",
        "date": "first"
    }).reset_index()

    segments.columns = [
        "MMSI", "segment_id",
        "mean_speed",
        "speed_std",
        "turning_rate",
        "mean_lat",
        "mean_lon",
        "date"
    ]

    # Drop invalid segments (single AIS point etc.)
    segments = segments.dropna(
        subset=["mean_speed", "speed_std", "turning_rate"]
    )

    return segments