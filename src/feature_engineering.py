import pandas as pd
import numpy as np

def compute_features(df):
    grouped = df.groupby(["mmsi", "segment_id"])

    features = grouped.agg({
        "speed": ["mean", "std"],
        "course": lambda x: (abs(x.diff())).mean(),
        "lat": "mean",
        "lon": "mean"
    })

    features.columns = ["mean_speed", "speed_std", "turning_rate", "lat", "lon"]
    features = features.reset_index()

    features["speed_std"] = features["speed_std"].fillna(0)
    features["turning_rate"] = features["turning_rate"].fillna(0)

    return features