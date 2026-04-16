import pandas as pd

def segment_trajectories(df, gap_minutes=60):
    df = df.sort_values(["mmsi", "timestamp"])

    df["time_diff"] = df.groupby("mmsi")["timestamp"].diff().dt.total_seconds() / 60
    df["new_segment"] = (df["time_diff"] > gap_minutes) | (df["time_diff"].isna())
    df["segment_id"] = df.groupby("mmsi")["new_segment"].cumsum()
    return df

