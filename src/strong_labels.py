import pandas as pd

def label_ais_points(ais_df, bw_df):
    ais_df["fishing"] = 0

    for _, row in bw_df.iterrows():
        mask = (
            (ais_df["mmsi"] == row["mmsi"]) &
            (ais_df["timestamp"] >= row["setupDateTime"]) &
            (
                (ais_df["timestamp"] <= row["removedDateTime"]) |
                pd.isna(row["removedDateTime"])
            )
        )
        ais_df.loc[mask, "fishing"] = 1

    return ais_df

def aggregate_segments(ais_df):
    grouped = ais_df.groupby(["mmsi", "segment_id"])

    agg = grouped["fishing"].mean().reset_index()
    agg = agg.rename(columns={"fishing": "fishing_ratio"})

    agg["strong_label"] = -1
    agg.loc[agg["fishing_ratio"] == 0, "strong_label"] = 0
    agg.loc[agg["fishing_ratio"] >= 0.5, "strong_label"] = 1

    agg = agg[agg["strong_label"].isin([0, 1])]

    return agg