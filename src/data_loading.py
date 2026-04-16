import pandas as pd
import json

def load_ais2024(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.lower()

    rename_map = {
        "basedatetime": "timestamp",
        "lat": "lat",
        "lon": "lon",
        "sog": "speed",
        "cog": "course"
    }

    df = df.rename(columns=rename_map)

    required = ["mmsi", "timestamp", "lat", "lon", "speed", "course"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["mmsi"] = df["mmsi"].astype(int)

    return df

def load_gfw(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.lower()

    required = ["cell_ll_lat", "cell_ll_lon", "fishing_hours"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"GFW missing columns: {missing}")

    df = df.rename(columns={
        "cell_ll_lat": "lat_bin",
        "cell_ll_lon": "lon_bin"
    })

    return df

def load_barentswatch(path):
    df = pd.read_csv(path, sep=";")
    df.columns = df.columns.str.strip().str.lower()

    required = ["mmsi", "setupdatetime", "removeddatetime"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df["mmsi"] = pd.to_numeric(df["mmsi"], errors="coerce")
    df = df.dropna(subset=["mmsi"])
    df["mmsi"] = df["mmsi"].astype(int)

    df["setupdatetime"] = pd.to_datetime(df["setupdatetime"], utc=True).dt.tz_convert(None)
    df["removeddatetime"] = pd.to_datetime(df["removeddatetime"], errors="coerce", utc=True).dt.tz_convert(None)

    df = df.rename(columns={
        "setupdatetime": "setupDateTime",
        "removeddatetime": "removedDateTime"
    })

    return df

def load_ais2026(path):
    with open(path, "r") as f:
        data = json.load(f)

    # extract features
    features = data["features"]

    records = []
    for f in features:
        props = f["properties"]
        coords = f["geometry"]["coordinates"]

        record = {
            "mmsi": props["mmsi"],
            "timestamp": props["msgtime"],
            "speed": props.get("speedOverGround", 0),
            "course": props.get("courseOverGround", 0),
            "lat": coords[1],
            "lon": coords[0],
        }

        records.append(record)

    df = pd.DataFrame(records)

    # normalize types
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="ISO8601").dt.tz_localize(None)
    df["mmsi"] = pd.to_numeric(df["mmsi"], errors="coerce")
    df = df.dropna(subset=["mmsi"])
    df["mmsi"] = df["mmsi"].astype(int)

    return df