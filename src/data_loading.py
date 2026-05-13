"""
data_loading.py

"""

import re
import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# AIS 2024  (MarineCadastre – US waters, weak-label training set)
# ---------------------------------------------------------------------------

def load_ais2024(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.lower()
    df = df.rename(columns={
        "basedatetime": "timestamp",
        "sog": "speed",
        "cog": "course",
    })
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["mmsi"] = pd.to_numeric(df["mmsi"], errors="coerce")
    df = df.dropna(subset=["mmsi", "timestamp", "lat", "lon"])
    df["mmsi"] = df["mmsi"].astype(int)
    df["speed"]  = pd.to_numeric(df["speed"],  errors="coerce").fillna(0.0)
    df["course"] = pd.to_numeric(df["course"], errors="coerce").fillna(0.0)
    return df[["mmsi", "timestamp", "lat", "lon", "speed", "course"]]


# ---------------------------------------------------------------------------
# AIS historic (Kystdatahuset – Norwegian waters, strong-label set)
# ---------------------------------------------------------------------------

def load_ais_historic(path: str) -> pd.DataFrame:
    """
    Loads bw_ais_historic.csv produced by the Kystdatahuset fetch script.
    Columns: MMSI, BaseDateTime, LAT, LON, SOG, COG
    """
    df = pd.read_csv(path)
    df.columns = df.columns.str.lower()
    df = df.rename(columns={
        "basedatetime": "timestamp",
        "sog": "speed",
        "cog": "course",
    })
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["mmsi"] = pd.to_numeric(df["mmsi"], errors="coerce")
    df = df.dropna(subset=["mmsi", "timestamp", "lat", "lon"])
    df["mmsi"] = df["mmsi"].astype(int)
    df["speed"]  = pd.to_numeric(df["speed"],  errors="coerce").fillna(0.0)
    df["course"] = pd.to_numeric(df["course"], errors="coerce").fillna(0.0)
    return df[["mmsi", "timestamp", "lat", "lon", "speed", "course"]]


# ---------------------------------------------------------------------------
# GFW  (Global Fishing Watch – weak label source)
# ---------------------------------------------------------------------------

def load_gfw(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.lower()
    df = df.rename(columns={
        "cell_ll_lat": "lat_bin",
        "cell_ll_lon": "lon_bin",
    })
    required = ["lat_bin", "lon_bin", "fishing_hours"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"GFW missing columns: {missing}")
    return df


# ---------------------------------------------------------------------------
# BarentsWatch  (gear deployment registry – strong label source)
# ---------------------------------------------------------------------------

def _parse_geo(s: str):
    """Parse '70° 8.814 N 30° 36.625 E' → (lat, lon) in decimal degrees."""
    if not isinstance(s, str) or not s.strip():
        return np.nan, np.nan
    m = re.match(
        r"(\d+)°\s*([\d.]+)\s*([NS])\s+(\d+)°\s*([\d.]+)\s*([EW])",
        s.strip()
    )
    if not m:
        return np.nan, np.nan
    lat = int(m.group(1)) + float(m.group(2)) / 60.0
    if m.group(3) == "S":
        lat = -lat
    lon = int(m.group(4)) + float(m.group(5)) / 60.0
    if m.group(6) == "W":
        lon = -lon
    return lat, lon


def load_barentswatch(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";")
    df.columns = df.columns.str.strip().str.lower()

    # MMSI
    df["mmsi"] = pd.to_numeric(df["mmsi"], errors="coerce")
    df = df.dropna(subset=["mmsi"])
    df["mmsi"] = df["mmsi"].astype(int)

    # Timestamps
    df["setup_dt"] = pd.to_datetime(
        df["setupdatetime"], errors="coerce", utc=True
    ).dt.tz_convert(None)
    df["removed_dt"] = pd.to_datetime(
        df["removeddatetime"], errors="coerce", utc=True
    ).dt.tz_convert(None)

    # Gear position (parse from DMS string)
    geo = df["geometry"].apply(_parse_geo)
    df["gear_lat"] = [g[0] for g in geo]
    df["gear_lon"] = [g[1] for g in geo]

    # Gear type
    df["gear_type"] = df["tooltypecode"].str.upper().fillna("UNKNOWN")

    result = df[[
        "mmsi", "setup_dt", "removed_dt",
        "gear_lat", "gear_lon", "gear_type"
    ]].copy()

    result = result.dropna(subset=["setup_dt"])
    print(f"  BW records loaded: {len(result)}")
    print(f"  BW unique MMSIs:   {result['mmsi'].nunique()}")
    print(f"  Gear types:        {result['gear_type'].value_counts().to_dict()}")
    return result
