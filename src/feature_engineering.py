import pandas as pd
import numpy as np

SEGMENT_FEATURES = [
    "mean_speed",
    "speed_std",
    "min_speed",
    "max_speed",
    "low_speed_fraction",
    "turning_rate",
    "cog_std",
    "sinuosity",
]

def _sinuosity(lats, lons):
    if len(lats) < 2:
        return 1.0
    lat_r = np.radians(lats)
    lon_r = np.radians(lons)
    dlat = np.diff(lat_r)
    dlon = np.diff(lon_r)
    avg_lat = (lat_r[:-1] + lat_r[1:]) / 2
    step_dist = np.sqrt(dlat**2 + (dlon * np.cos(avg_lat))**2)
    total_path = step_dist.sum()
    dlat_total = lat_r[-1] - lat_r[0]
    dlon_total = lon_r[-1] - lon_r[0]
    avg_lat_total = (lat_r[0] + lat_r[-1]) / 2
    displacement = np.sqrt(dlat_total**2 + (dlon_total * np.cos(avg_lat_total))**2)
    if displacement < 1e-10:
        return float(total_path / 1e-10) if total_path > 0 else 1.0
    return float(total_path / displacement)

def _low_speed_fraction(speeds, threshold=3.0):
    if len(speeds) == 0:
        return 0.0
    return float((speeds < threshold).sum() / len(speeds))

def _cog_std(courses):
    if len(courses) < 2:
        return 0.0
    rad = np.radians(courses)
    sin_mean = np.sin(rad).mean()
    cos_mean = np.cos(rad).mean()
    R = np.sqrt(sin_mean**2 + cos_mean**2)
    R = np.clip(R, 0.0, 1.0)
    return float(np.degrees(np.sqrt(-2 * np.log(R))))

def compute_features(df):
    records = []
    for (mmsi, segment_id), grp in df.groupby(["mmsi", "segment_id"]):
        speeds  = grp["speed"].values
        courses = grp["course"].values
        lats    = grp["lat"].values
        lons    = grp["lon"].values
        records.append({
            "mmsi":               mmsi,
            "segment_id":         segment_id,
            "mean_speed":         float(np.mean(speeds)),
            "speed_std":          float(np.std(speeds)) if len(speeds) > 1 else 0.0,
            "min_speed":          float(np.min(speeds)),
            "max_speed":          float(np.max(speeds)),
            "low_speed_fraction": _low_speed_fraction(speeds),
            "turning_rate":       float(np.mean(np.abs(((np.diff(courses) + 180) % 360) - 180))) if len(courses) > 1 else 0.0,
            "cog_std":            _cog_std(courses),
            "sinuosity":          _sinuosity(lats, lons),
            "lat":                float(np.mean(lats)),
            "lon":                float(np.mean(lons)),
            "seg_date":           grp["timestamp"].dt.date.iloc[0],
        })
    features = pd.DataFrame(records)
    features["sinuosity"] = features["sinuosity"].clip(upper=50.0)
    return features