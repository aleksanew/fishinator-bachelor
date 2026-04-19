import pandas as pd
import numpy as np


def _sinuosity(lats, lons):
    """
    Ratio of total path length to straight-line displacement.
    Returns 1.0 for a perfectly straight path, higher for winding paths.
    Returns 1.0 if the segment has fewer than 2 points or zero displacement.
    """
    if len(lats) < 2:
        return 1.0

    # Approximate distances using equirectangular projection (fine for short segments)
    lat_r = np.radians(lats)
    lon_r = np.radians(lons)

    dlat = np.diff(lat_r)
    dlon = np.diff(lon_r)
    avg_lat = (lat_r[:-1] + lat_r[1:]) / 2

    # Step distances in degrees-equivalent
    step_dist = np.sqrt(dlat**2 + (dlon * np.cos(avg_lat))**2)
    total_path = step_dist.sum()

    # Straight-line displacement between first and last point
    dlat_total = lat_r[-1] - lat_r[0]
    dlon_total = lon_r[-1] - lon_r[0]
    avg_lat_total = (lat_r[0] + lat_r[-1]) / 2
    displacement = np.sqrt(dlat_total**2 + (dlon_total * np.cos(avg_lat_total))**2)

    if displacement < 1e-10:
        # Vessel barely moved — very winding by definition
        return float(total_path / 1e-10) if total_path > 0 else 1.0

    return float(total_path / displacement)


def _low_speed_fraction(speeds, threshold=3.0):
    """Fraction of AIS points within a segment with speed below threshold knots."""
    if len(speeds) == 0:
        return 0.0
    return float((speeds < threshold).sum() / len(speeds))


def _cog_std(courses):
    """
    Circular standard deviation of course-over-ground.
    Handles the 0/360 wrap-around correctly.
    """
    if len(courses) < 2:
        return 0.0
    rad = np.radians(courses)
    sin_mean = np.sin(rad).mean()
    cos_mean = np.cos(rad).mean()
    R = np.sqrt(sin_mean**2 + cos_mean**2)   # mean resultant length
    # R close to 1 = concentrated headings; R close to 0 = spread out
    R = np.clip(R, 0.0, 1.0)
    return float(np.degrees(np.sqrt(-2 * np.log(R))))


def compute_features(df):
    records = []

    for (mmsi, segment_id), grp in df.groupby(["mmsi", "segment_id"]):
        speeds  = grp["speed"].values
        courses = grp["course"].values
        lats    = grp["lat"].values
        lons    = grp["lon"].values

        # --- existing features ---
        mean_speed   = float(np.mean(speeds))
        speed_std    = float(np.std(speeds)) if len(speeds) > 1 else 0.0
        turning_rate = float(np.mean(np.abs(np.diff(courses)))) if len(courses) > 1 else 0.0

        # --- new speed features ---
        min_speed  = float(np.min(speeds))
        max_speed  = float(np.max(speeds))
        low_speed_fraction = _low_speed_fraction(speeds, threshold=3.0)

        # --- new heading feature ---
        cog_std = _cog_std(courses)

        # --- trajectory shape ---
        sinuosity = _sinuosity(lats, lons)

        # --- spatial (keep for potential merging downstream) ---
        lat_mean = float(np.mean(lats))
        lon_mean = float(np.mean(lons))

        records.append({
            "mmsi":               mmsi,
            "segment_id":         segment_id,
            "mean_speed":         mean_speed,
            "speed_std":          speed_std,
            "min_speed":          min_speed,
            "max_speed":          max_speed,
            "low_speed_fraction": low_speed_fraction,
            "turning_rate":       turning_rate,
            "cog_std":            cog_std,
            "sinuosity":          sinuosity,
            "lat":                lat_mean,
            "lon":                lon_mean,
        })

    features = pd.DataFrame(records)

    # Clip sinuosity to a reasonable max to avoid extreme outliers
    # (e.g. vessels that barely move get astronomically high values)
    features["sinuosity"] = features["sinuosity"].clip(upper=50.0)

    return features