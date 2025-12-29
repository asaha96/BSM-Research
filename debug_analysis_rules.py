import pandas as pd
import os
import multiprocessing
import numpy as np
from datetime import datetime, timezone

# --- CONFIG (Ensure these match your analyzer script) ---
OUTPUT_DIR = os.environ.get("BSM_OUTPUT_DIR", os.path.join(os.getcwd(), "data", "analysis"))
STAGING_FILE = os.path.join(OUTPUT_DIR, "all_vehicle_data_2024_09.parquet")
NUM_WORKERS = 8 # Use a safe number of workers

# North Ave decision box (full)
LAT_MIN, LAT_MAX = 33.7710, 33.7717
LON_MIN, LON_MAX = -84.3907, -84.3887

# Crossing rules
NA_SLICES = 3; MIN_SLICES_TOUCHED = 3
WE_MIN_DEG, WE_MAX_DEG = 45.0, 135.0
MIN_WE_FRACTION = 0.50
GAP_S = 10.0

# --- HELPER FUNCTIONS ---
def within_box(df, lat_min, lat_max, lon_min, lon_max):
    return (df["Latitude"].between(lat_min, lat_max)) & (df["Longitude"].between(lon_min, lon_max))

def calc_bearing(lat1, lon1, lat2, lon2):
    lat1 = np.asarray(lat1, dtype="float64"); lon1 = np.asarray(lon1, dtype="float64")
    lat2 = np.asarray(lat2, dtype="float64"); lon2 = np.asarray(lon2, dtype="float64")
    mask = ~(np.isnan(lat1) | np.isnan(lon1) | np.isnan(lat2) | np.isnan(lon2))
    out = np.full_like(lat1, np.nan, dtype="float64")
    if mask.any():
        a1 = np.radians(lat1[mask]); o1 = np.radians(lon1[mask])
        a2 = np.radians(lat2[mask]); o2 = np.radians(lon2[mask])
        dlon = o2 - o1
        x = np.sin(dlon) * np.cos(a2)
        y = np.cos(a1)*np.sin(a2) - np.sin(a1)*np.cos(a2)*np.cos(dlon)
        brng = np.degrees(np.arctan2(x, y))
        with np.errstate(invalid="ignore"):
            brng = (brng + 360.0) % 360.0
        out[mask] = brng
    return out

def split_na_longitudinal(df, n=NA_SLICES):
    lmin, lmax = sorted([LON_MIN, LON_MAX])
    edges = np.linspace(lmin, lmax, n + 1)
    masks = []; base_lat = df["Latitude"].between(LAT_MIN, LAT_MAX)
    for i in range(n):
        m = base_lat & df["Longitude"].between(edges[i], edges[i+1])
        masks.append(m)
    return edges, masks

def safe_to_datetime(series):
    fast_format = '%Y-%m-%d %H:%M:%S.%f'
    try: return pd.to_datetime(series, format=fast_format, utc=True, errors="raise")
    except (ValueError, TypeError): return pd.to_datetime(series, utc=True, errors="coerce")

def contiguous_segments(mask: np.ndarray):
    idx = np.where(mask)[0]
    if idx.size == 0: return []
    splits = np.where(np.diff(idx) != 1)[0] + 1
    parts = np.split(idx, splits)
    return [p for p in parts if len(p) >= 2]

# --- WORKER FUNCTION (MODIFIED FOR DEBUGGING) ---
def debug_vehicle_worker(vid):
    try:
        filters = [('VehicleID', '==', vid)]
        g = pd.read_parquet(STAGING_FILE, filters=filters)

        g["DateTime_UTC"] = safe_to_datetime(g["DateTime_UTC"])
        g.dropna(subset=["Latitude", "Longitude", "DateTime_UTC"], inplace=True)
        g.sort_values("DateTime_UTC", inplace=True)
        if g.empty:
            return None

        g["Run"] = (g["DateTime_UTC"].astype("int64") // 10**9).diff().gt(GAP_S).cumsum()
        g["Run"] = g["Run"].fillna(0)

        total_runs = g['Run'].nunique()
        if total_runs == 0: return None
        
        na_overall = within_box(g, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX)
        runs_in_box = g.loc[na_overall, 'Run'].unique()
        count_in_box = len(runs_in_box)

        _, slices = split_na_longitudinal(g, NA_SLICES)
        tmp2 = pd.DataFrame({"Run": g["Run"]})
        for j, m in enumerate(slices): tmp2[f"s{j}"] = m.values
        hits = tmp2.groupby("Run").agg(**{f"s{j}":(f"s{j}","any") for j in range(NA_SLICES)})
        slice_ok_runs = set(hits.index[hits.sum(axis=1) >= MIN_SLICES_TOUCHED])
        count_slice_ok = len(slice_ok_runs)
        
        na_df = g.loc[na_overall].copy()
        if not na_df.empty:
            na_df["lat_next"] = na_df.groupby("Run")["Latitude"].shift(-1)
            na_df["lon_next"] = na_df.groupby("Run")["Longitude"].shift(-1)
            na_df["bearing"] = calc_bearing(na_df["Latitude"].values, na_df["Longitude"].values, na_df["lat_next"].values, na_df["lon_next"].values)
            bearing_stats = na_df.dropna(subset=["bearing"]).groupby("Run")["bearing"].apply(lambda s: np.mean((s>=WE_MIN_DEG)&(s<=WE_MAX_DEG)))
            bearing_ok_runs = set(bearing_stats[bearing_stats >= MIN_WE_FRACTION].index)
            count_bearing_ok = len(bearing_ok_runs)
        else:
            bearing_ok_runs = set()
            count_bearing_ok = 0

        good_runs = slice_ok_runs.intersection(bearing_ok_runs)
        count_final_crossings = len(good_runs)
        
        return {
            "total_runs": total_runs,
            "in_box": count_in_box,
            "slice_rule_passed": count_slice_ok,
            "bearing_rule_passed": count_bearing_ok,
            "final_crossings": count_final_crossings
        }

    except Exception:
        return None

# --- MAIN PROCESS ---
def main():
    if not os.path.exists(STAGING_FILE):
        raise SystemExit(f"Staging file not found: {STAGING_FILE}")

    all_vehicle_ids = pd.read_parquet(STAGING_FILE, columns=["VehicleID"])["VehicleID"].unique()
    print(f"Found {len(all_vehicle_ids)} unique vehicles. Starting diagnostic analysis...")

    with multiprocessing.Pool(NUM_WORKERS) as pool:
        results = pool.map(debug_vehicle_worker, all_vehicle_ids)
    
    total_counts = {"total_runs": 0, "in_box": 0, "slice_rule_passed": 0, "bearing_rule_passed": 0, "final_crossings": 0}
    for res in results:
        if res:
            for key in total_counts:
                total_counts[key] += res[key]
    
    print("\n--- Diagnostic Report ---")
    print(f"Total distinct vehicle trips found:       {total_counts['total_runs']}")
    print(f"Trips that entered the geographic box:    {total_counts['in_box']}")
    print("---")
    print(f"Trips that passed the SLICE rule:         {total_counts['slice_rule_passed']}")
    print(f"Trips that passed the BEARING rule:       {total_counts['bearing_rule_passed']}")
    print("---")
    print(f"Trips that passed BOTH rules (final #):   {total_counts['final_crossings']}")
    print("\n--- End of Report ---")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()