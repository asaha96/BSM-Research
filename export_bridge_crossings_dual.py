import os
import pandas as pd
import numpy as np
from datetime import datetime, timezone
import multiprocessing
from glob import glob
import pyarrow.parquet as pq # Added import
import math # Added import

# -----------------------
# CONFIG
# -----------------------
OUTPUT_DIR = os.environ.get("BSM_OUTPUT_DIR", os.path.join(os.getcwd(), "data", "analysis"))
# No longer needs a single STAGING_FILE path here
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Start with 1 worker to ensure memory stability
NUM_WORKERS = 1

# North Ave decision box (full) - Widened
LAT_MIN, LAT_MAX = 33.7710, 33.7717
LON_MIN, LON_MAX = -84.3907, -84.3887

# Crossing rules (Original W-E settings)
NA_SLICES = 3; MIN_SLICES_TOUCHED = 3
WE_MIN_DEG, WE_MAX_DEG = 45.0, 135.0
MIN_WE_FRACTION = 0.60
GAP_S = 10.0
MIN_POINTS_IN_BOX = 10 # Require at least 10 points in the box

# Viz-required columns
NEEDED_COLS = ["Latitude","Longitude","Speed_mps","Elevation_m","DateTime_UTC","Timestamp"]

# -----------------------
# Helpers
# -----------------------
def within_box(df, lat_min, lat_max, lon_min, lon_max):
    if "Latitude" not in df.columns or "Longitude" not in df.columns:
        return pd.Series([False] * len(df), index=df.index)
    # Ensure data is numeric before comparison
    lat = pd.to_numeric(df["Latitude"], errors='coerce')
    lon = pd.to_numeric(df["Longitude"], errors='coerce')
    return (lat.between(lat_min, lat_max)) & (lon.between(lon_min, lon_max))

def calc_bearing(lat1, lon1, lat2, lon2):
    lat1 = np.asarray(lat1, dtype="float64"); lon1 = np.asarray(lon1, dtype="float64")
    lat2 = np.asarray(lat2, dtype="float64"); lon2 = np.asarray(lon2, dtype="float64")
    # Mask out pairs where any value is NaN or where start/end points are identical
    mask = ~(np.isnan(lat1) | np.isnan(lon1) | np.isnan(lat2) | np.isnan(lon2) | ((lat1 == lat2) & (lon1 == lon2)))
    out = np.full_like(lat1, np.nan, dtype="float64")
    if mask.any():
        a1=np.radians(lat1[mask]); o1=np.radians(lon1[mask]); a2=np.radians(lat2[mask]); o2=np.radians(lon2[mask])
        dlon=o2-o1;
        # Use np functions for vectorized calculation
        x=np.sin(dlon)*np.cos(a2);
        y=np.cos(a1)*np.sin(a2)-np.sin(a1)*np.cos(a2)*np.cos(dlon)
        # Use np.arctan2 for numerical stability
        brng=np.degrees(np.arctan2(x,y));
        brng = (brng + 360.0) % 360.0;
        out[mask]=brng
    return out

def split_na_longitudinal(df, n=NA_SLICES):
    lmin, lmax = sorted([LON_MIN, LON_MAX]); edges = np.linspace(lmin, lmax, n + 1); masks = []
    if "Latitude" not in df.columns or "Longitude" not in df.columns: return np.array([]), []
    # Ensure numeric types
    lat = pd.to_numeric(df["Latitude"], errors='coerce')
    lon = pd.to_numeric(df["Longitude"], errors='coerce')
    base_lat = lat.between(LAT_MIN, LAT_MAX)
    for i in range(n): masks.append(base_lat & lon.between(edges[i], edges[i+1]))
    return edges, masks

def safe_to_datetime(series):
    if not pd.api.types.is_string_dtype(series) and not pd.api.types.is_object_dtype(series):
        try:
             if pd.api.types.is_numeric_dtype(series): return pd.to_datetime(series, unit='s', utc=True, errors='coerce')
             elif pd.api.types.is_datetime64_any_dtype(series):
                 if getattr(series.dt, 'tz', None) is None: return series.dt.tz_localize('UTC')
                 else: return series.dt.tz_convert('UTC')
             else: # Handle other unexpected types
                 return pd.to_datetime(pd.NA, utc=True)
        except Exception: return pd.to_datetime(pd.NA, utc=True)

    fast_format = '%Y-%m-%d %H:%M:%S.%f'
    try: return pd.to_datetime(series, format=fast_format, utc=True, errors="raise")
    except (ValueError, TypeError): return pd.to_datetime(series, utc=True, errors="coerce")

def contiguous_segments(mask: np.ndarray):
    idx = np.where(mask)[0];
    if idx.size == 0: return []
    splits = np.where(np.diff(idx) != 1)[0] + 1; parts = np.split(idx, splits)
    return [p for p in parts if len(p) >= 2]

# -----------------------
# WORKER FUNCTION (Further Memory Optimization for Viz Reload)
# -----------------------
def analyze_vehicle_worker(args):
    vid, staging_file_path = args
    try:
        # 1. Load only essential columns first
        initial_cols = ['VehicleID', 'Latitude', 'Longitude', 'DateTime_UTC', 'VehicleType']
        filters = [('VehicleID', '==', vid)]
        g = pd.read_parquet(staging_file_path, columns=initial_cols, filters=filters)

        if g.empty: return None
        if not all(col in g.columns for col in ['Latitude', 'Longitude', 'DateTime_UTC']): return None

        g["DateTime_UTC"] = safe_to_datetime(g["DateTime_UTC"])
        g = g[g["Latitude"].notna() & g["Longitude"].notna() & g["DateTime_UTC"].notna()]
        g = g.sort_values("DateTime_UTC")
        if g.empty: return None

        # 2. Early spatial filter
        g = g[g["Latitude"].between(LAT_MIN - 0.01, LAT_MAX + 0.01) & \
              g["Longitude"].between(LON_MIN - 0.01, LON_MAX + 0.01)]
        if g.empty: return None

        time_diff = g["DateTime_UTC"].astype("int64").diff() // 10**9
        run_col = (time_diff > GAP_S).cumsum().fillna(0)
        g = g.assign(Run=run_col)

        vehicle_crossings = []
        vehicle_viz_segments = []

        for run_id, run_df in g.groupby('Run'):
            df_in_box = run_df[within_box(run_df, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX)].copy()

            if len(df_in_box) < max(2, 10): # Using 10 as MIN_POINTS_IN_BOX implicitly
                 continue

            # --- Apply Slice Rule ---
            _, slices = split_na_longitudinal(df_in_box, NA_SLICES)
            if not slices or len(slices) != NA_SLICES: continue

            touches_required_slices = 0
            for slice_mask_series in slices:
                aligned_mask = slice_mask_series.reindex(df_in_box.index, fill_value=False)
                # Check if aligned_mask has any True values before using .any()
                if aligned_mask.any() and df_in_box.loc[aligned_mask].shape[0] > 0 : # Check existence before .any()
                     # If the above line gives error, revert to: if df_in_box[aligned_mask].any().any():
                    touches_required_slices += 1

            if touches_required_slices < MIN_SLICES_TOUCHED:
                 continue

            # --- Apply Bearing Rule ---
            df_in_box['lat_next'] = df_in_box['Latitude'].shift(-1)
            df_in_box['lon_next'] = df_in_box['Longitude'].shift(-1)
            valid_next = df_in_box['lat_next'].notna()
            bearings = np.full(len(df_in_box), np.nan)
            if valid_next.any():
                bearings[valid_next] = calc_bearing(
                    df_in_box.loc[valid_next, "Latitude"].values, df_in_box.loc[valid_next, "Longitude"].values,
                    df_in_box.loc[valid_next, "lat_next"].values, df_in_box.loc[valid_next, "lon_next"].values
                )
            df_in_box['bearing'] = bearings
            valid_bearings_series = df_in_box['bearing'].dropna()

            if valid_bearings_series.empty: continue

            we_frac = np.mean((valid_bearings_series >= WE_MIN_DEG) & (valid_bearings_series <= WE_MAX_DEG))

            if we_frac >= MIN_WE_FRACTION:
                 # It's a crossing!
                 vehicle_type = df_in_box["VehicleType"].iloc[0] if "VehicleType" in df_in_box.columns and not df_in_box.empty else "Unknown"

                 summary_dict = {
                      "VehicleID": vid, "VehicleType": vehicle_type,
                      "StartUTC": df_in_box["DateTime_UTC"].iloc[0].strftime("%Y-%m-%d %H:%M:%S"),
                      "EndUTC": df_in_box["DateTime_UTC"].iloc[-1].strftime("%Y-%m-%d %H:%M:%S"),
                      "Points": len(df_in_box),
                      "Pct_W_to_E": round(100 * we_frac, 2) if pd.notna(we_frac) else None
                 }
                 vehicle_crossings.append(summary_dict)

                 # --- Prepare Viz Rows (Optimized Reload) ---
                 crossing_indices = df_in_box.index # Get indices of the crossing points
                 
                 # Identify columns already loaded vs. those still needed for viz
                 cols_already_have = ['VehicleID', 'VehicleType', 'Latitude', 'Longitude', 'DateTime_UTC']
                 cols_to_reload = [col for col in NEEDED_COLS if col not in cols_already_have] + ['Timestamp'] # Include Timestamp if needed separately
                 cols_to_reload = list(set(cols_to_reload)) # Ensure uniqueness

                 viz_segment_df = df_in_box[cols_already_have].copy() # Start with data we have

                 if cols_to_reload: # Only reload if there are missing columns
                     try:
                         # Load ONLY the missing columns for this specific vehicle
                         missing_data = pd.read_parquet(
                             staging_file_path,
                             columns=cols_to_reload,
                             filters=[('VehicleID', '==', vid)]
                         )
                         # Filter reloaded data to only the rows matching the crossing indices
                         missing_data_segment = missing_data[missing_data.index.isin(crossing_indices)]
                         
                         # Join the missing data back to our segment dataframe
                         viz_segment_df = viz_segment_df.join(missing_data_segment, how='left')

                     except Exception as load_err:
                         print(f"Warning: Could not reload missing columns for viz segment for {vid}. Error: {load_err}")
                         # Fill missing columns with NaN if reload fails
                         for col in cols_to_reload:
                              if col not in viz_segment_df.columns: viz_segment_df[col] = np.nan

                 # Ensure Timestamp exists and is correct type AFTER potential reload/join
                 if "Timestamp" not in viz_segment_df.columns or viz_segment_df["Timestamp"].isna().any():
                     viz_segment_df["Timestamp"] = (viz_segment_df["DateTime_UTC"].astype("int64") // 10**9).astype("float64")
                 else:
                     # Ensure it's float64 if it exists
                     viz_segment_df["Timestamp"] = pd.to_numeric(viz_segment_df["Timestamp"], errors='coerce')


                 # Final check for all required columns
                 final_viz_cols = ["VehicleID", "VehicleType"] + NEEDED_COLS
                 for col in final_viz_cols:
                      if col not in viz_segment_df.columns:
                           if col == "VehicleID": viz_segment_df[col] = vid
                           elif col == "VehicleType": viz_segment_df[col] = vehicle_type
                           else: viz_segment_df[col] = np.nan

                 vehicle_viz_segments.append(viz_segment_df[final_viz_cols]) # Append with correct column order


        if not vehicle_crossings:
            return None

        consolidated_viz_df = pd.concat(vehicle_viz_segments, ignore_index=True) if vehicle_viz_segments else pd.DataFrame()
        return (vehicle_crossings, consolidated_viz_df)

    except Exception as e:
        print(f"Error processing vehicle {vid} from {os.path.basename(staging_file_path)}: {e}")
        # import traceback
        # print(traceback.format_exc())
        return None

# -----------------------
# MAIN PROCESS (Manager) - Uses memory-efficient ID reading
# -----------------------
def main():
    monthly_files = sorted(glob(os.path.join(OUTPUT_DIR, "all_vehicle_data_20*.parquet")))
    monthly_files = [f for f in monthly_files if "TEST" not in os.path.basename(f)]

    if not monthly_files:
        raise SystemExit(f"No monthly staging files found matching pattern in {OUTPUT_DIR}. Please run consolidate_data.py first.")

    print(f"Found {len(monthly_files)} monthly data files to analyze.")

    all_months_crossings = []
    all_months_viz_frames = [] # Collect DataFrames

    for staging_file_path in monthly_files:
        print(f"\n--- Processing file: {os.path.basename(staging_file_path)} ---")

        print("Reading unique vehicle IDs from this month (memory-efficiently)...")
        try:
            parquet_file = pq.ParquetFile(staging_file_path)
            unique_ids_this_month = set()
            # Check if file has row groups before iterating
            if parquet_file.num_row_groups > 0:
                for i in range(parquet_file.num_row_groups):
                    table = parquet_file.read_row_group(i, columns=['VehicleID'])
                    # Check if table read is valid and has rows
                    if table and table.num_rows > 0:
                         ids_in_group = table.column('VehicleID').to_pandas(zero_copy_only=False).unique()
                         unique_ids_this_month.update(ids_in_group)
            else:
                 # Handle empty or metadata-only Parquet files
                 print(f"Warning: Parquet file {os.path.basename(staging_file_path)} has no row groups.")
                 # Optionally try reading the whole file if small, or skip
                 # table = pq.read_table(staging_file_path, columns=['VehicleID'])
                 # if table and table.num_rows > 0: ... etc.


            all_vehicle_ids = list(unique_ids_this_month)
            if not all_vehicle_ids:
                print("No vehicles found in this month's file. Skipping.")
                continue

        except Exception as e:
            print(f"Error reading vehicle IDs from {os.path.basename(staging_file_path)}: {e}. Skipping this file.")
            continue

        print(f"Found {len(all_vehicle_ids)} unique vehicles. Starting analysis with {NUM_WORKERS} worker(s)...")

        tasks = [(vid, staging_file_path) for vid in all_vehicle_ids]

        # Choose execution path based on NUM_WORKERS
        if NUM_WORKERS > 1:
             try:
                 with multiprocessing.Pool(NUM_WORKERS) as pool:
                     # Using imap_unordered for better memory management with large result sets
                     results_iterator = pool.imap_unordered(analyze_vehicle_worker, tasks, chunksize=max(1, len(tasks) // (NUM_WORKERS * 4)))
                     for res in results_iterator:
                         if res is not None:
                             crossings, viz_df = res
                             all_months_crossings.extend(crossings)
                             if not viz_df.empty:
                                 all_months_viz_frames.append(viz_df) # Append DataFrame
             except Exception as e:
                 print(f"An error occurred during parallel processing for {os.path.basename(staging_file_path)}: {e}")
                 # Consider if you should stop or continue to the next month
                 # continue
        else: # Run sequentially if NUM_WORKERS is 1
             print("Running analysis sequentially (NUM_WORKERS=1)...")
             for task in tasks:
                 res = analyze_vehicle_worker(task)
                 if res is not None:
                     crossings, viz_df = res
                     all_months_crossings.extend(crossings)
                     if not viz_df.empty:
                         all_months_viz_frames.append(viz_df) # Append DataFrame


        print(f"--- Finished processing {os.path.basename(staging_file_path)} ---")

    # --- Consolidate and Save Final Results ---
    print("\n--- All months processed. Consolidating and saving final results ---")

    today = datetime.now(timezone.utc).strftime("%Y%m%d")
    essentials_path = os.path.join(OUTPUT_DIR, f"north_ave_bridge_crossings_ALL_MONTHS_{today}.csv")
    viz_path = os.path.join(OUTPUT_DIR, f"north_ave_bridge_crossing_rows_ALL_MONTHS_{today}.csv")

    if not all_months_crossings:
        print("No crossings found in any of the monthly files.")
        return # Exit if no crossings found

    # Save Essentials
    ess_df = pd.DataFrame(all_months_crossings)
    # Perform sort ensuring StartUTC is datetime if needed for correct sorting, or sort as string
    try:
        ess_df['StartUTC_dt'] = pd.to_datetime(ess_df['StartUTC'])
        ess_df.sort_values(["VehicleID", "StartUTC_dt"], inplace=True)
        ess_df.drop(columns=['StartUTC_dt'], inplace=True)
    except Exception:
        print("Warning: Could not sort essentials by datetime, sorting by string.")
        ess_df.sort_values(["VehicleID", "StartUTC"], inplace=True)

    ess_df.to_csv(essentials_path, index=False)
    print(f"\nSaved {len(ess_df)} total crossings → {essentials_path}")

    # Save Visualization Data if collected
    if all_months_viz_frames:
        print("Concatenating visualization dataframes (this might take time and memory)...")
        try:
             # Concatenate collected DataFrames
             viz_df = pd.concat(all_months_viz_frames, ignore_index=True)
             # Ensure DateTime_UTC is datetime for sorting
             viz_df["DateTime_UTC"] = pd.to_datetime(viz_df["DateTime_UTC"], errors='coerce') # Coerce errors just in case
             viz_df.sort_values(["VehicleID", "DateTime_UTC"], inplace=True)
             viz_df.to_csv(viz_path, index=False)
             print(f"Saved {len(viz_df)} total viz rows → {viz_path}")
        except Exception as e: # Catch potential memory error or other issues during concat/save
            print(f"Error saving consolidated visualization file: {e}")
            print("Consider saving visualization data per month or implementing chunked writing if memory issues persist.")
    else:
         print("No visualization rows were generated or collected.")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    # Ensure NUM_WORKERS is set correctly in CONFIG section above
    main()