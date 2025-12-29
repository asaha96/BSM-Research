import pandas as pd
import matplotlib.pyplot as plt
import os
from glob import glob
import numpy as np

# --- CONFIG ---
OUTPUT_DIR = os.environ.get("BSM_OUTPUT_DIR", os.path.join(os.getcwd(), "data", "analysis"))
PLOT_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "Verification_Dropped_Crossings")
os.makedirs(PLOT_OUTPUT_DIR, exist_ok=True)

# Bridge Box (Reference)
LAT_MIN, LAT_MAX = 33.7710, 33.7717
LON_MIN, LON_MAX = -84.3907, -84.3887

# --- HELPER FUNCTIONS ---

def calculate_bearing_vectorized(lat_arr, lon_arr, next_lat_arr, next_lon_arr):
    """Vectorized calculation of bearing between two arrays of coordinates."""
    lat1, lon1 = np.radians(lat_arr), np.radians(lon_arr)
    lat2, lon2 = np.radians(next_lat_arr), np.radians(next_lon_arr)
    
    dlon = lon2 - lon1
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - (np.sin(lat1) * np.cos(lat2) * np.cos(dlon))
    
    initial_bearing = np.arctan2(x, y)
    initial_bearing = np.degrees(initial_bearing)
    return (initial_bearing + 360) % 360

def normalize_id(vid):
    """Strips whitespace and ensures string."""
    return str(vid).strip()

def get_matching_rows(all_rows, vid, start, end):
    """
    Robustly finds rows for a vehicle ID, handling potential '0' padding mismatches.
    """
    vid = normalize_id(vid)
    
    # Try exact match first
    mask = (all_rows['VehicleID'] == vid)
    
    # If no match, try handling leading zero differences
    if not mask.any():
        # If vid is '4874', look for '04874'
        if len(vid) == 4:
             mask = (all_rows['VehicleID'] == '0' + vid)
        # If vid is '04874', look for '4874'
        elif vid.startswith('0'):
             mask = (all_rows['VehicleID'] == vid.lstrip('0'))
    
    # Combine with time filter
    points = all_rows[mask & (all_rows['DateTime_UTC'] >= start) & (all_rows['DateTime_UTC'] <= end)].copy()
    return points

def identify_dropped_crossings(summary_df, rows_file):
    print("Loading detailed rows (this takes a moment)...")
    cols_to_load = ['VehicleID', 'DateTime_UTC', 'Latitude', 'Longitude', 'Speed_mps']
    
    try:
        df_rows = pd.read_csv(rows_file, usecols=cols_to_load, low_memory=False)
        df_rows['DateTime_UTC'] = pd.to_datetime(df_rows['DateTime_UTC'], utc=True)
        df_rows['VehicleID'] = df_rows['VehicleID'].astype(str).str.strip()
        summary_df['VehicleID'] = summary_df['VehicleID'].astype(str).str.strip()
    except Exception as e:
        print(f"Error loading rows: {e}")
        return None, None

    # 1. Calculate Headings
    print("  Calculating headings...")
    df_rows.sort_values(['VehicleID', 'DateTime_UTC'], inplace=True)
    df_rows['next_lat'] = df_rows['Latitude'].shift(-1)
    df_rows['next_lon'] = df_rows['Longitude'].shift(-1)
    df_rows['next_vid'] = df_rows['VehicleID'].shift(-1)
    
    bearings = calculate_bearing_vectorized(
        df_rows['Latitude'].values, df_rows['Longitude'].values,
        df_rows['next_lat'].values, df_rows['next_lon'].values
    )
    df_rows['Heading'] = bearings
    
    # Handle last points
    mask_invalid = (df_rows['VehicleID'] != df_rows['next_vid'])
    df_rows.loc[mask_invalid, 'Heading'] = np.nan
    df_rows['Heading'] = df_rows['Heading'].ffill()

    # 2. Label Bridge vs Highway
    tolerance = 40
    is_east = (df_rows['Heading'] >= (90 - tolerance)) & (df_rows['Heading'] <= (90 + tolerance))
    is_west = (df_rows['Heading'] >= (270 - tolerance)) & (df_rows['Heading'] <= (270 + tolerance))
    is_slow = df_rows['Speed_mps'] < 1.0
    
    df_rows['Is_Bridge'] = is_east | is_west | is_slow
    
    # 3. Calculate Valid Duration per Crossing
    print("  Identifying false positives...")
    
    results = []
    for idx, row in summary_df.iterrows():
        vid = row['VehicleID']
        start = row['StartUTC']
        end = row['EndUTC']
        
        points = get_matching_rows(df_rows, vid, start, end)
        
        if points.empty:
            # If we can't find points, we assume it's invalid/dropped for safety (or data error)
            results.append({'Valid_Duration': 0.0, 'Total_Points': 0, 'Status': 'No Data Found'})
            continue
            
        valid_points = points[points['Is_Bridge']]
        
        if valid_points.empty:
            results.append({'Valid_Duration': 0.0, 'Total_Points': len(points), 'Status': 'Highway Only'})
        else:
            duration = (valid_points['DateTime_UTC'].max() - valid_points['DateTime_UTC'].min()).total_seconds()
            results.append({'Valid_Duration': duration, 'Total_Points': len(points), 'Status': 'Valid'})
    
    res_df = pd.DataFrame(results)
    summary_df['Valid_Duration'] = res_df['Valid_Duration']
    summary_df['Total_Points'] = res_df['Total_Points']
    summary_df['Status'] = res_df['Status']
    
    # DROPPED definition: Valid duration is effectively zero
    dropped_df = summary_df[summary_df['Valid_Duration'] <= 5.0].copy()
    
    return dropped_df, df_rows

def plot_dropped_crossing(crossing_row, all_rows):
    vid = str(crossing_row['VehicleID'])
    start = crossing_row['StartUTC']
    end = crossing_row['EndUTC']
    
    # Use robust matching
    trip_points = get_matching_rows(all_rows, vid, start, end)
    
    if trip_points.empty:
        print(f"  Skipping {vid}: No GPS points found matching this ID and Time.")
        return

    plt.figure(figsize=(10, 10))
    
    # 1. Draw Bridge Box
    plt.plot([LON_MIN, LON_MAX, LON_MAX, LON_MIN, LON_MIN],
             [LAT_MIN, LAT_MIN, LAT_MAX, LAT_MAX, LAT_MIN],
             'k--', linewidth=1.5, label='Bridge Boundary')
    
    # 2. Draw "False" Points (Highway) in RED
    highway_points = trip_points[~trip_points['Is_Bridge']]
    if not highway_points.empty:
        plt.plot(highway_points['Longitude'], highway_points['Latitude'], 
                 'rx', markersize=6, alpha=0.6, label='Highway Heading (Dropped)')
        
    # 3. Draw "Valid" Points (Bridge) in BLUE
    bridge_points = trip_points[trip_points['Is_Bridge']]
    if not bridge_points.empty:
        plt.plot(bridge_points['Longitude'], bridge_points['Latitude'], 
                 'b.', markersize=8, label='Bridge Heading (Kept)')

    # 4. Connect lines to show path flow
    plt.plot(trip_points['Longitude'], trip_points['Latitude'], 'k-', linewidth=0.5, alpha=0.3)

    # 5. Start/End
    plt.plot(trip_points.iloc[0]['Longitude'], trip_points.iloc[0]['Latitude'], 'go', label='Start')
    plt.plot(trip_points.iloc[-1]['Longitude'], trip_points.iloc[-1]['Latitude'], 'rs', label='End')

    plt.title(f"DROPPED CROSSING: {vid}\nStart: {start.strftime('%Y-%m-%d %H:%M')}\nPoints Found: {len(trip_points)}", 
              fontsize=12, fontweight='bold')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.axis('equal')
    
    filename = f"dropped_{vid}_{start.strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(os.path.join(PLOT_OUTPUT_DIR, filename))
    plt.close()
    print(f"  Generated plot: {filename}")

def main():
    print("Starting Dropped Crossings Verification...")

    # 1. Load Summary
    summary_files = glob(os.path.join(OUTPUT_DIR, "north_ave_bridge_crossings_ALL_MONTHS.csv"))
    rows_file = os.path.join(OUTPUT_DIR, "north_ave_bridge_crossing_rows_ALL_MONTHS_20251028.csv")
    
    if not summary_files or not os.path.exists(rows_file):
        print("Error: Missing input files.")
        return

    df_summary = pd.read_csv(summary_files[0])
    df_summary['StartUTC'] = pd.to_datetime(df_summary['StartUTC'], utc=True)
    df_summary['EndUTC'] = pd.to_datetime(df_summary['EndUTC'], utc=True)
    
    # Filter to relevant dates
    start_date = pd.Timestamp("2024-09-01", tz='UTC')
    end_date = pd.Timestamp("2025-08-31 23:59:59.999999", tz='UTC')
    df_summary = df_summary[(df_summary['StartUTC'] >= start_date) & (df_summary['StartUTC'] <= end_date)].copy()

    # 2. Identify Dropped Crossings
    dropped_df, all_rows = identify_dropped_crossings(df_summary, rows_file)
    
    if dropped_df is None or dropped_df.empty:
        print("No dropped crossings found.")
        return

    print(f"\nFound {len(dropped_df)} crossings identified as False Positives.")
    print(f"Generating verification plots in: {PLOT_OUTPUT_DIR}")
    
    # 3. Generate Plots
    count = 0
    for _, row in dropped_df.iterrows():
        plot_dropped_crossing(row, all_rows)
        count += 1
        if count % 10 == 0: print(f"  ... processed {count} plots ...")

    # Save list for reference
    dropped_df.to_csv(os.path.join(PLOT_OUTPUT_DIR, "dropped_crossings_list.csv"), index=False)
    print("\nVerification Complete.")

if __name__ == "__main__":
    main()