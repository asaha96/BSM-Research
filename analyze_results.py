import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from glob import glob

# --- CONFIG ---
OUTPUT_DIR = os.environ.get("BSM_OUTPUT_DIR", os.path.join(os.getcwd(), "data", "analysis"))
PLOT_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "Analysis_Long_Crossings_Final_3")
os.makedirs(PLOT_OUTPUT_DIR, exist_ok=True)

# Bridge Box
LAT_MIN, LAT_MAX = 33.7710, 33.7717
LON_MIN, LON_MAX = -84.3907, -84.3887

# --- HELPER FUNCTIONS ---

def calculate_bearing_vectorized(lat_arr, lon_arr, next_lat_arr, next_lon_arr):
    lat1, lon1 = np.radians(lat_arr), np.radians(lon_arr)
    lat2, lon2 = np.radians(next_lat_arr), np.radians(next_lon_arr)
    dlon = lon2 - lon1
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - (np.sin(lat1) * np.cos(lat2) * np.cos(dlon))
    return (np.degrees(np.arctan2(x, y)) + 360) % 360

def normalize_id(vid):
    return str(vid).strip().lstrip('0')

def get_matching_rows(all_rows, vid, start, end):
    """Robust ID matching."""
    target_id = normalize_id(vid)
    # Filter by time first for speed
    subset = all_rows[(all_rows['DateTime_UTC'] >= start) & (all_rows['DateTime_UTC'] <= end)].copy()
    # Then match ID
    mask = subset['VehicleID'].apply(normalize_id) == target_id
    return subset[mask]

def apply_duration_correction_and_get_rows(summary_df):
    """
    Applies the correction logic and returns BOTH the corrected summary 
    AND the full dataframe of rows with the 'Is_Bridge' tag.
    """
    rows_file = os.path.join(OUTPUT_DIR, "north_ave_bridge_crossing_rows_ALL_MONTHS_20251028.csv")
    if not os.path.exists(rows_file):
        print("Error: Rows file not found.")
        return summary_df, None

    print("Loading detailed rows...")
    cols = ['VehicleID', 'DateTime_UTC', 'Latitude', 'Longitude', 'Speed_mps']
    df_rows = pd.read_csv(rows_file, usecols=cols, low_memory=False)
    df_rows['DateTime_UTC'] = pd.to_datetime(df_rows['DateTime_UTC'], utc=True)
    df_rows['VehicleID'] = df_rows['VehicleID'].astype(str).str.strip()
    summary_df['VehicleID'] = summary_df['VehicleID'].astype(str).str.strip()

    print(f"  Calculating headings for {len(df_rows)} points...")
    df_rows.sort_values(['VehicleID', 'DateTime_UTC'], inplace=True)
    
    bearings = calculate_bearing_vectorized(
        df_rows['Latitude'].values, df_rows['Longitude'].values,
        df_rows['Latitude'].shift(-1).values, df_rows['Longitude'].shift(-1).values
    )
    df_rows['Heading'] = bearings
    
    # Handle last point of vehicle
    mask_invalid = (df_rows['VehicleID'] != df_rows['VehicleID'].shift(-1))
    df_rows.loc[mask_invalid, 'Heading'] = np.nan
    df_rows['Heading'] = df_rows['Heading'].ffill()
    
    # Filter: East/West OR Slow
    tol = 40
    is_bridge = (
        ((df_rows['Heading'] >= (90 - tol)) & (df_rows['Heading'] <= (90 + tol))) |
        ((df_rows['Heading'] >= (270 - tol)) & (df_rows['Heading'] <= (270 + tol))) |
        (df_rows['Speed_mps'] < 1.0)
    )
    df_rows['Is_Bridge'] = is_bridge
    
    # Create clean subset for duration calculation
    clean_rows = df_rows[is_bridge].copy()
    
    # Add match columns
    clean_rows['MatchID'] = clean_rows['VehicleID'].apply(normalize_id)
    summary_df['MatchID'] = summary_df['VehicleID'].apply(normalize_id)

    print("  Recalculating durations...")
    def get_corrected_duration(row):
        vid = row['MatchID']
        pts = clean_rows[
            (clean_rows['MatchID'] == vid) & 
            (clean_rows['DateTime_UTC'] >= row['StartUTC']) & 
            (clean_rows['DateTime_UTC'] <= row['EndUTC'])
        ]
        if pts.empty: return 0.0
        return (pts['DateTime_UTC'].max() - pts['DateTime_UTC'].min()).total_seconds()

    summary_df['Duration_sec'] = summary_df.apply(get_corrected_duration, axis=1)
    
    return summary_df, df_rows

def plot_trajectory(row, all_rows):
    vid = str(row['VehicleID'])
    points = get_matching_rows(all_rows, vid, row['StartUTC'], row['EndUTC'])
    
    if points.empty: return

    fig, ax = plt.subplots(figsize=(12, 12))
    
    # 1. Draw Bridge Box
    ax.plot([LON_MIN, LON_MAX, LON_MAX, LON_MIN, LON_MIN],
             [LAT_MIN, LAT_MIN, LAT_MAX, LAT_MAX, LAT_MIN], 
             'k--', linewidth=2, label='Bridge Boundary')

    # 2. Draw Points
    # Red X = Highway/Invalid (Heading North/South)
    # Blue Dot = Bridge/Valid (Heading East/West or Stopped)
    
    dropped = points[~points['Is_Bridge']]
    kept = points[points['Is_Bridge']]
    
    if not dropped.empty:
        ax.plot(dropped['Longitude'], dropped['Latitude'], 'rx', alpha=0.4, markersize=6, label='Ignored (Highway Heading)')
    
    if not kept.empty:
        # Connect the valid points to see the sequence
        ax.plot(kept['Longitude'], kept['Latitude'], 'b-', linewidth=1, alpha=0.5)
        ax.plot(kept['Longitude'], kept['Latitude'], 'bo', markersize=6, alpha=0.8, label='Valid Path (Bridge Heading)')
        
        # Mark Start and End of valid path
        ax.plot(kept.iloc[0]['Longitude'], kept.iloc[0]['Latitude'], 'g^', markersize=12, label='Start (Valid)', zorder=5)
        ax.plot(kept.iloc[-1]['Longitude'], kept.iloc[-1]['Latitude'], 'rs', markersize=12, label='End (Valid)', zorder=5)

    # 3. Add Info Text
    start_str = row['StartUTC'].strftime('%Y-%m-%d %H:%M')
    dur_min = row['Duration_sec'] / 60.0
    
    title_text = (f"Long Crossing Analysis: Vehicle {vid}\n"
                  f"Date: {start_str}\n"
                  f"Corrected Duration: {dur_min:.1f} minutes ({int(row['Duration_sec'])} sec)")
    
    ax.set_title(title_text, fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.axis('equal')
    
    # Save
    filename = f"long_crossing_{vid}_{int(row['Duration_sec'])}s.png"
    save_path = os.path.join(PLOT_OUTPUT_DIR, filename)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved plot: {filename}")

def main():
    print("Starting Long Crossing Analysis...")
    summary_files = glob(os.path.join(OUTPUT_DIR, "north_ave_bridge_crossings_ALL_MONTHS.csv"))
    if not summary_files: return
    
    df = pd.read_csv(summary_files[0])
    df['StartUTC'] = pd.to_datetime(df['StartUTC'], utc=True)
    df['EndUTC'] = pd.to_datetime(df['EndUTC'], utc=True)
    
    # 1. Apply Correction (Crucial step to filter out false positives)
    df, all_rows = apply_duration_correction_and_get_rows(df)
    
    if all_rows is None: return

    # 2. Filter for Long Crossings (> 1000 seconds)
    long_crossings = df[df['Duration_sec'] > 1000].copy()
    
    print(f"\nFound {len(long_crossings)} confirmed crossings > 1000 seconds.")
    
    if long_crossings.empty:
        print("No crossings meet the criteria.")
        return

    # 3. Plot them
    print(f"Generating trajectory maps in: {PLOT_OUTPUT_DIR}")
    
    # Sort by duration descending to see the longest first
    long_crossings = long_crossings.sort_values('Duration_sec', ascending=False)
    
    for _, row in long_crossings.iterrows():
        plot_trajectory(row, all_rows)
        
    # Save a CSV report of these specific 3 trips
    csv_path = os.path.join(PLOT_OUTPUT_DIR, "long_crossings_details.csv")
    long_crossings.to_csv(csv_path, index=False)
    print(f"Details saved to: {csv_path}")

if __name__ == "__main__":
    main()