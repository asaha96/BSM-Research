import pandas as pd
import matplotlib.pyplot as plt
import os
import io
import numpy as np

# --- CONFIG ---
OUTPUT_DIR = os.environ.get("BSM_OUTPUT_DIR", os.path.join(os.getcwd(), "data", "analysis"))
PLOT_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "Analysis_Long_Crossings_CORRECTED")
os.makedirs(PLOT_OUTPUT_DIR, exist_ok=True)

# Bridge Box (Reference)
LAT_MIN, LAT_MAX = 33.7710, 33.7717
LON_MIN, LON_MAX = -84.3907, -84.3887

# --- HEADING FILTER CONFIG ---
# North Ave is East-West (90 or 270). 
# Highway is North-South (0 or 180).
TOLERANCE = 40 

# --- RAW DATA INPUT (The 16 Problematic Crossings) ---
LONG_CROSSINGS_RAW = """VehicleID,VehicleType,StartUTC,EndUTC,Points,Pct_W_to_E,VehicleClass,Duration_sec,Duration_min
4874,,2025-08-08 19:43:58+00:00,2025-08-08 20:30:27+00:00,393,74.58,Ambulance,2789.0,46.48
4965,,2025-03-05 00:17:52+00:00,2025-03-05 01:11:48+00:00,494,65.82,Ambulance,3236.0,53.93
7922,,2025-03-08 12:52:36+00:00,2025-03-08 13:22:27+00:00,699,64.74,Ambulance,1791.0,29.85
04874,,2025-08-08 19:43:58+00:00,2025-08-08 20:30:27+00:00,391,80.0,Ambulance,2789.0,46.48
04888,,2024-09-17 23:57:46+00:00,2024-09-18 01:09:41+00:00,775,67.87,Ambulance,4315.0,71.92
04965,,2025-03-05 00:17:52+00:00,2025-03-05 01:11:48+00:00,479,71.43,Ambulance,3236.0,53.93
0510,,2024-09-17 18:11:45+00:00,2024-09-17 18:30:28+00:00,783,67.75,HERO,1123.0,18.72
07447,,2025-01-14 12:00:22+00:00,2025-01-14 12:28:55+00:00,317,65.49,Ambulance,1713.0,28.55
07449,,2024-11-02 17:39:40+00:00,2024-11-02 18:16:08+00:00,1045,68.66,Ambulance,2188.0,36.47
07450,,2024-09-03 14:54:12+00:00,2024-09-03 15:13:11+00:00,357,60.52,Ambulance,1139.0,18.98
07450,,2024-09-15 00:04:02+00:00,2024-09-15 00:58:54+00:00,1167,73.16,Ambulance,3292.0,54.87
07450,,2024-11-03 06:17:30+00:00,2024-11-03 07:22:15+00:00,448,69.74,Ambulance,3885.0,64.75
07450,,2025-04-18 01:03:06+00:00,2025-04-18 02:17:33+00:00,945,63.94,Ambulance,4467.0,74.45
07455,,2024-10-22 00:34:29+00:00,2024-10-22 01:17:10+00:00,677,68.45,Ambulance,2561.0,42.68
07902,,2024-12-11 03:39:09+00:00,2024-12-11 04:16:20+00:00,615,63.53,Ambulance,2231.0,37.18
07921,,2025-02-03 15:32:34+00:00,2025-02-03 16:29:40+00:00,644,69.19,Ambulance,3426.0,57.1"""

def get_target_crossings():
    df = pd.read_csv(io.StringIO(LONG_CROSSINGS_RAW))
    df['StartUTC'] = pd.to_datetime(df['StartUTC'], utc=True)
    df['EndUTC'] = pd.to_datetime(df['EndUTC'], utc=True)
    df['VehicleID'] = df['VehicleID'].astype(str).str.zfill(4)
    return df

def calculate_bearing(lat1, lon1, lat2, lon2):
    """Calculates bearing between two points."""
    lat1, lon1 = np.radians(lat1), np.radians(lon1)
    lat2, lon2 = np.radians(lat2), np.radians(lon2)
    dlon = lon2 - lon1
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - (np.sin(lat1) * np.cos(lat2) * np.cos(dlon))
    initial_bearing = np.arctan2(x, y)
    initial_bearing = np.degrees(initial_bearing)
    compass_bearing = (initial_bearing + 360) % 360
    return compass_bearing

def is_on_bridge_heading(heading):
    """
    Returns True if heading is roughly East or West (Bridge).
    Returns False if heading is roughly North or South (Highway).
    """
    if pd.isna(heading): return True 
    
    # East is 90, West is 270
    is_east = (90 - TOLERANCE) <= heading <= (90 + TOLERANCE)
    is_west = (270 - TOLERANCE) <= heading <= (270 + TOLERANCE)
    return is_east or is_west

def extract_rows_for_long_crossings(targets_df, all_rows_path):
    print(f"Reading huge rows file: {os.path.basename(all_rows_path)}...")
    df_rows = pd.read_csv(all_rows_path, low_memory=False)
    df_rows['DateTime_UTC'] = pd.to_datetime(df_rows['DateTime_UTC'], utc=True)
    df_rows['VehicleID'] = df_rows['VehicleID'].astype(str)
    
    extracted_dfs = []
    for _, target in targets_df.iterrows():
        vid = str(target['VehicleID'])
        mask = (
            (df_rows['VehicleID'].str.endswith(vid.lstrip('0'))) & 
            (df_rows['DateTime_UTC'] >= target['StartUTC']) &
            (df_rows['DateTime_UTC'] <= target['EndUTC'])
        )
        subset = df_rows[mask].copy()
        if not subset.empty:
            subset['Trip_Class'] = target['VehicleClass']
            subset['Original_Duration'] = target['Duration_min']
            extracted_dfs.append(subset)
    
    if extracted_dfs:
        return pd.concat(extracted_dfs, ignore_index=True)
    return None

def analyze_and_plot_corrected(df_rows, targets_df):
    print("\nApplying Heading Filter (Calculated) and Regenerating Plots...")
    
    results = []

    for _, target in targets_df.iterrows():
        vid = str(target['VehicleID'])
        start_time = target['StartUTC']
        end_time = target['EndUTC']
        v_class = target['VehicleClass']
        orig_dur = target['Duration_min']
        
        # Get the raw points
        raw_data = df_rows[
            (df_rows['VehicleID'].str.endswith(vid.lstrip('0'))) & 
            (df_rows['DateTime_UTC'] >= start_time) & 
            (df_rows['DateTime_UTC'] <= end_time)
        ].copy()
        
        if raw_data.empty: continue
        
        # Sort to ensure calculation works
        raw_data = raw_data.sort_values('DateTime_UTC')

        # --- CALCULATE HEADING ---
        # Shift latitude/longitude to compare row i with row i+1
        raw_data['next_lat'] = raw_data['Latitude'].shift(-1)
        raw_data['next_lon'] = raw_data['Longitude'].shift(-1)
        
        # Calculate bearing
        raw_data['Calculated_Heading'] = raw_data.apply(
            lambda x: calculate_bearing(x['Latitude'], x['Longitude'], x['next_lat'], x['next_lon']) 
            if pd.notnull(x['next_lat']) else np.nan, axis=1
        )
        
        # Forward fill the last value so the last point doesn't get dropped (it usually has NaN heading)
        raw_data['Calculated_Heading'] = raw_data['Calculated_Heading'].ffill()

        # --- APPLY FILTER ---
        # 1. Keep if Heading is East/West (Bridge)
        # 2. OR if Speed is very low (Stopped car = erratic heading)
        # 3. OR if Heading is NaN (Single points or errors)
        
        raw_data['Is_Bridge'] = raw_data.apply(
            lambda row: is_on_bridge_heading(row['Calculated_Heading']) or (row['Speed_mps'] < 1.0), axis=1
        )
        
        clean_data = raw_data[raw_data['Is_Bridge']].copy()
        rejected_data = raw_data[~raw_data['Is_Bridge']].copy()
        
        # --- RECALCULATE DURATION ---
        if not clean_data.empty:
            new_start = clean_data['DateTime_UTC'].min()
            new_end = clean_data['DateTime_UTC'].max()
            new_duration_sec = (new_end - new_start).total_seconds()
            new_duration_min = new_duration_sec / 60.0
        else:
            new_duration_min = 0
        
        print(f"Vehicle {vid}: Old Duration {orig_dur:.2f}m -> New Duration {new_duration_min:.2f}m")
        
        # --- PLOTTING ---
        plt.figure(figsize=(10, 10))
        
        # Bridge Box
        plt.plot([LON_MIN, LON_MAX, LON_MAX, LON_MIN, LON_MIN],
                 [LAT_MIN, LAT_MIN, LAT_MAX, LAT_MAX, LAT_MIN],
                 'k--', linewidth=1.5, label='Bridge Area')
        
        # Plot REJECTED points
        if not rejected_data.empty:
            plt.plot(rejected_data['Longitude'], rejected_data['Latitude'], 
                     'rx', markersize=6, alpha=0.4, label='Filtered Out (Heading N/S)')

        # Plot KEPT points
        color = '#ff7f0e' if v_class == 'Ambulance' else '#1f77b4'
        if not clean_data.empty:
            plt.plot(clean_data['Longitude'], clean_data['Latitude'], 
                     color=color, marker='.', markersize=8, linestyle='-', linewidth=1, label='Valid Path (Heading E/W)')
            
            plt.plot(clean_data.iloc[0]['Longitude'], clean_data.iloc[0]['Latitude'], 
                     'go', markersize=12, markeredgecolor='black', label='Corrected Start')
            plt.plot(clean_data.iloc[-1]['Longitude'], clean_data.iloc[-1]['Latitude'], 
                     'rX', markersize=12, markeredgecolor='black', label='Corrected End')

        plt.title(f"CORRECTED: {v_class} {vid}\nOld: {orig_dur:.1f} min -> New: {new_duration_min:.1f} min", 
                  fontsize=14, fontweight='bold')
        plt.xlabel('Longitude'); plt.ylabel('Latitude')
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        
        filename = f"corrected_{v_class}_{vid}_{start_time.strftime('%Y%m%d')}.png"
        plt.savefig(os.path.join(PLOT_OUTPUT_DIR, filename))
        plt.close()

        results.append({
            'VehicleID': vid,
            'Original_Min': orig_dur,
            'Corrected_Min': new_duration_min,
            'Reduction_Min': orig_dur - new_duration_min,
            'Status': 'Valid' if new_duration_min > 0.5 else 'Invalid/False Positive'
        })

    res_df = pd.DataFrame(results)
    report_path = os.path.join(PLOT_OUTPUT_DIR, "duration_correction_report.csv")
    res_df.to_csv(report_path, index=False)
    print(f"\nCorrection report saved to {report_path}")

def main():
    # 1. Load Targets
    targets_df = get_target_crossings()
    
    # 2. Locate Data
    extracted_file = os.path.join(OUTPUT_DIR, "Analysis_Long_Crossings", "long_crossings_detailed_rows.csv")
    big_file = os.path.join(OUTPUT_DIR, "north_ave_bridge_crossing_rows_ALL_MONTHS_20251028.csv")
    
    df_data = None
    if os.path.exists(extracted_file):
        print(f"Using extracted data: {extracted_file}")
        df_data = pd.read_csv(extracted_file)
        df_data['DateTime_UTC'] = pd.to_datetime(df_data['DateTime_UTC'], utc=True)
        df_data['VehicleID'] = df_data['VehicleID'].astype(str)
    elif os.path.exists(big_file):
        print("Extracted file not found. Reading from main file...")
        df_data = extract_rows_for_long_crossings(targets_df, big_file)
    else:
        print("Error: Data files not found.")
        return

    if df_data is not None:
        analyze_and_plot_corrected(df_data, targets_df)

if __name__ == "__main__":
    main()