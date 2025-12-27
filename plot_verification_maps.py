import pandas as pd
import matplotlib.pyplot as plt
import os
from glob import glob
from datetime import datetime

# --- CONFIG ---
OUTPUT_DIR = os.environ.get("BSM_OUTPUT_DIR", os.path.join(os.getcwd(), "data", "analysis"))
PLOT_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "Verification_Plots_Sept_2024")
os.makedirs(PLOT_OUTPUT_DIR, exist_ok=True)

LAT_MIN, LAT_MAX = 33.7710, 33.7717
LON_MIN, LON_MAX = -84.3907, -84.3887

def main():
    print("Starting verification plot generation...")

    try:
        # --- 1. Define the exact file paths ---
        summary_file_path = os.path.join(OUTPUT_DIR, "north_ave_bridge_crossings_ALL_MONTHS.csv")
        rows_file_path = os.path.join(OUTPUT_DIR, "north_ave_bridge_crossing_rows_ALL_MONTHS_20251028.csv")

        # --- 2. Check if both files exist ---
        if not os.path.exists(summary_file_path):
            print(f"Error: Missing summary file: {os.path.basename(summary_file_path)}")
            print("Please run the 'combine_csvs.py' script first.")
            return
            
        if not os.path.exists(rows_file_path):
            print(f"Error: Missing detailed rows file: {os.path.basename(rows_file_path)}")
            print("This file should have been created by 'export_bridge_crossings_dual.py'.")
            return

        # --- 3. Load the summary file ---
        print(f"Loading summary from: {os.path.basename(summary_file_path)}")
        df_summary = pd.read_csv(summary_file_path)
        
        # Set the variable for the next step
        latest_rows_file = rows_file_path 
        
    except Exception as e:
        print(f"Error loading files: {e}")
        return

    # --- 4. Filter summary for Sept 2024 ---
    
    # --- FIX 1: Add utc=True to make these timestamps tz-aware ---
    df_summary['StartUTC'] = pd.to_datetime(df_summary['StartUTC'], utc=True)
    df_summary['EndUTC'] = pd.to_datetime(df_summary['EndUTC'], utc=True)
    
    sept_crossings = df_summary[df_summary['StartUTC'].dt.strftime('%Y-%m') == '2024-09'].copy()
    
    if sept_crossings.empty:
        print("No crossings found in the summary file for September 2024.")
        return
        
    print(f"Found {len(sept_crossings)} crossings for September 2024. Loading detailed row data...")
    
    # --- 5. Load the (large) detailed rows file ---
    try:
        # --- FIX 2: Add low_memory=False to suppress the DtypeWarning ---
        df_rows = pd.read_csv(latest_rows_file, low_memory=False)
        
        # Add utc=True here as well to ensure consistency
        df_rows['DateTime_UTC'] = pd.to_datetime(df_rows['DateTime_UTC'], utc=True) 
        
    except Exception as e:
        print(f"Error loading detailed rows file: {e}")
        return
        
    print(f"Loaded {len(df_rows)} detailed GPS points. Generating plots...")

    # --- 6. Loop through each crossing and plot it ---
    for index, crossing in sept_crossings.iterrows():
        vid = crossing['VehicleID']
        start_time = crossing['StartUTC']
        end_time = crossing['EndUTC']
        
        # This comparison will now work
        plot_data = df_rows[
            (df_rows['VehicleID'] == vid) &
            (df_rows['DateTime_UTC'].between(start_time, end_time))
        ]
        
        if plot_data.empty:
            print(f"Warning: No GPS rows found for crossing {vid} at {start_time}. Skipping.")
            continue
            
        plt.figure(figsize=(10, 8))
        plt.plot(
            [LON_MIN, LON_MAX, LON_MAX, LON_MIN, LON_MIN],
            [LAT_MIN, LAT_MIN, LAT_MAX, LAT_MAX, LAT_MIN],
            'r--', label='Bridge Box'
        )
        plt.plot(plot_data['Longitude'], plot_data['Latitude'], 'b.-', label='Vehicle Path')
        plt.plot(plot_data.iloc[0]['Longitude'], plot_data.iloc[0]['Latitude'], 'go', markersize=10, label='Start')
        plt.plot(plot_data.iloc[-1]['Longitude'], plot_data.iloc[-1]['Latitude'], 'rs', markersize=10, label='End')
        
        plt.title(f"Crossing Verification: Vehicle {vid}\nTime: {start_time}")
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.legend()
        plt.grid(True)
        plt.axis('equal') 
        
        plot_filename = f"crossing_plot_{vid}_{start_time.strftime('%Y%m%d_%H%M%S')}.png"
        plot_path = os.path.join(PLOT_OUTPUT_DIR, plot_filename)
        
        try:
            plt.savefig(plot_path)
            print(f"  ... Saved plot: {plot_filename}")
        except Exception as e:
            print(f"Error saving plot {plot_filename}: {e}")
        plt.close()

    print("\nVerification plot generation complete.")

if __name__ == "__main__":
    main()