# inspect_data_location.py
import pandas as pd
import os

# --- CONFIG ---
OUTPUT_DIR = os.environ.get("BSM_OUTPUT_DIR", os.path.join(os.getcwd(), "data", "analysis"))
STAGING_FILE = os.path.join(OUTPUT_DIR, "all_vehicle_data_SEPT_2025_TEST.parquet")

# --- SCRIPT ---
print(f"Inspecting data location in: {os.path.basename(STAGING_FILE)}")

try:
    # Read only the necessary columns to save memory
    df = pd.read_parquet(STAGING_FILE, columns=['Latitude', 'Longitude'])
    df.dropna(inplace=True)

    if df.empty:
        print("The file contains no valid latitude or longitude data.")
    else:
        # --- Print the bounding box of the ACTUAL data ---
        print("\n--- Actual Data Bounding Box ---")
        print(f"Latitude Range:  {df['Latitude'].min()} to {df['Latitude'].max()}")
        print(f"Longitude Range: {df['Longitude'].min()} to {df['Longitude'].max()}")
        print("--------------------------------")
        
        # --- Print our TARGET box for easy comparison ---
        print("\n--- Target Bridge Bounding Box (for comparison) ---")
        print("Latitude Range:  33.7710 to 33.7717")
        print("Longitude Range: -84.3907 to -84.3887")
        print("-------------------------------------------------")

except Exception as e:
    print(f"An error occurred: {e}")