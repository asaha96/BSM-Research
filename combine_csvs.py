import pandas as pd
import os

# --- CONFIG ---

# This is the parent directory where the CSVs are located
BASE_DIR = os.environ.get("BSM_OUTPUT_DIR", os.path.join(os.getcwd(), "data", "analysis"))

# 1. List the exact filenames you want to combine
files_to_combine = [
    "north_ave_bridge_crossings_20251021.csv",
    "north_ave_bridge_crossings_MAR_AUG_2025_20251028.csv",
    "north_ave_bridge_crossings_Sept24_to_Feburary25.csv"
]

# 2. Define the new, combined output file name
output_filename = os.path.join(BASE_DIR, "north_ave_bridge_crossings_ALL_MONTHS.csv")

# --- SCRIPT ---

df_list = []
print("Starting to combine CSV files...")

for file in files_to_combine:
    file_path = os.path.join(BASE_DIR, file)
    
    if os.path.exists(file_path):
        print(f"Reading: {file}")
        try:
            # Read the CSV file
            df = pd.read_csv(file_path)
            df_list.append(df)
        except Exception as e:
            print(f"  Error reading {file}: {e}")
    else:
        print(f"Warning: File not found and will be skipped: {file_path}")

if not df_list:
    print("No files were successfully read. Exiting.")
else:
    # Concatenate all DataFrames in the list
    combined_df = pd.concat(df_list, ignore_index=True)
    
    # Save the combined DataFrame to the new CSV file
    try:
        combined_df.to_csv(output_filename, index=False)
        print("\n--- Success! ---")
        print(f"Combined {len(df_list)} files into one.")
        print(f"Total rows in new file: {len(combined_df)}")
        print(f"Saved to: {output_filename}")
    except Exception as e:
        print(f"Error saving combined file: {e}")