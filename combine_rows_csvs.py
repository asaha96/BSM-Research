import pandas as pd
import os
import glob

# --- CONFIG ---
# This is the parent directory where all the CSVs are located
BASE_DIR = os.environ.get("BSM_OUTPUT_DIR", os.path.join(os.getcwd(), "data", "analysis"))

# 1. Define the new, combined output file name
output_filename = os.path.join(BASE_DIR, "north_ave_bridge_crossing_rows_ALL_MONTHS.csv")

# --- SCRIPT ---
print("Starting to combine all DETAILED ROWS CSV files...")

# 2. Find all "rows" files, but exclude any file that might be a previous combined result
all_row_files = glob.glob(os.path.join(BASE_DIR, "north_ave_bridge_rows_*.csv"))
files_to_combine = [f for f in all_row_files if "ALL_MONTHS" not in f]

if not files_to_combine:
    print("No individual monthly 'rows' files were found to combine.")
    exit()

print(f"Found {len(files_to_combine)} files to combine.")

df_list = []
for file in files_to_combine:
    print(f"Reading: {os.path.basename(file)}")
    try:
        df = pd.read_csv(file)
        df_list.append(df)
    except Exception as e:
        print(f"  Error reading {os.path.basename(file)}: {e}")

if not df_list:
    print("No 'rows' files were successfully read. Exiting.")
else:
    print("\nCombining all files... (This may take a moment and use a lot of memory)")
    # Concatenate all DataFrames in the list
    combined_df = pd.concat(df_list, ignore_index=True)
    
    # Save the combined DataFrame to the new CSV file
    try:
        combined_df.to_csv(output_filename, index=False)
        print("\n--- Success! ---")
        print(f"Combined {len(df_list)} detailed 'rows' files into one.")
        print(f"Total rows in new file: {len(combined_df)}")
        print(f"Saved to: {output_filename}")
    except Exception as e:
        print(f"Error saving combined 'rows' file: {e}")