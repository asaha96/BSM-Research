import pandas as pd
import os
from glob import glob
import pyarrow.parquet as pq

# --- CONFIG ---
DATA_DIR = os.environ.get("BSM_DATA_DIR", os.path.join(os.getcwd(), "data", "csv"))
OUTPUT_DIR = os.environ.get("BSM_OUTPUT_DIR", os.path.join(os.getcwd(), "data", "analysis"))
CHUNK_SIZE = 1000 

# --- SCRIPT ---
MASTER_DTYPES = {
    "Timestamp": "float64", "SecMark": "Int64", "TempID": "string",
    "Latitude": "float64", "Longitude": "float64", "Elevation_m": "float64",
    "Speed_mps": "float64", "Heading_deg": "float64", "Transmission": "string",
    "DateTime_UTC": "string", "VehicleID": "string", "VehicleType": "string",
    "SourceFile": "string"
}
MASTER_COLUMNS = list(MASTER_DTYPES.keys())

# --- Define the 12 months you want to process ---
MONTHS_TO_PROCESS = [
    "2024-09", "2024-10", "2024-11", "2024-12",
    "2025-01", "2025-02", "2025-03", "2025-04",
    "2025-05", "2025-06", "2025-07", "2025-08"
]
# NOTE: I am leaving out 2025-09 since you already processed that in your test.
# You can add "2025-09" to the list if you want to re-process it.

print("Finding all CSV files (this may take a moment)...")
all_files = sorted(glob(os.path.join(DATA_DIR, "**", "*.csv"), recursive=True))
print(f"Found {len(all_files)} total files.")

# --- Outer loop to process each month ---
for month_prefix in MONTHS_TO_PROCESS:
    print(f"\n" + "="*50)
    print(f"Starting process for month: {month_prefix}")
    print("="*50)

    # 1. Define unique paths for this month
    STAGING_FILE = os.path.join(OUTPUT_DIR, f"all_vehicle_data_{month_prefix.replace('-', '_')}.parquet")
    temp_dir = os.path.join(OUTPUT_DIR, f"temp_chunks_{month_prefix.replace('-', '_')}")
    os.makedirs(temp_dir, exist_ok=True)

    # 2. Filter all_files for this specific month
    files_for_this_month = [p for p in all_files if f"{month_prefix}-" in os.path.basename(p)]
    
    if not files_for_this_month:
        print(f"No CSV files found for month {month_prefix}. Skipping.")
        continue
    
    print(f"Found {len(files_for_this_month)} files for {month_prefix}.")

    # 3. Run the chunking/resume logic for this month
    chunk_files = sorted(glob(os.path.join(temp_dir, "chunk_*.parquet")))
    processed_files = set()
    if chunk_files:
        print(f"Found {len(chunk_files)} existing chunks for {month_prefix}. Resuming...")
        for chunk_file in chunk_files:
            try:
                df_chunk = pd.read_parquet(chunk_file, columns=['SourceFile'])
                processed_files.update(df_chunk['SourceFile'].unique())
            except Exception as e:
                print(f"Warning: Could not read chunk {chunk_file}. It will be re-processed. Error: {e}")

    files_to_process = [f for f in files_for_this_month if f not in processed_files]
    if not files_to_process:
        print(f"All files for {month_prefix} are already processed into chunks.")
    else:
        print(f"Processing {len(files_to_process)} new files in chunks of {CHUNK_SIZE}...")
        
        frames_for_chunk = []
        chunk_num = len(chunk_files) 

        for i, p in enumerate(files_to_process):
            try:
                df = pd.read_csv(p, low_memory=False)
                if df.empty: continue

                df['SourceFile'] = p
                if "VehicleID" not in df.columns:
                    df["VehicleID"] = os.path.basename(os.path.dirname(p)).split("_",1)[0]
                
                for col in MASTER_COLUMNS:
                    if col not in df.columns: df[col] = pd.NA

                df = df[MASTER_COLUMNS].astype(MASTER_DTYPES)
                frames_for_chunk.append(df)

                if len(frames_for_chunk) >= CHUNK_SIZE:
                    chunk_df = pd.concat(frames_for_chunk, ignore_index=True)
                    chunk_path = os.path.join(temp_dir, f"chunk_{chunk_num}.parquet")
                    chunk_df.to_parquet(chunk_path, index=False)
                    print(f"  ... Wrote chunk {chunk_num} with {len(chunk_df)} rows.")
                    frames_for_chunk = [] 
                    chunk_num += 1
            except Exception as e:
                print(f"Skipping file {p} due to error: {e}")
                continue

        if frames_for_chunk:
            chunk_df = pd.concat(frames_for_chunk, ignore_index=True)
            chunk_path = os.path.join(temp_dir, f"chunk_{chunk_num}.parquet")
            chunk_df.to_parquet(chunk_path, index=False)
            print(f"  ... Wrote final chunk {chunk_num} with {len(chunk_df)} rows.")

    # --- 4. NEW: Memory-Efficient Merge Logic ---
    print(f"\nAll chunks for {month_prefix} processed. Merging...")
    chunk_files = sorted(glob(os.path.join(temp_dir, "chunk_*.parquet")))

    if not chunk_files:
        print(f"No data chunks found to merge for {month_prefix}.")
    else:
        # Delete any old/corrupted final file before starting the merge
        if os.path.exists(STAGING_FILE):
            os.remove(STAGING_FILE)
            
        schema = pq.read_schema(chunk_files[0])
        with pq.ParquetWriter(STAGING_FILE, schema, compression='snappy') as writer:
            for chunk_file in chunk_files:
                print(f"  ... Merging {os.path.basename(chunk_file)}")
                
                # This reads the chunk file in smaller pieces to save RAM
                p_file = pq.ParquetFile(chunk_file)
                for i in range(p_file.num_row_groups):
                    table = p_file.read_row_group(i)
                    writer.write_table(table)

        print(f"\nConsolidation complete for {month_prefix}. File created at: {STAGING_FILE}")

print("\n" + "="*50)
print("All monthly batch jobs complete.")
print("="*50)