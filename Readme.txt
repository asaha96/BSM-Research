README: Bridge Crossing Analysis Pipeline
This project contains scripts designed to analyze a large dataset of vehicle GPS CSV files to identify North Avenue bridge crossings, generate statistics, and create verification plots.

The pipeline is divided into two main parts:

Data Processing (consolidate_data.py, export_bridge_crossings_dual.py)

Post-Analysis (analyze_results.py, plot_verification_maps.py)

Scripts Overview ??
Data Processing
consolidate_data.py (Data Consolidation)

Purpose: Reads all raw CSV files from the DATA_DIR, cleans them, enforces a consistent structure, and saves the data into separate Parquet files for each month (from September 2024 to August 2025).

Why: This pre-processing step creates efficient, standardized data files that are much faster to analyze. It processes data in memory-safe chunks and is resumable.

How to Run: python consolidate_data.py. This script needs to be run only once or whenever new raw CSV data is added. It will take a long time to complete.

export_bridge_crossings_dual.py (Bridge Crossing Analysis)

Purpose: Reads the monthly Parquet files created by the first script, analyzes each vehicle's trajectory using defined geographic and bearing rules, and identifies West-to-East bridge crossings.

Why: This script performs the core scientific analysis. It uses multiprocessing (NUM_WORKERS) to speed up the analysis of the large monthly files.

How to Run: python export_bridge_crossings_dual.py. This script should be run after consolidate_data.py has successfully completed.

Post-Analysis & Visualization
analyze_results.py (Statistical Analysis)

Purpose: Reads the final summary CSV (...crossings_ALL_MONTHS...csv) to generate statistics and graphs.

Why: Calculates key metrics like total crossings per month, peak crossing times, and crossing duration.

How to Run: python analyze_results.py. Run this after the main analysis script is complete.

plot_verification_maps.py (Visual Verification)

Purpose: Reads both the summary and detailed rows CSVs to generate visual plots of individual crossings.

Why: Creates map plots for each crossing in a specified month (e.g., Sept. 2024) to allow for manual visual verification.

How to Run: python plot_verification_maps.py. Run this after the main analysis script is complete.

Workflow ??
Run consolidate_data.py first. Wait for it to complete processing all months.

Run export_bridge_crossings_dual.py second. This will analyze the consolidated data and produce the two final results CSVs.

Run the post-analysis scripts (analyze_results.py and plot_verification_maps.py) to generate statistics, graphs, and visual plots from the final CSV files.

Configuration ??
Key parameters are set at the top of each script:

DATA_DIR: Location of raw CSV files (used by consolidate_data.py).

OUTPUT_DIR: Where all output files (monthly Parquets, final CSVs, plots) are saved.

NUM_WORKERS: Number of parallel processes for the analysis script (adjust based on server memory).

Geographic Box & Crossing Rules: (LAT_MIN, WE_MIN_DEG, etc.) are defined in export_bridge_crossings_dual.py.

Analysis Scripts: The plotting and analysis scripts (analyze_results.py, plot_verification_maps.py) are configured to find the final ...ALL_MONTHS... CSV files in the OUTPUT_DIR.

Expected Outputs ??
1. Intermediate Monthly Parquet Files
From: consolidate_data.py

Location: OUTPUT_DIR

Format: all_vehicle_data_YYYY_MM.parquet (e.g., all_vehicle_data_2024_09.parquet)

Content: Cleaned and standardized vehicle data for one specific month.

2. Final Crossing CSV Files
From: export_bridge_crossings_dual.py

Location: OUTPUT_DIR

Files:

north_ave_bridge_crossings_ALL_MONTHS_...csv: A summary file where each row represents one detected bridge crossing (Vehicle ID, Start/End Time, etc.).

north_ave_bridge_crossing_rows_ALL_MONTHS_...csv: A detailed file containing the original GPS data points (Lat, Lon, Speed, etc.) for only the segments identified as valid crossings.

3. Analysis Plots and Graphs
From: analyze_results.py & plot_verification_maps.py

Location: OUTPUT_DIR/Analysis_Plots and OUTPUT_DIR/Verification_Plots_Sept_2024

Files:

crossings_per_month_bar_chart.png: Bar chart of total crossings per month.

crossing_duration_histogram.png: Histogram of how long crossings take.

crossing_plot_...png: A set of individual map plots showing the GPS path for each verified crossing.