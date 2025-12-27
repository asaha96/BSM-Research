# BSM Research — GDOT CV2X Bridge Crossing Analytics

Streamlit dashboard + analysis pipeline for identifying and reviewing **North Avenue Bridge** crossings from CV2X trajectory data (Georgia DOT).

## What’s in this repo

- **Dashboard (Streamlit)**: `visualization.py`
  - Dark “glass” UI
  - Fast loading mode using a consolidated crossings file
  - Filters (date/time, zones, vehicle type/ID) and analytics plots
  - “Confirmed crossings” table + monthly rollup
- **Batch processing pipeline**
  - `consolidate_data.py`: consolidate raw CSVs into monthly Parquet files
  - `export_bridge_crossings_dual.py`: apply crossing rules and export results
  - `analyze_results.py`, `plot_verification_maps.py`: post-analysis and QA plots

## Data sources

This project supports two data sources:

- **Consolidated (fast)** *(recommended for the dashboard)*:
  - `data/north_ave_bridge_crossing_rows_ALL_MONTHS_20251028.csv`
  - Used to stream/filter in chunks so the app loads quickly.
- **Raw folder**:
  - The dashboard can scan a folder of raw CSV files, but it’s slower.

> Note: The consolidated file contains **North Ave rows only**. Study Area / Ramp zone toggles are ignored in that mode.

## Quickstart

### 1) Create a virtual environment

```bash
cd "BSM Research"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Run the dashboard

```bash
source .venv/bin/activate
streamlit run visualization.py
```

Open the URL printed by Streamlit (typically `http://127.0.0.1:8501`).

## Configuration

The dashboard looks for data in `./data` by default, and can be configured using environment variables:

- **`BSM_DATA_DIR`**: directory containing raw CSVs (raw folder mode)
- **`BSM_CONSOLIDATED_CSV`**: path to consolidated crossings CSV

Example:

```bash
export BSM_DATA_DIR="/path/to/data"
export BSM_CONSOLIDATED_CSV="/path/to/north_ave_bridge_crossing_rows_ALL_MONTHS_20251028.csv"
streamlit run visualization.py
```

## Typical workflow (pipeline)

1. **Consolidate raw CSVs → monthly Parquet**

```bash
python consolidate_data.py
```

2. **Detect crossings**

```bash
python export_bridge_crossings_dual.py
```

3. **Post-analysis**

```bash
python analyze_results.py
python plot_verification_maps.py
```

## Repo hygiene notes

- This repo’s `.gitignore` excludes `data/` and `.venv/` to keep the GitHub repo lightweight and avoid committing environment-specific files.
- If you want to commit small sample data for demos, add a `data_sample/` folder and keep it under ~10–20MB.

## License

Internal

