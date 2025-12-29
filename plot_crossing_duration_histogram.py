import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
from glob import glob
import numpy as np

# --- CONFIG ---
OUTPUT_DIR = os.environ.get("BSM_OUTPUT_DIR", os.path.join(os.getcwd(), "data", "analysis"))
PLOT_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "Analysis_Plots_Final_2K")
os.makedirs(PLOT_OUTPUT_DIR, exist_ok=True)

# --- 2K RESOLUTION SETTINGS ---
DPI = 144
WIDTH_PX = 2560
HEIGHT_PX = 1440
FIG_SIZE_2K = (WIDTH_PX / DPI, HEIGHT_PX / DPI)

# --- VEHICLE ID LISTS ---
AMBULANCE_IDS = {
    '07902', '07917', '07909', '07912', '07906', '07904', '07915', '04932', 
    '07923', '07907', '07800', '07922', '07916', '04779', '07910', '07908', 
    '07920', '07913', '07924', '07918', '07905', '07921', '07454', '07452', 
    '07450', '07451', '06788', '07463', '10513', '04180', '04870', '04967', 
    '07919', '07455', '04874', '07461', '04872', '04888', '04866', '07458', 
    '07462', '07459', '07460', '07447', '07453', '04775', '07901', '04876', 
    '04963', '04875', '04965', '04966', '04962', '04777', '04784', '07449', 
    '07903', '07448'
}

HERO_IDS_SUFFIXES = {
    '0481', '0482', '0483', '0484', '0485', '0486', '0487', '0488', '0489', '0490',
    '0491', '0492', '0493', '0494', '0495', '0496', '0497', '0498', '0499', '0500',
    '0501', '0502', '0503', '0504', '0505', '0506', '0507', '0508', '0509', '0510',
    '0516', '0517', '0518', '0520', '0521', '0524', '0526', '0527', '0531', '0532',
    '0533', '0535', '0536', '0537', '0538', '0539', '0540', '0541', '0542', '0543',
    '0544', '0545', '0546', '0547', '0548', '0549', '0550', '0551', '04E548500F58'
}

# --- DATA PROCESSING FUNCTIONS ---

def calculate_bearing_vectorized(lat_arr, lon_arr, next_lat_arr, next_lon_arr):
    lat1, lon1 = np.radians(lat_arr), np.radians(lon_arr)
    lat2, lon2 = np.radians(next_lat_arr), np.radians(next_lon_arr)
    dlon = lon2 - lon1
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - (np.sin(lat1) * np.cos(lat2) * np.cos(dlon))
    return (np.degrees(np.arctan2(x, y)) + 360) % 360

def normalize_id(vid):
    return str(vid).strip().lstrip('0')

def apply_duration_correction(summary_df):
    rows_file = os.path.join(OUTPUT_DIR, "north_ave_bridge_crossing_rows_ALL_MONTHS_20251028.csv")
    if not os.path.exists(rows_file):
        print("Error: Rows file not found.")
        return summary_df

    print("Loading detailed rows (Heading Correction)...")
    cols = ['VehicleID', 'DateTime_UTC', 'Latitude', 'Longitude', 'Speed_mps']
    try:
        df_rows = pd.read_csv(rows_file, usecols=cols, low_memory=False)
        df_rows['DateTime_UTC'] = pd.to_datetime(df_rows['DateTime_UTC'], utc=True)
        df_rows['VehicleID'] = df_rows['VehicleID'].astype(str).str.strip()
        summary_df['VehicleID'] = summary_df['VehicleID'].astype(str).str.strip()
    except Exception as e:
        print(f"Error loading rows: {e}")
        return summary_df

    print(f"  Processing {len(df_rows)} GPS points...")
    df_rows.sort_values(['VehicleID', 'DateTime_UTC'], inplace=True)
    
    bearings = calculate_bearing_vectorized(
        df_rows['Latitude'].values, df_rows['Longitude'].values,
        df_rows['Latitude'].shift(-1).values, df_rows['Longitude'].shift(-1).values
    )
    df_rows['Heading'] = bearings
    mask_invalid = (df_rows['VehicleID'] != df_rows['VehicleID'].shift(-1))
    df_rows.loc[mask_invalid, 'Heading'] = np.nan
    df_rows['Heading'] = df_rows['Heading'].ffill()
    
    tol = 40
    is_bridge = (
        ((df_rows['Heading'] >= (90 - tol)) & (df_rows['Heading'] <= (90 + tol))) |
        ((df_rows['Heading'] >= (270 - tol)) & (df_rows['Heading'] <= (270 + tol))) |
        (df_rows['Speed_mps'] < 1.0)
    )
    clean_rows = df_rows[is_bridge].copy()
    
    print("  Normalizing IDs...")
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
    del summary_df['MatchID']
    return summary_df

def classify_vehicle(vid):
    vid_str = str(vid).strip()
    if vid_str in AMBULANCE_IDS: return "Ambulance"
    if vid_str in HERO_IDS_SUFFIXES: return "HERO"
    padded_4 = vid_str.zfill(4)
    if padded_4 in HERO_IDS_SUFFIXES: return "HERO"
    padded_5 = vid_str.zfill(5)
    if padded_5 in AMBULANCE_IDS: return "Ambulance"
    if vid_str.startswith("452-"):
        suffix = vid_str.split("-")[1]
        if suffix in HERO_IDS_SUFFIXES: return "HERO"
    return "Other"

# --- VISUALIZATION FUNCTIONS ---

def setup_2k_plot():
    """Returns figure and axis with 2K settings."""
    fig, ax = plt.subplots(figsize=FIG_SIZE_2K, dpi=DPI)
    return fig, ax

def add_violin_stats(ax, data_list, positions):
    """Adds Med/Q1/Q3 text labels to violin plots."""
    for i, data in enumerate(data_list):
        if len(data) == 0: continue
        
        quantiles = data.quantile([0.25, 0.5, 0.75])
        q1, median, q3 = quantiles[0.25], quantiles[0.5], quantiles[0.75]
        
        x_pos = positions[i]
        ax.scatter(x_pos, median, color='white', s=60, zorder=3, edgecolors='black')
        ax.vlines(x_pos, q1, q3, color='black', linestyle='-', lw=1.5, zorder=3)
        
        font_props = dict(fontsize=12, fontweight='bold', color='#333333')
        bbox_props = dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.6)
        
        offset = 0.15
        ax.text(x_pos + offset, median, f'Med: {median:.0f}', va='center', bbox=bbox_props, **font_props)
        ax.text(x_pos + offset, q3, f'Q3: {q3:.0f}', va='bottom', bbox=bbox_props, fontsize=11)
        ax.text(x_pos + offset, q1, f'Q1: {q1:.0f}', va='top', bbox=bbox_props, fontsize=11)

def filter_outliers(series):
    """Returns series without upper outliers (Q3 + 1.5*IQR)."""
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    upper_fence = q3 + 1.5 * iqr
    return series[series <= upper_fence]

def main():
    print("Starting Analysis...")
    summary_files = glob(os.path.join(OUTPUT_DIR, "north_ave_bridge_crossings_ALL_MONTHS.csv"))
    if not summary_files: return
    
    df = pd.read_csv(summary_files[0])
    df['StartUTC'] = pd.to_datetime(df['StartUTC'], utc=True)
    df['EndUTC'] = pd.to_datetime(df['EndUTC'], utc=True)
    
    start_date = pd.Timestamp("2024-09-01", tz='UTC')
    end_date = pd.Timestamp("2025-08-31 23:59:59.999999", tz='UTC')
    df = df[(df['StartUTC'] >= start_date) & (df['StartUTC'] <= end_date)].copy()
    
    # Data Correction
    df = apply_duration_correction(df)
    
    # Filter Noise (<5s)
    df_final = df[df['Duration_sec'] > 5.0].copy()
    df_final['VehicleClass'] = df_final['VehicleID'].apply(classify_vehicle)
    df_final['Month'] = df_final['StartUTC'].dt.to_period('M')
    
    print(f"Final Valid Crossings: {len(df_final)}")
    
    # =========================================================================
    # 1. Monthly Counts (Stacked Bar)
    # =========================================================================
    print("Generating Graph 1: Monthly Counts...")
    monthly_counts = df_final.groupby(['Month', 'VehicleClass']).size().unstack(fill_value=0)
    for c in ['HERO', 'Ambulance']: 
        if c not in monthly_counts: monthly_counts[c] = 0
    monthly_counts = monthly_counts[['HERO', 'Ambulance']]

    fig, ax = setup_2k_plot()
    monthly_counts.plot(kind='bar', stacked=True, ax=ax, color=['#1f77b4', '#ff7f0e'], width=0.7)
    
    ax.set_title('Bridge Crossings Per Month (Count)', fontsize=24, fontweight='bold', pad=20)
    ax.set_ylabel('Number of Crossings', fontsize=18)
    ax.set_xlabel('Month', fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=14)
    plt.xticks(rotation=0)
    ax.legend(title='Vehicle Type', fontsize=16, title_fontsize=18)

    # Labels (Top)
    totals = monthly_counts.sum(axis=1)
    y_off = totals.max() * 0.01
    for i, total in enumerate(totals):
        if total > 0: ax.text(i, total + y_off, str(int(total)), ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    # Labels (Inside Segments)
    for c in ax.containers:
        labels = [str(int(v.get_height())) if v.get_height() > 0 else '' for v in c]
        ax.bar_label(c, labels=labels, label_type='center', color='white', fontweight='bold', fontsize=12)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_OUTPUT_DIR, "1_monthly_counts_2k.png"), dpi=DPI)
    plt.close()

    # =========================================================================
    # 2. Monthly Total Seconds (Stacked Bar)
    # =========================================================================
    print("Generating Graph 2: Total Seconds per Month...")
    monthly_sec = df_final.groupby(['Month', 'VehicleClass'])['Duration_sec'].sum().unstack(fill_value=0)
    for c in ['HERO', 'Ambulance']: 
        if c not in monthly_sec: monthly_sec[c] = 0
    monthly_sec = monthly_sec[['HERO', 'Ambulance']]

    fig, ax = setup_2k_plot()
    monthly_sec.plot(kind='bar', stacked=True, ax=ax, color=['#1f77b4', '#ff7f0e'], width=0.7)
    
    ax.set_title('Total Seconds of Travel Per Month', fontsize=24, fontweight='bold', pad=20)
    ax.set_ylabel('Total Duration (Seconds)', fontsize=18)
    ax.set_xlabel('Month', fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=14)
    plt.xticks(rotation=0)
    ax.legend(title='Vehicle Type', fontsize=16, title_fontsize=18)

    # Labels (Top)
    totals = monthly_sec.sum(axis=1)
    y_off = totals.max() * 0.01
    for i, total in enumerate(totals):
        if total > 0: ax.text(i, total + y_off, f"{int(total):,}", ha='center', va='bottom', fontsize=13, fontweight='bold')
    
    # Labels (Inside Segments) - NEW ADDITION
    for c in ax.containers:
        # Use comma formatting for large numbers
        labels = [f"{int(v.get_height()):,}" if v.get_height() > 0 else '' for v in c]
        # Use slightly smaller font since numbers might be wide
        ax.bar_label(c, labels=labels, label_type='center', color='white', fontweight='bold', fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_OUTPUT_DIR, "2_monthly_total_seconds_2k.png"), dpi=DPI)
    plt.close()

    # =========================================================================
    # 3. Distribution Histogram (Detailed Labels)
    # =========================================================================
    print("Generating Graph 3: Distribution Histogram...")
    df_plot = df_final[df_final['VehicleClass'].isin(['HERO', 'Ambulance'])]
    heros = df_plot[df_plot['VehicleClass'] == 'HERO']['Duration_sec']
    ambs = df_plot[df_plot['VehicleClass'] == 'Ambulance']['Duration_sec']

    fig, ax = setup_2k_plot()
    
    bins = np.arange(0, 601, 15)
    
    counts, bin_edges, patches = ax.hist([heros, ambs], bins=bins, stacked=True, 
                                         color=['#1f77b4', '#ff7f0e'], label=['HERO', 'Ambulance'], 
                                         edgecolor='black', alpha=0.9)
    
    ax.set_title('Distribution of Crossing Durations (Detailed)', fontsize=24, fontweight='bold', pad=20)
    ax.set_xlabel('Duration (Seconds)', fontsize=18)
    ax.set_ylabel('Frequency', fontsize=18)
    ax.set_xlim(0, 600) 
    
    ax.set_xticks(np.arange(0, 601, 15))
    ax.tick_params(axis='x', rotation=45, labelsize=12)
    ax.tick_params(axis='y', labelsize=14)
    
    ax.legend(fontsize=16)
    ax.grid(axis='y', alpha=0.3)

    # Label Totals (Top)
    for bar in patches[-1]:
        height = bar.get_y() + bar.get_height()
        if height > 0:
            ax.text(bar.get_x() + bar.get_width()/2, height, str(int(height)), 
                    ha='center', va='bottom', fontweight='bold', fontsize=10)

    # Label Segments (Inside) - NEW ADDITION
    for dataset_patches in patches: # Loop through [HERO patches, Ambulance patches]
        for bar in dataset_patches:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_y() + height/2, 
                        str(int(height)), ha='center', va='center', 
                        color='white', fontweight='bold', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_OUTPUT_DIR, "3_duration_histogram_2k.png"), dpi=DPI)
    plt.close()

    # =========================================================================
    # 4. Violin Plots (Outliers Removed)
    # =========================================================================
    print("Generating Graph 4: Violin Plots (No Outliers)...")
    fig, ax = setup_2k_plot()
    
    heros_clean = filter_outliers(heros)
    ambs_clean = filter_outliers(ambs)
    
    data_clean = [heros_clean, ambs_clean]
    labels = ['HERO', 'Ambulance']
    
    parts = ax.violinplot(data_clean, showmeans=False, showmedians=False, showextrema=False)

    colors = ['#1f77b4', '#ff7f0e']
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_edgecolor('black')
        pc.set_alpha(0.7)

    positions = [1, 2]
    add_violin_stats(ax, data_clean, positions)

    ax.set_title('Crossing Duration Statistics (Violin Plot - No Outliers)', fontsize=24, fontweight='bold', pad=20)
    ax.set_ylabel('Duration (Seconds)', fontsize=18)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=18)
    ax.tick_params(axis='y', labelsize=14)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_OUTPUT_DIR, "4_duration_violin_2k.png"), dpi=DPI)
    plt.close()

    print("All 2K charts generated successfully.")

if __name__ == "__main__":
    main()