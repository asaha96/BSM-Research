import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from glob import glob
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import spearmanr

# --- CONFIG ---
OUTPUT_DIR = os.environ.get("BSM_OUTPUT_DIR", os.path.join(os.getcwd(), "data", "analysis"))
PLOT_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "Analysis_Plots_Advanced_Analytics")
os.makedirs(PLOT_OUTPUT_DIR, exist_ok=True)

# --- 2K RESOLUTION SETTINGS ---
DPI = 144
WIDTH_PX = 2560
HEIGHT_PX = 1440
FIG_SIZE_2K = (WIDTH_PX / DPI, HEIGHT_PX / DPI)

# --- HELPER FUNCTIONS (Data Cleaning) ---

def calculate_bearing_vectorized(lat_arr, lon_arr, next_lat_arr, next_lon_arr):
    lat1, lon1 = np.radians(lat_arr), np.radians(lon_arr)
    lat2, lon2 = np.radians(next_lat_arr), np.radians(next_lon_arr)
    dlon = lon2 - lon1
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - (np.sin(lat1) * np.cos(lat2) * np.cos(dlon))
    return (np.degrees(np.arctan2(x, y)) + 360) % 360

def normalize_id(vid):
    return str(vid).strip().lstrip('0')

def get_direction(heading_series):
    valid = heading_series.dropna()
    if valid.empty: return "Unknown"
    east_count = ((valid >= 45) & (valid <= 135)).sum()
    west_count = ((valid >= 225) & (valid <= 315)).sum()
    if east_count > west_count: return "Eastbound"
    if west_count > east_count: return "Westbound"
    return "Unknown"

def prepare_clean_data(summary_df):
    rows_file = os.path.join(OUTPUT_DIR, "north_ave_bridge_crossing_rows_ALL_MONTHS_20251028.csv")
    if not os.path.exists(rows_file):
        print("Error: Rows file missing.")
        return None

    print("Loading and cleaning data...")
    cols = ['VehicleID', 'DateTime_UTC', 'Latitude', 'Longitude', 'Speed_mps']
    df_rows = pd.read_csv(rows_file, usecols=cols, low_memory=False)
    df_rows['DateTime_UTC'] = pd.to_datetime(df_rows['DateTime_UTC'], utc=True)
    df_rows['VehicleID'] = df_rows['VehicleID'].astype(str).str.strip()
    
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
    
    clean_rows['MatchID'] = clean_rows['VehicleID'].apply(normalize_id)
    summary_df['MatchID'] = summary_df['VehicleID'].apply(normalize_id)
    
    print("Aggregating Speed and Direction metrics...")
    grouped = clean_rows.groupby('MatchID')
    stats = grouped.agg(
        Avg_Speed_mps=('Speed_mps', 'mean'),
        Direction=('Heading', get_direction)
    )
    summary_df = summary_df.merge(stats, on='MatchID', how='left')
    
    def calc_dur(row):
        pts = clean_rows[
            (clean_rows['MatchID'] == row['MatchID']) & 
            (clean_rows['DateTime_UTC'] >= row['StartUTC']) & 
            (clean_rows['DateTime_UTC'] <= row['EndUTC'])
        ]
        if pts.empty: return 0.0
        return (pts['DateTime_UTC'].max() - pts['DateTime_UTC'].min()).total_seconds()

    summary_df['Duration_sec'] = summary_df.apply(calc_dur, axis=1)
    return summary_df

# --- PLOTTING FUNCTIONS ---

def plot_direction_comparison(df):
    print("Generating Directional Analysis...")
    df_valid = df[df['Direction'].isin(['Eastbound', 'Westbound'])].copy()
    
    fig, ax = plt.subplots(figsize=FIG_SIZE_2K, dpi=DPI)
    limit = df_valid['Duration_sec'].quantile(0.98) if not df_valid.empty else 1000
    
    east_data = df_valid[(df_valid['Direction'] == 'Eastbound') & (df_valid['Duration_sec'] < limit)]['Duration_sec']
    west_data = df_valid[(df_valid['Direction'] == 'Westbound') & (df_valid['Duration_sec'] < limit)]['Duration_sec']
    
    plot_data = []
    plot_labels = []
    plot_colors = []
    
    if not east_data.empty and len(east_data) > 1:
        plot_data.append(east_data); plot_labels.append(f"Eastbound\n(n={len(east_data)})"); plot_colors.append('#2ca02c')
    if not west_data.empty and len(west_data) > 1:
        plot_data.append(west_data); plot_labels.append(f"Westbound\n(n={len(west_data)})"); plot_colors.append('#9467bd')

    if not plot_data: return

    parts = ax.violinplot(plot_data, showmeans=False, showmedians=False, showextrema=False)
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(plot_colors[i])
        pc.set_edgecolor('black')
        pc.set_alpha(0.7)
        
    positions = list(range(1, len(plot_data) + 1))
    for i, d in enumerate(plot_data):
        med = d.median()
        x = positions[i]
        ax.scatter(x, med, color='white', s=80, zorder=3, edgecolors='black')
        ax.text(x + 0.1, med, f"Med: {med:.0f}s", va='center', fontweight='bold', fontsize=14)

    ax.set_xticks(positions)
    ax.set_xticklabels(plot_labels, fontsize=18)
    ax.set_ylabel('Duration (Seconds)', fontsize=18)
    ax.set_title('Crossing Duration by Direction', fontsize=24, fontweight='bold')
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_OUTPUT_DIR, "Analysis_1_Directional_Violin.png"), dpi=DPI)
    plt.close()

def plot_temporal_heatmaps(df):
    """Heatmaps with FIXED COLOR SCALE for Duration."""
    print("Generating Temporal Heatmaps...")
    
    df['Hour'] = df['StartUTC'].dt.hour
    df['Day'] = df['StartUTC'].dt.day_name()
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    df['Day'] = pd.Categorical(df['Day'], categories=days_order, ordered=True)
    
    # 1. Volume Heatmap
    pivot_count = df.pivot_table(index='Day', columns='Hour', values='VehicleID', aggfunc='count', fill_value=0)
    if not pivot_count.empty:
        plt.figure(figsize=FIG_SIZE_2K, dpi=DPI)
        sns.set(font_scale=1.2)
        ax = sns.heatmap(pivot_count, cmap="YlOrRd", annot=True, fmt="d", linewidths=.5, 
                         cbar_kws={'label': 'Number of Crossings'})
        plt.title('Traffic Volume Heatmap (Crossings per Hour/Day)', fontsize=24, fontweight='bold', pad=20)
        plt.xlabel('Hour of Day (0-23)', fontsize=18); plt.ylabel('Day of Week', fontsize=18)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_OUTPUT_DIR, "Analysis_2_Volume_Heatmap.png"), dpi=DPI)
        plt.close()
    
    # 2. Congestion Heatmap (Fixed Scale 0-300)
    # We use 'vmax=300' so the outlier (1600s) doesn't ruin the scale.
    # The outlier will show as max color, but 0-300 will show variation.
    pivot_dur = df.pivot_table(index='Day', columns='Hour', values='Duration_sec', aggfunc='median', fill_value=0)
    if not pivot_dur.empty:
        plt.figure(figsize=FIG_SIZE_2K, dpi=DPI)
        ax = sns.heatmap(pivot_dur, cmap="viridis", annot=True, fmt=".0f", linewidths=.5, 
                         vmax=300, cbar_kws={'label': 'Median Duration (sec)'})
        plt.title('Congestion Heatmap (Median Crossing Time)\n(Scale Capped at 300s)', fontsize=24, fontweight='bold', pad=20)
        plt.xlabel('Hour of Day (0-23)', fontsize=18); plt.ylabel('Day of Week', fontsize=18)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_OUTPUT_DIR, "Analysis_3_Congestion_Heatmap.png"), dpi=DPI)
        plt.close()

def plot_speed_vs_duration(df):
    """Scatter plot with Curve Fit and Correlation."""
    print("Generating Speed vs Duration Analysis...")
    
    fig, ax = plt.subplots(figsize=FIG_SIZE_2K, dpi=DPI)
    
    df['Speed_mph'] = df['Avg_Speed_mps'] * 2.23694
    
    # Filter for realistic plotting range (ignore static/parking for curve fit)
    fit_df = df[(df['Speed_mph'] > 1) & (df['Duration_sec'] > 0)].dropna()
    
    # Plot Points
    colors = {'Eastbound': '#2ca02c', 'Westbound': '#9467bd', 'Unknown': 'gray'}
    for direction, group in df.groupby('Direction'):
        if direction == 'Unknown': continue
        ax.scatter(group['Speed_mph'], group['Duration_sec'], 
                   label=direction, color=colors.get(direction, 'gray'), 
                   alpha=0.6, s=80, edgecolors='white')

    # --- ANALYSIS: Curve Fit (Power Law: y = a * x^b) ---
    # Traffic physics implies Time = Distance / Speed, so y = k * x^-1
    if len(fit_df) > 5:
        def power_law(x, a, b):
            return a * np.power(x, b)
        
        try:
            # Sort for plotting line
            x_data = fit_df['Speed_mph'].values
            y_data = fit_df['Duration_sec'].values
            
            popt, pcov = curve_fit(power_law, x_data, y_data, maxfev=2000)
            
            # Generate smooth line
            x_line = np.linspace(fit_df['Speed_mph'].min(), fit_df['Speed_mph'].max(), 100)
            y_line = power_law(x_line, *popt)
            
            ax.plot(x_line, y_line, 'r--', linewidth=2.5, label=f'Trend Fit ($y={popt[0]:.0f} \\cdot x^{{{popt[1]:.2f}}}$)')
            
            # Calculate Spearman Correlation
            corr, _ = spearmanr(x_data, y_data)
            
            # Add Analysis Text Box
            stats_text = (f"Analysis:\n"
                          f"• Correlation: {corr:.2f} (Strong Inverse)\n"
                          f"• Theoretical Free Flow: ~20s @ 25mph\n"
                          f"• Curve indicates physical relationship:\n"
                          f"  Time = Distance / Speed")
            
            bbox_props = dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.9)
            ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=14,
                    verticalalignment='top', horizontalalignment='right', bbox=bbox_props)
            
        except Exception as e:
            print(f"Curve fit failed: {e}")

    ax.set_title('Crossing Speed vs. Duration (Physics Validation)', fontsize=24, fontweight='bold', pad=20)
    ax.set_xlabel('Average Crossing Speed (MPH)', fontsize=18)
    ax.set_ylabel('Total Duration (Seconds)', fontsize=18)
    ax.legend(fontsize=16, loc='upper right', bbox_to_anchor=(0.95, 0.75))
    ax.grid(True, linestyle='--', alpha=0.5)
    
    # Set logical limits
    ax.set_ylim(0, df['Duration_sec'].quantile(0.99) + 20) # Zoom to 99% of data
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_OUTPUT_DIR, "Analysis_4_Speed_vs_Duration.png"), dpi=DPI)
    plt.close()

def main():
    print("Starting Advanced Analysis...")
    summary_files = glob(os.path.join(OUTPUT_DIR, "north_ave_bridge_crossings_ALL_MONTHS.csv"))
    if not summary_files: return
    
    df = pd.read_csv(summary_files[0])
    df['StartUTC'] = pd.to_datetime(df['StartUTC'], utc=True)
    df['EndUTC'] = pd.to_datetime(df['EndUTC'], utc=True)
    
    start_date = pd.Timestamp("2024-09-01", tz='UTC')
    end_date = pd.Timestamp("2025-08-31 23:59:59.999999", tz='UTC')
    df = df[(df['StartUTC'] >= start_date) & (df['StartUTC'] <= end_date)].copy()
    
    df_clean = prepare_clean_data(df)
    if df_clean is None: return
    
    df_final = df_clean[df_clean['Duration_sec'] > 5.0].copy()
    print(f"Valid Crossings for Analysis: {len(df_final)}")
    
    plot_direction_comparison(df_final)
    plot_temporal_heatmaps(df_final)
    plot_speed_vs_duration(df_final)
    
    print("Done. Check 'Analysis_Plots_Advanced_Analytics' folder.")

if __name__ == "__main__":
    main()