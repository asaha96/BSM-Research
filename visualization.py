import streamlit as st
import pandas as pd
import pydeck as pdk
import os, math, datetime
import altair as alt
from glob import glob
import numpy as np
import colorsys, zlib  # [COLOR MAP]

# =============================
# PAGE CONFIG - Must be first Streamlit command
# =============================
st.set_page_config(
    page_title="GDOT CV2X Bridge Crossing Analytics",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "Georgia DOT Connected Vehicle Bridge Crossing Analysis Dashboard"
    }
)

# =============================
# DARK MODE GLASSMORPHISM CSS
# =============================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    :root {
        --bg-base: #0a0a0c;
        --bg-elevated: #141418;
        --glass-tint: rgba(255, 255, 255, 0.03);
        --glass-border: rgba(255, 255, 255, 0.08);
        --glass-glow: rgba(255, 255, 255, 0.12);
        --text-primary: #f5f5f7;
        --text-secondary: rgba(255, 255, 255, 0.6);
        --text-muted: rgba(255, 255, 255, 0.35);
        --accent: #3b82f6;
        --accent-glow: rgba(59, 130, 246, 0.4);
        --success: #22c55e;
        --success-dim: rgba(34, 197, 94, 0.15);
    }
    
    .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        background: var(--bg-base);
        color: var(--text-primary);
    }
    
    /* Glass Panel Mixin Style */
    .glass-panel {
        background: var(--glass-tint);
        backdrop-filter: blur(40px) saturate(150%);
        -webkit-backdrop-filter: blur(40px) saturate(150%);
        border: 1px solid var(--glass-border);
        border-radius: 16px;
    }
    
    /* Header */
    .main-header {
        background: var(--glass-tint);
        backdrop-filter: blur(40px) saturate(150%);
        -webkit-backdrop-filter: blur(40px) saturate(150%);
        border: 1px solid var(--glass-border);
        border-radius: 20px;
        padding: 2rem 2.5rem;
        margin: 0 0 2rem 0;
    }
    
    .main-header h1 {
        color: var(--text-primary);
        font-size: 1.6rem;
        font-weight: 600;
        margin: 0;
        letter-spacing: -0.3px;
    }
    
    .main-header p {
        color: var(--text-secondary);
        font-size: 0.9rem;
        font-weight: 400;
        margin: 0.4rem 0 0 0;
    }
    
    .org-badge {
        display: inline-block;
        background: var(--glass-tint);
        border: 1px solid var(--glass-border);
        color: var(--text-secondary);
        padding: 0.35rem 0.9rem;
        border-radius: 100px;
        font-size: 0.7rem;
        font-weight: 500;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        margin-top: 1rem;
    }
    
    /* Metric Cards */
    .metric-card {
        background: var(--glass-tint);
        backdrop-filter: blur(40px) saturate(150%);
        -webkit-backdrop-filter: blur(40px) saturate(150%);
        border: 1px solid var(--glass-border);
        border-radius: 16px;
        padding: 1.25rem 1rem;
        text-align: center;
        transition: all 0.25s ease;
    }
    
    .metric-card:hover {
        border-color: var(--glass-glow);
        background: rgba(255, 255, 255, 0.05);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 600;
        color: var(--text-primary);
        line-height: 1.1;
        letter-spacing: -0.5px;
    }
    
    .metric-label {
        font-size: 0.7rem;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-top: 0.5rem;
        font-weight: 500;
    }
    
    /* Section Headers */
    .section-header {
        color: var(--text-primary);
        font-size: 1.1rem;
        font-weight: 600;
        letter-spacing: -0.2px;
        margin: 2.5rem 0 1rem 0;
        padding-bottom: 0.6rem;
        border-bottom: 1px solid var(--glass-border);
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: var(--bg-elevated) !important;
        border-right: 1px solid var(--glass-border);
    }
    
    [data-testid="stSidebar"] > div:first-child {
        background: transparent;
        padding-top: 1.5rem;
    }
    
    .sidebar-header {
        text-align: center;
        padding: 0 1rem 1.25rem 1rem;
        border-bottom: 1px solid var(--glass-border);
        margin-bottom: 1.25rem;
    }
    
    .sidebar-header h2 {
        color: var(--text-primary);
        font-size: 1rem;
        font-weight: 600;
        margin: 0;
    }
    
    .sidebar-header p {
        color: var(--text-muted);
        font-size: 0.75rem;
        margin: 0.2rem 0 0 0;
    }
    
    [data-testid="stSidebar"] .stMarkdown h3 {
        color: var(--text-muted);
        font-size: 0.65rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin: 1.5rem 0 0.6rem 0;
    }

    /* Sidebar section titles (custom) */
    .control-title {
        color: rgba(255,255,255,0.72);
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        margin: 1.1rem 0 0.65rem 0;
    }

    .control-subtle {
        color: var(--text-muted);
        font-size: 0.75rem;
        margin: 0.25rem 0 0.75rem 0;
    }

    /* Buttons (pill style) */
    [data-testid="stSidebar"] button {
        border-radius: 999px !important;
        border: 1px solid var(--glass-border) !important;
        background: rgba(255,255,255,0.035) !important;
        color: rgba(255,255,255,0.75) !important;
        transition: background 150ms ease, border-color 150ms ease, transform 150ms ease;
    }

    [data-testid="stSidebar"] button:hover {
        background: rgba(255,255,255,0.06) !important;
        border-color: rgba(255,255,255,0.14) !important;
        transform: translateY(-1px);
    }

    [data-testid="stSidebar"] button:active {
        transform: translateY(0px);
    }

    /* Make small button rows tighter */
    .control-button-row {
        margin: 0.25rem 0 0.75rem 0;
    }
    
    /* Status Badge */
    .status-badge {
        background: var(--success-dim);
        border: 1px solid rgba(34, 197, 94, 0.25);
        border-radius: 10px;
        padding: 0.6rem 0.9rem;
        margin-bottom: 1rem;
    }
    
    .status-badge span {
        color: var(--success);
        font-weight: 500;
        font-size: 0.8rem;
    }
    
    /* All Streamlit inputs - dark theme */
    .stTextInput > div > div > input,
    .stSelectbox > div > div,
    .stMultiSelect > div > div,
    .stDateInput > div > div > input,
    .stTimeInput > div > div > input {
        background: var(--glass-tint) !important;
        border: 1px solid var(--glass-border) !important;
        border-radius: 10px !important;
        color: var(--text-primary) !important;
    }
    
    .stSlider > div > div > div {
        background: var(--glass-border) !important;
    }
    
    .stSlider [data-baseweb="slider"] div {
        background: var(--accent) !important;
    }
    
    /* Checkbox */
    .stCheckbox label {
        color: var(--text-secondary) !important;
    }
    
    /* Data Table */
    .stDataFrame {
        background: var(--glass-tint);
        border-radius: 14px;
        overflow: hidden;
        border: 1px solid var(--glass-border);
    }
    
    .stDataFrame [data-testid="stDataFrameResizable"] {
        background: transparent;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: var(--glass-tint);
        border: 1px solid var(--glass-border);
        border-radius: 12px;
        padding: 4px;
        gap: 4px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 8px;
        padding: 0.55rem 1.1rem;
        font-weight: 500;
        font-size: 0.85rem;
        color: var(--text-muted);
        border: none;
    }
    
    .stTabs [aria-selected="true"] {
        background: rgba(255, 255, 255, 0.08);
        color: var(--text-primary);
        border: 1px solid var(--glass-border);
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: var(--glass-tint) !important;
        border: 1px solid var(--glass-border) !important;
        border-radius: 10px !important;
        color: var(--text-secondary) !important;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2.5rem 2rem;
        color: var(--text-muted);
        font-size: 0.75rem;
        margin-top: 3rem;
        border-top: 1px solid var(--glass-border);
    }
    
    .footer strong {
        color: var(--text-secondary);
        font-weight: 500;
    }
    
    /* Hide Streamlit defaults (keep header visible so sidebar toggle works) */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Keep Streamlit header (contains sidebar toggle). Make it visually minimal. */
    [data-testid="stHeader"] {
        background: transparent;
        height: 3rem;
    }
    [data-testid="stToolbar"] {
        right: 0.5rem;
    }
    
    /* Info/Warning/Error boxes */
    .stAlert {
        background: var(--glass-tint) !important;
        border: 1px solid var(--glass-border) !important;
        border-radius: 12px !important;
        color: var(--text-secondary) !important;
    }
    
    /* Map container */
    [data-testid="stDeckGlJsonChart"] {
        border-radius: 16px;
        overflow: hidden;
        border: 1px solid var(--glass-border);
    }
    
    /* Column config table header */
    [data-testid="stDataFrame"] th {
        background: rgba(255, 255, 255, 0.03) !important;
        color: var(--text-muted) !important;
        font-size: 0.7rem !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    [data-testid="stDataFrame"] td {
        color: var(--text-secondary) !important;
        border-color: var(--glass-border) !important;
    }
    
    /* Progress column in table */
    [data-testid="stDataFrame"] [role="progressbar"] {
        background: var(--accent) !important;
    }
</style>
""", unsafe_allow_html=True)


# =============================
# CONFIGURATION / CONSTANTS
# =============================
# NOTE: The original project used a Windows network path. To make this portable
# (e.g. on macOS), allow configuring the data directory via env var or sidebar.
DEFAULT_DATA_DIR = os.environ.get("BSM_DATA_DIR", os.path.join(os.getcwd(), "data"))
DEFAULT_CONSOLIDATED_CSV = os.environ.get(
    "BSM_CONSOLIDATED_CSV",
    os.path.join(DEFAULT_DATA_DIR, "north_ave_bridge_crossing_rows_ALL_MONTHS_20251028.csv"),
)

# North Ave decision box (matching export_bridge_crossings_dual.py)
LAT_MIN, LAT_MAX = 33.7710, 33.7717
LON_MIN, LON_MAX = -84.3907, -84.3887

# Study Area
STUDY_MIN_LAT, STUDY_MAX_LAT = 33.76751010976017, 33.77720144115842
STUDY_MIN_LON, STUDY_MAX_LON = -84.39631547182263, -84.38199135710357

# Ramp
RAMP_MIN_LAT, RAMP_MAX_LAT = 33.772288, 33.774335
RAMP_MIN_LON, RAMP_MAX_LON = -84.390657, -84.390581

# Stop bar
STOP_BAR_LAT, STOP_BAR_LON = 33.771431, -84.390662

# Degree to meter conversions
METERS_PER_DEG_LAT = 111320
METERS_PER_DEG_LON = 111320 * math.cos(math.radians(STOP_BAR_LAT))

# Trip segmentation and NA segment thresholds
GAP_S = 10.0
MIN_TRIP_POINTS = 5
MIN_NA_POINTS = 10
MIN_NA_DURATION_S = 10.0

# Crossing detection rules (matching export_bridge_crossings_dual.py)
NA_SLICES = 3
MIN_SLICES_TOUCHED = 3
WE_MIN_DEG, WE_MAX_DEG = 45.0, 135.0
MIN_WE_FRACTION = 0.60
MIN_POINTS_IN_BOX = 10

# Bearing based W to E check (for visualization)
WE_MIN_BEARING, WE_MAX_BEARING = 45.0, 135.0
WE_MIN_PCT = 0.30

# Smoothing short flips inside NA
RUN_MIN_POINTS = 5
RUN_MIN_SECONDS = 8.0

# --- [SCATTER CONFIG] tweak these values for dot appearance ---
DOT_RADIUS_M = 0.2          # meter radius used by deck.gl before pixel clamps
DOT_RADIUS_MIN_PX = 0.1       # smallest on-screen size
DOT_RADIUS_MAX_PX = 0.5       # largest on-screen size
DOT_OPACITY = 0.1           # 0..1
DOT_STROKE_PX = 0.3         # outline width in pixels
# --------------------------------------------------------------

# =============================
# PAGE HEADER
# =============================
st.markdown("""
<div class="main-header">
    <h1>Connected Vehicle Bridge Analytics</h1>
    <p>CV2X trajectory analysis for North Avenue Bridge — Atlanta, Georgia</p>
    <span class="org-badge">Georgia Department of Transportation</span>
</div>
""", unsafe_allow_html=True)

# Sidebar branding
st.sidebar.markdown("""
<div class="sidebar-header">
    <h2>Analysis Controls</h2>
    <p>Configure filters and parameters</p>
</div>
""", unsafe_allow_html=True)

# Quick actions (UI only)
qa_cols = st.sidebar.columns([1, 1])
with qa_cols[0]:
    if st.button("Reset", use_container_width=True, help="Reset all filters to defaults"):
        # Clear only the widget states we own
        for k in [
            "ui_sel_types",
            "ui_sel_ids",
            "ui_start_date",
            "ui_end_date",
            "ui_start_time",
            "ui_end_time",
            "ui_use_na",
            "ui_use_study",
            "ui_use_ramp",
            "ui_slider_dt",
        ]:
            if k in st.session_state:
                del st.session_state[k]
        st.rerun()
with qa_cols[1]:
    st.caption("Tip: use the pills under Vehicle Selection for quick filtering.")

# =============================
# HELPERS
# =============================
def calc_bearing(lat1, lon1, lat2, lon2):
    """Robust bearing calculation. Handles NaN values and identical points."""
    lat1 = np.asarray(lat1, dtype="float64")
    lon1 = np.asarray(lon1, dtype="float64")
    lat2 = np.asarray(lat2, dtype="float64")
    lon2 = np.asarray(lon2, dtype="float64")
    mask = ~(np.isnan(lat1) | np.isnan(lon1) | np.isnan(lat2) | np.isnan(lon2) | 
             ((lat1 == lat2) & (lon1 == lon2)))
    out = np.full_like(lat1, np.nan, dtype="float64")
    if mask.any():
        a1, o1 = np.radians(lat1[mask]), np.radians(lon1[mask])
        a2, o2 = np.radians(lat2[mask]), np.radians(lon2[mask])
        dlon = o2 - o1
        x = np.sin(dlon) * np.cos(a2)
        y = np.cos(a1) * np.sin(a2) - np.sin(a1) * np.cos(a2) * np.cos(dlon)
        brng = np.degrees(np.arctan2(x, y))
        out[mask] = (brng + 360.0) % 360.0
    return out

def within_box(df, lat_min, lat_max, lon_min, lon_max):
    """
    Robust within_box matching export_bridge_crossings_dual.py
    Ensures data is numeric before comparison.
    """
    if "Latitude" not in df.columns or "Longitude" not in df.columns:
        return pd.Series([False] * len(df), index=df.index)
    # Ensure data is numeric before comparison
    lat = pd.to_numeric(df["Latitude"], errors='coerce')
    lon = pd.to_numeric(df["Longitude"], errors='coerce')
    return (lat.between(lat_min, lat_max)) & (lon.between(lon_min, lon_max))

def split_na_longitudinal(df, n=NA_SLICES):
    """
    Split North Ave box into longitudinal slices for crossing detection.
    Returns edges and masks for each slice.
    """
    lmin, lmax = sorted([LON_MIN, LON_MAX])
    edges = np.linspace(lmin, lmax, n + 1)
    masks = []
    if "Latitude" not in df.columns or "Longitude" not in df.columns:
        return np.array([]), []
    # Ensure numeric types
    lat = pd.to_numeric(df["Latitude"], errors='coerce')
    lon = pd.to_numeric(df["Longitude"], errors='coerce')
    base_lat = lat.between(LAT_MIN, LAT_MAX)
    for i in range(n):
        masks.append(base_lat & lon.between(edges[i], edges[i+1]))
    return edges, masks

def color_for_id(vid, alpha=200):
    # stable hue per ID
    h = (zlib.adler32(str(vid).encode("utf-8")) % 360) / 360.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.65, 0.95)
    return [int(r*255), int(g*255), int(b*255), alpha]

@st.cache_data(show_spinner=True)
def load_data(data_dir: str):
    files = glob(os.path.join(data_dir, "**", "*.csv"), recursive=True)
    frames = []
    for path in files:
        try:
            df = pd.read_csv(path, low_memory=False)
        except Exception:
            continue

        # numeric lat lon speed elev
        df["Latitude"]  = pd.to_numeric(df.get("Latitude"),  errors="coerce")
        df["Longitude"] = pd.to_numeric(df.get("Longitude"), errors="coerce")
        df["Speed_mps"] = pd.to_numeric(df.get("Speed_mps"), errors="coerce")
        if "Elevation_m" in df.columns:
            df["Elevation_m"] = pd.to_numeric(df["Elevation_m"], errors="coerce")
        else:
            df["Elevation_m"] = np.nan
        
        # Handle Heading_deg column (use if available, otherwise calculate later)
        if "Heading_deg" in df.columns:
            df["Heading_deg"] = pd.to_numeric(df["Heading_deg"], errors="coerce")
        else:
            df["Heading_deg"] = np.nan

        df.dropna(subset=["Latitude", "Longitude"], inplace=True)
        if df.empty:
            continue

        # datetime - handle DateTime_UTC column (already in CSV format)
        if "DateTime_UTC" not in df.columns:
            if "Timestamp" in df.columns:
                df["DateTime_UTC"] = pd.to_datetime(df["Timestamp"], unit="s", utc=True)
            else:
                df["DateTime_UTC"] = pd.to_datetime(os.path.getmtime(path), unit="s", utc=True)
        df["DateTime_UTC"] = pd.to_datetime(df["DateTime_UTC"], errors="coerce", utc=True)
        df.dropna(subset=["DateTime_UTC"], inplace=True)
        if df.empty:
            continue

        # Vehicle ID: Use TempID from CSV if available, otherwise fall back to folder name
        if "TempID" in df.columns:
            # Use TempID as VehicleID (convert to string, normalize)
            df["VehicleID"] = df["TempID"].astype(str).str.strip()
        else:
            # Fall back to folder name method
            folder = os.path.basename(os.path.dirname(path))
            vid = folder.split("_", 1)[0]
            df["VehicleID"] = vid
        
        # Vehicle type classification
        df["VehicleID_str"] = df["VehicleID"].astype(str).str.strip().str.lstrip('0')
        df["VehicleType"] = df["VehicleID_str"].apply(
            lambda x: "Ambulance" if x.isdigit() and len(x) == 5 
            else ("HERO" if x.isdigit() and len(x) == 4 else "Unknown")
        )

        # distance to stop bar
        df["dlat"] = (df["Latitude"] - STOP_BAR_LAT) * METERS_PER_DEG_LAT
        df["dlon"] = (df["Longitude"] - STOP_BAR_LON) * METERS_PER_DEG_LON
        df["distance_m"] = np.sqrt(df["dlat"]**2 + df["dlon"]**2)

        frames.append(df)

    if not frames:
        return pd.DataFrame()

    big = pd.concat(frames, ignore_index=True)
    big.sort_values(["VehicleID", "DateTime_UTC"], inplace=True)
    return big


def _infer_vehicle_type_from_id(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip().str.lstrip("0")
    return s.apply(
        lambda x: "Ambulance"
        if x.isdigit() and len(x) == 5
        else ("HERO" if x.isdigit() and len(x) == 4 else "Unknown")
    )


@st.cache_data(show_spinner=False)
def consolidated_timestamp_bounds(csv_path: str) -> tuple[int, int]:
    """Return (min_ts, max_ts) epoch seconds for the consolidated CSV."""
    ts = pd.read_csv(csv_path, usecols=["Timestamp"])["Timestamp"]
    ts = pd.to_numeric(ts, errors="coerce").dropna().astype("int64")
    if ts.empty:
        return (0, 0)
    return int(ts.min()), int(ts.max())


@st.cache_data(show_spinner=False)
def consolidated_available_months(csv_path: str) -> list[str]:
    """Return sorted list of YYYY-MM strings present in the consolidated CSV."""
    ts = pd.read_csv(csv_path, usecols=["Timestamp"])["Timestamp"]
    ts = pd.to_numeric(ts, errors="coerce").dropna().astype("int64")
    if ts.empty:
        return []
    dt = pd.to_datetime(ts, unit="s", utc=True)
    months = dt.dt.strftime("%Y-%m").unique().tolist()
    months.sort()
    return months


@st.cache_data(show_spinner=True)
def load_consolidated_filtered(
    csv_path: str,
    start_ts_s: int,
    end_ts_s: int,
    start_tod_s: int,
    end_tod_s: int,
    use_na: bool,
    use_study: bool,
    use_ramp: bool,
    vehicle_ids: list[str] | None = None,
    vehicle_types: list[str] | None = None,
) -> pd.DataFrame:
    """
    Stream/filter the consolidated CSV without reading everything into memory at once.
    Filters applied:
    - timestamp range (epoch seconds)
    - time-of-day (UTC seconds of day)
    - zones (NOTE: consolidated file is North Ave-only; other zones are ignored)
    - optional vehicle IDs/types
    """
    usecols = ["VehicleID", "VehicleType", "Latitude", "Longitude", "Speed_mps", "Elevation_m", "Timestamp"]
    id_set = set(vehicle_ids) if vehicle_ids else None
    type_set = set(vehicle_types) if vehicle_types else None

    # time-of-day helper (supports wrap-around)
    def tod_mask(ts_s: pd.Series) -> pd.Series:
        t = (ts_s % 86400).astype("int64")
        if end_tod_s >= start_tod_s:
            return (t >= start_tod_s) & (t <= end_tod_s)
        # wraps midnight
        return (t >= start_tod_s) | (t <= end_tod_s)

    out_frames = []
    for chunk in pd.read_csv(
        csv_path,
        usecols=usecols,
        chunksize=250_000,
        dtype={"VehicleID": "string"},
        low_memory=False,
    ):
        chunk["Timestamp"] = pd.to_numeric(chunk["Timestamp"], errors="coerce")
        chunk = chunk.dropna(subset=["Timestamp"])
        chunk["Timestamp"] = chunk["Timestamp"].astype("int64")

        chunk = chunk[(chunk["Timestamp"] >= start_ts_s) & (chunk["Timestamp"] <= end_ts_s)]
        if chunk.empty:
            continue

        chunk = chunk[tod_mask(chunk["Timestamp"])]
        if chunk.empty:
            continue

        # NOTE: consolidated file contains North Ave crossings rows; keep zone checkboxes for UX but ignore non-NA.
        if not use_na and not (use_study or use_ramp):
            # user turned off NA explicitly => empty by definition for this file
            continue

        # normalize VehicleID to string
        chunk["VehicleID"] = chunk["VehicleID"].astype(str).str.strip()
        # Heading is not present in consolidated file; keep column for downstream bearing logic.
        chunk["Heading_deg"] = np.nan

        # VehicleType may be missing; infer as needed
        if "VehicleType" not in chunk.columns or chunk["VehicleType"].isna().all():
            chunk["VehicleType"] = _infer_vehicle_type_from_id(chunk["VehicleID"])
        else:
            # coerce numeric NaNs to strings is bad; keep NaN then fill
            chunk["VehicleType"] = chunk["VehicleType"].astype("object")
            chunk.loc[chunk["VehicleType"].isna(), "VehicleType"] = _infer_vehicle_type_from_id(
                chunk.loc[chunk["VehicleType"].isna(), "VehicleID"]
            )

        if id_set is not None:
            chunk = chunk[chunk["VehicleID"].isin(id_set)]
            if chunk.empty:
                continue

        if type_set is not None:
            chunk = chunk[chunk["VehicleType"].isin(type_set)]
            if chunk.empty:
                continue

        # numeric coercions
        chunk["Latitude"] = pd.to_numeric(chunk["Latitude"], errors="coerce")
        chunk["Longitude"] = pd.to_numeric(chunk["Longitude"], errors="coerce")
        chunk["Speed_mps"] = pd.to_numeric(chunk["Speed_mps"], errors="coerce")
        chunk["Elevation_m"] = pd.to_numeric(chunk["Elevation_m"], errors="coerce")
        chunk = chunk.dropna(subset=["Latitude", "Longitude"])
        if chunk.empty:
            continue

        out_frames.append(chunk)

    if not out_frames:
        return pd.DataFrame()

    df = pd.concat(out_frames, ignore_index=True)
    df["DateTime_UTC"] = pd.to_datetime(df["Timestamp"], unit="s", utc=True)
    df.sort_values(["VehicleID", "DateTime_UTC"], inplace=True)
    return df

def segment_trips(df_vehicle, gap_s=GAP_S):
    dt = df_vehicle["DateTime_UTC"].astype("int64") // 10**9
    prev = dt.shift(1)
    gap = (dt - prev)
    new_trip = (gap.isna()) | (gap > gap_s)
    tripnum = new_trip.cumsum()
    return tripnum

def compute_elev_threshold(na_df):
    vals = na_df["Elevation_m"].dropna()
    if len(vals) < 100:
        return None
    q25 = np.nanpercentile(vals, 25)
    q75 = np.nanpercentile(vals, 75)
    if (q75 - q25) < 2.0:
        return None
    low_med  = np.nanmedian(vals[vals <= q25])
    high_med = np.nanmedian(vals[vals >= q75])
    thr = (low_med + high_med) / 2.0
    return thr

def split_runs_by_level(na_trip_df, elev_thr):
    if na_trip_df.empty:
        return []
    if elev_thr is not None and na_trip_df["Elevation_m"].notna().sum() >= MIN_NA_POINTS:
        na_trip_df = na_trip_df.copy()
        na_trip_df["LevelCand"] = np.where(na_trip_df["Elevation_m"] >= elev_thr, "Bridge", "Underpass")
    else:
        na_trip_df = na_trip_df.copy()
        na_trip_df["LevelCand"] = None

    ts = na_trip_df["DateTime_UTC"].astype("int64") // 10**9
    gap = ts.diff().fillna(0)
    new_block = (gap > GAP_S)
    block_id = new_block.cumsum()
    na_trip_df["BlockID"] = block_id

    runs = []
    for _, block in na_trip_df.groupby("BlockID"):
        if len(block) < RUN_MIN_POINTS:
            continue
        dur = (block["DateTime_UTC"].iloc[-1] - block["DateTime_UTC"].iloc[0]).total_seconds()
        if dur < RUN_MIN_SECONDS:
            continue

        if block["LevelCand"].notna().any():
            maj = block["LevelCand"].mode()
            level = maj.iloc[0] if not maj.empty else None
        else:
            level = None

        if level is None:
            br = calc_bearing(
                block["Latitude"].to_numpy(),
                block["Longitude"].to_numpy(),
                block["Latitude"].shift(-1).to_numpy(),
                block["Longitude"].shift(-1).to_numpy()
            )
            valid = pd.Series(br).dropna()
            we_pct = np.mean((valid >= WE_MIN_BEARING) & (valid <= WE_MAX_BEARING)) if len(valid) else 0.0
            level = "Bridge" if we_pct >= WE_MIN_PCT else "Underpass"

        runs.append((block, level))

    # merge very short middle runs between two same level runs
    merged = []
    i = 0
    while i < len(runs):
        block, lvl = runs[i]
        if i > 0 and i < len(runs) - 1:
            prev_block, prev_lvl = merged[-1]
            next_block, next_lvl = runs[i + 1]
            dur_mid = (block["DateTime_UTC"].iloc[-1] - block["DateTime_UTC"].iloc[0]).total_seconds()
            if dur_mid < (RUN_MIN_SECONDS * 1.5) and prev_lvl == next_lvl:
                i += 1
                continue
        merged.append((block, lvl))
        i += 1
    return merged

 # =============================
 # LOAD & FILTER
 # =============================
with st.sidebar.expander("Data Source", expanded=False):
    source_mode = st.radio(
        "Source",
        options=["Consolidated (fast)", "Raw folder"],
        index=0,
        help="Use the consolidated crossings file for faster loading, or scan all raw CSVs.",
    )

    if source_mode == "Consolidated (fast)":
        CONSOLIDATED_CSV = st.text_input(
            "Crossings CSV",
            value=DEFAULT_CONSOLIDATED_CSV,
            help="Path to consolidated North Ave crossings CSV",
        ).strip()
        if not CONSOLIDATED_CSV or not os.path.isfile(CONSOLIDATED_CSV):
            st.error("Invalid consolidated CSV path")
            st.stop()

        months = consolidated_available_months(CONSOLIDATED_CSV)
        month_pick = st.selectbox(
            "Month",
            options=["All months"] + months,
            index=0,
            help="Restrict loading to a single month for faster analysis.",
            key="ui_month_pick",
        )
        min_ts_s, max_ts_s = consolidated_timestamp_bounds(CONSOLIDATED_CSV)
        if min_ts_s == 0 and max_ts_s == 0:
            st.error("No valid timestamps in consolidated CSV")
            st.stop()

        # If a month is selected, clamp bounds to that month.
        if month_pick != "All months":
            m_start = pd.Timestamp(f"{month_pick}-01", tz="UTC")
            m_end = (m_start + pd.offsets.MonthEnd(1)).replace(
                hour=23, minute=59, second=59, microsecond=999999
            )
            min_ts_s = max(min_ts_s, int(m_start.timestamp()))
            max_ts_s = min(max_ts_s, int(m_end.timestamp()))

    else:
        DATA_DIR = st.text_input(
            "Directory Path",
            value=DEFAULT_DATA_DIR,
            help="Path to folder containing CV2X CSV files",
        ).strip()
        if not DATA_DIR or not os.path.isdir(DATA_DIR):
            st.error("Invalid directory path")
            st.stop()

        _df = load_data(DATA_DIR)
        if _df.empty:
            st.sidebar.error("No valid data found")
            st.stop()

        valid_dt = _df["DateTime_UTC"].dropna()
        if valid_dt.empty:
            st.sidebar.error("No valid DateTime_UTC values in data.")
            st.stop()

        min_ts_s = int(valid_dt.min().timestamp())
        max_ts_s = int(valid_dt.max().timestamp())

# Sidebar badge for what we're loading
st.sidebar.markdown(
    f"""
<div class="status-badge">
    <span>Source: {source_mode}</span>
</div>
""",
    unsafe_allow_html=True,
)

default_start_date = pd.to_datetime(min_ts_s, unit="s", utc=True).date()
default_end_date = pd.to_datetime(max_ts_s, unit="s", utc=True).date()

st.sidebar.markdown('<div class="control-title">Time Filters</div>', unsafe_allow_html=True)
st.sidebar.markdown('<div class="control-subtle">Limit records by date and time-of-day.</div>', unsafe_allow_html=True)

st.sidebar.markdown("### Date Range")
col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.date_input(
        "From",
        value=default_start_date,
        min_value=default_start_date,
        max_value=default_end_date,
        key="ui_start_date",
    )
with col2:
    end_date = st.date_input(
        "To",
        value=default_end_date,
        min_value=default_start_date,
        max_value=default_end_date,
        key="ui_end_date",
    )

start_ts = pd.Timestamp.combine(start_date, datetime.time.min).tz_localize("UTC")
end_ts = pd.Timestamp.combine(end_date, datetime.time.max).tz_localize("UTC")

st.sidebar.markdown("### Time of Day")
col1, col2 = st.sidebar.columns(2)
with col1:
    start_time = st.time_input("From", datetime.time(0, 0), key="ui_start_time")
with col2:
    end_time = st.time_input("To", datetime.time(23, 59), key="ui_end_time")

# convert to epoch seconds for fast mode
start_ts_s = int(start_ts.timestamp())
end_ts_s = int(end_ts.timestamp())
start_tod_s = start_time.hour * 3600 + start_time.minute * 60 + start_time.second
end_tod_s = end_time.hour * 3600 + end_time.minute * 60 + end_time.second

if source_mode == "Raw folder":
    # apply date/time filter in-memory
    mask = (_df["DateTime_UTC"] >= start_ts) & (_df["DateTime_UTC"] <= end_ts)
    df = _df.loc[mask].copy()

    mask_t = (df["DateTime_UTC"].dt.time >= start_time) & (df["DateTime_UTC"].dt.time <= end_time)
    df = df.loc[mask_t].copy()

# spatial union
st.sidebar.markdown('<div class="control-title">Spatial Filters</div>', unsafe_allow_html=True)
st.sidebar.markdown("### Analysis Zones")
use_na    = st.sidebar.checkbox("North Ave Bridge", True, help="Primary study area", key="ui_use_na")
use_study = st.sidebar.checkbox("Extended Study Area", False, key="ui_use_study")
use_ramp  = st.sidebar.checkbox("Highway Ramp", False, key="ui_use_ramp")

if source_mode == "Consolidated (fast)":
    if use_study or use_ramp:
        st.sidebar.info("Note: the consolidated file only contains North Ave rows. Study Area / Ramp filters are ignored.")

    df = load_consolidated_filtered(
        CONSOLIDATED_CSV,
        start_ts_s=start_ts_s,
        end_ts_s=end_ts_s,
        start_tod_s=start_tod_s,
        end_tod_s=end_tod_s,
        use_na=use_na,
        use_study=use_study,
        use_ramp=use_ramp,
        vehicle_ids=None,
        vehicle_types=None,
    )
    if df.empty:
        st.warning("No data matches the current time filters (and month selection, if set).")
        st.stop()

    st.sidebar.markdown(
        f"""
<div class="status-badge">
    <span>{df.shape[0]:,} rows loaded</span>
</div>
""",
        unsafe_allow_html=True,
    )

if any([use_na, use_study, use_ramp]):
    mask_union = pd.Series(False, index=df.index)
    if use_na:
        mask_union |= within_box(df, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX)
    if use_study:
        mask_union |= within_box(df, STUDY_MIN_LAT, STUDY_MAX_LAT, STUDY_MIN_LON, STUDY_MAX_LON)
    if use_ramp:
        mask_union |= within_box(df, RAMP_MIN_LAT, RAMP_MAX_LAT, RAMP_MIN_LON, RAMP_MAX_LON)
    df = df[mask_union].copy()

# vehicle filters
st.sidebar.markdown('<div class="control-title">Vehicle Filters</div>', unsafe_allow_html=True)
st.sidebar.markdown("### Vehicle Selection")
types = sorted(df["VehicleType"].dropna().unique().tolist())
type_btns = st.sidebar.columns(3)
with type_btns[0]:
    if st.button("All", use_container_width=True):
        st.session_state["ui_sel_types"] = types
with type_btns[1]:
    if st.button("Ambulance", use_container_width=True, disabled=("Ambulance" not in types)):
        st.session_state["ui_sel_types"] = ["Ambulance"]
with type_btns[2]:
    if st.button("HERO", use_container_width=True, disabled=("HERO" not in types)):
        st.session_state["ui_sel_types"] = ["HERO"]

sel_types = st.sidebar.multiselect(
    "Vehicle Types",
    types,
    default=types,
    help="Filter by vehicle classification",
    key="ui_sel_types",
)
df = df[df["VehicleType"].isin(sel_types)].copy()

ids = sorted(df["VehicleID"].dropna().unique().tolist())
id_btns = st.sidebar.columns(3)
default_top = ids[:5] if len(ids) >= 5 else ids
with id_btns[0]:
    if st.button("Top 5", use_container_width=True):
        st.session_state["ui_sel_ids"] = default_top
with id_btns[1]:
    if st.button("All IDs", use_container_width=True):
        st.session_state["ui_sel_ids"] = ids
with id_btns[2]:
    if st.button("Clear", use_container_width=True):
        st.session_state["ui_sel_ids"] = []

sel_ids = st.sidebar.multiselect(
    "Vehicle IDs",
    ids,
    default=default_top,
    help="Select specific vehicles to analyze",
    key="ui_sel_ids",
)
if not sel_ids:
    st.sidebar.error("Please select at least one vehicle")
    st.stop()
df = df[df["VehicleID"].isin(sel_ids)].copy()

# color map per vehicle [COLOR MAP]
veh_color_map = {vid: color_for_id(vid, alpha=200) for vid in df["VehicleID"].unique()}
df["veh_color"] = df["VehicleID"].map(veh_color_map)

# animation slider
if df["DateTime_UTC"].isna().all():
    st.info("No valid timestamps after filters")
    st.stop()
min_dt = df["DateTime_UTC"].min().to_pydatetime()
max_dt = df["DateTime_UTC"].max().to_pydatetime()

st.sidebar.markdown('<div class="control-title">Playback</div>', unsafe_allow_html=True)
st.sidebar.markdown("### Timeline")
slider_dt = st.sidebar.slider(
    "Analysis Window",
    min_value=min_dt,
    max_value=max_dt,
    value=max_dt,
    format="MM/DD/YY HH:mm",
    help="Drag to filter by time",
    key="ui_slider_dt",
)
df_plot = df[df["DateTime_UTC"] <= pd.to_datetime(slider_dt, utc=True)].copy()

if df_plot.empty:
    st.warning("No data matches the current filter criteria")
    st.stop()

# =============================
# TRIP SEGMENTATION (matching export_bridge_crossings_dual.py)
# =============================
df_plot.sort_values(["VehicleID", "DateTime_UTC"], inplace=True)
# Use time_diff method matching analysis script (per vehicle)
df_plot["time_diff"] = df_plot.groupby("VehicleID")["DateTime_UTC"].transform(
    lambda x: x.astype("int64").diff() // 10**9
)
run_col = (df_plot["time_diff"] > GAP_S).groupby(df_plot["VehicleID"]).cumsum().fillna(0)
df_plot["Run"] = run_col.astype(int)
df_plot["TripNum"] = df_plot["Run"]  # Use Run as TripNum for consistency
df_plot.drop(columns=["time_diff"], inplace=True)  # Clean up temporary column

# drop ultra short trips
trip_sizes = df_plot.groupby(["VehicleID", "TripNum"]).size().rename("trip_npts")
df_plot = df_plot.join(trip_sizes, on=["VehicleID", "TripNum"])
df_plot = df_plot[df_plot["trip_npts"] >= MIN_TRIP_POINTS].copy()

# =============================
# CROSSING DETECTION (matching export_bridge_crossings_dual.py)
# =============================
summary_rows = []
df_plot["Level"] = "Outside"
df_plot["Is_Crossing"] = False
df_plot["in_NA"] = within_box(df_plot, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX)

# --- [TOOLTIP FIELD] format time for map tooltip ---
df_plot["Time_UTC_str"] = [d.strftime("%Y-%m-%d %H:%M:%S") if pd.notna(d) else "" for d in df_plot["DateTime_UTC"]]

# Calculate bearings if Heading_deg not available
if df_plot["Heading_deg"].isna().all():
    # Calculate bearings from lat/lon
    df_plot['lat_next'] = df_plot.groupby("VehicleID")['Latitude'].shift(-1)
    df_plot['lon_next'] = df_plot.groupby("VehicleID")['Longitude'].shift(-1)
    valid_next = df_plot['lat_next'].notna() & (df_plot['VehicleID'] == df_plot['VehicleID'].shift(-1))
    bearings = np.full(len(df_plot), np.nan)
    if valid_next.any():
        bearings[valid_next] = calc_bearing(
            df_plot.loc[valid_next, "Latitude"].values,
            df_plot.loc[valid_next, "Longitude"].values,
            df_plot.loc[valid_next, "lat_next"].values,
            df_plot.loc[valid_next, "lon_next"].values
        )
    df_plot['bearing'] = bearings
    # Clean up temporary columns
    df_plot.drop(columns=['lat_next', 'lon_next'], inplace=True, errors='ignore')
else:
    # Use Heading_deg from CSV (convert to bearing format if needed)
    df_plot['bearing'] = df_plot['Heading_deg'].values

for vid in sel_ids:
    gveh = df_plot[df_plot["VehicleID"] == vid].copy()
    if gveh.empty:
        continue
    vtype = gveh["VehicleType"].iloc[0]

    for run_id, run_df in gveh.groupby('Run'):
        run_df = run_df.sort_values("DateTime_UTC").copy()
        
        # Find points in North Ave box
        df_in_box = run_df[within_box(run_df, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX)].copy()
        
        if len(df_in_box) < max(2, MIN_POINTS_IN_BOX):
            continue

        # --- Apply Slice Rule (matching export_bridge_crossings_dual.py) ---
        _, slices = split_na_longitudinal(df_in_box, NA_SLICES)
        if not slices or len(slices) != NA_SLICES:
                continue

        touches_required_slices = 0
        for slice_mask_series in slices:
            aligned_mask = slice_mask_series.reindex(df_in_box.index, fill_value=False)
            if aligned_mask.any() and df_in_box.loc[aligned_mask].shape[0] > 0:
                touches_required_slices += 1
        
        if touches_required_slices < MIN_SLICES_TOUCHED:
            continue

        # --- Apply Bearing Rule ---
        valid_bearings = df_in_box['bearing'].dropna()
        if valid_bearings.empty:
                continue

        we_frac = np.mean((valid_bearings >= WE_MIN_DEG) & (valid_bearings <= WE_MAX_DEG))
        
        if we_frac >= MIN_WE_FRACTION:
            # It's a crossing!
            df_plot.loc[df_in_box.index, "Is_Crossing"] = True
            df_plot.loc[df_in_box.index, "Level"] = "Bridge"

            summary_rows.append({
                "VehicleID": vid,
                "VehicleType": vtype,
                "TripNum": int(run_id),
                "Level": "Bridge",
                "StartNA": df_in_box["DateTime_UTC"].iloc[0],
                "EndNA": df_in_box["DateTime_UTC"].iloc[-1],
                "%W->E": round(we_frac * 100.0, 2),
                "NA_Points": len(df_in_box),
                "NA_Duration_s": int((df_in_box["DateTime_UTC"].iloc[-1] - df_in_box["DateTime_UTC"].iloc[0]).total_seconds()),
                "Slices_Touched": touches_required_slices
            })

# =============================
# DASHBOARD METRICS
# =============================
st.markdown('<div class="section-header">Overview</div>', unsafe_allow_html=True)

# Calculate metrics
total_points = len(df_plot)
unique_vehicles = df_plot["VehicleID"].nunique()
crossings_detected = len(summary_rows) if summary_rows else 0
date_range_str = f"{start_date.strftime('%b %d')} – {end_date.strftime('%b %d, %Y')}"

# Display metrics in columns
metric_cols = st.columns(4)
with metric_cols[0]:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{total_points:,}</div>
        <div class="metric-label">Data Points</div>
    </div>
    """, unsafe_allow_html=True)
with metric_cols[1]:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{unique_vehicles}</div>
        <div class="metric-label">Vehicles</div>
    </div>
    """, unsafe_allow_html=True)
with metric_cols[2]:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{crossings_detected}</div>
        <div class="metric-label">Crossings</div>
    </div>
    """, unsafe_allow_html=True)
with metric_cols[3]:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value" style="font-size: 1.1rem;">{date_range_str}</div>
        <div class="metric-label">Period</div>
    </div>
    """, unsafe_allow_html=True)

# =============================
# MAP VISUALIZATION
# =============================
st.markdown('<div class="section-header">Trajectory Map</div>', unsafe_allow_html=True)

if df_plot.empty:
    st.info("No data points to display")
else:
    # Ensure veh_color column exists and is properly formatted
    if "veh_color" not in df_plot.columns:
        veh_color_map = {vid: color_for_id(vid, alpha=200) for vid in df_plot["VehicleID"].unique()}
        df_plot["veh_color"] = df_plot["VehicleID"].map(veh_color_map)
    
    # Filter out any rows with invalid coordinates
    df_map = df_plot[
        df_plot["Latitude"].notna() & 
        df_plot["Longitude"].notna() &
        df_plot["Latitude"].between(-90, 90) &
        df_plot["Longitude"].between(-180, 180)
    ].copy()
    
    if df_map.empty:
        st.warning("No valid coordinates to display")
    else:
        scatter_layer = pdk.Layer(
            "ScatterplotLayer",
            df_map,
            get_position="[Longitude, Latitude]",
            get_fill_color="veh_color",
            stroked=True,
            get_line_color=[255, 255, 255, 60],
            line_width_min_pixels=1,
            get_radius=4,
            radius_min_pixels=5,
            radius_max_pixels=12,
            opacity=0.85,
            pickable=True,
            auto_highlight=True,
        )
        bridge_boundary = pdk.Layer(
            "PolygonLayer",
            data=[
                {
                    "polygon": [
                        [LON_MIN, LAT_MIN],
                        [LON_MAX, LAT_MIN],
                        [LON_MAX, LAT_MAX],
                        [LON_MIN, LAT_MAX],
                        [LON_MIN, LAT_MIN],
                    ]
                }
            ],
            get_polygon="polygon",
            get_fill_color=[59, 130, 246, 20],
            get_line_color=[59, 130, 246, 200],
            line_width_min_pixels=2,
            stroked=True,
            filled=True,
            pickable=False,
        )
        mid_lat = df_map["Latitude"].mean()
        mid_lon = df_map["Longitude"].mean()
        deck = pdk.Deck(
            map_style="dark",
            initial_view_state=pdk.ViewState(
                latitude=mid_lat, longitude=mid_lon, zoom=16, pitch=45, bearing=0
            ),
            layers=[bridge_boundary, scatter_layer],
            tooltip={
                "html": "<div style='font-family: Inter, -apple-system, sans-serif; padding: 14px 16px; min-width: 180px;'><div style='font-weight: 600; color: #f5f5f7; font-size: 13px; margin-bottom: 10px;'>Vehicle {VehicleID}</div><div style='color: rgba(255,255,255,0.55); font-size: 12px; line-height: 1.7;'>Type: {VehicleType}<br/>Trip: {TripNum}<br/>Status: {Level}</div><div style='font-size: 10px; color: rgba(255,255,255,0.3); margin-top: 10px; padding-top: 10px; border-top: 1px solid rgba(255,255,255,0.08);'>{Time_UTC_str} UTC</div></div>",
                "style": {
                    "backgroundColor": "rgba(20, 20, 24, 0.92)",
                    "backdropFilter": "blur(40px)",
                    "border": "1px solid rgba(255,255,255,0.08)",
                    "borderRadius": "14px",
                    "boxShadow": "0 12px 40px rgba(0,0,0,0.5)",
                },
            },
        )
        st.pydeck_chart(deck, use_container_width=True)

# =============================
# CROSSING SUMMARY TABLE
# =============================
st.markdown('<div class="section-header">Crossing Events</div>', unsafe_allow_html=True)

if summary_rows:
    seg_df = pd.DataFrame(summary_rows)
    seg_df = seg_df.sort_values(["VehicleID", "TripNum", "StartNA"])
    seg_df["Month"] = seg_df["StartNA"].dt.strftime("%Y-%m")

    # Sidebar: confirmed crossings controls (filter + sort)
    st.sidebar.markdown('<div class="control-title">Confirmed Crossings</div>', unsafe_allow_html=True)
    cross_month = st.sidebar.selectbox(
        "Crossings month",
        options=["All months"] + sorted(seg_df["Month"].dropna().unique().tolist()),
        index=0,
        key="ui_cross_month",
        help="Filter confirmed crossings table by month.",
    )
    cross_sort = st.sidebar.selectbox(
        "Sort crossings by",
        options=["Start time", "Duration (desc)", "Direction % (desc)", "Points (desc)"],
        index=0,
        key="ui_cross_sort",
    )

    seg_df_view = seg_df.copy()
    if cross_month != "All months":
        seg_df_view = seg_df_view[seg_df_view["Month"] == cross_month].copy()

    if cross_sort == "Start time":
        seg_df_view = seg_df_view.sort_values(["StartNA"])
    elif cross_sort == "Duration (desc)":
        seg_df_view = seg_df_view.sort_values(["NA_Duration_s"], ascending=False)
    elif cross_sort == "Direction % (desc)":
        seg_df_view = seg_df_view.sort_values(["%W->E"], ascending=False)
    else:
        seg_df_view = seg_df_view.sort_values(["NA_Points"], ascending=False)

    # Monthly rollup
    monthly = (
        seg_df_view.groupby("Month", as_index=False)
        .agg(
            Crossings=("Month", "size"),
            Vehicles=("VehicleID", "nunique"),
            AvgDuration_s=("NA_Duration_s", "mean"),
            AvgDirectionPct=("%W->E", "mean"),
        )
        .sort_values("Month")
    )

    st.markdown('<div class="section-header">Crossings by Month</div>', unsafe_allow_html=True)
    st.dataframe(monthly, use_container_width=True, hide_index=True)

    seg_df_display = seg_df_view.copy()
    seg_df_display["StartNA"] = seg_df_display["StartNA"].dt.strftime("%Y-%m-%d %H:%M:%S")
    seg_df_display["EndNA"] = seg_df_display["EndNA"].dt.strftime("%Y-%m-%d %H:%M:%S")
    
    # Rename columns for display
    seg_df_display = seg_df_display.rename(columns={
        "VehicleID": "Vehicle",
        "VehicleType": "Type",
        "TripNum": "Trip",
        "Month": "Month",
        "StartNA": "Entry",
        "EndNA": "Exit",
        "%W->E": "Direction %",
        "NA_Points": "Points",
        "NA_Duration_s": "Duration",
        "Slices_Touched": "Zones"
    })
    
    st.dataframe(
        seg_df_display.reset_index(drop=True),
        use_container_width=True,
        hide_index=True,
        column_config={
            "Vehicle": st.column_config.TextColumn("Vehicle", width="small"),
            "Direction %": st.column_config.ProgressColumn("W→E", format="%.0f%%", min_value=0, max_value=100),
            "Duration": st.column_config.NumberColumn("Duration", format="%d s"),
        }
    )
else:
    st.info("No crossings detected with current filters")

# =============================
# ANALYTICS CHARTS
# =============================
st.markdown('<div class="section-header">Analytics</div>', unsafe_allow_html=True)

# Create tabs for different chart types
chart_tabs = st.tabs(["Distance", "Speed", "Position"])

# Dark mode chart configuration
chart_config = {
    "axis": {
        "labelFontSize": 10, 
        "titleFontSize": 11, 
        "titleColor": "rgba(255,255,255,0.6)", 
        "labelColor": "rgba(255,255,255,0.35)",
        "gridColor": "rgba(255,255,255,0.06)",
        "domainColor": "rgba(255,255,255,0.1)"
    },
    "title": {"fontSize": 12, "fontWeight": 500, "color": "rgba(255,255,255,0.85)"},
    "view": {"strokeWidth": 0},
    "background": "transparent"
}

with chart_tabs[0]:
    for vid in sel_ids:
        uf = df_plot[df_plot["VehicleID"] == vid].sort_values("DateTime_UTC")
        if uf.empty:
            continue

        vtype = uf["VehicleType"].iloc[0]
        chart_dist = (
            alt.Chart(uf)
            .mark_area(
                line={"color": "#3b82f6", "strokeWidth": 2},
                color=alt.Gradient(
                    gradient="linear",
                    stops=[
                        alt.GradientStop(color="rgba(59, 130, 246, 0.35)", offset=0),
                        alt.GradientStop(color="rgba(59, 130, 246, 0.0)", offset=1),
                    ],
                    x1=1,
                    x2=1,
                    y1=1,
                    y2=0,
                ),
            )
            .encode(
                x=alt.X(
                    "DateTime_UTC:T",
                    title="Time",
                    axis=alt.Axis(format="%H:%M:%S", labelAngle=0),
                ),
                y=alt.Y("distance_m:Q", title="Distance (m)"),
            )
            .properties(title=f"{vid} · {vtype}", height=200)
            .configure(**chart_config)
        )
        st.altair_chart(chart_dist, use_container_width=True)

with chart_tabs[1]:
    for vid in sel_ids:
        uf = df_plot[df_plot["VehicleID"] == vid].sort_values("DateTime_UTC")
        if uf.empty or "Speed_mps" not in uf.columns:
            continue

        vtype = uf["VehicleType"].iloc[0]
        chart_speed = (
            alt.Chart(uf)
            .mark_line(color="#22c55e", strokeWidth=2)
            .encode(
                x=alt.X(
                    "DateTime_UTC:T",
                    title="Time",
                    axis=alt.Axis(format="%H:%M:%S", labelAngle=0),
                ),
                y=alt.Y("Speed_mps:Q", title="Speed (m/s)"),
            )
            .properties(title=f"{vid} · {vtype}", height=200)
            .configure(**chart_config)
        )
        st.altair_chart(chart_speed, use_container_width=True)

with chart_tabs[2]:
    for vid in sel_ids:
        uf = df_plot[df_plot["VehicleID"] == vid].sort_values("DateTime_UTC")
        if uf.empty:
            continue

        pos_cols = st.columns(2)
        lat_min, lat_max = uf["Latitude"].min(), uf["Latitude"].max()
        lon_min, lon_max = uf["Longitude"].min(), uf["Longitude"].max()
        lat_m = (lat_max - lat_min) * 0.1 or 1e-5
        lon_m = (lon_max - lon_min) * 0.1 or 1e-5

        with pos_cols[0]:
            chart_lat = (
                alt.Chart(uf)
                .mark_line(color="#a855f7", strokeWidth=2)
                .encode(
                    x=alt.X(
                        "DateTime_UTC:T",
                        title="Time",
                        axis=alt.Axis(format="%H:%M", labelAngle=0),
                    ),
                    y=alt.Y(
                        "Latitude:Q",
                        title="Latitude",
                        scale=alt.Scale(domain=[lat_min - lat_m, lat_max + lat_m]),
                    ),
                )
                .properties(title=f"{vid} · Latitude", height=170)
                .configure(**chart_config)
            )
            st.altair_chart(chart_lat, use_container_width=True)

        with pos_cols[1]:
            chart_lon = (
                alt.Chart(uf)
                .mark_line(color="#f97316", strokeWidth=2)
                .encode(
                    x=alt.X(
                        "DateTime_UTC:T",
                        title="Time",
                        axis=alt.Axis(format="%H:%M", labelAngle=0),
                    ),
                    y=alt.Y(
                        "Longitude:Q",
                        title="Longitude",
                        scale=alt.Scale(domain=[lon_min - lon_m, lon_max + lon_m]),
                    ),
                )
                .properties(title=f"{vid} · Longitude", height=170)
                .configure(**chart_config)
            )
            st.altair_chart(chart_lon, use_container_width=True)

# =============================
# FOOTER
# =============================
st.markdown("""
<div class="footer">
    <p><strong>Georgia Department of Transportation</strong></p>
    <p>Connected Vehicle Analytics · North Avenue Bridge · Atlanta</p>
</div>
""", unsafe_allow_html=True)
