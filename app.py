# app.py
import warnings
import pandas as pd
import numpy as np
import streamlit as st
import traceback  # NEW: for detailed debug

import view_trends
import view_runs
import view_quality
import mapping_ui
import inspector
import data_loader
import processing
import data_resolution
from utils import safe_div


# --- FIX: Console Error Suppression ---
warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")
warnings.filterwarnings("ignore", category=RuntimeWarning
                        , message="invalid value encountered in scalar divide")
warnings.filterwarnings("ignore", category=RuntimeWarning
                        , message="divide by zero encountered in divide")
warnings.filterwarnings("ignore", category=FutureWarning
                        , message="errors='ignore' is deprecated and will raise in a future version. Use to_numeric without passing `errors` and catch exceptions explicitly instead")


# ---------------------------------------------------------------
# Page & Session Initialisation
# ---------------------------------------------------------------

st.set_page_config(
    page_title="Heat Pump Analytics Dashboard",
    layout="wide",
    page_icon="🔥"
)

if "system_config" not in st.session_state:
    st.session_state["system_config"] = {
        "profile_name": "My Heat Pump",
        "mapping": {},
        "units": {},
        "rooms_per_zone": {},
    }

if "csv_uploader_version" not in st.session_state:
    st.session_state["csv_uploader_version"] = 0

if "capabilities" not in st.session_state:
    st.session_state["capabilities"] = {}

if "debug_log" not in st.session_state:
    st.session_state["debug_log"] = []


def log_debug(message: str):
    """Simple helper to add messages into an in-app debug log."""
    st.session_state["debug_log"].append(message)


# ---------------------------------------------------------------
# Sidebar: Mode + Profile
# ---------------------------------------------------------------

st.sidebar.title("Navigation")
mode = st.sidebar.radio(
    "Select View",
    (
        "System Setup & Mapping",
        "Pre-flight Inspector",
        "Long-Term Trends",
        "Run Inspector",
        "Data Quality Audit",
    ),
)


def render_sidebar_profile():
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Active Profile")

    profile_name = st.session_state["system_config"].get("profile_name", "My Heat Pump")
    st.sidebar.write(f"**Profile:** {profile_name}")

    if st.sidebar.button("Edit System Setup"):
        st.session_state["mapping_mode"] = "edit"


render_sidebar_profile()


# ---------------------------------------------------------------
# Manual Data Cache
# ---------------------------------------------------------------

# === MANUAL CACHING LOGIC ===
def get_processed_data(files, user_config):
    """
    Manually manages the cache to prevent UI re-renders of the loading screen.
    """
    if not files:
        return None

    files_key = tuple(sorted((f.name, f.size) for f in files))
    config_key = str(user_config)
    combined_key = (files_key, config_key)

    if "cached" in st.session_state:
        if st.session_state["cached"]["key"] == combined_key:
            return st.session_state["cached"]

    status_container = st.status("Processing Data...", expanded=True)
    try:
        # 1. Load & Clean
        status_container.write("Loading and merging files (Numeric + State)...")
        progress_cb = lambda t, p: status_container.write(f"Reading: {t}")

        res = data_loader.load_and_clean_data(files, user_config, progress_cb)
        if not res:
            status_container.update(label="Error: No data found", state="error")
            return None

        # 2. Hydraulics
        status_container.write("Applying physics engine...")
        df = processing.apply_gatekeepers(res["df"], user_config)

        # 3. Runs
        status_container.write("Detecting Runs (DHW/Heating)...")
        runs = processing.detect_runs(df, user_config)

        # 4. Daily Stats
        status_container.write("Calculating daily stats...")
        daily = processing.get_daily_stats(df)

        # --- NEW: capability detection ---
        has_flowrate = "FlowRate" in df.columns and df["FlowRate"].notna().any()
        has_heat_sensor = "Heat" in df.columns and pd.to_numeric(df["Heat"], errors="coerce").fillna(0).abs().sum() > 0
        caps = st.session_state.get("capabilities", {})
        caps["has_flowrate"] = has_flowrate
        caps["has_heat_sensor"] = has_heat_sensor
        caps["has_energy_channel"] = has_flowrate or has_heat_sensor
        st.session_state["capabilities"] = caps

        status_container.update(label="Processing Complete!", state="complete", expanded=False)

        # 5. Cache everything
        cache = {
            "key": combined_key,
            "df": df,
            "runs": runs,
            "daily": daily,
            "patterns": res["patterns"],
        }
        st.session_state["cached"] = cache

        st.session_state["heartbeat_baseline"] = res["baselines"]
        st.session_state["heartbeat_baseline_path"] = res.get("baseline_path")

        return cache

    except Exception as e:
        status_container.update(label="Processing failed", state="error")
        st.error(f"An error occurred while loading data:\n{e}")
        st.code(traceback.format_exc())
        return None


# ---------------------------------------------------------------
# Top-level layout helpers
# ---------------------------------------------------------------

def render_debug_panel():
    with st.expander("Developer Debug Log", expanded=False):
        if not st.session_state["debug_log"]:
            st.caption("_No debug messages yet._")
        else:
            for line in st.session_state["debug_log"]:
                st.text(line)


def render_sidebar_file_upload():
    st.sidebar.markdown("### Upload CSV Files")
    uploader_key = f"csv_upload_v{st.session_state['csv_uploader_version']}"
    files = st.sidebar.file_uploader(
        "Grafana exports + Home Assistant history",
        type=["csv"],
        accept_multiple_files=True,
        key=uploader_key,
    )
    return files


def render_sidebar_mode_help():
    st.sidebar.markdown("---")
    if mode == "System Setup & Mapping":
        st.sidebar.info("Configure your system profile, sensor mappings, and zones.")
    elif mode == "Pre-flight Inspector":
        st.sidebar.info("Quickly inspect raw files and sensor availability before analysis.")
    elif mode == "Long-Term Trends":
        st.sidebar.info("View seasonal trends and performance over time.")
    elif mode == "Run Inspector":
        st.sidebar.info("Deep dive into individual heating and DHW runs.")
    elif mode == "Data Quality Audit":
        st.sidebar.info("Audit sensor health, retention, and resolution.")


# ---------------------------------------------------------------
# MAIN UI
# ---------------------------------------------------------------

files = render_sidebar_file_upload()
render_sidebar_mode_help()

if mode == "System Setup & Mapping":
    st.title("System Setup & Mapping")

    mapping_ui.render_mapping_ui(
        st.session_state["system_config"],
        files,
    )

    render_debug_panel()

else:
    if files:
        data = get_processed_data(files, st.session_state["system_config"])
    else:
        data = st.session_state.get("cached")

    if data:
        # --- Overview Header ---
        st.title("Heat Pump Analytics Dashboard")

        # Mode-specific content
        if mode == "Pre-flight Inspector":
            st.markdown("### Pre-flight Inspector")
            with st.expander("Raw File Summary", expanded=True):
                inspector.render_preflight_inspector(files)

        else:
            # --- Global Stats (GREEN BLOCK ONLY, shared between LT Trends + Run Inspector) ---
            try:
                # Canonical global stats from processing engine
                stats = processing.compute_global_stats(data["df"])
                total_heat = stats["total_heat_kwh"]
                total_elec = stats["total_elec_kwh"]
                global_cop = stats["global_cop"]

                runs_detected = len(data.get("runs") or [])
            except Exception as e:
                st.error("Error computing global stats; see debug log.")
                log_debug(f"Global stats error: {e}")
                total_heat = total_elec = global_cop = 0.0
                runs_detected = len(data.get("runs") or [])

            # --- Global Stats (GREEN BLOCK ONLY, shared between LT Trends + Run Inspector) ---
            if data and mode in ("Long-Term Trends", "Run Inspector"):
                st.markdown("### Global Stats")

                # Runs Detected as headline
                st.metric("Runs Detected", f"{runs_detected}")

                # Detailed stats
                st.metric("Total Heat Output", f"{total_heat:.1f} kWh")
                st.metric("Total Electricity Input", f"{total_elec:.1f} kWh")
                st.metric("Global COP", f"{global_cop:.2f}")

            # --- Configuration block ---
            st.markdown("### Configuration")
            profile_name = st.session_state["system_config"].get("profile_name", "Unnamed Profile")
            st.caption(f"Profile: **{profile_name}**")

            col1, col2 = st.columns(2)
            with col1:
                caps = st.session_state.get("capabilities", {})
                has_flowrate = caps.get("has_flowrate", False)
                has_heat_sensor = caps.get("has_heat_sensor", False)
                has_energy_channel = caps.get("has_energy_channel", False)

                st.write("**Capabilities Detected:**")
                st.write(f"- Flow rate sensor: {'✅' if has_flowrate else '❌'}")
                st.write(f"- Direct heat sensor: {'✅' if has_heat_sensor else '❌'}")
                st.write(f"- Energy channel: {'✅' if has_energy_channel else '❌'}")

            with col2:
                if st.button("Reset Cached Data"):
                    for key in [
                        "cached",
                        "heartbeat_baseline",
                        "heartbeat_baseline_path",
                        "debug_log",
                        "system_config",
                        "cached",
                        "uploaded_filenames",
                        "capabilities",
                    ]:
                        st.session_state.pop(key, None)

                    # Force the file_uploader to re-mount with a fresh key
                    st.session_state["csv_uploader_version"] += 1

                    st.rerun()

            # --- Main View Switcher ---
            st.markdown("---")

            if mode == "Long-Term Trends":
                view_trends.render_long_term_trends(data["daily"], data["df"], data["runs"])
            elif mode == "Run Inspector":
                view_runs.render_run_inspector(data["df"], data["runs"])
            elif mode == "Data Quality Audit":
                # New: compute resolution metadata for the loaded dataset
                resolution_meta = data_resolution.analyze_resolution(data["df"])
                # New: streamlined Data Quality renderer (df + resolution + config)
                view_quality.render_data_quality(
                    data["df"],
                    resolution_meta,
                    st.session_state["system_config"],
                )

    else:
        st.info("Upload CSV files to begin.")
        st.sidebar.markdown("---")
        with st.sidebar.expander("About therm"):
            st.markdown("**therm v2.0** - Heat Pump Performance Analysis")
