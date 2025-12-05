# app.py
#
# Samsung Heat Pump Analytics Dashboard v25.x (Modularised)
# ---------------------------------------------------------
# Updated to integrate the new 3-module Data Quality system:
#
# - view_quality_core.py
# - view_quality_components.py
# - view_quality.py (orchestrator)
# - data_resolution.py   ← NEW dependency
#
# Only minimal changes were made — no structural refactor.
#

import streamlit as st
import pandas as pd
import numpy as np
import json
import datetime
from datetime import datetime as dt
from pathlib import Path

import data_loader
import processing
import data_resolution                 # ← NEW MODULE IMPORT
from utils import safe_div

import view_quality
import view_trends
import view_runs


# -------------------------------------------------------------------
# Sidebar configuration
# -------------------------------------------------------------------

st.set_page_config(page_title="Heat Pump Analytics", layout="wide")

st.sidebar.title("Navigation")

modes = [
    "Upload Data",
    "Trends",
    "Daily Runs",
    "Data Quality Audit",
]

mode = st.sidebar.radio("Select Mode", modes)

# Load system config from session state
if "system_config" not in st.session_state:
    st.session_state["system_config"] = {
        "profile_name": "",
        "mapping": {},
    }

# -------------------------------------------------------------------
# File upload section
# -------------------------------------------------------------------

if mode == "Upload Data":
    st.title("Upload System Data")

    uploaded_files = st.file_uploader(
        "Upload Grafana CSV files and Home Assistant CSV",
        type=["csv"],
        accept_multiple_files=True
    )

    if uploaded_files:
        with st.spinner("Processing uploaded files…"):
            data = data_loader.process_uploaded_files(uploaded_files)
            st.session_state["loaded_data"] = data

        st.success("Files loaded successfully.")

        st.write("### System Profile")
        st.json(st.session_state["system_config"])

        st.write("### Data Summary")
        st.write(f"Grafana rows: {len(data['df'])}")
        st.write(f"Daily rows:   {len(data['daily'])}")

# -------------------------------------------------------------------
# Load data for all non-upload pages
# -------------------------------------------------------------------

if mode != "Upload Data":
    if "loaded_data" not in st.session_state:
        st.warning("Please upload data first.")
        st.stop()

    data = st.session_state["loaded_data"]

# -------------------------------------------------------------------
# Trends Page
# -------------------------------------------------------------------

if mode == "Trends":
    st.title("Long-Term Trends")
    view_trends.render_trends(data["df"], st.session_state["system_config"])

# -------------------------------------------------------------------
# Daily Runs Page
# -------------------------------------------------------------------

elif mode == "Daily Runs":
    st.title("Daily Run Explorer")
    view_runs.render_daily_runs(data["daily"], data["df"], st.session_state["system_config"])

# -------------------------------------------------------------------
# DATA QUALITY AUDIT (UPDATED SECTION)
# -------------------------------------------------------------------

elif mode == "Data Quality Audit":
    st.title("Data Quality Audit")

    # NEW — compute resolution metadata on-demand
    resolution_meta = data_resolution.analyze_resolution(data["df"])

    # NEW — updated interface (3 arguments only)
    view_quality.render_data_quality(
        data["df"],
        resolution_meta,
        st.session_state["system_config"]
    )

# -------------------------------------------------------------------
# End of file
# -------------------------------------------------------------------
