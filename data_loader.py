"""
data_loader.py
---------------
Main orchestration layer for data ingestion.

Pipeline (DL-A architecture):
  1. Read uploaded CSV files           → file_reader.read_files
  2. Reshape to wide-format            → dataframe_reshaper.reshape_data
  3. Resample to uniform 1-minute grid → resampler.resample_wide
  4. Apply user mappings               → data_mapper.map_and_clean
  5. Compute resolution + confidence   → data_resolution.analyze_resolution
  6. Return df + metadata for app use
"""

import pandas as pd
from typing import List, Dict

from file_reader import read_files
from dataframe_reshaper import reshape_data
from resampler import resample_wide
from data_mapper import map_and_clean
from data_resolution import analyze_resolution


# ================================================================
# Master API: load_and_clean_data
# ================================================================
def load_and_clean_data(uploaded_files: List,
                        user_config: Dict,
                        progress_callback=None):
    """
    Master function used by the application to process uploads.

    Inputs:
      uploaded_files: list of file-like objects
      user_config: mapping + other config from System Settings UI
      progress_callback: optional function(step_name, pct)

    Returns dict:
      {
         "df": cleaned, resampled, mapped dataframe,
         "raw_history": dataframe before resampling,
         "resolution": resolution_metadata,
      }
    """

    def _step(name, pct):
        if progress_callback:
            progress_callback(name, pct)

    # ------------------------------------------------------------
    # STEP 1 — Read files (long-format)
    # ------------------------------------------------------------
    _step("Reading files", 5)
    parsed = read_files(uploaded_files)

    state_frames = parsed.get("state", [])
    numeric_frames = parsed.get("numeric", [])

    # ------------------------------------------------------------
    # STEP 2 — Reshape → wide-format merged DF
    # ------------------------------------------------------------
    _step("Reshaping data", 20)
    df_wide = reshape_data(state_frames, numeric_frames)

    if df_wide is None or df_wide.empty:
        return {
            "df": None,
            "raw_history": None,
            "resolution": {
                "error": "No valid data found in uploaded files."
            },
        }

    # Keep a copy before resampling for debugging / UI reference
    raw_history = df_wide.copy()

    # ------------------------------------------------------------
    # STEP 3 — Resample
    # ------------------------------------------------------------
    _step("Resampling", 40)
    df_resampled = resample_wide(df_wide, freq="1min")
    df_resampled.index.name = "timestamp"

    # ------------------------------------------------------------
    # STEP 4 — Apply user mapping + type coercion
    # ------------------------------------------------------------
    _step("Applying mappings", 60)
    df_mapped = map_and_clean(df_resampled, user_config)

    # ------------------------------------------------------------
    # STEP 5 — Compute resolution metadata
    # ------------------------------------------------------------
    _step("Analyzing resolution", 80)

    # resolution engine expects a long-style timestamp column
    df_for_resolution = df_mapped.reset_index().rename(columns={"timestamp": "timestamp"})

    resolution_meta = analyze_resolution(df_for_resolution)

    # Attach metadata to dataframe for downstream modules
    df_mapped.attrs["resolution"] = resolution_meta

    # ------------------------------------------------------------
    # DONE
    # ------------------------------------------------------------
    _step("Completed", 100)

    return {
        "df": df_mapped,
        "raw_history": raw_history,
        "resolution": resolution_meta,
    }
