"""
view_quality.py
---------------
Top-level orchestrator for the Data Quality tab.

This module does:
  ✔ Imports logic engine (view_quality_core)
  ✔ Imports UI renderer (view_quality_components)
  ✔ Defines one function for the app to call:
        render_data_quality(df, resolution_meta, user_config)

This module performs NO core logic. All logic lives in view_quality_core,
and all UI structural blocks live in view_quality_components.
"""

import streamlit as st

from view_quality_core import build_quality_model
from view_quality_components import (
    render_quality_tab
)


# ================================================================
# MAIN PUBLIC ENTRY POINT
# ================================================================

def render_data_quality(df, resolution_meta, user_config):
    """
    Called by app.py to display the Data Quality tab.

    Inputs:
      df             → cleaned/resampled/mapped dataframe
      resolution_meta → output of data_resolution.analyze_resolution()
      user_config    → mapping + settings from System Settings

    Steps:
      1. Build structured quality model using core logic
      2. Render UI using components module
    """

    st.title("Data Quality Overview")

    # Build data model (pure logic)
    quality_model = build_quality_model(df, resolution_meta, user_config)

    # Tabs (future expansion: heartbeats, patterns, advanced quality)
    tabs = st.tabs(["Summary & Quality"])

    with tabs[0]:
        render_quality_tab(quality_model)

    # Additional tabs may be added later for:
    #   • Advanced Patterns
    #   • Heartbeat History
    #   • Resolution Diagnostics
