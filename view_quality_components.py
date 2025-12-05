"""
view_quality_components.py
--------------------------
Streamlit UI components for data quality visualisation.

This module renders:
  - Summary panels
  - Category quality cards
  - Sensor-level availability matrix
  - Heartbeat baseline summaries
  - Unmapped sensor listings

Takes structured data produced by view_quality_core.build_quality_model().
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import List         # ← FIX


# ================================================================
# UTILITIES — COLOURS & BADGES
# ================================================================

def _badge_for_quality(q: str) -> str:
    """HTML badge for 'good', 'medium', 'poor'."""
    colours = {
        "good": "#2ECC71",        # green
        "medium": "#F1C40F",      # yellow
        "poor": "#E74C3C",        # red
    }
    colour = colours.get(q, "#95A5A6")  # grey fallback
    return f"<span style='background:{colour}; padding:2px 6px; border-radius:4px; color:white; font-size:0.75rem;'>{q}</span>"


def _badge_for_confidence(conf: str) -> str:
    """HTML badge for B1 confidence."""
    map_ = {
        "high": "#2ECC71",
        "medium": "#F1C40F",
        "low": "#E67E22",
        "very_low": "#D35400",
        "minimal": "#E74C3C",
        "unknown": "#95A5A6",
    }
    colour = map_.get(conf, "#95A5A6")
    return f"<span style='background:{colour}; padding:3px 8px; border-radius:4px; color:white; font-weight:600;'>{conf}</span>"


def _styled_table(df: pd.DataFrame):
    """Apply Streamlit-friendly styling to sensor matrix."""
    def colour_map(val, expected):
        if val >= expected:
            return "background-color:#2ecc71; color:white;"
        if val >= expected * 0.6:
            return "background-color:#f1c40f; color:black;"
        return "background-color:#e74c3c; color:white;"

    styled = df.style.format({
        "Availability": "{:.1%}",
        "Expected": "{:.1%}",
    })

    styled = styled.apply(
        lambda row: [
            colour_map(row["Availability"], row["Expected"])
            if col == "Availability" else ""
            for col in row.index
        ],
        axis=1,
    )

    return styled


# ================================================================
# SUMMARY PANEL
# ================================================================

def render_summary_panel(summary: dict):
    """Render main header: expected availability, on-minutes, confidence, etc."""
    st.subheader("Data Quality Summary")

    cols = st.columns(3)

    with cols[0]:
        exp = summary.get("expected", None)
        if exp is not None:
            st.metric("Expected Availability", f"{exp:.0%}")
        else:
            st.metric("Expected Availability", "N/A")

    with cols[1]:
        minutes = summary.get("on_minutes", 0)
        st.metric("System Active (min)", f"{minutes:.0f}")

    with cols[2]:
        # resolution_meta passed through summary
        res = summary.get("resolution", {})
        global_conf = res.get("global_confidence", {})
        heat_conf = global_conf.get("heat_confidence", "unknown")
        cop_conf = global_conf.get("cop_confidence", "unknown")
        html = (
            f"Heat: {_badge_for_confidence(heat_conf)}<br>"
            f"COP: {_badge_for_confidence(cop_conf)}"
        )
        st.markdown(html, unsafe_allow_html=True)


# ================================================================
# CATEGORY CARDS (groups)
# ================================================================

def render_category_cards(categories: dict):
    """Show group-level availability summaries."""
    st.subheader("Sensor Categories")

    for group, info in categories.items():
        sensors = info["sensors"]

        with st.expander(f"{group.upper()} ({len(sensors)} sensors)"):
            if sensors:
                st.write(f"**Sensors:** {', '.join(sensors)}")
            else:
                st.write("_No sensors detected for this category_")

            av = info["availability"]
            if av is not None:
                st.write(f"**Average Availability:** {av:.1%}")
            else:
                st.write("**Average Availability:** N/A")

            st.write(f"Good: {info['good']}, Medium: {info['medium']}, Poor: {info['poor']}")


# ================================================================
# SENSOR MATRIX TABLE
# ================================================================

def render_sensor_matrix(matrix_df: pd.DataFrame):
    """Display styled sensor availability matrix."""
    st.subheader("Sensor-Level Data Availability")

    if matrix_df.empty:
        st.write("_No sensor data available_")
        return

    styled = _styled_table(matrix_df)
    st.dataframe(styled, use_container_width=True)


# ================================================================
# HEARTBEAT PANEL
# ================================================================

def render_heartbeat_panel(heartbeat: dict):
    """Show sensor heartbeat info."""
    st.subheader("Sensor Heartbeats")

    rows = []
    for sensor, hb in heartbeat.items():
        rows.append({
            "Sensor": sensor,
            "First Seen": hb["first_seen"],
            "Last Seen": hb["last_seen"],
            "Samples": hb["count"],
        })

    hb_df = pd.DataFrame(rows)
    hb_df = hb_df.sort_values("Sensor")

    st.dataframe(hb_df, use_container_width=True)


# ================================================================
# UNMAPPED SENSORS
# ================================================================

def render_unmapped_sensors(unmapped: List[str]):
    """List unmapped sensors."""
    st.subheader("Unmapped Sensors")

    if not unmapped:
        st.success("All sensors mapped correctly.")
        return

    st.warning("Some sensors were not mapped:")
    st.write(", ".join(unmapped))


# ================================================================
# MASTER RENDER FUNCTION FOR TAB
# ================================================================

def render_quality_tab(quality_model: dict):
    """
    Renders a full tab view using the outputs from view_quality_core.
    """
    render_summary_panel(quality_model["summary"])
    st.markdown("---")

    render_category_cards(quality_model["categories"])
    st.markdown("---")

    render_sensor_matrix(quality_model["sensor_matrix"])
    st.markdown("---")

    render_heartbeat_panel(quality_model["heartbeats"])
    st.markdown("---")

    render_unmapped_sensors(quality_model["unmapped"])
