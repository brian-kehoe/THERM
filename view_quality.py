# view_quality.py

import streamlit as st
import pandas as pd
import numpy as np
import os

from config import (
    SENSOR_EXPECTATION_MODE,
    SENSOR_GROUPS,
    SENSOR_ROLES,
    BASELINE_JSON_PATH,
)
from utils import availability_pct
import baselines


def render_data_quality(
    daily_df: pd.DataFrame,
    df: pd.DataFrame,
    unmapped_entities: list,
    patterns: dict | None,
    heartbeat_path: str | None,
) -> None:
    """
    Data Quality Studio:
    - Overview scorecard (DQ_Tier + category availability)
    - Category drill-down
    - Master sensor matrix
    - Heartbeat baselines
    - Unmapped data
    """
    st.title("️ Data Quality Studio")

    if daily_df is None or daily_df.empty:
        st.warning("No data loaded.")
        return

    # ------------------------------------------------------------------
    # Partial Day Logic
    # ------------------------------------------------------------------
    # Denominator is based on actual system-on minutes per day rather than
    # blindly hardcoding 1440 mins. This allows 100% uptime on partial days.
    system_on_minutes = daily_df.apply(
        lambda r: max(
            r.get("Recorded_Minutes", 0),
            r.get("DQ_Power_Count", 0),
            1,
        ),
        axis=1,
    )

    def expected_window_series(sensor_name: str, system_on_minutes_series: pd.Series):
        """
        Compute the expected sample count per day for a sensor, based on:
        - Learned heartbeat baselines (if available in session state)
        - SENSOR_EXPECTATION_MODE (system/heating/dhw/system_slow/event_only)
        """
        baseline_all = st.session_state.get("heartbeat_baseline", {})
        baseline_data = (baseline_all or {}).get(sensor_name)

        # Scenario A: Baseline exists (e.g. sparse sensor like OWM)
        if baseline_data and baseline_data.get("expected_minutes"):
            # Scale expected minutes by partial-day ratio
            baseline_ratio = baseline_data["expected_minutes"] / 1440.0
            base = system_on_minutes_series * baseline_ratio
            return (
                base.replace(0, np.nan)
                .apply(lambda x: max(1.0, x) if x > 0 else np.nan)
            )

        # Scenario B: Mode-driven expectation
        mode = SENSOR_EXPECTATION_MODE.get(sensor_name, "system")

        if mode == "heating_active":
            base = daily_df.get("Active_Mins", system_on_minutes_series)
        elif mode == "dhw_active":
            base = daily_df.get("Total_DHW_Mins", system_on_minutes_series)
        elif mode == "system_slow":
            # Very slow sensors: effectively "hourly-ish"
            base = (system_on_minutes_series / 60.0).apply(np.ceil)
        else:
            base = system_on_minutes_series

        return base.replace(0, np.nan)

    def format_dq_df(df_in: pd.DataFrame) -> pd.DataFrame:
        df_out = df_in.copy()
        df_out.index.name = "Date"
        try:
            df_out.index = df_out.index.strftime("%d-%m-%Y")
        except Exception:
            # If not a datetime index, leave as-is
            pass
        return df_out

    dq_tab1, dq_tab2, dq_tab3, dq_tab4, dq_tab5 = st.tabs(
        ["Overview", "Category Drill-Down", "All Sensors", "Heartbeats", "⚠️ Unmapped Data"]
    )

    # ------------------------------------------------------------------
    # TAB 1: Overview
    # ------------------------------------------------------------------
    with dq_tab1:
        st.markdown("### System Health Scorecard")

        dq_avg = float(daily_df.get("DQ_Score", 0).mean())
        c1, c2, c3 = st.columns(3)
        c1.metric("Average Data Health", f"{dq_avg:.1f}%")

        gold = len(
            daily_df[
                daily_df.get("DQ_Tier", "")
                .astype(str)
                .str.contains("Gold", na=False)
            ]
        )
        silver = len(
            daily_df[
                daily_df.get("DQ_Tier", "")
                .astype(str)
                .str.contains("Silver", na=False)
            ]
        )
        c2.metric("Gold Days", gold)
        c3.metric("Silver Days", silver)

        overview_df = daily_df[["DQ_Tier"]].copy()
        group_cols: list[str] = []

        for group_name, sensors in SENSOR_GROUPS.items():
            if "Events" in group_name or "Event" in group_name:
                continue

            # Identify sensors that actually have count columns
            valid_sensors = [
                s
                for s in sensors
                if f"DQ_{s}_Count" in daily_df.columns
                or f"{s}_count" in daily_df.columns
            ]
            if not valid_sensors:
                continue

            group_pcts = []
            for s in valid_sensors:
                col = (
                    f"DQ_{s}_Count"
                    if f"DQ_{s}_Count" in daily_df.columns
                    else f"{s}_count"
                )
                pct = availability_pct(
                    daily_df[col],
                    expected_window_series(s, system_on_minutes),
                )
                group_pcts.append(pct)

            if group_pcts:
                overview_df[group_name] = (
                    pd.concat(group_pcts, axis=1).mean(axis=1).round(0)
                )
                group_cols.append(group_name)

        overview_disp = format_dq_df(overview_df[["DQ_Tier"] + group_cols])

        st.dataframe(
            overview_disp.style.background_gradient(
                subset=group_cols,
                cmap="RdYlGn",
                vmin=0,
                vmax=100,
            ).format("{:.0f}", subset=group_cols),
            width="stretch",
        )

    # ------------------------------------------------------------------
    # TAB 2: Category Drill-Down
    # ------------------------------------------------------------------
    with dq_tab2:
        st.markdown("### Category Inspector")

        cat = st.selectbox("Select System Category", list(SENSOR_GROUPS.keys()))
        selected = SENSOR_GROUPS.get(cat, [])

        cat_df = pd.DataFrame(index=daily_df.index)
        valid_cols: list[str] = []

        for sensor in selected:
            col_name = (
                f"DQ_{sensor}_Count"
                if f"DQ_{sensor}_Count" in daily_df.columns
                else f"{sensor}_count"
            )
            if col_name not in daily_df.columns:
                continue

            mode = SENSOR_EXPECTATION_MODE.get(sensor, "system")

            if mode == "event_only" or "defrost" in sensor.lower():
                # Event-style sensors: report raw count
                cat_df[sensor] = daily_df[col_name].fillna(0).astype(int)
            else:
                exp = expected_window_series(sensor, system_on_minutes)
                cat_df[sensor] = availability_pct(
                    daily_df[col_name], exp
                ).round(0)

            valid_cols.append(sensor)

        cat_disp = format_dq_df(cat_df)

        if not cat_disp.empty:
            normal_cols = [
                c
                for c in valid_cols
                if SENSOR_EXPECTATION_MODE.get(c, "system") != "event_only"
            ]
            styler = cat_disp.style.format("{:.0f}", na_rep="-")
            if normal_cols:
                styler = styler.background_gradient(
                    subset=normal_cols,
                    cmap="RdYlGn",
                    vmin=0,
                    vmax=100,
                )
            st.dataframe(styler, width="stretch")
        else:
            st.info("No data for this category.")

    # ------------------------------------------------------------------
    # TAB 3: Master Sensor Matrix
    # ------------------------------------------------------------------
    with dq_tab3:
        st.markdown("### Master Sensor Matrix")

        count_cols = [
            c
            for c in daily_df.columns
            if c.endswith("_count") or c.endswith("_Count")
        ]
        if count_cols:
            flat_data = {}
            for c in count_cols:
                clean_name = (
                    c.replace("DQ_", "")
                    .replace("_Count", "")
                    .replace("_count", "")
                )
                if "short_cycle" in clean_name.lower():
                    continue

                mode = SENSOR_EXPECTATION_MODE.get(clean_name, "system")
                series = daily_df[c]

                if mode == "event_only":
                    flat_data[clean_name] = series.fillna(0).astype(int)
                else:
                    flat_data[clean_name] = availability_pct(
                        series,
                        expected_window_series(clean_name, system_on_minutes),
                    ).round(0)

            df_flat = pd.DataFrame(flat_data, index=daily_df.index)
            matrix_disp = format_dq_df(df_flat)

            normal_cols = [
                c
                for c in df_flat.columns
                if SENSOR_EXPECTATION_MODE.get(c, "system") != "event_only"
            ]
            styler = matrix_disp.style.format("{:.0f}", na_rep="-")
            if normal_cols:
                styler = styler.background_gradient(
                    subset=normal_cols,
                    cmap="RdYlGn",
                    vmin=0,
                    vmax=100,
                )

            st.dataframe(styler, width="stretch")
        else:
            st.info("No count-based columns found in daily data.")

    # ------------------------------------------------------------------
    # TAB 4: Heartbeats
    # ------------------------------------------------------------------
    with dq_tab4:
        st.markdown("### Sensor Heartbeat Baselines")

        # The heartbeat baselines may come from:
        #  - st.session_state["heartbeat_baseline"] (already loaded)
        #  - A JSON file on disk (BASELINE_JSON_PATH / heartbeat_path)
        baseline = st.session_state.get("heartbeat_baseline")

        if not baseline:
            baseline_path = (
                heartbeat_path
                if heartbeat_path
                else BASELINE_JSON_PATH
            )

            if baseline_path and os.path.exists(baseline_path):
                try:
                    st.info(f"Loaded heartbeat baselines from: {baseline_path}")
                    baseline = baselines.load_baseline_file(baseline_path)
                    st.session_state["heartbeat_baseline"] = baseline
                except Exception as e:
                    st.error(f"Error loading baseline file: {e}")
                    baseline = {}
            else:
                baseline = {}

        if not baseline:
            st.warning(
                "No heartbeat baseline is available yet. "
                "Run the baseline builder to generate expected sensor windows."
            )
        else:
            # Show a simple table of baselines
            rows = []
            for sensor_name, meta in baseline.items():
                rows.append(
                    {
                        "Sensor": sensor_name,
                        "Role": meta.get("role", SENSOR_ROLES.get(sensor_name, "")),
                        "Expected Minutes / Day": meta.get(
                            "expected_minutes", np.nan
                        ),
                        "Mode": SENSOR_EXPECTATION_MODE.get(
                            sensor_name, "system"
                        ),
                    }
                )
            hb_df = pd.DataFrame(rows).sort_values("Sensor")
            st.dataframe(hb_df, width="stretch")

    # ------------------------------------------------------------------
    # TAB 5: Unmapped Data
    # ------------------------------------------------------------------
    with dq_tab5:
        st.markdown("### Unmapped / Dropped Entities")

        if not unmapped_entities:
            st.info("No unmapped entities found in the source files.")
        else:
            st.write(
                "The following entities were present in the upload but not "
                "mapped to any internal sensor role:"
            )
            st.dataframe(
                pd.DataFrame(
                    sorted(set(unmapped_entities)), columns=["Entity ID"]
                ),
                width="stretch",
            )
