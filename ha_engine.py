"""
ha_engine.py

Home Assistant–only physics engine for the heat pump.

This module:
- Takes a long-form Home Assistant history dataframe (Time, entity_id, value)
- Pivots to wide format
- Resamples to 1-minute frequency with limited forward fill
- Applies your profile mapping
- Derives:
    - DeltaT_HA
    - is_active_HA
    - is_DHW_HA
    - is_heating_HA
    - Heat_HA, Heat_Heating_HA, Heat_DHW_HA
    - COP_Real_HA, COP_Graph_HA
- Returns:
    - ha_engine_df: indexed by Time, with *_HA columns
    - ha_debug: dict with coverage statistics
"""

from __future__ import annotations

from typing import Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd


# --- Constants (tune if needed) ---

# Minimum power to consider the HP "really on" (W)
MIN_ACTIVE_POWER_W = 200.0

# Minimum compressor frequency (Hz) to consider HP "really on"
MIN_ACTIVE_FREQ_HZ = 7.5

# Minimum and maximum DeltaT (°C) for valid heat calculations
MIN_DELTAT_C = 0.2
MAX_DELTAT_C = 15.0

# Minimum FlowRate (L/min) for valid heat
MIN_FLOW_LPM = 3.0

# Forward-fill horizon in minutes for HA resample
DEFAULT_FFILL_LIMIT = 120


def _pivot_ha_long_to_wide(
    df_ha_long: pd.DataFrame,
    ffill_limit: int = DEFAULT_FFILL_LIMIT,
) -> pd.DataFrame:
    """
    Convert long-form HA data (Time, entity_id, value) to a wide, 1-min resampled dataframe.

    Expected input columns:
        - 'Time': datetime-like
        - 'entity_id': string
        - 'value': numeric (floatable) or string

    Returns:
        df_wide: Time-indexed, 1-minute frequency, numeric values.
    """
    if "Time" not in df_ha_long.columns or "entity_id" not in df_ha_long.columns:
        raise ValueError("df_ha_long must contain 'Time' and 'entity_id' columns")

    # Make a copy to avoid mutating caller data
    df = df_ha_long.copy()

    # Ensure Time is datetime and sorted
    df["Time"] = pd.to_datetime(df["Time"])
    df = df.sort_values("Time")

    # Coerce value to numeric where possible
    # Non-numeric values become NaN here; text series (e.g., DHW_Mode) will be handled separately.
    if "value" in df.columns:
        df["value_num"] = pd.to_numeric(df["value"], errors="coerce")
    else:
        # Some HA exports use 'state' as the value column; fall back
        df["value_num"] = pd.to_numeric(df["state"], errors="coerce")

    # Pivot numeric values: Time x entity_id -> value_num
    df_numeric = (
        df[["Time", "entity_id", "value_num"]]
        .dropna(subset=["value_num"])
        .groupby(["Time", "entity_id"])["value_num"]
        .mean()
        .unstack()
    )

    # Resample to 1 minute and forward-fill short gaps
    df_numeric = (
        df_numeric
        .resample("1min")
        .mean()
        .ffill(limit=ffill_limit)
    )

    return df_numeric


def _extract_text_series(
    df_ha_long: pd.DataFrame,
    entity_id: str,
    ffill_limit: int = DEFAULT_FFILL_LIMIT,
) -> pd.Series:
    """
    Extract a *text* series (like ValveMode or DHW_Mode) from HA history.

    Returns a 1-min resampled, forward-filled pandas Series indexed by Time.
    If the entity does not exist, returns an empty series aligned to no index.
    """
    mask = df_ha_long["entity_id"] == entity_id
    if not mask.any():
        return pd.Series(dtype="object")

    df_sub = df_ha_long.loc[mask, ["Time", "value"]].copy()
    df_sub["Time"] = pd.to_datetime(df_sub["Time"])
    df_sub = df_sub.sort_values("Time")
    df_sub = df_sub.set_index("Time")

    # Resample text by taking last known state each minute
    s = (
        df_sub["value"]
        .resample("1min")
        .last()
        .ffill(limit=ffill_limit)
    )

    return s


def build_ha_engine(
    df_ha_long: pd.DataFrame,
    mapping: Dict[str, str],
    ffill_limit: int = DEFAULT_FFILL_LIMIT,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Build a Home Assistant–only engine dataframe from long-form HA history.

    Args:
        df_ha_long: long-form HA dataframe with columns:
            - 'Time' (datetime-like)
            - 'entity_id' (string)
            - 'value' (string/numeric)
        mapping: profile mapping dict containing at least:
            - Power, FlowTemp, ReturnTemp, FlowRate, OutdoorTemp,
              ValveMode, Freq, DHW_Mode, DHW_Temp, Indoor_Power (optional)
        ffill_limit: forward-fill horizon in minutes for numeric/text resample.

    Returns:
        ha_engine_df: Time-indexed dataframe with columns:
            - Power_HA
            - FlowTemp_HA, ReturnTemp_HA, FlowRate_HA
            - DeltaT_HA
            - Freq_HA
            - ValveMode_HA, DHW_Mode_HA
            - DHW_Temp_HA
            - is_active_HA, is_DHW_HA, is_heating_HA
            - Heat_HA, Heat_Heating_HA, Heat_DHW_HA
            - COP_Real_HA, COP_Graph_HA
        ha_debug: dict with coverage stats (rows, active rows, etc.)
    """
    # 1) Pivot numeric HA data to wide 1-min frame
    df_wide = _pivot_ha_long_to_wide(df_ha_long, ffill_limit=ffill_limit)

    # 2) Extract numeric core series via mapping
    def _get_numeric(col_key: str) -> pd.Series:
        ent = mapping.get(col_key)
        if not ent or ent not in df_wide.columns:
            return pd.Series(index=df_wide.index, dtype="float64")
        return df_wide[ent].astype("float64")

    power = _get_numeric("Power")
    flow_temp = _get_numeric("FlowTemp")
    return_temp = _get_numeric("ReturnTemp")
    flow_rate = _get_numeric("FlowRate")
    freq = _get_numeric("Freq")
    dhw_temp = _get_numeric("DHW_Temp")

    # 3) Extract text series (ValveMode, DHW_Mode) directly from HA long-form
    valve_ent = mapping.get("ValveMode")
    dhw_mode_ent = mapping.get("DHW_Mode")

    valve_mode = (
        _extract_text_series(df_ha_long, valve_ent, ffill_limit=ffill_limit)
        if valve_ent
        else pd.Series(index=df_wide.index, dtype="object")
    )

    dhw_mode = (
        _extract_text_series(df_ha_long, dhw_mode_ent, ffill_limit=ffill_limit)
        if dhw_mode_ent
        else pd.Series(index=df_wide.index, dtype="object")
    )

    # Align text series index to df_wide if needed
    for series_name, s in [("valve_mode", valve_mode), ("dhw_mode", dhw_mode)]:
        if not s.empty and not s.index.equals(df_wide.index):
            # Reindex to numeric frame timeline
            s = s.reindex(df_wide.index).ffill(limit=ffill_limit)
        if series_name == "valve_mode":
            valve_mode = s
        else:
            dhw_mode = s

    # 4) Compute DeltaT
    delta_t = flow_temp - return_temp

    # 5) Activity flags (HA engine version)
    is_active = (
        (power >= MIN_ACTIVE_POWER_W) |
        (freq >= MIN_ACTIVE_FREQ_HZ)
    )

    # DHW vs heating inference from ValveMode and DHW_Mode text
    def classify_dhw(valve: Optional[str], mode: Optional[str]) -> bool:
        if isinstance(valve, str):
            v = valve.lower()
            if "dhw" in v or "tank" in v or "water" in v:
                return True
            if "heat" in v or "space" in v or "room" in v:
                return False
        if isinstance(mode, str):
            m = mode.lower()
            if "dhw" in m or "tank" in m or "water" in m or "std" in m or "standard" in m:
                return True
        return False

    dhw_flags = []
    heating_flags = []

    for v, m in zip(valve_mode.fillna(""), dhw_mode.fillna("")):
        is_dhw = classify_dhw(v, m)
        dhw_flags.append(is_dhw)
        heating_flags.append(not is_dhw)

    is_DHW = pd.Series(dhw_flags, index=df_wide.index, dtype=bool)
    is_heating = pd.Series(heating_flags, index=df_wide.index, dtype=bool)

    # 6) Heat calculation (HA engine)
    # NOTE: This uses the same structure as your existing engine:
    # - Gate on is_active, FlowRate > MIN_FLOW_LPM, MIN_DELTAT_C <= |DeltaT| <= MAX_DELTAT_C
    # - Heat in W with a 4.186 factor (L/min * °C * 4.186 ≈ W-ish; this matches your existing debug ranges).
    valid_heat = (
        is_active &
        (flow_rate > MIN_FLOW_LPM) &
        (delta_t.abs() >= MIN_DELTAT_C) &
        (delta_t.abs() <= MAX_DELTAT_C)
    )

    heat = np.where(
        valid_heat,
        flow_rate * delta_t * 4.186,  # keep consistent with existing engine
        0.0,
    )

    heat_series = pd.Series(heat, index=df_wide.index, name="Heat_HA")

    # Split into space vs DHW contributions
    heat_heating = np.where(is_heating, heat, 0.0)
    heat_dhw = np.where(is_DHW, heat, 0.0)

    heat_heating_series = pd.Series(heat_heating, index=df_wide.index, name="Heat_Heating_HA")
    heat_dhw_series = pd.Series(heat_dhw, index=df_wide.index, name="Heat_DHW_HA")

    # 7) COP (real) – avoid divide-by-zero
    # Power is in W; convert to kW for COP: Heat (W) / Power (W) ~= dimensionless
    with np.errstate(divide="ignore", invalid="ignore"):
        cop_real = np.where(
            (power > 10) & (heat > 0),
            heat / power,
            np.nan,
        )

    cop_real_series = pd.Series(cop_real, index=df_wide.index, name="COP_Real_HA")

    # For simplicity, use same series for graph-ready COP; you can smooth later if desired
    cop_graph_series = cop_real_series.copy()
    cop_graph_series.name = "COP_Graph_HA"

    # 8) Build HA engine dataframe
    ha_engine_df = pd.DataFrame(
        {
            "Power_HA": power,
            "FlowTemp_HA": flow_temp,
            "ReturnTemp_HA": return_temp,
            "FlowRate_HA": flow_rate,
            "DeltaT_HA": delta_t,
            "Freq_HA": freq,
            "ValveMode_HA": valve_mode,
            "DHW_Mode_HA": dhw_mode,
            "DHW_Temp_HA": dhw_temp,
            "is_active_HA": is_active,
            "is_DHW_HA": is_DHW,
            "is_heating_HA": is_heating,
            "Heat_HA": heat_series,
            "Heat_Heating_HA": heat_heating_series,
            "Heat_DHW_HA": heat_dhw_series,
            "COP_Real_HA": cop_real_series,
            "COP_Graph_HA": cop_graph_series,
        }
    )

    ha_engine_df.index.name = "Time"

    # 9) Debug stats
    # Coverage numbers for sanity checks and UI display
    active_rows = int(is_active.sum())
    dhw_rows = int(is_DHW.sum())
    heating_rows = int(is_heating.sum())

    ha_debug: Dict[str, Any] = {
        "rows": int(len(ha_engine_df)),
        "active_rows": active_rows,
        "heating_rows": heating_rows,
        "dhw_rows": dhw_rows,
        "flowrate_nonzero": int((ha_engine_df["FlowRate_HA"] > 0).sum()),
        "deltat_nonzero": int(ha_engine_df["DeltaT_HA"].ne(0).sum()),
        "heat_nonzero": int(ha_engine_df["Heat_HA"].abs().gt(0).sum()),
        "heat_heating_nonzero": int(ha_engine_df["Heat_Heating_HA"].abs().gt(0).sum()),
        "heat_dhw_nonzero": int(ha_engine_df["Heat_DHW_HA"].abs().gt(0).sum()),
        "power_min": float(np.nanmin(power.values)) if len(power) else np.nan,
        "power_max": float(np.nanmax(power.values)) if len(power) else np.nan,
        "flowrate_min": float(np.nanmin(flow_rate.values)) if len(flow_rate) else np.nan,
        "flowrate_max": float(np.nanmax(flow_rate.values)) if len(flow_rate) else np.nan,
        "deltat_min": float(np.nanmin(delta_t.values)) if len(delta_t) else np.nan,
        "deltat_max": float(np.nanmax(delta_t.values)) if len(delta_t) else np.nan,
        "freq_min": float(np.nanmin(freq.values)) if len(freq) else np.nan,
        "freq_max": float(np.nanmax(freq.values)) if len(freq) else np.nan,
    }

    return ha_engine_df, ha_debug
