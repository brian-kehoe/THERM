"""
processing.py
--------------
Core heat-pump physics + daily statistics + confidence integration.

This module consumes:
    df: wide-format, resampled dataframe from data_loader
    df.attrs["resolution"]: metadata from data_resolution.analyze_resolution()

Outputs:
    - df_out: same df with extra physics columns (Heat_kW, COP, etc.)
    - daily_stats: rolled-up summary suitable for UI and charts

Key Behaviours:
    ✔ Heat, COP always computed (Option B1)
    ✔ Confidence derived from resolution engine (minimal → high)
    ✔ Physics protected against missing sensors
    ✔ Daily stats annotated with confidence classes
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional


# ============================================================================
# Core Physics Helpers
# ============================================================================

def _compute_delta_t(df: pd.DataFrame):
    """Compute ΔT = FlowTemp - ReturnTemp safely."""
    if "FlowTemp" not in df.columns or "ReturnTemp" not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return df["FlowTemp"] - df["ReturnTemp"]


def _compute_heat_kw(df: pd.DataFrame):
    """
    Compute instantaneous heat output (kW).
    Formula (typical for hydronic HP systems):
        Heat_kW = 4.186 * FlowRate(L/min) * ΔT / 60

    If FlowRate missing, fallback to proxy:
        Heat_kW ≈ c * Power  (not physically accurate, but retains trend)
    """

    dt = _compute_delta_t(df)

    # Case 1 — full formula available
    if "FlowRate" in df.columns and df["FlowRate"].notna().any():
        fr = df["FlowRate"].astype(float)
        heat_kw = 4.186 * fr * dt / 60.0
        return heat_kw

    # Case 2 — no flow rate → fallback trend proxy
    if "Power" in df.columns:
        # Rough correlation proxy (not physical)
        return df["Power"].astype(float) * 0.25 / 1000.0  # 0.25 W→W multiplier placeholder

    # Case 3 — cannot compute anything
    return pd.Series(np.nan, index=df.index)


def _compute_cop(df: pd.DataFrame, heat_kw: pd.Series):
    """Compute COP = Heat_kW / Power_kW with safety checks."""
    if "Power" not in df.columns:
        return pd.Series(np.nan, index=df.index)

    power_kw = df["Power"].astype(float) / 1000.0
    with np.errstate(divide='ignore', invalid='ignore'):
        cop = heat_kw / power_kw
    cop[power_kw <= 0] = np.nan
    return cop


# ============================================================================
# Confidence Integration (Option B1)
# ============================================================================

CONF_RANK = {
    "minimal": 0,
    "very_low": 1,
    "low": 2,
    "medium": 3,
    "high": 4,
    "unknown": -1,
}

def _badge_level(conf: str):
    """Map confidence level to UI badge class."""
    if conf == "high":       return "badge-green"
    if conf == "medium":     return "badge-yellow"
    if conf == "low":        return "badge-orange"
    if conf == "very_low":   return "badge-orange-dark"
    if conf == "minimal":    return "badge-red"
    return "badge-grey"


def _apply_b1_confidence(df: pd.DataFrame, resolution_meta: Dict):
    """
    Attach global heat/COP confidence to the dataframe.
    Adds:
        df.attrs["heat_confidence"]
        df.attrs["cop_confidence"]
        df.attrs["confidence_badges"] = {...}
    """

    if resolution_meta is None:
        df.attrs["heat_confidence"] = "unknown"
        df.attrs["cop_confidence"] = "unknown"
        df.attrs["confidence_badges"] = {
            "heat": _badge_level("unknown"),
            "cop":  _badge_level("unknown"),
        }
        return df

    global_conf = resolution_meta.get("global_confidence", {})
    heat_conf = global_conf.get("heat_confidence", "unknown")
    cop_conf = global_conf.get("cop_confidence", "unknown")

    df.attrs["heat_confidence"] = heat_conf
    df.attrs["cop_confidence"] = cop_conf
    df.attrs["confidence_badges"] = {
        "heat": _badge_level(heat_conf),
        "cop":  _badge_level(cop_conf),
    }
    return df


# ============================================================================
# Daily Statistics
# ============================================================================

def _compute_daily_stats(df: pd.DataFrame, heat_kw: pd.Series, cop: pd.Series):
    """
    Produce per-day totals:
        Heat_kWh
        Elec_kWh
        Mean_COP
        Max_FlowTemp
        Min_OutdoorTemp
        Daily confidence (min of all sensors involved)
    """

    df_daily = pd.DataFrame(index=df.resample("D").mean().index)

    # Heat energy
    heat_kwh = (heat_kw.resample("D").sum() / 60.0)  # minute resolution → kWh
    df_daily["Heat_kWh"] = heat_kwh

    # Electrical consumption
    if "Power" in df.columns:
        elec_kwh = df["Power"].astype(float).resample("D").sum() / 60_000.0
        df_daily["Elec_kWh"] = elec_kwh
    else:
        df_daily["Elec_kWh"] = np.nan

    # COP
    df_daily["COP"] = cop.resample("D").mean()

    # Outdoor temp (optional)
    if "OutdoorTemp" in df.columns:
        df_daily["OutdoorTemp_Min"] = df["OutdoorTemp"].astype(float).resample("D").min()
        df_daily["OutdoorTemp_Max"] = df["OutdoorTemp"].astype(float).resample("D").max()

    # Flow temperature bounds
    if "FlowTemp" in df.columns:
        df_daily["FlowTemp_Max"] = df["FlowTemp"].astype(float).resample("D").max()

    # Daily confidence = min(global heat, cop confidence)
    heat_conf = df.attrs.get("heat_confidence", "unknown")
    cop_conf = df.attrs.get("cop_confidence", "unknown")

    # Pick the lower confidence
    def _min_conf(a, b):
        if a not in CONF_RANK or b not in CONF_RANK:
            return "unknown"
        return min((a, b), key=lambda c: CONF_RANK.get(c, -1))

    daily_conf = _min_conf(heat_conf, cop_conf)
    df_daily["Confidence"] = daily_conf

    return df_daily


# ============================================================================
# MASTER PUBLIC API
# ============================================================================

def process_data(df: pd.DataFrame):
    """
    Main processing entry point called by the app.

    Input:
        df — final mapped + resampled dataframe from data_loader

    Output dict:
        {
            "df": df_with_physics,
            "daily_stats": dataframe,
            "confidence": {
                "heat": ...,
                "cop": ...
            }
        }
    """

    if df is None or df.empty:
        return {
            "df": None,
            "daily_stats": None,
            "confidence": {"heat": "unknown", "cop": "unknown"},
        }

    # ------------------------------------------------------------
    # Physics calculations
    # ------------------------------------------------------------
    delta_t = _compute_delta_t(df)
    heat_kw = _compute_heat_kw(df)
    cop = _compute_cop(df, heat_kw)

    df_out = df.copy()
    df_out["DeltaT"] = delta_t
    df_out["Heat_kW"] = heat_kw
    df_out["COP"] = cop

    # ------------------------------------------------------------
    # Confidence integration
    # ------------------------------------------------------------
    resolution_meta = df.attrs.get("resolution", None)
    df_out = _apply_b1_confidence(df_out, resolution_meta)

    # ------------------------------------------------------------
    # Daily rollups
    # ------------------------------------------------------------
    daily_stats = _compute_daily_stats(df_out, heat_kw, cop)

    return {
        "df": df_out,
        "daily_stats": daily_stats,
        "confidence": {
            "heat": df_out.attrs["heat_confidence"],
            "cop": df_out.attrs["cop_confidence"],
        },
    }
