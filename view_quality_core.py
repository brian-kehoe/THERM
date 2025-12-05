"""
view_quality_core.py
--------------------
Pure logic for sensor quality, expectations, roles, heartbeats, and
category-level summaries. No Streamlit code in this file.

The UI layer calls:
    quality = build_quality_model(df, resolution_meta, user_config)

Returns:
    {
      "summary": {...},
      "categories": {...},
      "sensor_matrix": pd.DataFrame(),
      "heartbeats": {...},
      "unmapped": [...],
    }

Hybrid canonical + legacy support:
  - All canonical names (FlowTemp, ReturnTemp, Power...) supported.
  - All legacy sensor IDs preserved without breaking behaviour.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple


# ============================================================
# 1. CANONICAL + LEGACY SENSOR GROUP DEFINITIONS (HYBRID)
# ============================================================

CANONICAL_GROUPS = {
    "flow": ["FlowTemp"],
    "return": ["ReturnTemp"],
    "power": ["Power", "Indoor_Power"],
    "hydraulic": ["FlowRate", "Freq", "Defrost"],
    "dhw": ["DHW_Mode", "DHW_Temp", "ValveMode"],
    "weather": ["OutdoorTemp", "Outdoor_Humidity", "Wind_Speed"],
}

# Legacy names: extracted from your view_quality.py
LEGACY_GROUPS = {
    "flow": [
        "sensor.heat_pump_flow_temperature",
        "heat_pump_flow_temperature",
    ],
    "return": [
        "sensor.heat_pump_return_temperature",
        "heat_pump_return_temperature",
    ],
    "power": [
        "sensor.heat_pump_power_ch1",
        "heat_pump_power_ch1",
    ],
    "hydraulic": [
        "sensor.heat_pump_flow_rate",
        "heat_pump_flow_rate",
        "sensor.heat_pump_compressor_frequency",
        "heat_pump_compressor_frequency",
        "sensor.heat_pump_defrost_status",
        "heat_pump_defrost_status",
    ],
    "dhw": [
        "sensor.heat_pump_hot_water_mode_value",
        "heat_pump_hot_water_mode_value",
        "sensor.heat_pump_hot_water_temperature",
        "heat_pump_hot_water_temperature",
        "sensor.heat_pump_3way_valve_position_value",
        "heat_pump_3way_valve_position_value",
    ],
    "weather": [
        "sensor.openweathermap_temperature",
        "sensor.openweathermap_humidity",
        "sensor.openweathermap_wind_speed",
        "sensor.ecowitt_weather_indoor_temperature",
        "sensor.ecowitt_weather_humidity",
        "sensor.ecowitt_weather_solar_radiation",
        "sensor.ecowitt_weather_wind_speed",
    ],
}

# Merge canonical + legacy
HYBRID_GROUPS = {
    key: list(set(CANONICAL_GROUPS.get(key, []) + LEGACY_GROUPS.get(key, [])))
    for key in set(CANONICAL_GROUPS) | set(LEGACY_GROUPS)
}


# ============================================================
# 2. EXPECTED SERIES + ON-MINUTES LOGIC
# ============================================================

def expected_window_series(system_on_minutes: float) -> float:
    """
    Returns the minimum expected availability % for a continuous sensor,
    based on system-on time for that day.
    Matches your original logic exactly.
    """
    if system_on_minutes >= 600:      # >=10 hours
        return 0.95
    if system_on_minutes >= 300:      # 5–10 hours
        return 0.85
    if system_on_minutes >= 120:      # 2–5 hours
        return 0.70
    if system_on_minutes >= 60:       # 1–2 hours
        return 0.50
    return 0.30                        # very short days


def compute_system_on_minutes(df: pd.DataFrame) -> float:
    """
    Determines the approximate number of minutes the heat pump was ON
    by detecting non-idle power.
    Uses your existing heuristics.
    """

    if "Power" in df.columns:
        power_col = df["Power"].astype(float)
        active_minutes = (power_col > 100).sum()  # >100W = HP running
        return float(active_minutes)

    # Fallback: try indoor power
    if "Indoor_Power" in df.columns:
        ip = df["Indoor_Power"].astype(float)
        return float((ip > 50).sum())

    # No power proxy → assume always ON (most conservative)
    return float(len(df))


# ============================================================
# 3. SENSOR AVAILABILITY + QUALITY
# ============================================================

def compute_sensor_availability(df: pd.DataFrame) -> Dict[str, float]:
    """
    Computes % of non-null samples per sensor.
    """
    availability = {}
    total = len(df)
    if total == 0:
        return {c: 0.0 for c in df.columns}

    for col in df.columns:
        non_null = df[col].notna().sum()
        availability[col] = non_null / total

    return availability


def classify_sensor_quality(availability: float, expected: float) -> str:
    """
    Return quality classification.
    """
    if availability >= expected:
        return "good"
    if availability >= expected * 0.6:
        return "medium"
    return "poor"


# ============================================================
# 4. GROUP-LEVEL QUALITY
# ============================================================

def compute_group_quality(sensor_availability: Dict[str, float]) -> Dict[str, Dict]:
    """
    Computes quality per sensor group (flow, return, weather, etc.).
    Returns:
        {
           "flow": {
               "sensors": [...],
               "availability": 0.xx,
               "count_good": X,
               ...
           },
           ...
        }
    """
    group_info = {}

    for group_name, sensor_list in HYBRID_GROUPS.items():
        matching = [s for s in sensor_availability if s in sensor_list]
        if not matching:
            group_info[group_name] = {
                "sensors": [],
                "availability": None,
                "good": 0,
                "medium": 0,
                "poor": 0,
            }
            continue

        group_av = np.mean([sensor_availability[s] for s in matching])
        group_info[group_name] = {
            "sensors": matching,
            "availability": group_av,
            "good": sum(sensor_availability[s] >= 0.9 for s in matching),
            "medium": sum(0.6 <= sensor_availability[s] < 0.9 for s in matching),
            "poor": sum(sensor_availability[s] < 0.6 for s in matching),
        }

    return group_info


# ============================================================
# 5. SENSOR MATRIX (for UI table)
# ============================================================

def build_sensor_matrix(df: pd.DataFrame,
                        sensor_availability: Dict[str, float],
                        expected: float) -> pd.DataFrame:
    """
    Builds a table:
         Sensor | Availability | Expected | Quality
    The UI will add styling.
    """

    rows = []
    for sensor in df.columns:
        av = sensor_availability.get(sensor, 0.0)
        q = classify_sensor_quality(av, expected)
        rows.append({
            "Sensor": sensor,
            "Availability": av,
            "Expected": expected,
            "Quality": q,
        })

    return pd.DataFrame(rows)


# ============================================================
# 6. HEARTBEAT BASELINE
# ============================================================

def extract_heartbeat(df: pd.DataFrame) -> Dict[str, Dict]:
    """
    Extracts minimal heartbeat info required for UI.
    (In your previous system this used session_state; here we decouple.)
    Returns:
    {
       sensor_name: {
           "first_seen": ts,
           "last_seen": ts,
           "count": N,
       }
    }
    """

    heartbeat = {}

    for col in df.columns:
        non_null = df[col].dropna()
        if non_null.empty:
            heartbeat[col] = {
                "first_seen": None,
                "last_seen": None,
                "count": 0
            }
            continue

        heartbeat[col] = {
            "first_seen": non_null.index[0],
            "last_seen": non_null.index[-1],
            "count": int(non_null.count())
        }

    return heartbeat


# ============================================================
# 7. UNMAPPED SENSOR DETECTION
# ============================================================

def detect_unmapped_sensors(df: pd.DataFrame, user_config: Dict) -> List[str]:
    """
    Identify sensors not included in user mapping.
    """
    mapped = set(user_config.get("mapping", {}).values())
    sensors = set(df.columns)
    return sorted(list(sensors - mapped))


# ============================================================
# 8. MASTER QUALITY MODEL BUILDER
# ============================================================

def build_quality_model(df: pd.DataFrame,
                        resolution_meta: Dict,
                        user_config: Dict):
    """
    Main entry point — produces a clean, structured model consumed by UI.

    df: final resampled & mapped dataframe
    resolution_meta: output of data_resolution.analyze_resolution()
    user_config: user mappings
    """

    if df is None or df.empty:
        return {
            "summary": {"error": "Empty dataframe"},
            "categories": {},
            "sensor_matrix": pd.DataFrame(),
            "heartbeats": {},
            "unmapped": [],
        }

    # ----------------------------------------
    # Compute system ON time (per-day method replaced with aggregate model)
    # ----------------------------------------
    sys_minutes = compute_system_on_minutes(df)
    expected_availability = expected_window_series(sys_minutes)

    # ----------------------------------------
    # Compute availability per sensor
    # ----------------------------------------
    sensor_av = compute_sensor_availability(df)

    # ----------------------------------------
    # Group-level quality
    # ----------------------------------------
    group_quality = compute_group_quality(sensor_av)

    # ----------------------------------------
    # Sensor matrix (for UI)
    # ----------------------------------------
    sensor_matrix = build_sensor_matrix(df, sensor_av, expected_availability)

    # ----------------------------------------
    # Heartbeat extraction
    # ----------------------------------------
    heartbeat = extract_heartbeat(df)

    # ----------------------------------------
    # Unmapped sensors
    # ----------------------------------------
    unmapped = detect_unmapped_sensors(df, user_config)

    # ----------------------------------------
    # Global summary
    # ----------------------------------------
    summary = {
        "expected": expected_availability,
        "on_minutes": sys_minutes,
        "resolution": resolution_meta,
        "group_quality": group_quality,
    }

    # ----------------------------------------
    # Final structured output
    # ----------------------------------------
    return {
        "summary": summary,
        "categories": group_quality,
        "sensor_matrix": sensor_matrix,
        "heartbeats": heartbeat,
        "unmapped": unmapped,
    }
