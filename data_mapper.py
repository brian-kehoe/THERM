"""
data_mapper.py
---------------
Handles mapping of entity IDs to canonical internal names and performs
type coercion / data cleanup.

Responsibilities:
  - Apply user-specified mappings from System Setup
  - Convert mapped sensors to consistent names (e.g., "FlowTemp", "ReturnTemp")
  - Preserve original entity_id names for room/zone sensors (no Room_1 etc.)
  - Coerce numeric sensors to floats where possible
  - Normalise binary/state sensors ("on"/"off") → 1/0
  - Protect text-only sensors (ValveMode, DHW_Mode, etc.)
  - Return a clean, analysis-ready dataframe
"""

import pandas as pd
import numpy as np
from typing import Dict


# ================================================================
# Helpers: Identify likely binary/state columns
# ================================================================
BINARY_TRUE = {"on", "true", "1", 1, True}
BINARY_FALSE = {"off", "false", "0", 0, False}


def _coerce_binary(col: pd.Series):
    """
    Convert string-like binary values to 0/1.
    Unrecognised values remain as-is (handled by string coercion later).
    """
    lowered = col.astype(str).str.lower()

    out = pd.Series(index=col.index, dtype="float64")

    out.loc[lowered.isin(BINARY_TRUE)] = 1.0
    out.loc[lowered.isin(BINARY_FALSE)] = 0.0

    # Leave non-binary values as NaN → later becomes string column if needed
    return out


# ================================================================
# Type coercion function
# ================================================================
def _coerce_column_types(df: pd.DataFrame, protected_text_cols: set):
    """
    Coerce dataframe column types:
      - numeric sensors → float
      - binary/state sensors → 0/1
      - protected text sensors stay as string
    """

    df_out = df.copy()

    for col in df_out.columns:
        if col in protected_text_cols:
            df_out[col] = df_out[col].astype(str)
            continue

        series = df_out[col]

        # Identify binary
        # If >90% of first 20 samples look like binary → treat as binary
        sample_vals = series.dropna().astype(str).str.lower().head(20)
        if len(sample_vals) > 0:
            if sample_vals.isin(BINARY_TRUE | BINARY_FALSE).mean() >= 0.9:
                df_out[col] = _coerce_binary(series)
                continue

        # Try numeric
        numeric_series = pd.to_numeric(series, errors="coerce")

        # If numeric conversion succeeds for ≥ 80% of samples → numeric column
        if numeric_series.notna().mean() >= 0.8:
            df_out[col] = numeric_series.astype(float)
        else:
            # Fallback to string
            df_out[col] = series.astype(str)

    return df_out


# ================================================================
# Mapping Application
# ================================================================
def apply_mapping(df: pd.DataFrame, user_config: Dict):
    """
    Applies user-defined mappings from System Settings.

    user_config["mapping"] looks like:
    {
        "Power": "sensor.heat_pump_power_ch1",
        "FlowTemp": "sensor.heat_pump_flow_temperature",
        "Room_1": "sensor.main_bedroom_temperature",
        ...
    }

    Behaviour:
      ✔ Standardised internal names (Power, FlowTemp, ReturnTemp, etc.) will appear
      ✔ Rooms/Zones keep their actual sensor names (no placeholders)
      ✔ Unmapped sensors remain available
    """

    if df is None or df.empty:
        return df

    df = df.copy()

    mapping = user_config.get("mapping", {})
    protected_text_cols = set()

    # Canonical names the app expects
    canonical_targets = [
        "Power", "FlowTemp", "ReturnTemp", "FlowRate",
        "OutdoorTemp", "Indoor_Power",
        "ValveMode", "DHW_Mode", "Defrost", "Freq", "DHW_Temp",
        "Outdoor_Humidity", "Wind_Speed",
    ]

    for canonical_name, sensor_id in mapping.items():

        if sensor_id not in df.columns:
            continue  # sensor missing in upload

        if canonical_name in canonical_targets:
            # Rename column to canonical name
            df = df.rename(columns={sensor_id: canonical_name})

            # Protect textual sensors from numeric coercion
            if canonical_name in ["ValveMode", "DHW_Mode"]:
                protected_text_cols.add(canonical_name)

        else:
            # Room_X or Zone_X → keep real sensor name, do nothing
            # (the All Sensors / UI displays actual sensor names)
            pass

    # Coerce types after renaming
    df = _coerce_column_types(df, protected_text_cols)

    return df


# ================================================================
# Exported API
# ================================================================
def map_and_clean(df: pd.DataFrame, user_config: Dict) -> pd.DataFrame:
    """
    Public entry point for the mapper.

    Steps:
      1. Apply mapping (renaming)
      2. Coerce types (binary/numeric/string)
    """
    if df is None or df.empty:
        return df

    df_out = apply_mapping(df, user_config)
    return df_out
