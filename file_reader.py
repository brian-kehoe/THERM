"""
file_reader.py
---------------
Reads uploaded CSV files and returns normalised long-format DataFrames.

Responsibilities:
  - Identify timestamp column
  - Normalise timestamps to pandas datetime
  - Classify file as:
        • state/history (entity_id, state, timestamp)
        • numeric timeseries (single entity_id or multiple columns)
  - Return clean long-format dataframes with uniform schema:
        timestamp | entity_id | value
  - No pivoting, no resampling, no mapping (handled in later modules)
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Tuple


# ================================================================
# Helper: Identify timestamp column in a CSV
# ================================================================
def detect_timestamp_column(df: pd.DataFrame):
    """
    Identify the best timestamp column.
    Priority:
      1. "time"
      2. columns containing "timestamp"
      3. "date" or "datetime"
    Returns column name or None.
    """
    candidates = [c for c in df.columns if c.lower() in ("time", "timestamp", "date", "datetime")]

    if candidates:
        return candidates[0]

    # fuzzy match
    for c in df.columns:
        cl = c.lower()
        if "time" in cl or "date" in cl:
            return c

    return None


# ================================================================
# Helper: Normalise timestamps
# ================================================================
def normalise_timestamp_column(df: pd.DataFrame, time_col: str):
    """
    Convert time column to pandas datetime.
    Returns df with 'timestamp' column standardised.
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df[time_col], errors="coerce", utc=False)
    df = df.dropna(subset=["timestamp"])
    df = df.drop(columns=[time_col], errors="ignore")
    return df


# ================================================================
# Detect if CSV is Home Assistant long-format state file
# ================================================================
def is_home_assistant_state_format(df: pd.DataFrame):
    """
    HA history/state files typically have:
        entity_id | state | last_changed | last_updated
    or sometimes:
        entity_id | state | time
    """
    cols = [c.lower() for c in df.columns]

    if "entity_id" in cols and "state" in cols:
        return True

    # Some HA exports rename state to value
    if "entity_id" in cols and "value" in cols:
        return True

    return False


# ================================================================
# Detect if CSV is numeric timeseries (Grafana/Influx)
# ================================================================
def is_numeric_timeseries(df: pd.DataFrame):
    """
    Grafana/Influx files typically look like:
        Time | sensor.heat_pump_flow_temperature | sensor.heat_pump_return_temperature | ...
    """
    cols = [c.lower() for c in df.columns]

    if any("time" == c or "timestamp" in c for c in cols):
        # If more than 1 additional numeric column → numeric timeseries
        numeric_cols = [c for c in df.columns if c not in ["Time", "time", "timestamp"]]
        if len(numeric_cols) >= 1:
            return True

    return False


# ================================================================
# Convert HA long-format file → standard long-format
# ================================================================
def parse_home_assistant_file(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert Home Assistant long-format CSV into our canonical form:
        timestamp | entity_id | value
    """
    df = df.copy()

    # Determine time column
    time_col = detect_timestamp_column(df)
    if time_col is None:
        raise ValueError("Cannot find timestamp column in Home Assistant file.")

    df = normalise_timestamp_column(df, time_col)

    # Determine value column
    if "state" in df.columns:
        value_col = "state"
    elif "value" in df.columns:
        value_col = "value"
    else:
        raise ValueError("Home Assistant file missing 'state' or 'value' column.")

    # Standardise the output format
    df = df.rename(columns={value_col: "value"})
    df = df[["timestamp", "entity_id", "value"]]

    # Clean up
    df["entity_id"] = df["entity_id"].astype(str)
    df["value"] = df["value"].astype(str)  # numeric conversion happens later

    return df


# ================================================================
# Convert numeric timeseries → standard long-format
# ================================================================
def parse_numeric_timeseries(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert a Grafana/Influx numeric CSV into canonical long-format:
        timestamp | entity_id | value
    One row in input becomes many rows (long format).
    """
    df = df.copy()

    # Identify timestamp
    time_col = detect_timestamp_column(df)
    if time_col is None:
        raise ValueError("Numeric timeseries file missing timestamp column.")

    df = normalise_timestamp_column(df, time_col)
    df = df.sort_values("timestamp")

    value_cols = [c for c in df.columns if c not in ["timestamp"]]

    # Melt → long format
    df_long = df.melt(id_vars=["timestamp"], value_vars=value_cols,
                      var_name="entity_id", value_name="value")

    # Drop missing/nan values
    df_long = df_long.dropna(subset=["value"])
    df_long["entity_id"] = df_long["entity_id"].astype(str)

    return df_long


# ================================================================
# Main entry: read multiple files and classify each
# ================================================================
def read_files(uploaded_files: List) -> Dict[str, pd.DataFrame]:
    """
    Reads a list of uploaded CSV files and returns dict:
        {
            "state": [df1, df2, ...],
            "numeric": [df3, df4, ...],
        }
    Each df is a long-format (timestamp | entity_id | value)
    """
    state_files = []
    numeric_files = []

    for f in uploaded_files:
        try:
            f.seek(0)
            df_raw = pd.read_csv(f)
        except Exception as e:
            print(f"Error reading CSV: {f.name} → {e}")
            continue

        # Classify file
        if is_home_assistant_state_format(df_raw):
            try:
                state_files.append(parse_home_assistant_file(df_raw))
            except Exception as e:
                print(f"Failed to parse HA file {f.name}: {e}")
        elif is_numeric_timeseries(df_raw):
            try:
                numeric_files.append(parse_numeric_timeseries(df_raw))
            except Exception as e:
                print(f"Failed to parse numeric file {f.name}: {e}")
        else:
            print(f"Unknown file type (skipping): {f.name}")

    return {
        "state": state_files,
        "numeric": numeric_files,
    }
