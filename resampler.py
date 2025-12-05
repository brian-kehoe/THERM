"""
resampler.py
-------------
Handles time-index resampling for wide-format numeric and state data.

Responsibilities:
  - Ensure a continuous datetime index
  - Resample numeric columns (mean or interpolate)
  - Resample state/binary columns (forward fill)
  - Avoid filling very large gaps (gap protection)
  - Prepare dataframe for processing and resolution analysis
"""

import pandas as pd
import numpy as np
from typing import Optional


# ================================================================
# Helper: Identify likely binary/state columns
# ================================================================
def _detect_state_columns(df: pd.DataFrame):
    """
    Columns that:
        - contain few unique values, OR
        - look like strings "on"/"off", OR
        - are non-numeric most of the time
    """
    state_cols = []
    numeric_cols = []

    for col in df.columns:
        series = df[col].dropna()

        if series.empty:
            # assume numeric; mapper will handle text cases
            numeric_cols.append(col)
            continue

        # If values look like "on"/"off"
        lower_vals = set(str(v).lower() for v in series.head(20))
        if lower_vals <= {"on", "off", "true", "false", "0", "1"}:
            state_cols.append(col)
            continue

        # If very few unique values → a state sensor
        if series.nunique() <= 4:
            state_cols.append(col)
            continue

        # Otherwise treat as numeric
        numeric_cols.append(col)

    return numeric_cols, state_cols


# ================================================================
# Numeric resampling: mean + gap-safe interpolation
# ================================================================
def _resample_numeric(df: pd.DataFrame, freq="1min"):
    """
    Numeric resampling pipeline:
       - resample to minute grid using mean
       - interpolate small gaps
       - avoid filling very long gaps
    """
    df_res = df.resample(freq).mean()

    for col in df_res.columns:
        series = df_res[col]

        # Determine gap size (in minutes)
        is_na = series.isna()
        if not is_na.any():
            continue

        na_groups = is_na.astype(int).groupby((is_na != is_na.shift()).cumsum()).sum()
        max_gap = na_groups.max()

        # Only interpolate small gaps (≤ 15 minutes)
        if max_gap <= 15:
            df_res[col] = series.interpolate(limit_direction="both")
        else:
            # Fill only edges, not the whole big missing chunk
            df_res[col] = series.interpolate(limit=15, limit_direction="both")

    return df_res


# ================================================================
# State / binary resampling
# ================================================================
def _resample_state(df: pd.DataFrame, freq="1min"):
    """
    State sensors should be forward-filled (ffill).
    Resampling simply keeps last known state for each minute.
    """
    df_res = df.resample(freq).last()

    # Forward-fill for continuity
    df_res = df_res.ffill()

    return df_res


# ================================================================
# Public API: resample unified wide-format dataframe
# ================================================================
def resample_wide(df: pd.DataFrame, freq="1min") -> pd.DataFrame:
    """
    Takes the wide-format dataframe (timestamp index) and returns a resampled version.

    Steps:
      1. Ensure datetime index
      2. Split into numeric vs state columns
      3. Resample each category appropriately
      4. Merge back

    Returns:
      clean_resampled_df
    """

    if df is None or df.empty:
        return df

    df = df.copy()
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    numeric_cols, state_cols = _detect_state_columns(df)

    # Subset
    df_num = df[numeric_cols] if numeric_cols else None
    df_state = df[state_cols] if state_cols else None

    # Resample numeric part
    if df_num is not None:
        df_num_res = _resample_numeric(df_num, freq=freq)
    else:
        df_num_res = None

    # Resample state/binary part
    if df_state is not None:
        df_state_res = _resample_state(df_state, freq=freq)
    else:
        df_state_res = None

    # Merge back
    if df_num_res is not None and df_state_res is not None:
        df_out = df_num_res.join(df_state_res, how="outer")
    elif df_num_res is not None:
        df_out = df_num_res
    else:
        df_out = df_state_res

    df_out = df_out.sort_index()
    df_out.index.name = "timestamp"

    return df_out
