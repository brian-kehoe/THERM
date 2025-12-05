"""
dataframe_reshaper.py
---------------------
This module takes the long-format dataframes from file_reader.py and converts them
into wide-format tables suitable for resampling and processing.

Responsibilities:
  - Pivot numeric values:   timestamp | entity → wide table
  - Pivot state values:     timestamp | entity → wide table
  - Merge numeric + state into unified wide-format dataframe
  - Handle duplicate timestamps cleanly
  - Ensure sorted datetime index and clean structure
"""

import pandas as pd
import numpy as np
from typing import List, Optional


# ================================================================
# Helper: pivot numeric dataframe
# ================================================================
def pivot_numeric(df_list: List[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """
    Convert list of long-format numeric dataframes into a single wide-format df:
        timestamp becomes index
        each numeric entity → separate column
    """
    if not df_list:
        return None

    # Combine all numeric long-format frames
    df = pd.concat(df_list, ignore_index=True)
    if df.empty:
        return None

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])

    # Attempt numeric coercion; values that fail stay as NaN
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    # Pivot to wide format
    df_wide = df.pivot_table(
        index="timestamp",
        columns="entity_id",
        values="value",
        aggfunc="mean"  # average duplicates
    )

    df_wide = df_wide.sort_index()
    df_wide.index = pd.to_datetime(df_wide.index)

    # Drop entirely empty columns
    df_wide = df_wide.dropna(axis=1, how="all")

    return df_wide


# ================================================================
# Helper: pivot Home Assistant long-format state dataframe
# ================================================================
def pivot_state(df_list: List[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """
    Convert list of HA state/history dataframes into wide-format:
        timestamp becomes index
        entity_id → column
        state values stored as strings (numeric conversion handled later)
    """
    if not df_list:
        return None

    df = pd.concat(df_list, ignore_index=True)
    if df.empty:
        return None

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])

    # Keep string states; numeric coercion happens later in mapper
    df_wide = df.pivot_table(
        index="timestamp",
        columns="entity_id",
        values="value",
        aggfunc="last"  # last state if multiple occur at same timestamp
    )

    df_wide = df_wide.sort_index()
    df_wide.index = pd.to_datetime(df_wide.index)

    # Drop columns that are completely empty
    df_wide = df_wide.dropna(axis=1, how="all")

    return df_wide


# ================================================================
# Merge numeric + state into single wide dataframe
# ================================================================
def merge_numeric_state(df_numeric: Optional[pd.DataFrame],
                        df_state: Optional[pd.DataFrame],
                        ) -> Optional[pd.DataFrame]:
    """
    Merge numeric and state wide-format dataframes.

    Rules:
      - Outer join on timestamp
      - Sort final index
      - Deduplicate timestamps
      - Consistent datetime index
    """
    if df_numeric is None and df_state is None:
        return None

    if df_numeric is None:
        df = df_state.copy()
        df.index = pd.to_datetime(df.index)
        return df.sort_index()

    if df_state is None:
        df = df_numeric.copy()
        df.index = pd.to_datetime(df.index)
        return df.sort_index()

    # Both numeric + state exist → merge outer
    df_merged = df_numeric.join(df_state, how="outer", lsuffix="_num", rsuffix="_state")
    df_merged = df_merged.sort_index()

    # Ensure no duplicate timestamps (rare)
    df_merged = df_merged[~df_merged.index.duplicated(keep="last")]

    df_merged.index = pd.to_datetime(df_merged.index)

    return df_merged


# ================================================================
# Public API: reshape long-format into unified wide-format
# ================================================================
def reshape_data(state_frames: List[pd.DataFrame],
                 numeric_frames: List[pd.DataFrame]):
    """
    Main API function used by data_loader.py.

    Returns:
        unified_df_wide
    """

    df_numeric_wide = pivot_numeric(numeric_frames)
    df_state_wide = pivot_state(state_frames)

    df = merge_numeric_state(df_numeric_wide, df_state_wide)

    return df
