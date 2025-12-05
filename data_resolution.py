"""
data_resolution.py
------------------
Module for adaptive, sensor-specific resolution detection, retention estimation,
and confidence scoring for Heat Pump Analytics.

This module performs:
1. Automatic high-resolution window detection (up to ~30 days)
2. Per-sensor baseline frequency analysis from the high-resolution period
3. Detection of downsampled / degraded data outside the HR window
4. Confidence scoring (B1 model: High → Minimal)
5. Metadata export for processing + UI warning banners

Input expectations (wide format):
    - Either:
        a) A DataFrame with a 'timestamp' column (any type convertible to datetime)
       or
        b) A DataFrame with a DatetimeIndex (will be reset to 'timestamp')
    - All other columns are treated as sensor value columns.
"""

import numpy as np
import pandas as pd
from datetime import timedelta


# ================================================================
# Utility: Compute timestamp deltas safely
# ================================================================
def _compute_intervals(series: pd.Series):
    """Return array of time deltas (in seconds) between consecutive timestamps."""
    if series.empty:
        return np.array([])
    dt = series.diff().dropna()
    if dt.empty:
        return np.array([])
    return dt.dt.total_seconds().values


# ================================================================
# STEP 0 — Normalise input dataframe
# ================================================================
def _ensure_timestamp_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure there is a 'timestamp' column of dtype datetime64[ns].

    Accepts:
      - Wide df with DatetimeIndex
      - Wide df with a 'timestamp' column

    Returns:
      df_with_ts: copy of df with a proper 'timestamp' column.
    """
    if df is None or df.empty:
        return df

    df_work = df.copy()

    if "timestamp" in df_work.columns:
        df_work["timestamp"] = pd.to_datetime(df_work["timestamp"], errors="coerce")
        df_work = df_work.dropna(subset=["timestamp"])
        return df_work

    # Fallback: use DatetimeIndex if available
    if isinstance(df_work.index, pd.DatetimeIndex):
        idx_name = df_work.index.name or "timestamp"
        df_work = df_work.reset_index().rename(columns={idx_name: "timestamp"})
        df_work["timestamp"] = pd.to_datetime(df_work["timestamp"], errors="coerce")
        df_work = df_work.dropna(subset=["timestamp"])
        return df_work

    raise ValueError(
        "data_resolution.analyze_resolution() expects either a 'timestamp' column "
        "or a DatetimeIndex on the input dataframe."
    )


# ================================================================
# STEP 1 — Detect High-Resolution Window (HR window)
# ================================================================
def detect_high_res_window(df: pd.DataFrame, max_days: int = 30):
    """
    Automatically detect the highest-resolution window within the last N days.
    - Looks for dense timestamp regions
    - Window size dynamically adapts (3–10 days)

    df must contain a 'timestamp' column of dtype datetime64[ns].

    Returns:
        (start_ts, end_ts) or (None, None)
    """
    if df is None or df.empty:
        return None, None

    df_sorted = df.sort_values("timestamp")
    end = df_sorted["timestamp"].iloc[-1]
    start_limit = end - timedelta(days=max_days)

    # restrict search
    recent = df_sorted[df_sorted["timestamp"] >= start_limit]
    if recent.empty:
        return None, None

    # Use sliding window sizes: 3 → 10 days
    window_sizes = [3, 5, 7, 10]
    best_score = -1
    best_range = (None, None)

    timestamps = recent["timestamp"].values

    for win in window_sizes:
        win_delta = np.timedelta64(win, "D")
        # 2-pointer method
        left = 0
        for right in range(len(timestamps)):
            while timestamps[right] - timestamps[left] > win_delta:
                left += 1
            count = right - left + 1
            if count <= 2:
                continue

            subset = recent.iloc[left : right + 1]
            intervals = _compute_intervals(subset["timestamp"])
            if len(intervals) == 0:
                continue

            density = count / win   # entries per day
            median_dt = np.median(intervals)
            score = density / median_dt  # higher density + lower dt = better resolution

            if score > best_score:
                best_score = score
                best_range = (subset["timestamp"].iloc[0], subset["timestamp"].iloc[-1])

    return best_range


# ================================================================
# STEP 2 — Determine per-sensor baseline frequency (from HR window)
# ================================================================
def compute_baseline_intervals(df: pd.DataFrame, hr_start, hr_end):
    """
    For each sensor column, compute:
    - baseline median interval (seconds)
    - classification: periodic_high / periodic_med / periodic_low /
                      event_based / hourly_native / irregular

    df must contain a 'timestamp' column; all other columns are treated as sensors.
    """
    if hr_start is None or hr_end is None or df is None or df.empty:
        return {}

    baseline = {}
    hr_df = df[(df["timestamp"] >= hr_start) & (df["timestamp"] <= hr_end)]

    # All columns except timestamp/entity_id are treated as values (wide format)
    value_cols = [c for c in df.columns if c not in ["timestamp", "entity_id"]]

    for col in value_cols:
        series = hr_df[hr_df[col].notna()][["timestamp", col]]
        if series.empty:
            baseline[col] = {
                "baseline_interval": None,
                "baseline_type": "no_data",
            }
            continue

        # Compute median dt
        intervals = _compute_intervals(series["timestamp"])
        if len(intervals) == 0:
            baseline[col] = {
                "baseline_interval": None,
                "baseline_type": "event_based",
            }
            continue

        m_dt = np.median(intervals)

        # Classify baseline type
        if m_dt <= 20:
            btype = "periodic_high"
        elif m_dt <= 90:
            btype = "periodic_medium"
        elif m_dt <= 300:
            btype = "periodic_low"
        elif abs(m_dt - 3600) < 120:  # aware of OWM hourly-native sensors
            btype = "hourly_native"
        else:
            # Could be event-based or irregular
            if len(series[col].unique()) <= 6:
                btype = "event_based"
            else:
                btype = "irregular"

        baseline[col] = {
            "baseline_interval": float(m_dt),
            "baseline_type": btype,
        }

    return baseline


# ================================================================
# STEP 3 — Compute degradation outside HR window
# ================================================================
def classify_resolution_vs_baseline(full_df: pd.DataFrame, baseline: dict):
    """
    For each sensor, compare full data intervals vs baseline intervals.
    Produces:
        resolution_status[col] = {
            baseline_interval,
            observed_interval,
            confidence,
            degraded
        }

    full_df must contain 'timestamp' and the same value columns used for baseline.
    """

    resolution_map = {}
    if full_df is None or full_df.empty:
        return resolution_map

    value_cols = [c for c in full_df.columns if c not in ["timestamp", "entity_id"]]

    for col in value_cols:
        base_info = baseline.get(col, None)

        if base_info is None or base_info["baseline_interval"] is None:
            # No usable baseline → unknown
            resolution_map[col] = {
                "baseline_interval": None,
                "observed_interval": None,
                "confidence": "unknown",
                "degraded": False,
            }
            continue

        base_dt = base_info["baseline_interval"]
        base_type = base_info["baseline_type"]

        series = full_df[full_df[col].notna()][["timestamp", col]]
        intervals = _compute_intervals(series["timestamp"])
        if len(intervals) == 0:
            resolution_map[col] = {
                "baseline_interval": base_dt,
                "observed_interval": None,
                "confidence": "unknown",
                "degraded": False,
            }
            continue

        obs_dt = float(np.median(intervals))

        # B1 — Confidence scoring
        ratio = obs_dt / base_dt if base_dt > 0 else 999

        if ratio <= 1.5:
            conf = "high"
        elif ratio <= 3:
            conf = "medium"
        elif ratio <= 8:
            conf = "low"
        elif ratio <= 20:
            conf = "very_low"
        else:
            conf = "minimal"

        degraded = conf in ["low", "very_low", "minimal"]

        # If hourly but baseline is much faster → minimal
        if obs_dt >= 3000 and base_dt < 300:
            conf = "minimal"
            degraded = True

        resolution_map[col] = {
            "baseline_interval": base_dt,
            "observed_interval": obs_dt,
            "confidence": conf,
            "degraded": degraded,
        }

    return resolution_map


# ================================================================
# STEP 4 — Estimate retention window
# ================================================================
def estimate_retention_days(df: pd.DataFrame):
    """
    Estimate how long high-resolution data is kept before downsampling.
    Looks for the earliest point where dt jumps significantly for multiple sensors.

    df must contain 'timestamp'.
    """
    if df is None or df.empty:
        return None

    df_sorted = df.sort_values("timestamp")
    ts = df_sorted["timestamp"]
    oldest = ts.iloc[0]
    newest = ts.iloc[-1]

    total_days = (newest - oldest).total_seconds() / 86400
    if total_days < 2:
        return total_days

    # Look for dt jumps across multiple sensors
    change_points = []

    for col in df.columns:
        if col in ["timestamp", "entity_id"]:
            continue
        series = df[df[col].notna()][["timestamp", col]]
        intervals = _compute_intervals(series["timestamp"])
        if len(intervals) == 0:
            continue
        # look for big jumps (≥ 1 hr)
        idx = np.where(intervals >= 3600)[0]
        if len(idx) > 0:
            change_points.append(series["timestamp"].iloc[idx[0]])

    if change_points:
        cutoff = min(change_points)
        return (newest - cutoff).total_seconds() / 86400

    return total_days


# ================================================================
# STEP 5 — Confidence for Heat & COP (B1 Propagation)
# ================================================================
def compute_global_confidence(res_map: dict):
    """
    Determine global heat & COP confidence (B1):
        heat_conf = min(FT_conf, RT_conf, Power_conf)
        cop_conf  = min(heat_conf, Power_conf)

    Only canonical names are used here:
        FlowTemp, ReturnTemp, Power
    """

    def _get(col):
        if col not in res_map:
            return "unknown"
        return res_map[col]["confidence"]

    flow = _get("FlowTemp")
    ret = _get("ReturnTemp")
    power = _get("Power")

    # Simple priority order
    order = ["minimal", "very_low", "low", "medium", "high"]

    def score(c):
        if c not in order:
            return -1
        return order.index(c)

    heat_score = min(score(flow), score(ret), score(power))
    heat_conf = order[heat_score] if heat_score >= 0 else "unknown"

    cop_score = min(heat_score, score(power))
    cop_conf = order[cop_score] if cop_score >= 0 else "unknown"

    return {
        "heat_confidence": heat_conf,
        "cop_confidence": cop_conf,
    }


# ================================================================
# Main entry point
# ================================================================
def analyze_resolution(df: pd.DataFrame):
    """
    Unified API for:
        HR window detection
        baseline calculation
        resolution comparison
        retention estimation
        global confidence scoring

    Input:
      df  — wide-format dataframe with either:
              • a 'timestamp' column, or
              • a DatetimeIndex.

    Returns:
        {
          "hr_start": ...,
          "hr_end": ...,
          "baseline": {...},
          "resolution_map": {...},
          "retention_days": float,
          "global_confidence": {...}
        }
    """
    if df is None or df.empty:
        return {
            "hr_start": None,
            "hr_end": None,
            "baseline": {},
            "resolution_map": {},
            "retention_days": None,
            "global_confidence": {
                "heat_confidence": "unknown",
                "cop_confidence": "unknown",
            },
        }

    # Normalise input to have a proper 'timestamp' column
    df_ts = _ensure_timestamp_column(df)

    hr_start, hr_end = detect_high_res_window(df_ts)

    baseline = compute_baseline_intervals(df_ts, hr_start, hr_end)

    resolution_map = classify_resolution_vs_baseline(df_ts, baseline)

    retention_days = estimate_retention_days(df_ts)

    global_conf = compute_global_confidence(resolution_map)

    return {
        "hr_start": hr_start,
        "hr_end": hr_end,
        "baseline": baseline,
        "resolution_map": resolution_map,
        "retention_days": retention_days,
        "global_confidence": global_conf,
    }
