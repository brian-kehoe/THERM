# ha_loader.py
# Home Assistant → Engine-ready dataframe loader for THERM
# Updated for public-beta-v4.3
#
# Key behaviours:
#   - Loads HA long-form CSV (entity_id, state, last_changed/last_updated/time)
#   - Infers dtype per entity (binary / numeric / string)
#   - Converts to wide, 1-minute resampled dataframe
#   - Interprets Samsung Modbus state/mode sensors into synthetic columns
#       * DHW_Mode_Label, DHW_Mode_Is_Active
#       * DHW_Status_Is_On
#       * Defrost_Is_Active
#       * Immersion_Is_On
#       * Valve_Is_DHW, Valve_Is_Heating, Valve_Position_Label
#   - NEW: Interpretation can use either:
#       * Raw Modbus entities (default), or
#       * Optional "HA mapped" entities (template / interpreted sensors),
#         controlled via user_config["state_mode_sources"].
#   - Applies user mapping for core engine roles (Power, FlowTemp, etc.)
#   - Runs engine gatekeepers, run detection, and daily stats.
#
# All raw columns from HA remain untouched. Synthetic columns are added
# after pivot/resample and BEFORE physics, and are not renamed by mapping.


import pandas as pd
import numpy as np
import time
from typing import List, Dict, Any, Callable, Optional

from inspector import is_binary_sensor, safe_smart_parse
import processing


# ----------------------------------------------------------------------
# CONSTANTS: default raw Modbus entity IDs for state/mode roles
# ----------------------------------------------------------------------

DEFAULT_RAW_ENTITIES = {
    "DHW_Mode": "sensor.heat_pump_hot_water_mode",
    "DHW_Status": "sensor.heat_pump_hot_water_status",
    "Defrost": "sensor.heat_pump_defrost_status",
    "Immersion": "sensor.heat_pump_immersion_heater_mode",
    "Valve": "sensor.heat_pump_3way_valve_position",
}


# ----------------------------------------------------------------------
# HELPERS: dtype inference and value conversion
# ----------------------------------------------------------------------

def _infer_dtype(series: pd.Series) -> str:
    """
    Infer dtype for a single sensor based on HA long-form 'state' series.
    Returns: "binary", "numeric", or "string".

    Improvements for Modbus / HA:
        - Classic HA binary detection via is_binary_sensor().
        - Numeric series that contain ONLY {0, 1} are treated as binary.
    """
    values = series.dropna().astype(str)

    # 1. Classic HA binary semantics (on/off, true/false, etc.)
    if is_binary_sensor(values):
        return "binary"

    # 2. Mostly numeric? (including Modbus integers, floats)
    parsed, mostly_numeric = safe_smart_parse(values)
    if mostly_numeric:
        try:
            num = pd.to_numeric(parsed, errors="coerce").dropna()
            uniq = set(num.unique().tolist())
            # Treat pure 0/1 numeric registers as binary
            if uniq and uniq.issubset({0, 1}):
                return "binary"
        except Exception:
            return "numeric"
        return "numeric"

    # 3. Fallback: treat as string
    return "string"


def _convert_value(val, dtype: str):
    """
    Convert HA 'state' values to proper Python values based on inferred dtype.
    """
    if dtype == "binary":
        s = str(val).strip().lower()
        return 1 if s in ("on", "true", "1", "yes") else 0

    elif dtype == "numeric":
        try:
            return float(val)
        except Exception:
            return np.nan

    else:
        # string / categorical
        return val


# ----------------------------------------------------------------------
# MODBUS INTERPRETATION (dual-source: raw or HA-mapped)
# ----------------------------------------------------------------------

def _pick_state_source(
    df: pd.DataFrame,
    user_config: Dict[str, Any],
    role_key: str,
) -> tuple[Optional[str], Optional[str]]:
    """
    Select which column to use for a given logical role (DHW_Mode, DHW_Status, etc.).

    Priority:
        1) If state_mode_sources[role_key]["source"] == "mapped" and mapped_entity_id
           exists in df.columns → use mapped.
        2) Else if raw_entity_id exists in df.columns → use raw.
        3) Else → (None, None) meaning "no data available for this role".

    user_config["state_mode_sources"] is optional. If absent or incomplete,
    this function defaults to raw Modbus entity IDs from DEFAULT_RAW_ENTITIES.
    """
    sources_cfg = user_config.get("state_mode_sources", {})
    cfg = sources_cfg.get(role_key, {})

    default_raw = DEFAULT_RAW_ENTITIES.get(role_key)
    source_mode = cfg.get("source", "raw")  # "raw" or "mapped"
    raw_id = cfg.get("raw_entity_id", default_raw)
    mapped_id = cfg.get("mapped_entity_id", None)

    # Prefer mapped if explicitly requested and present in df
    if source_mode == "mapped" and mapped_id and mapped_id in df.columns:
        return "mapped", mapped_id

    # Otherwise fall back to raw if present
    if raw_id and raw_id in df.columns:
        return "raw", raw_id

    # Nothing available
    return None, None


def enrich_modbus_interpretation(df: pd.DataFrame, user_config: Dict[str, Any]) -> pd.DataFrame:
    """
    Insert interpreted columns for Samsung Modbus state/mode sensors (or their
    HA-mapped equivalents) INTO THE WIDE DATAFRAME, AFTER pivot/resample.

    Logical roles and their synthetic outputs:

        Role "DHW_Mode":
            - DHW_Mode_Label
            - DHW_Mode_Is_Active

        Role "DHW_Status":
            - DHW_Status_Is_On
            (DHW_Status_Raw is implicitly the underlying column if raw used)

        Role "Defrost":
            - Defrost_Is_Active

        Role "Immersion":
            - Immersion_Is_On

        Role "Valve":
            - Valve_Is_DHW
            - Valve_Is_Heating
            - Valve_Position_Label

    Behaviour:
        - If source == "raw": interpret numeric Modbus codes.
        - If source == "mapped": trust the mapped sensor as already-interpreted
          and derive booleans/labels with conservative logic.
        - Raw HA columns remain untouched.
    """

    df = df.copy()

    # ------------------------------------------------------------
    # 1. DHW Mode (logical role "DHW_Mode")
    # ------------------------------------------------------------
    mode_src, mode_col = _pick_state_source(df, user_config, "DHW_Mode")
    if mode_src and mode_col in df.columns:
        s = df[mode_col]

        if mode_src == "raw":
            # Raw Modbus register (73): 0 Eco, 1 Standard, 2 Power, 3 Force (on some units)
            mode_map = {
                0: "Eco",
                1: "Standard",
                2: "Power",
                3: "Force",
            }
            df["DHW_Mode_Label"] = s.map(mode_map).fillna("Unknown")
            df["DHW_Mode_Is_Active"] = s.apply(
                lambda v: 0 if pd.isna(v) else int(v != 0)
            )
        else:
            # Mapped: assume s is already label/enum-like
            # We still provide a boolean "is active" guard.
            df["DHW_Mode_Label"] = s.astype("string")
            df["DHW_Mode_Is_Active"] = s.apply(
                lambda v: 0 if pd.isna(v) else int(str(v).strip().lower() not in ("0", "off", "idle", "eco-off"))
            )

    # ------------------------------------------------------------
    # 2. DHW Status (logical role "DHW_Status")
    # ------------------------------------------------------------
    status_src, status_col = _pick_state_source(df, user_config, "DHW_Status")
    if status_src and status_col in df.columns:
        s = df[status_col]

        if status_src == "raw":
            # Raw Modbus register (72): 0, 1, 360 etc.
            # Treat > 0 as "on".
            df["DHW_Status_Is_On"] = s.apply(
                lambda v: 0 if pd.isna(v) else int(v > 0)
            )
            # Raw values remain in the original column; if you want an
            # explicit alias for debugging, you can add it here:
            # df["DHW_Status_Raw"] = s
        else:
            # Mapped: expect boolean/int-like or "on"/"off"
            def _mapped_dhw_on(val):
                if pd.isna(val):
                    return 0
                sval = str(val).strip().lower()
                if sval in ("on", "true", "1", "yes", "active", "heating"):
                    return 1
                try:
                    return 1 if float(sval) > 0 else 0
                except Exception:
                    return 0

            df["DHW_Status_Is_On"] = s.apply(_mapped_dhw_on)

    # ------------------------------------------------------------
    # 3. Defrost Status (logical role "Defrost")
    # ------------------------------------------------------------
    defrost_src, defrost_col = _pick_state_source(df, user_config, "Defrost")
    if defrost_src and defrost_col in df.columns:
        s = df[defrost_col]

        if defrost_src == "raw":
            # Raw defrost status register (2): 0 idle, non-zero various active / transitions.
            df["Defrost_Is_Active"] = s.apply(
                lambda v: 0 if pd.isna(v) else int(v != 0)
            )
        else:
            # Mapped: assume boolean-ish values (on/off, true/false, etc.)
            def _mapped_defrost(val):
                if pd.isna(val):
                    return 0
                sval = str(val).strip().lower()
                if sval in ("on", "true", "1", "yes", "defrost", "active"):
                    return 1
                try:
                    return 1 if float(sval) > 0 else 0
                except Exception:
                    return 0

            df["Defrost_Is_Active"] = s.apply(_mapped_defrost)

    # ------------------------------------------------------------
    # 4. Immersion Heater Mode (logical role "Immersion")
    # ------------------------------------------------------------
    imm_src, imm_col = _pick_state_source(df, user_config, "Immersion")
    if imm_src and imm_col in df.columns:
        s = df[imm_col]

        if imm_src == "raw":
            # Raw Modbus (85): 0 off, 1 on.
            df["Immersion_Is_On"] = s.apply(
                lambda v: 0 if pd.isna(v) else int(v == 1)
            )
        else:
            # Mapped: boolean-ish
            def _mapped_imm(val):
                if pd.isna(val):
                    return 0
                sval = str(val).strip().lower()
                if sval in ("on", "true", "1", "yes", "heating"):
                    return 1
                try:
                    return 1 if float(sval) > 0 else 0
                except Exception:
                    return 0

            df["Immersion_Is_On"] = s.apply(_mapped_imm)

    # ------------------------------------------------------------
    # 5. 3-way Valve Position (logical role "Valve")
    # ------------------------------------------------------------
    valve_src, valve_col = _pick_state_source(df, user_config, "Valve")
    if valve_src and valve_col in df.columns:
        s = df[valve_col]

        if valve_src == "raw":
            # Raw Modbus (89): 0 = heating, 1 = DHW.
            df["Valve_Is_DHW"] = s.apply(
                lambda v: 0 if pd.isna(v) else int(v == 1)
            )
            df["Valve_Is_Heating"] = s.apply(
                lambda v: 0 if pd.isna(v) else int(v == 0)
            )

            def _raw_valve_label(v):
                if pd.isna(v):
                    return None
                if v == 0:
                    return "Heating"
                if v == 1:
                    return "DHW"
                return f"Unknown({int(v)})"

            df["Valve_Position_Label"] = s.apply(_raw_valve_label)

        else:
            # Mapped: we try both numeric and string possibilities.
            def _mapped_valve_flags(val):
                if pd.isna(val):
                    return 0, 0, None
                sval = str(val).strip().lower()
                # Common label-style values
                if sval in ("heating", "space", "rad", "rads", "space_heat"):
                    return 0, 1, "Heating"
                if sval in ("dhw", "hot water", "tank", "cylinder"):
                    return 1, 0, "DHW"
                # Numeric-ish values
                try:
                    fv = float(sval)
                    if fv == 0:
                        return 0, 1, "Heating"
                    if fv == 1:
                        return 1, 0, "DHW"
                    return 0, 0, f"Unknown({fv:g})"
                except Exception:
                    return 0, 0, f"Unknown({sval})"

            mapped = s.apply(_mapped_valve_flags)
            df["Valve_Is_DHW"] = mapped.apply(lambda x: x[0])
            df["Valve_Is_Heating"] = mapped.apply(lambda x: x[1])
            df["Valve_Position_Label"] = mapped.apply(lambda x: x[2])

    return df


# ----------------------------------------------------------------------
# SAMPLING QUALITY DETECTION
# ----------------------------------------------------------------------

def _infer_sampling_meta(
    grouped: pd.DataFrame,
    ts_col: str,
    user_config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Infer whether the HA feed is sparse/hourly so we can avoid per-run logic.

    Heuristic:
      - Look at the mapped Power entity in the pre-resample grouped data.
      - Compute median spacing between samples and overall coverage ratio
        (events / expected per-minute rows over the timespan).
      - Flag as sparse if median spacing is >= 45 minutes OR coverage < 0.2.
    """
    mapping = user_config.get("mapping", {}) if isinstance(user_config, dict) else {}
    power_entity = mapping.get("Power")

    meta = {
        "power_entity": power_entity,
        "median_sample_seconds": None,
        "coverage_ratio": None,
        "low_res_days_ratio": None,
        "low_res_day_count": None,
        "total_day_count": None,
        "high_res_day_count": None,
        "high_res_threshold_per_day": 300,
        "high_res_run_window_start": None,
        "high_res_run_window_end": None,
        "high_res_run_window_rows": None,
        "is_sparse_power_sampling": False,
    }

    if not power_entity or grouped.empty:
        return meta

    power_rows = grouped[grouped["entity_id"] == power_entity]
    if power_rows.empty:
        return meta

    ts = pd.to_datetime(power_rows[ts_col], errors="coerce").dropna().sort_values()
    if len(ts) < 3:
        return meta

    deltas = ts.diff().dropna().dt.total_seconds()
    if deltas.empty:
        return meta

    median_secs = float(deltas.median())
    span_secs = (ts.iloc[-1] - ts.iloc[0]).total_seconds()
    expected_minutes = max(1.0, span_secs / 60.0)
    coverage_ratio = float(len(ts) / expected_minutes)

    per_day_counts = ts.dt.floor("D").value_counts().sort_index()
    low_res_days = (per_day_counts < 300).sum()  # ~5-min cadence or worse
    total_days = len(per_day_counts)
    low_res_ratio = float(low_res_days / total_days) if total_days else None

    # High-resolution days (sufficient samples to run per-run analysis)
    high_res_threshold = meta["high_res_threshold_per_day"]
    high_res_days = [d for d, c in per_day_counts.items() if c >= high_res_threshold]
    meta["high_res_day_count"] = len(high_res_days)

    # Compute the first contiguous block of high-res days (assumes retention starts high-res then drops)
    run_start_day = None
    run_end_day = None
    if high_res_days:
        sorted_days = sorted(high_res_days)
        run_start_day = sorted_days[0]
        run_end_day = run_start_day
        for day in sorted_days[1:]:
            if (day - run_end_day) <= pd.Timedelta(days=1):
                run_end_day = day
            else:
                break  # stop at first gap; we only use the initial high-res retention window

    # If no high-res block exists, treat as sparse; otherwise keep runs enabled even if later days are low-res
    is_sparse = False
    if run_start_day is None:
        is_sparse = (median_secs >= 45 * 60) or (coverage_ratio < 0.2) or (low_res_ratio is not None and low_res_ratio >= 0.5)

    meta["median_sample_seconds"] = median_secs
    meta["coverage_ratio"] = coverage_ratio
    meta["low_res_days_ratio"] = low_res_ratio
    meta["low_res_day_count"] = int(low_res_days)
    meta["total_day_count"] = int(total_days)
    if run_start_day is not None:
        meta["high_res_run_window_start"] = run_start_day.isoformat()
        # include full final day (up to 23:59)
        run_end_ts = run_end_day + pd.Timedelta(days=1) - pd.Timedelta(minutes=1)
        meta["high_res_run_window_end"] = run_end_ts.isoformat()
    meta["is_sparse_power_sampling"] = is_sparse

    return meta


# ----------------------------------------------------------------------
# MAIN ENTRY POINT
# ----------------------------------------------------------------------

def process_ha_files(
    files: List[Any],
    user_config: Dict[str, Any],
    progress_cb: Optional[Callable[[str, float], None]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Load Home Assistant long-form CSV(s), convert to wide 1-minute data,
    apply Modbus/HA-mapped interpretation, apply user mapping, run physics,
    detect runs, and compute daily stats.

    Returns dict with:
        df          → engine dataframe (post-physics)
        raw_history → wide dataframe after interpretation, before physics
        runs        → list of runs
        daily       → daily energy table
        patterns    → None (HA mode does not use patterns yet)
        baselines   → None (HA mode does not use heartbeats yet)
    """

    # ------------------------------------------------------------------
    # 1. LOAD CSVs
    # ------------------------------------------------------------------
    dfs = []
    t_start = time.time()
    total = len(files)

    for i, f in enumerate(files):
        if progress_cb:
            progress_cb(f"Reading HA CSV {i+1}/{total}", (i + 1) / total)
        t_read = time.time()
        f.seek(0)
        df = pd.read_csv(f)
        read_secs = time.time() - t_read
        try:
            import sys
            sys.stdout.write(f"[ha_loader] read_csv secs={read_secs:.3f} rows={len(df)}\n")
        except Exception:
            pass
        dfs.append(df)

    if not dfs:
        return None

    long = pd.concat(dfs, ignore_index=True)

    # ------------------------------------------------------------------
    # 2. VERIFY REQUIRED COLUMNS
    # ------------------------------------------------------------------
    # HA files usually have: entity_id, state, last_changed
    time_cols = [c for c in ("last_changed", "last_updated", "time") if c in long.columns]
    if "entity_id" not in long.columns or "state" not in long.columns or not time_cols:
        return None

    ts_col = time_cols[0]
    long[ts_col] = pd.to_datetime(long[ts_col], errors="coerce")
    long = long.dropna(subset=[ts_col])

    # ------------------------------------------------------------------
    # 3. INFER DTYPES PER ENTITY
    # ------------------------------------------------------------------
    dtype_map: Dict[str, str] = {}
    t_dtype = time.time()
    for eid, grp in long.groupby("entity_id"):
        dtype_map[eid] = _infer_dtype(grp["state"])
    dtype_secs = time.time() - t_dtype

    # ------------------------------------------------------------------
    # 4. CONVERT VALUES
    # ------------------------------------------------------------------
    t_convert = time.time()
    long["value"] = [
        _convert_value(v, dtype_map.get(eid, "string"))
        for v, eid in zip(long["state"], long["entity_id"])
    ]
    convert_secs = time.time() - t_convert

    # ------------------------------------------------------------------
    # 5. GROUP DUPLICATES  ONE VALUE PER (timestamp, entity_id)
    # ------------------------------------------------------------------
    # Optimised: split numeric vs string to avoid slow per-group apply().
    t_group = time.time()
    numeric_entities = {eid for eid, dt in dtype_map.items() if dt != 'string'}
    long['entity_id'] = long['entity_id'].astype('category')

    grouped_frames = []

    if numeric_entities:
        num_mask = long['entity_id'].isin(numeric_entities)
        num_df = long.loc[num_mask, [ts_col, 'entity_id', 'value']]
        grouped_num = (
            num_df.groupby([ts_col, 'entity_id'], sort=False, observed=True)['value']
            .mean()
        )
        grouped_frames.append(grouped_num)

    str_mask = ~long['entity_id'].isin(numeric_entities)
    if str_mask.any():
        str_df = long.loc[str_mask, [ts_col, 'entity_id', 'value']]
        grouped_str = (
            str_df.groupby([ts_col, 'entity_id'], sort=False, observed=True)['value']
            .last()
        )
        grouped_frames.append(grouped_str)

    if grouped_frames:
        grouped = pd.concat(grouped_frames).reset_index()
    else:
        grouped = pd.DataFrame(columns=[ts_col, 'entity_id', 'value'])
    group_secs = time.time() - t_group
    # 6. PIVOT TO WIDE
    # ------------------------------------------------------------------
    t_pivot = time.time()
    wide = grouped.pivot(index=ts_col, columns="entity_id", values="value")
    wide.index = pd.to_datetime(wide.index)
    pivot_secs = time.time() - t_pivot

    # ------------------------------------------------------------------
    # 6b. SAMPLING QUALITY CHECK (pre-resample)
    # ------------------------------------------------------------------
    sampling_meta = _infer_sampling_meta(grouped, ts_col, user_config)

    # ------------------------------------------------------------------
    # 7. RESAMPLE TO 1-MINUTE
    # ------------------------------------------------------------------
    t_resample = time.time()
    df_wide = wide.resample("1min").last().ffill(limit=120)
    resample_secs = time.time() - t_resample

    # ------------------------------------------------------------------
    # 8. APPLY MODBUS / HA-MAPPED INTERPRETATION
    # ------------------------------------------------------------------
    df_wide = enrich_modbus_interpretation(df_wide, user_config)

    # ------------------------------------------------------------------
    # 9. APPLY USER SENSOR MAPPING (core engine roles)
    # ------------------------------------------------------------------
    # user_config["mapping"] maps THERM keys → entity_id strings
    mapping = user_config.get("mapping", {})

    # Reverse: entity_id → THERM key
    rename_dict: Dict[str, str] = {}
    for therm_key, entity_id in mapping.items():
        if entity_id in df_wide.columns:
            rename_dict[entity_id] = therm_key

    df = df_wide.rename(columns=rename_dict)

    # ------------------------------------------------------------------
    # 10. COMPUTE DeltaT IF MAPPED
    # ------------------------------------------------------------------
    if "FlowTemp" in df.columns and "ReturnTemp" in df.columns:
        df["DeltaT"] = df["FlowTemp"] - df["ReturnTemp"]

    # Ensure numeric types for engine core numeric columns
    numeric_cols = ["Power", "FlowTemp", "ReturnTemp", "FlowRate", "Freq", "DeltaT"]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # ------------------------------------------------------------------
    # 11. RUN PHYSICS ENGINE
    # ------------------------------------------------------------------
    t_gate = time.time()
    df_engine = processing.apply_gatekeepers(df, user_config)
    gate_secs = time.time() - t_gate
    if df_engine is None or df_engine.empty:
        return None

    # ------------------------------------------------------------------
    # 12. RUN DETECTOR + DAILY STATS
    # ------------------------------------------------------------------
    t_runs = time.time()
    run_detection_disabled = False
    runs = []

    if sampling_meta.get("is_sparse_power_sampling"):
        run_detection_disabled = True
        try:
            import sys
            med = sampling_meta.get("median_sample_seconds")
            cov = sampling_meta.get("coverage_ratio")
            low = sampling_meta.get("low_res_day_count")
            tot = sampling_meta.get("total_day_count")
            med_str = f"{med:.0f}s" if med is not None else "unknown"
            cov_str = f"{cov:.3f}" if cov is not None else "unknown"
            low_str = f"{low}/{tot}" if low is not None and tot is not None else "unknown"
            sys.stdout.write(
                "[ha_loader] skipping run detection: sparse/hourly sampling "
                f"(median={med_str} coverage={cov_str} low_res_days={low_str})\n"
            )
        except Exception:
            pass
    else:
        run_df = df_engine
        # If we have a high-res retention window, limit run detection to that period
        start_str = sampling_meta.get("high_res_run_window_start")
        end_str = sampling_meta.get("high_res_run_window_end")
        if start_str and end_str:
            start_ts = pd.to_datetime(start_str)
            end_ts = pd.to_datetime(end_str)
            try:
                tz = run_df.index.tz
                if tz is not None:
                    if start_ts.tzinfo is None:
                        start_ts = start_ts.tz_localize(tz)
                    if end_ts.tzinfo is None:
                        end_ts = end_ts.tz_localize(tz)
            except Exception:
                pass
            run_df = run_df.loc[start_ts:end_ts]
            sampling_meta["high_res_run_window_rows"] = int(len(run_df))
        runs = processing.detect_runs(run_df, user_config) if not run_df.empty else []

    t_daily = time.time()
    daily = processing.get_daily_stats(df_engine)
    daily_secs = time.time() - t_daily
    runs_secs = t_daily - t_runs

    total_secs = time.time() - t_start
    try:
        import sys
        sys.stdout.write(
            "[ha_loader] timing "
            f"read={read_secs:.3f}s dtype={dtype_secs:.3f}s convert={convert_secs:.3f}s "
            f"group={group_secs:.3f}s pivot={pivot_secs:.3f}s resample={resample_secs:.3f}s "
            f"gate={gate_secs:.3f}s runs={runs_secs:.3f}s daily={daily_secs:.3f}s total={total_secs:.3f}s\n"
        )
    except Exception:
        pass

    # ------------------------------------------------------------------
    # 13. OUTPUT STRUCTURE (same as Grafana loader)
    # ------------------------------------------------------------------
    return {
        "df": df_engine,
        "raw_history": df,   # wide dataframe after interpretation, before physics
        "runs": runs,
        "daily": daily,
        "sampling_meta": sampling_meta,
        "run_detection_disabled": run_detection_disabled,
        "patterns": None,
        "baselines": None,
    }
