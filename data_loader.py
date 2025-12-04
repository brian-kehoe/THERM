# data_loader.py
import pandas as pd
import numpy as np
import re
from datetime import datetime, timedelta
from config import ENTITY_MAP, SENSOR_ROLES, BASELINE_JSON_PATH
from baselines import build_sensor_baselines, analyze_sensor_reporting_patterns, smart_forward_fill, load_saved_heartbeat_baseline
from utils import _log_warn

# --- FALLBACK MAPPING ---
FALLBACK_OWM_MAP = {
    'sensor.openweathermap_wind_speed': 'Wind_Speed_OWM',
    'sensor.openweathermap_humidity': 'Outdoor_Humidity_OWM',
    'sensor.openweathermap_uv_index': 'UV_Index_OWM',
    'sensor.openweathermap_temperature': 'OutdoorTemp_OWM'
}

def _parse_grafana_timestamp(filename):
    match = re.search(r'-(\d{2}_\d{2}_\d{4} \d{2}_\d{2}_\d{2})\.csv$', filename)
    if match:
        try:
            return datetime.strptime(match.group(1), '%d_%m_%Y %H_%M_%S')
        except ValueError:
            return None
    return None

def find_grafana_pairs(uploaded_files):
    numeric_files = {}
    state_files = {}
    for file_obj in uploaded_files:
        filename = file_obj['fileName']
        ts = _parse_grafana_timestamp(filename)
        if ts is None: continue
        indexed = {'name': filename, 'timestamp': ts, 'handle': file_obj['handle']}
        if "Numeric-data" in filename: numeric_files[ts] = indexed
        elif "State-data" in filename: state_files[ts] = indexed

    paired_files = []
    unmatched_files = [f for f in uploaded_files if _parse_grafana_timestamp(f['fileName']) is None]
    used_state_timestamps = set()
    for num_ts, num_file in numeric_files.items():
        match = None
        for state_ts, state_file in state_files.items():
            if state_file['timestamp'] in used_state_timestamps: continue
            if abs(num_ts - state_ts) <= timedelta(minutes=2):
                match = state_file
                break
        if match:
            paired_files.append((num_file['name'], match['name']))
            used_state_timestamps.add(match['timestamp'])
        else:
            unmatched_files.append({'fileName': num_file['name'], 'handle': num_file['handle']})
    for state_ts, state_file in state_files.items():
        if state_file['timestamp'] not in used_state_timestamps:
            unmatched_files.append({'fileName': state_file['name'], 'handle': state_file['handle']})
    return paired_files, unmatched_files

def _normalize_df_columns(df):
    cleaned_cols = []
    for c in df.columns:
        c = str(c).strip().strip('"').strip("'").replace('\r', '').replace('\n', '').replace('\ufeff', '')
        cleaned_cols.append(c)
    df.columns = cleaned_cols
    if 'Time' in df.columns: df = df.rename(columns={'Time': 'last_changed'})
    df.columns = [c.lower() for c in df.columns]
    if 'last_changed' not in df.columns:
        if 'time' in df.columns: df = df.rename(columns={'time': 'last_changed'})
        elif 'timestamp' in df.columns: df = df.rename(columns={'timestamp': 'last_changed'})
        elif len(df.columns) > 0: df = df.rename(columns={df.columns[0]: 'last_changed'})
    if 'state' not in df.columns and 'value' in df.columns:
        df = df.rename(columns={'value': 'state'})
    return df

def process_grafana_pair(numeric_file_handle, state_file_handle):
    numeric_file_handle.seek(0)
    df_num = pd.read_csv(numeric_file_handle)
    df_num = _normalize_df_columns(df_num)
    df_num = df_num.filter(items=['last_changed', 'state', 'entity_id'])

    state_file_handle.seek(0)
    df_state = pd.read_csv(state_file_handle)
    df_state = _normalize_df_columns(df_state)
    
    if 'state' in df_state.columns and 'value' in df_state.columns:
        df_state['state'] = df_state['state'].astype(str).replace('nan', np.nan).fillna(df_state['value'])
    elif 'value' in df_state.columns:
        df_state = df_state.rename(columns={'value': 'state'})
        
    df_state = df_state.filter(items=['last_changed', 'state', 'entity_id'])
    full_df = pd.concat([df_num, df_state], ignore_index=True)
    full_df['entity_id'] = full_df['entity_id'].astype(str).str.lower()
    return full_df

import pandas as pd
import numpy as np
import re
from datetime import datetime, timedelta
from config import ENTITY_MAP, SENSOR_ROLES, BASELINE_JSON_PATH
from baselines import build_sensor_baselines, analyze_sensor_reporting_patterns, smart_forward_fill, load_saved_heartbeat_baseline
from utils import _log_warn

# --- FALLBACK MAPPING ---
FALLBACK_OWM_MAP = {
    'sensor.openweathermap_wind_speed': 'Wind_Speed_OWM',
    'sensor.openweathermap_humidity': 'Outdoor_Humidity_OWM',
    'sensor.openweathermap_uv_index': 'UV_Index_OWM',
    'sensor.openweathermap_temperature': 'OutdoorTemp_OWM'
}

def _parse_grafana_timestamp(filename):
    match = re.search(r'-(\d{2}_\d{2}_\d{4} \d{2}_\d{2}_\d{2})\.csv$', filename)
    if match:
        try:
            return datetime.strptime(match.group(1), '%d_%m_%Y %H_%M_%S')
        except ValueError:
            return None
    return None

def find_grafana_pairs(uploaded_files):
    numeric_files = {}
    state_files = {}
    for file_obj in uploaded_files:
        filename = file_obj['fileName']
        ts = _parse_grafana_timestamp(filename)
        if ts is None: continue
        indexed = {'name': filename, 'timestamp': ts, 'handle': file_obj['handle']}
        if "Numeric-data" in filename: numeric_files[ts] = indexed
        elif "State-data" in filename: state_files[ts] = indexed

    paired_files = []
    unmatched_files = [f for f in uploaded_files if _parse_grafana_timestamp(f['fileName']) is None]
    used_state_timestamps = set()
    for num_ts, num_file in numeric_files.items():
        match = None
        for state_ts, state_file in state_files.items():
            if state_file['timestamp'] in used_state_timestamps: continue
            if abs(num_ts - state_ts) <= timedelta(minutes=2):
                match = state_file
                break
        if match:
            paired_files.append((num_file['name'], match['name']))
            used_state_timestamps.add(match['timestamp'])
        else:
            unmatched_files.append({'fileName': num_file['name'], 'handle': num_file['handle']})
    for state_ts, state_file in state_files.items():
        if state_file['timestamp'] not in used_state_timestamps:
            unmatched_files.append({'fileName': state_file['name'], 'handle': state_file['handle']})
    return paired_files, unmatched_files

def _normalize_df_columns(df):
    cleaned_cols = []
    for c in df.columns:
        c = str(c).strip().strip('"').strip("'").replace('\r', '').replace('\n', '').replace('\ufeff', '')
        cleaned_cols.append(c)
    df.columns = cleaned_cols
    if 'Time' in df.columns: df = df.rename(columns={'Time': 'last_changed'})
    df.columns = [c.lower() for c in df.columns]
    if 'last_changed' not in df.columns:
        if 'time' in df.columns: df = df.rename(columns={'time': 'last_changed'})
        elif 'timestamp' in df.columns: df = df.rename(columns={'timestamp': 'last_changed'})
        elif len(df.columns) > 0: df = df.rename(columns={df.columns[0]: 'last_changed'})
    if 'state' not in df.columns and 'value' in df.columns:
        df = df.rename(columns={'value': 'state'})
    return df

def process_grafana_pair(numeric_file_handle, state_file_handle):
    numeric_file_handle.seek(0)
    df_num = pd.read_csv(numeric_file_handle)
    df_num = _normalize_df_columns(df_num)
    df_num = df_num.filter(items=['last_changed', 'state', 'entity_id'])

    state_file_handle.seek(0)
    df_state = pd.read_csv(state_file_handle)
    df_state = _normalize_df_columns(df_state)
    
    if 'state' in df_state.columns and 'value' in df_state.columns:
        df_state['state'] = df_state['state'].astype(str).replace('nan', np.nan).fillna(df_state['value'])
    elif 'value' in df_state.columns:
        df_state = df_state.rename(columns={'value': 'state'})
        
    df_state = df_state.filter(items=['last_changed', 'state', 'entity_id'])
    full_df = pd.concat([df_num, df_state], ignore_index=True)
    full_df['entity_id'] = full_df['entity_id'].astype(str).str.lower()
    return full_df

def load_and_clean_data(uploaded_files, progress_cb=None):
    # --- Performance Timer Helper ---
    t_start = datetime.now()
    t_last = t_start
    
    def log_step(step_name):
        nonlocal t_last
        now = datetime.now()
        duration = (now - t_last).total_seconds()
        print(f"[{now.strftime('%H:%M:%S')}] {step_name} ({duration:.2f}s)")
        t_last = now

    print(f"[{t_start.strftime('%H:%M:%S')}] START: load_and_clean_data")
    
    uploaded_file_index = [{'fileName': f.name, 'handle': f} for f in uploaded_files]
    if progress_cb: progress_cb("Pairing files...", 10)
    
    paired_files, unmatched_files = find_grafana_pairs(uploaded_file_index)
    all_dfs = []
    total_files = len(paired_files) * 2 + len(unmatched_files)
    processed = 0

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Loading {len(paired_files)} pairs and {len(unmatched_files)} unmatched files...")

    # 1. Load Paired Grafana Files (IMMEDIATE DATE PARSING)
    for num_file_name, state_file_name in paired_files:
        try:
            num_h = next(f['handle'] for f in uploaded_file_index if f['fileName'] == num_file_name)
            state_h = next(f['handle'] for f in uploaded_file_index if f['fileName'] == state_file_name)
            
            df_pair = process_grafana_pair(num_h, state_h)
            
            # FAST PATH: Grafana is typically DD/MM/YYYY.
            if 'last_changed' in df_pair.columns:
                df_pair['last_changed'] = pd.to_datetime(df_pair['last_changed'], dayfirst=True, utc=True, errors='coerce')
                
            all_dfs.append(df_pair)
            processed += 2
            if progress_cb:
                pct = 10 + int((processed / max(total_files, 1)) * 25)
                progress_cb(f"Loading paired files ({processed}/{total_files})...", pct)
        except Exception: pass

    # 2. Load Unmatched Files (IMMEDIATE DATE PARSING)
    for file_obj in unmatched_files:
        try:
            file_obj['handle'].seek(0)
            df = pd.read_csv(file_obj['handle'])
            df = _normalize_df_columns(df)
            
            if 'last_changed' in df.columns and 'state' in df.columns:
                if 'entity_id' in df.columns:
                    df['entity_id'] = df['entity_id'].astype(str).str.lower()
                    
                    # FAST PATH: HA History is typically ISO8601.
                    df['last_changed'] = pd.to_datetime(df['last_changed'], utc=True, errors='coerce')
                    
                    all_dfs.append(df.filter(items=['last_changed', 'state', 'entity_id']))
            processed += 1
            if progress_cb:
                pct = 10 + int((processed / max(total_files, 1)) * 25)
                progress_cb(f"Loading unmatched files ({processed}/{total_files})...", pct)
        except Exception: pass

    log_step("File I/O & Initial Parse")

    if not all_dfs: 
        print("ABORT: No valid dataframes found.")
        return None
        
    full_df = pd.concat(all_dfs, ignore_index=True).sort_values('last_changed')
    log_step(f"Merging {len(full_df)} rows")

    if progress_cb: progress_cb("Mapping entity IDs...", 40)

    # 3. Entity Normalization (CATEGORICAL + O(1) LOOKUP)
    ACTIVE_MAP = {**FALLBACK_OWM_MAP, **ENTITY_MAP}
    active_map_lower = {k.strip().lower(): k.strip().lower() for k in ACTIVE_MAP.keys()}
    
    unique_ids = full_df['entity_id'].unique()
    id_map = {}
    
    for raw_id in unique_ids:
        s_id = str(raw_id).strip().lower()
        if s_id in active_map_lower:
            id_map[raw_id] = s_id
            continue
        
        found_prefix = False
        prefixes = ['sensor.', 'binary_sensor.', 'switch.']
        for prefix in prefixes:
            if prefix + s_id in active_map_lower:
                id_map[raw_id] = prefix + s_id
                found_prefix = True
                break
        
        if not found_prefix:
            id_map[raw_id] = s_id

    # O(1) Mapping: Convert to categorical -> Rename categories -> Convert back
    full_df['entity_id'] = full_df['entity_id'].astype('category')
    full_df['entity_id'] = full_df['entity_id'].map(id_map).astype(str)
    
    log_step("Entity Mapping (Categorical)")

    # Identify Unmapped Entities
    valid_keys = set(k.strip().lower() for k in ACTIVE_MAP.keys())
    final_unique = set(full_df['entity_id'].unique())
    unmapped_entities = sorted(list(final_unique - valid_keys))
    
    if not full_df.empty:
        # Timezone Cleanup (No parsing needed, fast)
        if full_df['last_changed'].dt.tz is not None:
            full_df['last_changed'] = full_df['last_changed'].dt.tz_localize(None)
        full_df = full_df.dropna(subset=['last_changed'])
        
    log_step("Timezone Cleanup")

    if progress_cb: progress_cb("Cleaning numeric data (Vectorized)...", 50)

    # 4. Binary/Numeric Cleaning (VECTORIZED - NEW!)
    # We replace the slow .apply() loop with fast vectorized operations
    
    # A. Identify rows to clean
    # Using regex=False for simple string matching is faster
    binary_mask = (
        full_df['entity_id'].str.startswith('binary_', na=False) | 
        full_df['entity_id'].str.contains('defrost', regex=False, na=False)
    )

    if binary_mask.any():
        # B. Extract subset
        binary_subset = full_df.loc[binary_mask, 'state'].astype(str).str.lower()
        
        # C. Map 'on'/'off' strings to numbers (Fast Dictionary Map)
        # This handles the bulk of the work instantly
        mapped_vals = binary_subset.map({'on': 1, 'off': 0})
        
        # D. For anything not 'on'/'off', try standard numeric conversion
        # The 'fillna' handles the cases where map returned NaN (i.e. it wasn't on/off)
        numeric_vals = pd.to_numeric(binary_subset, errors='coerce')
        final_vals = mapped_vals.fillna(numeric_vals).fillna(0.0)
        
        # E. Assign back
        full_df.loc[binary_mask, 'state'] = final_vals

    log_step("Cleaning Numeric Data (Vectorized)")

    if full_df.empty: return None

    raw_history_df = full_df[['last_changed', 'entity_id']].copy()

    # 5. Load Baselines
    try:
        if not full_df.empty:
            month_values = full_df["last_changed"].dt.month.dropna()
            current_month = int(month_values.mode()[0]) if len(month_values) > 0 else datetime.now().month
        else:
            current_month = datetime.now().month
        saved_baselines, source_path = load_saved_heartbeat_baseline(BASELINE_JSON_PATH, current_month=current_month)
    except Exception as exc:
        _log_warn(f"Failed to load seasonal baseline: {exc}")
        saved_baselines, source_path = {}, None

    if progress_cb: progress_cb("Resampling & filling gaps...", 70)
    
    # Pivot
    df_pivot = full_df.pivot_table(index='last_changed', columns='entity_id', values='state', aggfunc='last')
    log_step("Pivoting")
    
    # 6. Resample
    df_res = df_pivot.resample('1min').last()

    runtime_baselines = build_sensor_baselines(full_df, SENSOR_ROLES)
    baselines = {**saved_baselines, **runtime_baselines}
    
    sensor_patterns = analyze_sensor_reporting_patterns(df_pivot, baselines=baselines)
    
    # Smart Forward Fill
    df_res = smart_forward_fill(df_res, sensor_patterns)
    log_step("Smart Filling")
    
    # 7. Rename Entities
    clean_map = {k.strip().lower(): v for k, v in ACTIVE_MAP.items()}
    df_res = df_res.rename(columns=clean_map)

    # 8. Post-Processing & Types
    if 'Immersion_Mode' in df_res.columns:
        df_res['Immersion_Mode'] = df_res['Immersion_Mode'].apply(lambda x: 1 if str(x).lower() == 'on' else 0)
    if 'Quiet_Mode' in df_res.columns:
        df_res['Quiet_Mode'] = df_res['Quiet_Mode'].apply(lambda x: 1 if str(x).lower() == 'on' else 0)
    
    if 'DHW_Mode' in df_res.columns:
        def _norm_dhw_mode(val):
            if pd.isna(val): return np.nan
            s = str(val).strip().lower()
            if s in ("economic", "eco"): return "Economic"
            if s in ("standard", "std"): return "Standard"
            if s in ("power", "boost", "forced"): return "Power"
            return np.nan
        df_res['DHW_Mode'] = df_res['DHW_Mode'].apply(_norm_dhw_mode)

    # Final robust numeric conversion
    for col in [c for c in df_res.columns if c not in ['ValveMode', 'DHW_Mode']]:
        try:
            df_res[col] = pd.to_numeric(df_res[col], errors='coerce')
        except Exception: pass
        
    log_step("Final Clean")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] DONE: load_and_clean_data complete.")
        
    return {
        "df": df_res,
        "raw_history": raw_history_df,
        "baselines": baselines,
        "baseline_path": source_path,
        "patterns": sensor_patterns,
        "unmapped_entities": unmapped_entities
    }