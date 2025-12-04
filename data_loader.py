# data_loader.py
import pandas as pd
import streamlit as st

def load_and_clean_data(files, user_config, progress_cb=None):
    """
    Robust data loader that handles both Numeric (Float) and State (Text) CSV files.
    """
    if not files:
        return None
    
    numeric_dfs = []
    state_dfs = []
    
    # 1. Read and Classify Files
    for i, f in enumerate(files):
        try:
            # Read CSV
            f.seek(0)
            temp = pd.read_csv(f)
            
            # Parse Time
            if 'Time' in temp.columns:
                temp['Time'] = pd.to_datetime(temp['Time'], dayfirst=True, errors='coerce')
                temp = temp.dropna(subset=['Time'])
            
            # Classify based on columns
            if 'state' in temp.columns:
                state_dfs.append(temp)
            elif 'value' in temp.columns:
                numeric_dfs.append(temp)
                
            if progress_cb: 
                progress_cb(f"Read {f.name}", (i / len(files)) * 0.2)
                
        except Exception as e:
            st.error(f"Error reading {f.name}: {e}")
            
    # 2. Process Numeric Data (Resample: Mean)
    df_numeric_wide = pd.DataFrame()
    if numeric_dfs:
        if progress_cb: progress_cb("Processing numeric data...", 0.3)
        df_num = pd.concat(numeric_dfs)
        # Pivot: Index=Time, Columns=entity_id, Values=value
        # We group by Time/entity_id first to handle any duplicate timestamps
        df_numeric_wide = df_num.groupby(['Time', 'entity_id'])['value'].mean().unstack()
        # Resample to 1 minute, interpolating missing values
        df_numeric_wide = df_numeric_wide.resample('1min').mean().interpolate(limit=30)
        
    # 3. Process State Data (Resample: FFill)
    df_state_wide = pd.DataFrame()
    if state_dfs:
        if progress_cb: progress_cb("Processing state data...", 0.5)
        df_state = pd.concat(state_dfs)
        # Pivot: Index=Time, Columns=entity_id, Values=state
        # For state, we take the 'last' known state if duplicates exist
        df_state_wide = df_state.groupby(['Time', 'entity_id'])['state'].last().unstack()
        # Resample to 1 minute, FORWARD FILLING the state (state persists until changed)
        df_state_wide = df_state_wide.resample('1min').ffill()
        
    # 4. Merge
    if progress_cb: progress_cb("Merging datasets...", 0.6)
    if df_numeric_wide.empty and df_state_wide.empty:
        return None
    elif df_numeric_wide.empty:
        combined_df = df_state_wide
    elif df_state_wide.empty:
        combined_df = df_numeric_wide
    else:
        # Outer join to align timestamps
        combined_df = df_numeric_wide.join(df_state_wide, how='outer')

    # 5. Apply Mapping
    if progress_cb: progress_cb("Applying sensor mapping...", 0.8)
    if user_config and "mapping" in user_config:
        # Create reverse map: {Raw_ID: Friendly_Column}
        # The config has {Friendly: Raw}
        forward_map = user_config["mapping"]
        reverse_map = {v: k for k, v in forward_map.items()}
        combined_df = combined_df.rename(columns=reverse_map)
    
    # 6. Final Cleanup
    # Ensure index is sorted
    combined_df = combined_df.sort_index()
    # Fill small gaps
    combined_df = combined_df.ffill(limit=5)
    
    return {
        "df": combined_df,
        "raw_history": combined_df.copy(),
        "baselines": None,
        "patterns": None
    }