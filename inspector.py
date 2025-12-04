# inspector.py
import pandas as pd
import streamlit as st

def inspect_raw_files(uploaded_files):
    """
    Reads files purely for diagnostic purposes without modifying them.
    Returns a summary dataframe and a detail dictionary.
    """
    summary_stats = []
    file_details = {}

    for uploaded_file in uploaded_files:
        # Reset pointer to read from start
        uploaded_file.seek(0)
        
        file_stat = {
            "Filename": uploaded_file.name,
            "Size (KB)": round(uploaded_file.size / 1024, 1)
        }
        
        try:
            # Read first few lines to sniff structure
            df_raw = pd.read_csv(uploaded_file)
            
            # 1. Basic Counts
            file_stat["Rows"] = len(df_raw)
            file_stat["Columns"] = len(df_raw.columns)
            
            # 2. Timestamp Detection (Improved for Home Assistant)
            time_col = None
            # Standard HA headers + Grafana headers
            candidates = ['time', 'date', 'last_changed', 'last_updated', 'created']
            
            for col in df_raw.columns:
                if any(c in col.lower() for c in candidates):
                    time_col = col
                    break
            
            if time_col:
                try:
                    # FIX: Robust mixed format parsing + explicit string conversion
                    # format='mixed' allows combining ISO8601 (HA) and DayFirst (Grafana)
                    start_ts = pd.to_datetime(df_raw[time_col].iloc[0], dayfirst=True, errors='coerce', format='mixed')
                    end_ts = pd.to_datetime(df_raw[time_col].iloc[-1], dayfirst=True, errors='coerce', format='mixed')
                    file_stat["Start Time"] = str(start_ts)
                    file_stat["End Time"] = str(end_ts)
                except:
                    file_stat["Start Time"] = "Parse Error"
                    file_stat["End Time"] = "Parse Error"
            else:
                file_stat["Start Time"] = "Not Found"
                file_stat["End Time"] = "Not Found"

            # 3. Entity Detection
            entities_found = []
            cols_lower = [str(c).lower() for c in df_raw.columns]
            
            if 'entity_id' in cols_lower:
                structure_type = "Long Format (State)"
                actual_col = df_raw.columns[cols_lower.index('entity_id')]
                # Force entity_id to string to avoid object type issues
                entities_found = df_raw[actual_col].dropna().astype(str).unique().tolist()
            else:
                structure_type = "Wide Format (Table)"
                ignore = ['time', 'value', 'series', 'timestamp', 'last_changed', 'last_updated']
                entities_found = [c for c in df_raw.columns if c.lower() not in ignore]

            file_details[uploaded_file.name] = {
                "structure": structure_type,
                "columns_raw": list(df_raw.columns),
                "entities_found": sorted([str(e) for e in entities_found])
            }
            
        except Exception as e:
            file_stat["Rows"] = "Error"
            file_stat["Start Time"] = "Error"
            file_stat["End Time"] = "Error"
            file_details[uploaded_file.name] = {"error": str(e)}

        summary_stats.append(file_stat)
        # CRITICAL: Reset pointer so the actual data_loader can read it later
        uploaded_file.seek(0)

    # Return as string to avoid PyArrow mixed-type crashes in Streamlit
    return pd.DataFrame(summary_stats).astype(str), file_details