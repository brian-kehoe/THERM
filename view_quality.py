# view_quality.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

def expected_window_series(daily_df, mode='active'):
    """
    Generates a boolean series where True = Data is expected to be present.
    mode: 'active' (any heating/DHW) or 'dhw_active' (only DHW)
    """
    # Create default 'always true' mask
    mask = pd.Series(True, index=daily_df.index)
    
    # If we want to filter based on system usage:
    # We look for days where the system was actually doing something.
    
    system_on_minutes = daily_df.get('Active_Mins', pd.Series(0, index=daily_df.index))
    
    if mode == 'active':
        # Expect data if system was active for > 10 mins
        mask = system_on_minutes > 10
    elif mode == 'dhw_active':
        # FIX: Changed 'Total_DHW_Mins' to 'DHW_Mins' to match processing.py output
        base = daily_df.get('DHW_Mins', system_on_minutes)
        mask = base > 5
        
    return mask

def check_missing_blocks(df, col_name, threshold_mins=60):
    """Detects continuous blocks of missing data > threshold."""
    if col_name not in df.columns:
        return []
        
    series = df[col_name]
    is_missing = series.isna()
    
    # Group consecutive missing values
    # We use a trick: (is_missing != is_missing.shift()).cumsum() creates a group ID
    groups = is_missing.groupby((is_missing != is_missing.shift()).cumsum())
    
    missing_blocks = []
    
    for _, group in groups:
        if group.iloc[0]: # If this group is 'True' (Missing)
            duration = len(group) # Assuming 1-minute frequency
            if duration >= threshold_mins:
                missing_blocks.append({
                    "start": group.index[0],
                    "end": group.index[-1],
                    "duration": duration
                })
                
    return missing_blocks

def render_data_quality(daily, df, unmapped, patterns, baseline_path):
    st.header("🛡️ Data Quality Audit")
    
    if daily is None or df is None:
        st.error("No data available for audit.")
        return

    # 1. High Level Scores
    col1, col2, col3 = st.columns(3)
    
    score = daily['DQ_Score'].mean()
    tier = "Tier 1 (Gold)" if score > 95 else "Tier 2 (Silver)" if score > 80 else "Tier 3 (Bronze)"
    
    col1.metric("Overall completeness", f"{score:.1f}%")
    col2.metric("Data Tier", tier)
    col3.metric("Days Analyzed", len(daily))
    
    st.divider()
    
    # 2. Sensor Health Table
    st.subheader("Sensor Health Check")
    
    sensors = ['Power', 'FlowTemp', 'ReturnTemp', 'FlowRate', 'OutdoorTemp']
    health_data = []
    
    for s in sensors:
        if s in df.columns:
            missing = df[s].isna().sum()
            total = len(df)
            pct = 100 - (missing/total * 100)
            
            # Check logic blocks
            blocks = check_missing_blocks(df, s)
            
            health_data.append({
                "Sensor": s,
                "Completeness": f"{pct:.1f}%",
                "Missing Blocks (>1h)": len(blocks),
                "Status": "✅ OK" if pct > 98 else "⚠️ Check"
            })
    
    st.dataframe(pd.DataFrame(health_data), use_container_width=True)
    
    # 3. Baseline Comparison
    if baseline_path:
        st.success(f"Comparing against baseline: {baseline_path}")
    else:
        st.info("No baseline signature found. (This feature learns from your specific heat pump over time)")