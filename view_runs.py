# view_runs.py
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

def get_friendly_name(internal_key, system_config):
    """
    Helper to resolve 'Room_1' -> 'Living Room' using the session config.
    """
    if not system_config or "mapping" not in system_config:
        return internal_key
    
    # The config mapping is { "Friendly Name": "internal_key" } usually, 
    # but strictly speaking the config passed to backend was { "Friendly": "Raw Entity" }.
    # The processing.py already handled the internal column renaming.
    # We need to find the key in the user's config that maps to this internal column.
    
    # However, in therm v2, the 'mapping' in system_config is:
    # { "Power": "entity_id_1", "Room_1": "entity_id_2" }
    # It does NOT store the user's custom text label (e.g. "Living Room").
    # Wait - the user interface mapping_ui.py usually asks for a label?
    # Inspecting the user's provided JSON: "Room_1": "ecowitt_weather_indoor_temperature"
    # It seems the entity ID IS the friendly name in Home Assistant.
    
    # We will try to make it prettier by cleaning the entity ID.
    
    raw_entity = system_config.get("mapping", {}).get(internal_key, "")
    if raw_entity:
        # distinct "sensor.kitchen_temperature" -> "Kitchen Temperature"
        friendly = raw_entity.replace("sensor.", "").replace("_", " ").title()
        # Remove common suffixes for brevity
        friendly = friendly.replace(" Temperature", "").replace(" Value", "")
        return friendly
        
    return internal_key

def render_run_inspector(df, runs):
    st.subheader("Run Inspector")
    
    if not runs:
        st.info("No runs detected.")
        return

    # Select Run
    run_ids = [r['id'] for r in runs]
    
    # Create a label for the dropdown that shows date + type
    run_options = {r['id']: f"Run {r['id']} | {r['start'].strftime('%d %b %H:%M')} | {r['run_type']}" for r in runs}
    
    selected_id = st.selectbox("Select Run", run_ids, format_func=lambda x: run_options[x])
    
    # Get Data
    run = next(r for r in runs if r['id'] == selected_id)
    mask = (df.index >= run['start']) & (df.index <= run['end'])
    run_df = df.loc[mask]
    
    # Config for labels
    config = st.session_state.get("system_config", {})

    # Metrics Row
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Type", run['run_type'])
    c2.metric("Duration", f"{run['duration_mins']} mins")
    c3.metric("COP", run['run_cop'])
    c4.metric("Zones", run['active_zones'])
    
    # Charts
    
    # 1. Temperatures (Flow/Return/Outdoor)
    fig_temp = go.Figure()
    fig_temp.add_trace(go.Scatter(x=run_df.index, y=run_df['FlowTemp'], name='Flow', line=dict(color='red')))
    fig_temp.add_trace(go.Scatter(x=run_df.index, y=run_df['ReturnTemp'], name='Return', line=dict(color='orange')))
    if 'OutdoorTemp' in run_df.columns:
        fig_temp.add_trace(go.Scatter(x=run_df.index, y=run_df['OutdoorTemp'], name='Outdoor', line=dict(color='blue', dash='dot')))
    
    fig_temp.update_layout(title="System Temperatures", height=300, margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig_temp, use_container_width=True)
    
    # 2. Power & Heat
    fig_pwr = go.Figure()
    fig_pwr.add_trace(go.Scatter(x=run_df.index, y=run_df['Power'], name='Electric (W)', fill='tozeroy', line=dict(color='purple')))
    fig_pwr.add_trace(go.Scatter(x=run_df.index, y=run_df['Heat'], name='Heat (W)', line=dict(color='gold')))
    
    fig_pwr.update_layout(title="Power & Heat Output", height=300, margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig_pwr, use_container_width=True)
    
    # 3. Room Response
    # Find relevant room columns
    room_cols = [c for c in run_df.columns if c.startswith("Room_")]
    
    if room_cols:
        fig_rooms = go.Figure()
        for r in room_cols:
            # Check if this room has data
            if run_df[r].mean() > 0:
                # GET FRIENDLY NAME
                label = get_friendly_name(r, config)
                fig_rooms.add_trace(go.Scatter(x=run_df.index, y=run_df[r], name=label))
                
        fig_rooms.update_layout(title="Room Temperatures", height=300, margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig_rooms, use_container_width=True)
    else:
        st.caption("No room sensors mapped.")