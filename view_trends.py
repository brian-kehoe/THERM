# view_trends.py
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from utils import safe_div
from schema_defs import check_feature_availability

def get_friendly_name(internal_key, system_config):
    """Helper to resolve internal keys to friendly names."""
    if not system_config or "mapping" not in system_config:
        return internal_key
    
    raw_entity = system_config.get("mapping", {}).get(internal_key, "")
    if raw_entity:
        # Clean up entity IDs to look like titles
        friendly = raw_entity.replace("sensor.", "").replace("_", " ").title()
        friendly = friendly.replace(" Temperature", "").replace(" Value", "")
        return friendly
    return internal_key

def render_long_term_trends(daily, df, runs):
    st.header("📈 Long-Term Trends")
    
    if daily is None or daily.empty:
        st.info("No daily statistics available.")
        return

    # Check Features
    features = check_feature_availability(df)
    
    # === TAB SELECTION ===
    tabs = ["Overview", "Efficiency (COP)", "Temperatures"]
    
    # Unlock DHW tab if we have ANY DHW data (relaxed check)
    has_dhw_data = (daily['DHW_Mins'].sum() > 0) or features.get('dhw_analysis', False)
    if has_dhw_data:
        tabs.append("DHW Analysis")
        
    if "Active_Zones_Count" in df.columns:
        tabs.append("Zone Analysis")

    selected_tab = st.radio("Select View", tabs, horizontal=True)
    st.divider()

    # === 1. OVERVIEW ===
    if selected_tab == "Overview":
        # KPIs
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Heat", f"{daily['Total_Heat_kWh'].sum():.1f} kWh")
        c2.metric("Total Elec", f"{daily['Total_Electricity_kWh'].sum():.1f} kWh")
        
        # SCOP Calculation
        scop = safe_div(daily['Total_Heat_kWh'].sum(), daily['Total_Electricity_kWh'].sum())
        c3.metric("Period SCOP", f"{scop:.2f}")
        c4.metric("Active Days", len(daily))

        # Stacked Bar: Energy
        fig = go.Figure()
        fig.add_trace(go.Bar(x=daily.index, y=daily['Electricity_Heating_kWh'], name='Heating Elec'))
        if has_dhw_data:
            fig.add_trace(go.Bar(x=daily.index, y=daily['Electricity_DHW_kWh'], name='DHW Elec'))
        
        fig.update_layout(title="Daily Electricity Consumption (kWh)", barmode='stack')
        st.plotly_chart(fig, use_container_width=True)

    # === 2. EFFICIENCY (COP) ===
    elif selected_tab == "Efficiency (COP)":
        fig = px.scatter(daily, x='Outdoor_Avg', y='Global_SCOP', 
                         size='Total_Heat_kWh', color='Total_Heat_kWh',
                         title="Daily COP vs Outdoor Temperature",
                         labels={'Global_SCOP': 'COP', 'Outdoor_Avg': 'Outdoor Temp (°C)'})
        fig.add_hline(y=3.0, line_dash="dot", annotation_text="Target COP 3.0")
        st.plotly_chart(fig, use_container_width=True)
        
        st.caption("Bubble size represents total heat output for the day.")

    # === 3. TEMPERATURES ===
    elif selected_tab == "Temperatures":
        # User Config for Friendly Names
        config = st.session_state.get("system_config", {})
        
        # Find Room Columns
        room_cols = [c for c in daily.columns if c.startswith('Room_') and c.endswith('_mean')]
        
        if room_cols:
            fig = go.Figure()
            for r in room_cols:
                # Clean key: "Room_1_mean" -> "Room_1"
                base_key = r.replace("_mean", "")
                label = get_friendly_name(base_key, config)
                fig.add_trace(go.Scatter(x=daily.index, y=daily[r], name=label))
            
            fig.update_layout(title="Average Daily Room Temperatures")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No room temperature sensors mapped.")

    # === 4. DHW ANALYSIS ===
    elif selected_tab == "DHW Analysis":
        col1, col2 = st.columns(2)
        
        # Chart 1: DHW Cycles per Day
        # We need to count runs per day from the 'runs' list
        if runs:
            run_df = pd.DataFrame(runs)
            if not run_df.empty:
                dhw_runs = run_df[run_df['run_type'] == 'DHW']
                if not dhw_runs.empty:
                    daily_counts = dhw_runs.set_index('start').resample('D').size()
                    
                    fig1 = px.bar(x=daily_counts.index, y=daily_counts.values, 
                                  title="DHW Cycles per Day", labels={'y': 'Cycles'})
                    col1.plotly_chart(fig1, use_container_width=True)
                else:
                    col1.info("No DHW runs detected in run list.")
            else:
                col1.info("No runs data.")
        
        # Chart 2: Energy for DHW
        fig2 = px.bar(daily, x=daily.index, y='Electricity_DHW_kWh', title="DHW Electricity (kWh)")
        col2.plotly_chart(fig2, use_container_width=True)

    # === 5. ZONE ANALYSIS ===
    elif selected_tab == "Zone Analysis":
        st.subheader("Zone Usage")
        
        # We want to see how often different zone configurations are used
        if 'Zone_Config' in df.columns:
            # We need to aggregate the raw DF, not daily
            # Count minutes for each config
            config_counts = df['Zone_Config'].value_counts()
            
            # Filter out "None" if desired, or keep it to see idle time
            config_counts = config_counts[config_counts.index != "None"]
            
            if not config_counts.empty:
                fig = px.pie(values=config_counts.values, names=config_counts.index, 
                             title="Heating Time Distribution by Zone Configuration")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("System was effectively idle (No active zones detected).")