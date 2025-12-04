# mapping_ui.py
"""
Handles the user interface for sensor mapping and configuration.
"""
import streamlit as st
import pandas as pd
import json
from datetime import datetime
from schema_defs import (
    REQUIRED_SENSORS, RECOMMENDED_SENSORS, OPTIONAL_SENSORS, 
    ZONE_SENSORS, ENVIRONMENTAL_SENSORS, AI_CONTEXT_PROMPTS
)
import config_manager

def get_all_unique_entities(uploaded_files):
    """Fast scan of headers in uploaded files."""
    found_entities = set()
    for file_obj in uploaded_files:
        file_obj.seek(0)
        try:
            # Read header only
            df_head = pd.read_csv(file_obj, nrows=2)
            if 'entity_id' in df_head.columns:
                # Long format: need to scan the entity_id column
                file_obj.seek(0)
                df_ent = pd.read_csv(file_obj, usecols=['entity_id'])
                found_entities.update(df_ent['entity_id'].dropna().unique())
            else:
                # Wide format: columns are sensors
                cols = [c for c in df_head.columns if c.lower() not in ['time', 'date', 'timestamp', 'last_changed', 'series']]
                found_entities.update(cols)
        except Exception:
            pass
        file_obj.seek(0)
    return sorted(list(found_entities))

def render_sensor_row(label, internal_key, options, defaults, required=False, help_text=None):
    """Renders a single row of the mapping interface."""
    c1, c2 = st.columns([2, 1])
    
    # 1. Sensor Selection
    saved_val = defaults["mapping"].get(internal_key, "None")
    idx = options.index(saved_val) if saved_val in options else 0
    
    with c1:
        label_text = f"**{label}**" + (" *" if required else "")
        sel = st.selectbox(
            label_text, 
            options, 
            index=idx, 
            key=f"map_{internal_key}",
            help=help_text
        )
    
    # 2. Unit Selection (Conditional)
    unit_val = None
    from schema_defs import get_unit_options
    u_opts = get_unit_options(internal_key)
    
    if len(u_opts) > 1 and u_opts != ["unknown"]:
        with c2:
            saved_unit = defaults["units"].get(internal_key, u_opts[0])
            u_idx = u_opts.index(saved_unit) if saved_unit in u_opts else 0
            unit_val = st.selectbox(
                "Unit", 
                u_opts, 
                index=u_idx, 
                key=f"unit_{internal_key}", 
                label_visibility="visible"
            )
    else:
        # Default to first option if only one available
        if u_opts and u_opts != ["unknown"]:
            unit_val = u_opts[0]
            
    return sel, unit_val

def render_configuration_interface(uploaded_files):
    """Main function to render the configuration wizard."""
    st.markdown("## 🛠️ System Setup")
    st.info("Please map your data columns to the standardized therm sensors.")

    # --- 1. Load Existing Config ---
    col_load, col_name = st.columns([1, 2])
    with col_load:
        uploaded_config = st.file_uploader("📂 Load Profile (JSON)", type="json", key="cfg_upload")
    
    defaults = {
        "mapping": {},
        "units": {},
        "ai_context": {},
        "profile_name": "My Heat Pump"
    }

    if uploaded_config:
        loaded = json.load(uploaded_config)
        defaults.update(loaded)
        st.success(f"Loaded profile: {defaults.get('profile_name')}")

    with col_name:
        profile_name = st.text_input("Profile Name", value=defaults["profile_name"])

    # --- 2. Scan Files ---
    if "available_sensors" not in st.session_state:
        with st.spinner("Scanning files for sensors..."):
            st.session_state["available_sensors"] = get_all_unique_entities(uploaded_files)
    
    options = ["None"] + st.session_state.get("available_sensors", [])

    # --- 3. The Form ---
    with st.form("config_wizard"):
        user_map = {}
        user_units = {}

        # -- REQUIRED --
        st.subheader("1. Critical Sensors (Required)")
        for key, details in REQUIRED_SENSORS.items():
            sel, unit = render_sensor_row(
                details['label'], key, options, defaults, required=True, help_text=details['description']
            )
            if sel and sel != "None":
                user_map[key] = sel
                if unit: user_units[key] = unit

        # -- RECOMMENDED --
        st.subheader("2. Recommended Sensors")
        for key, details in RECOMMENDED_SENSORS.items():
            sel, unit = render_sensor_row(
                details['label'], key, options, defaults, help_text=details['description']
            )
            if sel and sel != "None":
                user_map[key] = sel
                if unit: user_units[key] = unit

        # -- ZONES --
        with st.expander("🏠 Heating Zones & Pumps"):
            st.caption("Map binary sensors (On/Off) that indicate when a zone is active.")
            for key, details in ZONE_SENSORS.items():
                sel, unit = render_sensor_row(
                    details['label'], key, options, defaults, help_text=details['description']
                )
                if sel and sel != "None":
                    user_map[key] = sel
                    if unit: user_units[key] = unit

        # -- OPTIONAL --
        with st.expander("➕ Advanced / Optional Sensors"):
            all_optional = {**OPTIONAL_SENSORS, **ENVIRONMENTAL_SENSORS}
            for key, details in all_optional.items():
                sel, unit = render_sensor_row(
                    details['label'], key, options, defaults, help_text=details['description']
                )
                if sel and sel != "None":
                    user_map[key] = sel
                    if unit: user_units[key] = unit

        # -- AI CONTEXT --
        st.divider()
        st.subheader("🤖 AI Analysis Context")
        ai_inputs = {}
        for key, prompts in AI_CONTEXT_PROMPTS.items():
            val = st.text_area(
                prompts['label'],
                value=defaults["ai_context"].get(key, ""),
                placeholder=prompts['placeholder'],
                help=prompts['help']
            )
            ai_inputs[key] = val

        submitted = st.form_submit_button("✅ Save Configuration & Process")

        if submitted:
            # Validate
            missing = [k for k in REQUIRED_SENSORS if k not in user_map]
            if missing:
                st.error(f"Missing required sensors: {', '.join(missing)}")
                return None
            
            # Build Config Object
            config_object = {
                "profile_name": profile_name,
                "created_at": datetime.now().isoformat(),
                "mapping": user_map,   # Internal: Raw
                "units": user_units,
                "ai_context": ai_inputs,
                "therm_version": "2.0"
            }
            return config_object
            
    return None

def render_config_download(config):
    """Renders the download button for the current config."""
    # We use config_manager to format it nicely for export
    export_data = config_manager.export_config_for_sharing(config)
    json_str = json.dumps(export_data, indent=2)
    
    st.download_button(
        label="💾 Download Configuration Profile",
        data=json_str,
        file_name=f"therm_profile_{config.get('profile_name', 'default').replace(' ', '_')}.json",
        mime="application/json"
    )