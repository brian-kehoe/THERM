# config_manager.py
"""
Manages configuration profiles and export logic.
"""
import json
from datetime import datetime
from schema_defs import REQUIRED_SENSORS, check_feature_availability

def validate_config(config):
    """
    Validates that a configuration meets minimum requirements.
    """
    errors = []
    if "mapping" not in config:
        return False, ["Missing 'mapping' section"]
    
    mapped_sensors = set(config["mapping"].keys())
    required_sensors = set(REQUIRED_SENSORS.keys())
    
    missing = required_sensors - mapped_sensors
    if missing:
        errors.append(f"Missing required sensors: {', '.join(missing)}")
    
    return len(errors) == 0, errors

def export_config_for_sharing(config):
    """
    Creates a clean version of the config for download.
    """
    # We essentially just want to ensure it has the right keys and maybe add a timestamp
    export_data = {
        "therm_version": config.get("therm_version", "2.0"),
        "profile_name": config.get("profile_name", "My Profile"),
        "mapping": config.get("mapping", {}),
        "units": config.get("units", {}),
        "ai_context": config.get("ai_context", {}),
        "exported_at": datetime.now().isoformat()
    }
    return export_data