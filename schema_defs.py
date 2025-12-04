# schema_defs.py
"""
Defines the internal sensor schema, units, and user configuration prompts.
This module contains no logic - only data structures.
"""

# =============================================================================
# SENSOR SCHEMA: Internal names that processing.py expects
# =============================================================================

REQUIRED_SENSORS = {
    "Power": {
        "label": "⚡ Electricity Input (Total Heat Pump)",
        "description": "Total electrical power consumption of the heat pump",
        "unit_options": ["W", "kW"],
        "standard_unit": "W"
    },
    "FlowTemp": {
        "label": "🌡️ Flow Temperature",
        "description": "Temperature of water leaving the heat pump",
        "unit_options": ["°C", "°F"],
        "standard_unit": "°C"
    },
    "ReturnTemp": {
        "label": "🌡️ Return Temperature", 
        "description": "Temperature of water entering the heat pump",
        "unit_options": ["°C", "°F"],
        "standard_unit": "°C"
    },
    "FlowRate": {
        "label": "💧 Flow Rate",
        "description": "Water flow rate through the heat pump",
        "unit_options": ["L/min", "m³/hr", "L/sec", "gal/min"],
        "standard_unit": "L/min"
    }
}

RECOMMENDED_SENSORS = {
    "Heat": {
        "label": "🔥 Heat Output",
        "description": "Thermal energy output (can be calculated if missing)",
        "unit_options": ["W", "kW"],
        "standard_unit": "W"
    },
    "OutdoorTemp": {
        "label": "🌤️ Outdoor Temperature",
        "description": "Ambient outdoor temperature for efficiency analysis",
        "unit_options": ["°C", "°F"],
        "standard_unit": "°C"
    }
}

OPTIONAL_SENSORS = {
    "Indoor_Power": {
        "label": "⚡ Indoor Unit Power",
        "description": "Electrical consumption of indoor controller/pumps",
        "unit_options": ["W", "kW"],
        "standard_unit": "W"
    },
    "DHW_Temp": {
        "label": "🚿 Hot Water Tank Temperature",
        "description": "Domestic hot water cylinder temperature",
        "unit_options": ["°C", "°F"],
        "standard_unit": "°C"
    },
    "Freq": {
        "label": "🔧 Compressor Frequency",
        "description": "Compressor modulation frequency",
        "unit_options": ["Hz"],
        "standard_unit": "Hz"
    },
    "Defrost": {
        "label": "❄️ Defrost Status",
        "description": "Binary indicator of defrost cycle (0/1 or off/on)",
        "unit_options": ["binary"],
        "standard_unit": "binary"
    },
    "ValveMode": {
        "label": "🔀 3-Way Valve Mode",
        "description": "Mode indicator (e.g., 'Heating', 'DHW', 'Hot Water')",
        "unit_options": ["text"],
        "standard_unit": "text"
    },
    "DHW_Mode": {
        "label": "🚿 DHW Operating Mode",
        "description": "Hot water mode (e.g., 'Standard', 'Economic', 'Power')",
        "unit_options": ["text"],
        "standard_unit": "text"
    },
    "Quiet_Mode": {
        "label": "🔇 Quiet Mode",
        "description": "Quiet/Night mode status (0/1 or off/on)",
        "unit_options": ["binary"],
        "standard_unit": "binary"
    },
    "Immersion_Mode": {
        "label": "🔌 Immersion Heater Status",
        "description": "Backup immersion heater status (0/1 or off/on)",
        "unit_options": ["binary"],
        "standard_unit": "binary"
    }
}

# Zone Configuration
# Supports 1-4 heating zones dynamically
ZONE_SENSORS = {
    "Zone_1": {
        "label": "🏠 Heating Zone 1",
        "description": "Primary heating zone (e.g., Downstairs, Main Floor, UFH)",
        "unit_options": ["binary"],
        "standard_unit": "binary"
    },
    "Zone_2": {
        "label": "🏠 Heating Zone 2",
        "description": "Secondary heating zone (e.g., Upstairs, Radiators)",
        "unit_options": ["binary"],
        "standard_unit": "binary"
    },
    "Zone_3": {
        "label": "🏠 Heating Zone 3",
        "description": "Tertiary heating zone",
        "unit_options": ["binary"],
        "standard_unit": "binary"
    },
    "Zone_4": {
        "label": "🏠 Heating Zone 4",
        "description": "Quaternary heating zone",
        "unit_options": ["binary"],
        "standard_unit": "binary"
    }
}

# Environmental Sensors
ENVIRONMENTAL_SENSORS = {
    "Solar_Rad": {
        "label": "☀️ Solar Radiation",
        "description": "Solar irradiance for solar gain analysis",
        "unit_options": ["W/m²"],
        "standard_unit": "W/m²"
    },
    "Wind_Speed": {
        "label": "💨 Wind Speed",
        "description": "Wind speed affecting heat loss",
        "unit_options": ["m/s", "km/h", "mph"],
        "standard_unit": "m/s"
    },
    "Outdoor_Humidity": {
        "label": "💧 Outdoor Humidity",
        "description": "Relative humidity",
        "unit_options": ["%"],
        "standard_unit": "%"
    }
}

# Room Temperature Sensors (Dynamic - user can map 0 to N rooms)
ROOM_SENSOR_PREFIX = "Room_"

# =============================================================================
# AI CONTEXT PROMPTS
# =============================================================================

AI_CONTEXT_PROMPTS = {
    "hp_model": {
        "label": "🔧 Heat Pump Model & Specifications",
        "placeholder": "e.g., Samsung EHS Mono 8kW (Gen 6), Mitsubishi Ecodan 11kW",
        "help": "Model name and rated capacity help the AI understand expected performance curves and operating characteristics.",
        "default": ""
    },
    "tariff_structure": {
        "label": "💰 Electricity Tariff Structure",
        "placeholder": "e.g., Day rate €0.35/kWh (08:00-23:00), Night rate €0.15/kWh (23:00-08:00), Peak €0.45/kWh (17:00-19:00)",
        "help": "Describe your electricity pricing in plain text. Include time-of-use bands, peak rates, and any standing charges. The AI uses this to calculate operating costs and suggest optimization strategies.",
        "default": ""
    },
    "property_context": {
        "label": "🏠 Property & Heating System Description",
        "placeholder": "e.g., 150m² detached house, A2-rated, underfloor heating downstairs (50m²), radiators upstairs (100m²), 250L DHW cylinder",
        "help": "Property size, insulation rating, heat emitter types, and DHW capacity. This context helps the AI understand thermal inertia, heat loss characteristics, and system behavior.",
        "default": ""
    },
    "operational_goals": {
        "label": "🎯 Operational Goals & Constraints",
        "placeholder": "e.g., Maximize efficiency, minimize night-time noise, maintain 21°C living room temperature, DHW ready by 07:00",
        "help": "What are you trying to achieve? Cost savings, comfort targets, noise constraints? The AI tailors its recommendations to your priorities.",
        "default": ""
    }
}

# =============================================================================
# UNIT CONVERSION FACTORS
# =============================================================================

UNIT_CONVERSIONS = {
    # Temperature conversions (to Celsius)
    "°F": lambda x: (x - 32) * 5/9,
    "°C": lambda x: x,
    
    # Power conversions (to Watts)
    "kW": lambda x: x * 1000,
    "W": lambda x: x,
    
    # Flow rate conversions (to L/min)
    "L/min": lambda x: x,
    "m³/hr": lambda x: x * 16.6667,
    "L/sec": lambda x: x * 60.0,
    "gal/min": lambda x: x * 3.785,  # US gallons
    
    # Wind speed conversions (to m/s)
    "m/s": lambda x: x,
    "km/h": lambda x: x / 3.6,
    "mph": lambda x: x * 0.44704,
    
    # Binary/Text (no conversion)
    "binary": lambda x: x,
    "text": lambda x: x,
    "%": lambda x: x,
    "W/m²": lambda x: x,
    "Hz": lambda x: x
}

# =============================================================================
# FEATURE DEPENDENCIES
# =============================================================================

FEATURE_REQUIREMENTS = {
    "basic_efficiency": ["Power", "FlowTemp", "ReturnTemp", "FlowRate"],
    "cop_calculation": ["Power", "Heat"],  # Heat can be derived if missing
    "weather_compensation": ["OutdoorTemp", "FlowTemp"],
    "dhw_analysis": ["DHW_Temp", "ValveMode"],
    "zone_analysis": ["Zone_1"],  # At least one zone
    "run_detection": ["Power"],
    "cost_analysis": ["Power"],  # Requires tariff data from AI context
    "defrost_detection": ["Defrost", "OutdoorTemp"],
    "room_response": [ROOM_SENSOR_PREFIX]  # At least one room sensor
}

# =============================================================================
# VALIDATION RULES
# =============================================================================

VALIDATION_RULES = {
    "Power": {"min": 0, "max": 50000, "type": "numeric"},
    "Heat": {"min": 0, "max": 100000, "type": "numeric"},
    "FlowTemp": {"min": -20, "max": 80, "type": "numeric"},
    "ReturnTemp": {"min": -20, "max": 80, "type": "numeric"},
    "FlowRate": {"min": 0, "max": 100, "type": "numeric"},
    "OutdoorTemp": {"min": -40, "max": 50, "type": "numeric"},
    "DHW_Temp": {"min": 0, "max": 90, "type": "numeric"},
    "Freq": {"min": 0, "max": 120, "type": "numeric"},
    "Indoor_Power": {"min": 0, "max": 5000, "type": "numeric"},
    "Defrost": {"values": [0, 1, "on", "off"], "type": "binary"},
    "ValveMode": {"type": "text"},
    "DHW_Mode": {"type": "text"}
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_all_sensor_definitions():
    """Returns combined dict of all sensor definitions."""
    return {
        **REQUIRED_SENSORS,
        **RECOMMENDED_SENSORS,
        **OPTIONAL_SENSORS,
        **ZONE_SENSORS,
        **ENVIRONMENTAL_SENSORS
    }

def get_sensor_label(internal_name):
    """Get user-friendly label for an internal sensor name."""
    all_sensors = get_all_sensor_definitions()
    if internal_name in all_sensors:
        return all_sensors[internal_name]["label"]
    elif internal_name.startswith(ROOM_SENSOR_PREFIX):
        room_name = internal_name.replace(ROOM_SENSOR_PREFIX, "")
        return f"🌡️ Room Temperature: {room_name}"
    return internal_name

def get_unit_options(internal_name):
    """Get valid unit options for a sensor."""
    all_sensors = get_all_sensor_definitions()
    if internal_name in all_sensors:
        return all_sensors[internal_name]["unit_options"]
    return ["unknown"]

def check_feature_availability(mapped_sensors):
    """
    Check which features are available based on mapped sensors.
    Returns dict of {feature: bool}
    """
    availability = {}
    for feature, requirements in FEATURE_REQUIREMENTS.items():
        if feature == "room_response":
            # Special case: check for any room sensor
            has_room = any(s.startswith(ROOM_SENSOR_PREFIX) for s in mapped_sensors)
            availability[feature] = has_room
        else:
            availability[feature] = all(req in mapped_sensors for req in requirements)
    return availability