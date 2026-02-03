"""
Shared constants for SF Muni Metro station detection.

This module contains station definitions, ordering, and section mappings
used by both the detection library and development scripts.
"""

# Station order (left to right on display board)
# Includes internal maintenance platforms needed for spatial detection
STATION_ORDER = [
    ("WE", "West Portal"),
    ("FH", "Forest Hill"),
    ("CA", "Castro"),
    ("CH", "Church"),
    ("VN", "Van Ness"),
    ("CC", "Civic Center"),
    ("PO", "Powell"),
    ("MO", "Montgomery"),
    ("EM", "Embarcadero"),
    ("MN", "Main"),        # internal maintenance platform
    ("FP", "4th & King"),  # internal maintenance platform
    ("TT", "Third St"),    # internal maintenance platform
    ("CT", "Chinatown"),
    ("US", "Union Square"),
    ("YB", "Yerba Buena"),
]

# Internal maintenance platforms - detected for spatial positioning but
# not surfaced to users and segments involving them are filtered out
INTERNAL_STATIONS = {"MN", "FP", "TT"}

# Direction sections: the display has two track sections with different
# direction naming conventions
# Left section (West Portal -> Embarcadero): upper=Westbound, lower=Eastbound
# Right section (Chinatown -> Yerba Buena): upper=Northbound, lower=Southbound
SUBWAY_CODES = {"WE", "FH", "CA", "CH", "VN", "CC", "PO", "MO", "EM"}
CENTRAL_CODES = {"CT", "US", "YB"}

# Public station names (for UI dropdowns etc.)
PUBLIC_STATIONS = [
    (code, name) for code, name in STATION_ORDER
    if code not in INTERNAL_STATIONS
]

# North-south stations (use Northbound/Southbound instead of Westbound/Eastbound)
NORTH_SOUTH_STATIONS = {'CT', 'US', 'YB'}


def get_section_directions(from_code, to_code):
    """Get (upper_direction, lower_direction) for a segment based on section.

    Returns:
        Tuple of (upper_dir, lower_dir) strings.
    """
    if from_code in SUBWAY_CODES and to_code in SUBWAY_CODES:
        return ('Westbound', 'Eastbound')
    elif from_code in CENTRAL_CODES and to_code in CENTRAL_CODES:
        return ('Northbound', 'Southbound')
    else:
        # Cross-section or internal - shouldn't appear in output
        return ('Outbound', 'Inbound')
