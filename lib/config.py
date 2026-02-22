"""
Centralized configuration for the Muni Metro detection system.

This module contains all configurable constants used throughout the codebase,
organized into logical categories. Import from here instead of hardcoding values.
"""

import numpy as np

# =============================================================================
# IMAGE SOURCE
# =============================================================================

# URL for the Muni status display image
IMAGE_URL = "http://sfmunicentral.com/sfmunicentral_Snapshot_Objects/Mimic1_A7SE582P.jpg"

# Webpage containing the image (for Selenium fallback)
WEBPAGE_URL = "http://sfmunicentral.com/Enterprise/MetroLO.htm"
WEBPAGE_IMAGE_ID = "snapshotImage"
WEBPAGE_WAIT_TIME = 10  # seconds

# =============================================================================
# IMAGE DIMENSIONS
# =============================================================================

# Reference image dimensions (pixels)
# All position calculations are based on these dimensions and scaled proportionally
REFERENCE_WIDTH = 1860
REFERENCE_HEIGHT = 800

# Expected dimensions for validation
EXPECTED_WIDTH = REFERENCE_WIDTH
EXPECTED_HEIGHT = REFERENCE_HEIGHT


# =============================================================================
# Y-POSITION PERCENTAGES (relative to image height)
# =============================================================================

# Track Y positions (where trains travel)
UPPER_TRACK_Y_PCT = 0.51      # ~408px at 800px height
LOWER_TRACK_Y_PCT = 0.5875    # ~470px at 800px height
TRACK_HEIGHT_PCT = 0.04       # ~32px band height

# Label Y positions (station name labels)
UPPER_LABEL_Y_PCT = 0.375     # ~300px at 800px height
LOWER_LABEL_Y_PCT = 0.68      # ~544px at 800px height

# Platform Y positions for color detection
# Western stations (WE, FH, CA, CH) have different platform positions
PLATFORM_Y_WESTERN_UPPER_PCT = 0.475   # ~380px at 800px
PLATFORM_Y_WESTERN_LOWER_PCT = 0.625   # ~500px at 800px
PLATFORM_Y_EASTERN_UPPER_PCT = 0.53    # ~424px at 800px
PLATFORM_Y_EASTERN_LOWER_PCT = 0.5625  # ~450px at 800px

# Y-bands for region-based detection
Y_BANDS = {
    'upper_platform': (0.35, 0.50),
    'lower_platform': (0.58, 0.72),
    'upper_track': (0.50, 0.54),
    'lower_track': (0.56, 0.60),
    'upper_labels': (0.42, 0.50),
    'lower_labels': (0.60, 0.68),
}

# Train text detection bands (for OCR)
UPPER_TRAIN_BAND = (0.25, 0.48)
LOWER_TRAIN_BAND = (0.56, 0.80)

# Track Y ranges for train position classification
UPPER_TRACK_Y_RANGE = (0.48, 0.54)
LOWER_TRACK_Y_RANGE = (0.54, 0.62)


# =============================================================================
# HSV COLOR RANGES
# =============================================================================

# All HSV ranges are numpy arrays: [H, S, V]
# OpenCV uses H: 0-180, S: 0-255, V: 0-255

HSV_RANGES = {
    # Platform colors
    'platform_blue': {
        'lower': np.array([100, 100, 60]),
        'upper': np.array([130, 255, 255]),
    },
    'platform_yellow': {
        'lower': np.array([18, 70, 150]),
        'upper': np.array([45, 255, 255]),
    },

    # Track segment colors
    'track_cyan': {
        'lower': np.array([80, 100, 180]),
        'upper': np.array([100, 255, 255]),
    },
    'track_red_low': {  # Red wraps around in HSV
        'lower': np.array([0, 80, 120]),
        'upper': np.array([12, 255, 255]),
    },
    'track_red_high': {
        'lower': np.array([168, 80, 120]),
        'upper': np.array([180, 255, 255]),
    },

    # Train text colors (for colored label detection)
    'text_yellow': {
        'lower': np.array([15, 80, 120]),
        'upper': np.array([40, 255, 255]),
    },
    'text_green': {
        'lower': np.array([35, 80, 120]),
        'upper': np.array([85, 255, 255]),
    },

    # Train symbol colors (for train rectangle detection)
    'train_yellow': {
        'lower': np.array([20, 100, 150]),
        'upper': np.array([35, 255, 255]),
        'max_width': 45,
        'max_height': 25,
    },
    'train_green': {
        'lower': np.array([40, 80, 120]),
        'upper': np.array([65, 255, 255]),
        'max_width': 45,
        'max_height': 25,
    },
}


# =============================================================================
# DETECTION THRESHOLDS
# =============================================================================

# Platform color detection
PLATFORM_COLOR_MIN_THRESHOLD = 50  # Min pixels to detect platform color
PLATFORM_COLOR_SAMPLE_SIZE = 15    # Radius around detection point

# Track segment detection
SEGMENT_MIN_PIXEL_THRESHOLD = 100  # Min pixels to detect segment color
SEGMENT_MIN_PERCENTAGE = 0.05      # Min 5% of segment must be colored
SEGMENT_RATIO_THRESHOLD = 0.15     # 15% threshold for color classification

# Platform detection constraints
PLATFORM_MIN_AREA = 100
PLATFORM_MAX_AREA = 2000
PLATFORM_MIN_WIDTH = 15
PLATFORM_MAX_WIDTH = 50
PLATFORM_MIN_HEIGHT = 6
PLATFORM_MAX_HEIGHT = 25

# Station clustering
STATION_CLUSTER_THRESHOLD = 40  # Max X-distance to group as same station


# =============================================================================
# TRAIN DETECTION
# =============================================================================

# OCR configuration
OCR_CONFIG = '--oem 3 --psm 6'  # Tesseract: LSTM+legacy, uniform text block
OCR_SCALE_FACTOR = 4  # Upscale images for better OCR accuracy

# Train ID validation
VALID_TRAIN_PREFIXES = set('MBCDEFIJKLNRSTW')
VALID_SINGLE_SUFFIXES = set('JKLMNSTX*')
VALID_DOUBLE_SUFFIXES = {'JJ', 'KK', 'LL', 'MM', 'NN', 'SS', 'TT'}

# Revenue route classification (for status calculation, not OCR validation)
# Revenue trains are in passenger service; non-revenue are maintenance/deadhead
REVENUE_SINGLE_SUFFIXES = set('JKLMNST')
REVENUE_DOUBLE_SUFFIXES = {'JJ', 'KK', 'LL', 'MM', 'NN', 'SS', 'TT'}
NON_REVENUE_MARKERS = {'X', '*'}

# Text column detection
TEXT_COLUMN_GAP_THRESHOLD = 7  # Pixels between separate columns
TEXT_MIN_ROWS = 5  # Min rows to consider valid text
TEXT_DARK_THRESHOLD = 120  # Grayscale threshold for dark text
COLORED_TEXT_THRESHOLD = 100  # Column sum threshold for colored text

# Component detection
COMPONENT_X_TOLERANCE = 8  # Pixels to group components into columns
COMPONENT_MIN_AREA = 20
COMPONENT_MIN_WIDTH = 5
COMPONENT_MAX_WIDTH = 25
COMPONENT_MIN_HEIGHT = 40
COMPONENT_MAX_HEIGHT = 130
COMPONENT_MIN_TOTAL_AREA = 150

# Train symbol detection
SYMBOL_MIN_AREA = 200
SYMBOL_MIN_RECT_AREA = 250
SYMBOL_MIN_WIDTH = 18
SYMBOL_MAX_WIDTH = 45
SYMBOL_MIN_HEIGHT = 10
SYMBOL_MAX_HEIGHT = 20
SYMBOL_MIN_RECTANGULARITY = 0.5
SYMBOL_MIN_ASPECT = 1.5
SYMBOL_MAX_ASPECT = 3.5

# Cluster detection for pileups
CLUSTER_MIN_TRAINS = 3
CLUSTER_MAX_SPREAD = 300
CLUSTER_ZONE_MAX_SPREAD = 200

# Deduplication thresholds
DEDUP_PROXIMITY_THRESHOLD = 40  # Pixels to consider same train


# =============================================================================
# BUNCHING DETECTION
# =============================================================================

BUNCHING_CLUSTER_DISTANCE = 70   # Max pixels between trains in a cluster
BUNCHING_THRESHOLD = 4           # Min trains to count as bunching
BUNCHING_DEFAULT_ZONE_LENGTH = 300  # Default detection zone in pixels

# Excluded stations (internal only, not for bunching detection)
BUNCHING_EXCLUDED_STATIONS = {'MN', 'FP', 'TT'}

# Station-specific zone lengths (where default doesn't apply)
BUNCHING_ZONE_LENGTH_UPPER = {
    'CT': 150,  # Chinatown - limited space at terminus
    'CH': 75,   # Church - shorter zone
    'EM': 75,   # Embarcadero - limited space
}
BUNCHING_ZONE_LENGTH_LOWER = {
    'CT': 75,   # Chinatown - don't extend past subway end
    'VN': 150,  # Van Ness - half length
    'EM': 75,   # Embarcadero - limited space at terminus
    'MO': 150,  # Montgomery - near terminus
}


# =============================================================================
# STATUS CLASSIFICATION
# =============================================================================

# Minimum trains with valid route suffixes to consider system operating
MIN_TRAINS_OPERATING = 2

# Hysteresis transition thresholds (consecutive checks needed to change status)
HYSTERESIS_THRESHOLDS = {
    # Worsening transitions (slower - avoid false alarms)
    ('green', 'yellow'): 3,
    ('green', 'red'): 3,
    ('yellow', 'red'): 3,
    # Improving transitions (faster - report good news quickly)
    ('red', 'yellow'): 2,
    ('red', 'green'): 2,
    ('yellow', 'green'): 2,
}


# =============================================================================
# API / CACHING
# =============================================================================

# Cache staleness thresholds (seconds)
CACHE_MAX_AGE = 300          # 5 minutes - max age before fallback
STALENESS_FRESH = 300        # 0-5 min: fresh
STALENESS_AGING = 900        # 5-15 min: aging
STALENESS_STALE = 1800       # 15-30 min: stale
STALENESS_EXPIRED = 3600     # >60 min: expired (was incorrectly 1800)

# Status check interval
DEFAULT_CHECK_INTERVAL = 30  # seconds (local development)
CLOUD_CHECK_INTERVAL = 180   # seconds (Cloud Run - matches Cloud Scheduler)

# Analytics report cache
REPORT_CACHE_MAX_AGE = 1800  # 30 minutes


# =============================================================================
# HTTP / NETWORK
# =============================================================================

# Image download settings
HTTP_TIMEOUT = 10            # seconds
HTTP_MAX_RETRIES = 3
# Exponential backoff: delay = 2^attempt (1s, 2s, 4s)


# =============================================================================
# CONFIDENCE THRESHOLDS
# =============================================================================

CONFIDENCE_CACHE_VALID = 0.7     # Min confidence to use cached detection
CONFIDENCE_DETECTION_VALID = 0.7  # Min confidence for valid detection


# =============================================================================
# IMAGE ARCHIVAL
# =============================================================================

DEFAULT_ARCHIVE_BUCKET = 'munimetro-image-archive'
BASELINE_ARCHIVE_INTERVAL = 3600  # 1 hour between baseline archives
