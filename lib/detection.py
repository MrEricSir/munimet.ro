#!/usr/bin/env python3
"""
OpenCV-based detection library for SF Muni Metro status.

This module provides the detection API used by the status service,
combining station detection, train detection, and delay analysis.
"""

from pathlib import Path

import cv2

# Import detection components from lib/
from lib.station_detector import (
    StationDetector,
    STATION_X_POSITIONS,
    UPPER_TRACK_Y_PCT,
    LOWER_TRACK_Y_PCT,
    TRACK_HEIGHT_PCT,
    UPPER_LABEL_Y_PCT,
    LOWER_LABEL_Y_PCT,
    REFERENCE_IMAGE_WIDTH,
    REFERENCE_IMAGE_HEIGHT,
    HSV_RANGES,
)
from lib.station_constants import STATION_ORDER, INTERNAL_STATIONS

# Try to import train detector (requires tesseract)
try:
    from lib.train_detector import TrainDetector, TESSERACT_AVAILABLE
except ImportError:
    TrainDetector = None
    TESSERACT_AVAILABLE = False

# Path resolution
LIB_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = LIB_DIR.parent

# Station full names for display
STATION_NAMES = {
    'WE': 'West Portal',
    'FH': 'Forest Hill',
    'CA': 'Castro',
    'CH': 'Church',
    'VN': 'Van Ness',
    'CC': 'Civic Center',
    'PO': 'Powell',
    'MO': 'Montgomery',
    'EM': 'Embarcadero',
    'MN': 'Main',
    'FP': 'Folsom',
    'TT': 'Temporary Terminal',
    'CT': 'Chinatown',
    'US': 'Union Square',
    'YB': 'Yerba Buena',
}

# Direction terminology
NORTH_SOUTH_STATIONS = {'CT', 'US', 'YB'}

# Western stations have different Y positions for platforms
WESTERN_STATIONS = {'WE', 'FH', 'CA', 'CH'}


def get_platform_y(code, img_height):
    """Get platform Y positions for a station based on image height."""
    if code in WESTERN_STATIONS:
        upper_y = int(img_height * 0.475)   # ~380 in 800px
        lower_y = int(img_height * 0.625)   # ~500 in 800px
    else:
        upper_y = int(img_height * 0.53)    # ~424 in 800px
        lower_y = int(img_height * 0.5625)  # ~450 in 800px
    return upper_y, lower_y

# Initialize detectors (lazy loaded)
_station_detector = None
_train_detector = None


def _get_station_detector():
    """Get or create station detector (singleton)."""
    global _station_detector
    if _station_detector is None:
        _station_detector = StationDetector()
    return _station_detector


def _get_train_detector():
    """Get or create train detector (singleton)."""
    global _train_detector
    if _train_detector is None and TrainDetector is not None and TESSERACT_AVAILABLE:
        _train_detector = TrainDetector()
    return _train_detector


def detect_platform_color(hsv, x, y, size=15, min_threshold=50):
    """
    Detect platform color (blue=normal, yellow=hold) at given position.

    Returns:
        tuple: (color_name, pixel_count) where color_name is 'blue', 'yellow', or 'unknown'
    """
    h, w = hsv.shape[:2]

    # Define region around point
    x1 = max(0, x - size)
    x2 = min(w, x + size)
    y1 = max(0, y - size)
    y2 = min(h, y + size)

    roi = hsv[y1:y2, x1:x2]

    # Count blue pixels (normal platform)
    blue_lower = HSV_RANGES['platform_blue']['lower']
    blue_upper = HSV_RANGES['platform_blue']['upper']
    blue_mask = cv2.inRange(roi, blue_lower, blue_upper)
    blue_pixels = cv2.countNonZero(blue_mask)

    # Count yellow pixels (hold mode)
    yellow_lower = HSV_RANGES['platform_yellow']['lower']
    yellow_upper = HSV_RANGES['platform_yellow']['upper']
    yellow_mask = cv2.inRange(roi, yellow_lower, yellow_upper)
    yellow_pixels = cv2.countNonZero(yellow_mask)

    # Return dominant color
    if blue_pixels >= yellow_pixels and blue_pixels >= min_threshold:
        return 'blue', blue_pixels
    elif yellow_pixels >= min_threshold:
        return 'yellow', yellow_pixels
    else:
        return 'unknown', 0


def detect_segment_color(hsv, bounds):
    """
    Detect track segment color (cyan=normal, red=disabled) in bounding box.

    Returns:
        tuple: (color_name, pixel_count) where color_name is 'cyan', 'red', or 'unknown'
    """
    # bounds is a dict with x_min, x_max, y_min, y_max
    x_min = int(bounds['x_min'])
    x_max = int(bounds['x_max'])
    y_min = int(bounds['y_min'])
    y_max = int(bounds['y_max'])
    roi = hsv[y_min:y_max, x_min:x_max]

    # Count cyan pixels (normal)
    cyan_lower = HSV_RANGES['track_cyan']['lower']
    cyan_upper = HSV_RANGES['track_cyan']['upper']
    cyan_mask = cv2.inRange(roi, cyan_lower, cyan_upper)
    cyan_pixels = cv2.countNonZero(cyan_mask)

    # Count red pixels (disabled) - red wraps around in HSV
    red_low_mask = cv2.inRange(roi, HSV_RANGES['track_red_low']['lower'], HSV_RANGES['track_red_low']['upper'])
    red_high_mask = cv2.inRange(roi, HSV_RANGES['track_red_high']['lower'], HSV_RANGES['track_red_high']['upper'])
    red_mask = cv2.bitwise_or(red_low_mask, red_high_mask)
    red_pixels = cv2.countNonZero(red_mask)

    # Return dominant color
    min_threshold = 100
    if cyan_pixels > red_pixels and cyan_pixels >= min_threshold:
        return 'cyan', cyan_pixels
    elif red_pixels >= min_threshold:
        return 'red', red_pixels
    else:
        return 'unknown', 0


def detect_train_bunching(trains, threshold=4, cluster_distance=85):
    """
    Detect train bunching (multiple trains clustered close together).

    Returns:
        list: List of bunching incidents [{station, track, direction, train_count}]
    """
    # Internal stations to always exclude
    INTERNAL_STATIONS = {'MN', 'FP', 'TT'}

    # Track-specific exclusions for turnaround points where bunching is normal:
    # - Upper track (Westbound/Northbound): CT is the northern terminus
    # - Lower track (Eastbound/Southbound): EM is the eastern terminus, MO is adjacent
    EXCLUDED_UPPER = INTERNAL_STATIONS | {'CT'}
    EXCLUDED_LOWER = INTERNAL_STATIONS | {'EM', 'MO'}  # Eastbound trains bunch at Embarcadero turnaround

    def get_direction(station_code, track):
        if track == 'upper':
            return 'Northbound' if station_code in NORTH_SOUTH_STATIONS else 'Westbound'
        else:
            return 'Southbound' if station_code in NORTH_SOUTH_STATIONS else 'Eastbound'

    bunching_incidents = []

    # Separate trains by track and sort by x (exclude UNKNOWN trains - often false positives)
    upper_trains = sorted([t for t in trains if t.get('track') == 'upper' and not t.get('id', '').startswith('UNKNOWN')], key=lambda t: t['x'])
    lower_trains = sorted([t for t in trains if t.get('track') == 'lower' and not t.get('id', '').startswith('UNKNOWN')], key=lambda t: t['x'])

    def find_clusters(sorted_trains):
        if len(sorted_trains) < threshold:
            return []
        clusters = []
        current_cluster = [sorted_trains[0]]
        for i in range(1, len(sorted_trains)):
            if sorted_trains[i]['x'] - sorted_trains[i-1]['x'] <= cluster_distance:
                current_cluster.append(sorted_trains[i])
            else:
                if len(current_cluster) >= threshold:
                    clusters.append(current_cluster)
                current_cluster = [sorted_trains[i]]
        if len(current_cluster) >= threshold:
            clusters.append(current_cluster)
        return clusters

    # Find clusters on each track
    # For upper track (westbound): trains queue to the RIGHT of stations they're entering
    # Find the nearest station to the front (left edge) of the cluster
    for cluster in find_clusters(upper_trains):
        cluster_left_x = min(t['x'] for t in cluster)
        nearest_station = None
        min_distance = float('inf')
        for station_code, station_x in STATION_X_POSITIONS.items():
            if station_code in EXCLUDED_UPPER:
                continue
            # Find nearest station to cluster front (absolute distance)
            distance = abs(station_x - cluster_left_x)
            if distance < min_distance:
                min_distance = distance
                nearest_station = station_code
        if nearest_station and min_distance < 300:
            bunching_incidents.append({
                'station': nearest_station,
                'track': 'upper',
                'direction': get_direction(nearest_station, 'upper'),
                'train_count': len(cluster),
            })

    # For lower track (eastbound): trains queue to the LEFT of stations they're entering
    # Find the nearest station at or to the RIGHT of the cluster front
    for cluster in find_clusters(lower_trains):
        cluster_right_x = max(t['x'] for t in cluster)
        nearest_station = None
        min_distance = float('inf')
        for station_code, station_x in STATION_X_POSITIONS.items():
            if station_code in EXCLUDED_LOWER:
                continue
            # Station must be at or to the right of the cluster front
            if station_x >= cluster_right_x:
                distance = station_x - cluster_right_x
                if distance < min_distance:
                    min_distance = distance
                    nearest_station = station_code
        if nearest_station and min_distance < 300:
            bunching_incidents.append({
                'station': nearest_station,
                'track': 'lower',
                'direction': get_direction(nearest_station, 'lower'),
                'train_count': len(cluster),
            })

    return bunching_incidents


def calculate_system_status(trains, delays_platforms, delays_segments, bunching_incidents=None):
    """
    Calculate overall system status based on detection data.

    Returns:
        str: 'red', 'yellow', or 'green'
    """
    import re

    if bunching_incidents is None:
        bunching_incidents = []

    # Check how many trains have valid route suffixes
    suffix_pattern = re.compile(r'\d{4}([A-Z]{1,2})$')
    trains_with_routes = 0

    for train in trains:
        train_id = train.get('id', '')
        if train_id.startswith('UNKNOWN'):
            continue
        match = suffix_pattern.search(train_id)
        if match:
            trains_with_routes += 1

    # Red: Fewer than 2 trains with route suffixes (not operating)
    if trains_with_routes < 2:
        return 'red'

    # Yellow: 2+ platforms in hold OR any track sections disabled OR bunching
    platforms_in_hold = len(delays_platforms)
    tracks_disabled = len(delays_segments)
    has_bunching = len(bunching_incidents) > 0

    if platforms_in_hold >= 2 or tracks_disabled > 0 or has_bunching:
        return 'yellow'

    # Green: Normal operation
    return 'green'


# Station order for determining consecutive stations in direction of travel
# Westbound: travels right-to-left (EM -> WE), so list in that order
# Eastbound: travels left-to-right (WE -> EM), so list in that order
WESTBOUND_ORDER = ['EM', 'MO', 'PO', 'CC', 'VN', 'CH', 'CA', 'FH', 'WE']
EASTBOUND_ORDER = ['WE', 'FH', 'CA', 'CH', 'VN', 'CC', 'PO', 'MO', 'EM']
# Northbound: travels right-to-left (YB -> CT)
# Southbound: travels left-to-right (CT -> YB)
NORTHBOUND_ORDER = ['YB', 'US', 'CT']
SOUTHBOUND_ORDER = ['CT', 'US', 'YB']


def generate_delay_summaries(delays_platforms, delays_segments, bunching_incidents):
    """
    Generate human-readable delay summaries for display.

    Examples:
    - Single station hold: "Northbound delay at Union Square"
    - Multiple consecutive holds: "Westbound delay from Church to West Portal"
    - Red track segment: "Eastbound service not running between Powell and Montgomery"
    - Train bunching: "Westbound backup at Powell (4 trains)"

    Returns:
        list: List of human-readable summary strings
    """
    summaries = []

    # Get the station order for each direction
    direction_orders = {
        'Westbound': WESTBOUND_ORDER,
        'Eastbound': EASTBOUND_ORDER,
        'Northbound': NORTHBOUND_ORDER,
        'Southbound': SOUTHBOUND_ORDER,
    }

    # Group platform delays by direction
    platforms_by_direction = {}
    for delay in delays_platforms:
        direction = delay['direction']
        if direction not in platforms_by_direction:
            platforms_by_direction[direction] = []
        platforms_by_direction[direction].append(delay)

    # Process each direction's platform delays
    for direction, delays in platforms_by_direction.items():
        station_codes = [d['station'] for d in delays]
        order = direction_orders.get(direction, [])

        # Sort stations by their position in the direction order
        station_positions = {code: order.index(code) for code in station_codes if code in order}
        sorted_codes = sorted(station_codes, key=lambda c: station_positions.get(c, 999))

        # Find consecutive groups
        groups = []
        current_group = []

        for code in sorted_codes:
            if code not in order:
                continue
            pos = order.index(code)

            if not current_group:
                current_group = [code]
            elif order.index(current_group[-1]) + 1 == pos:
                # Consecutive
                current_group.append(code)
            else:
                # Gap - start new group
                groups.append(current_group)
                current_group = [code]

        if current_group:
            groups.append(current_group)

        # Generate summaries for each group
        for group in groups:
            if len(group) == 1:
                # Single station
                station_name = STATION_NAMES.get(group[0], group[0])
                summaries.append(f"{direction} delay at {station_name}")
            else:
                # Multiple consecutive stations - from first to last in direction
                first_name = STATION_NAMES.get(group[0], group[0])
                last_name = STATION_NAMES.get(group[-1], group[-1])
                summaries.append(f"{direction} delay from {first_name} to {last_name}")

    # Process red track segments
    for segment in delays_segments:
        from_name = STATION_NAMES.get(segment['from'], segment['from'])
        to_name = STATION_NAMES.get(segment['to'], segment['to'])
        direction = segment['direction']
        summaries.append(f"{direction} service not running between {from_name} and {to_name}")

    # Process train bunching
    for bunching in bunching_incidents:
        station_name = STATION_NAMES.get(bunching['station'], bunching['station'])
        direction = bunching['direction']
        count = bunching['train_count']
        summaries.append(f"{direction} backup at {station_name} ({count} trains)")

    return summaries


def generate_description(status, trains, delays_platforms, delays_segments, bunching_incidents):
    """Generate a human-readable description of the system status."""
    if status == 'red':
        return "System not operating - insufficient train activity detected"

    if status == 'yellow':
        # Use the new readable summaries
        summaries = generate_delay_summaries(delays_platforms, delays_segments, bunching_incidents)
        if summaries:
            return "Delays detected - " + "; ".join(summaries)
        return "Delays detected"

    return f"Normal operation - {len(trains)} trains detected, all systems running"


def detect_system_status(image_path):
    """
    Detect subway system status from an image.

    Args:
        image_path: Path to the subway display image

    Returns:
        dict: Detection results including:
            - system_status: 'red', 'yellow', or 'green'
            - confidence: Always 1.0 (deterministic detection)
            - description: Human-readable status description
            - stations: Station detection data
            - segments: Track segment data
            - trains: Train detection data
            - delays_platforms: Platforms in hold mode
            - delays_segments: Red track sections
            - delays_bunching: Train bunching incidents
    """
    # Load image
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")

    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Get station detector and positions
    detector = _get_station_detector()
    positions = detector.get_hardcoded_positions(w, h, STATION_ORDER)

    # Detect stations and platforms
    stations = []
    delays_platforms = []

    for code, name in STATION_ORDER:
        if code in INTERNAL_STATIONS:
            continue

        pos = positions['stations'].get(code)
        if not pos:
            continue

        x = pos['center_x']
        # Use platform Y positions (not label positions) for color detection
        upper_y, lower_y = get_platform_y(code, h)

        # Detect platform colors
        upper_color, _ = detect_platform_color(hsv, x, upper_y)
        lower_color, _ = detect_platform_color(hsv, x, lower_y)

        stations.append({
            'code': code,
            'name': name,
            'x': x,
            'upper_y': upper_y,
            'lower_y': lower_y,
            'upper_color': upper_color,
            'lower_color': lower_color,
        })

        # Track delays (yellow = hold)
        if upper_color == 'yellow' and code != 'CT':  # CT upper is normally in hold
            direction = 'Northbound' if code in NORTH_SOUTH_STATIONS else 'Westbound'
            delays_platforms.append({
                'station': code,
                'name': name,
                'track': 'upper',
                'direction': direction,
            })

        if lower_color == 'yellow':
            direction = 'Southbound' if code in NORTH_SOUTH_STATIONS else 'Eastbound'
            delays_platforms.append({
                'station': code,
                'name': name,
                'track': 'lower',
                'direction': direction,
            })

    # Detect track segments
    segments = []
    delays_segments = []

    for seg_key, seg_data in positions['track_segments'].items():
        bounds = seg_data['bounds']
        color, _ = detect_segment_color(hsv, bounds)

        segments.append({
            'key': seg_key,
            'from_code': seg_data['from_code'],
            'to_code': seg_data['to_code'],
            'direction': seg_data['direction'],
            'color': color,
        })

        if color == 'red':
            delays_segments.append({
                'from': seg_data['from_code'],
                'to': seg_data['to_code'],
                'direction': seg_data['direction'],
                'key': seg_key,
            })

    # Detect trains
    trains = []
    train_detector = _get_train_detector()
    if train_detector is not None:
        raw_trains = train_detector.detect_trains(img)
        for t in raw_trains:
            trains.append({
                'id': t['id'],
                'x': int(t['x']),
                'y': int(t['y']),
                'track': t['track'],
                'confidence': t.get('confidence', 'high'),
            })

    # Detect train bunching
    bunching_incidents = detect_train_bunching(trains)

    # Calculate overall status
    system_status = calculate_system_status(trains, delays_platforms, delays_segments, bunching_incidents)

    # Generate description and summaries
    description = generate_description(system_status, trains, delays_platforms, delays_segments, bunching_incidents)
    delay_summaries = generate_delay_summaries(delays_platforms, delays_segments, bunching_incidents)

    return {
        'system_status': system_status,
        'confidence': 1.0,  # Deterministic detection
        'description': description,
        'delay_summaries': delay_summaries,
        'stations': stations,
        'segments': segments,
        'trains': trains,
        'delays_platforms': delays_platforms,
        'delays_segments': delays_segments,
        'delays_bunching': bunching_incidents,
        'image_dimensions': {'width': w, 'height': h},
    }


# For testing
if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        result = detect_system_status(sys.argv[1])
        import json
        print(json.dumps(result, indent=2))
    else:
        print("Usage: python detection.py <image_path>")
