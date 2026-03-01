#!/usr/bin/env python3
"""
OpenCV-based detection library for SF Muni Metro status.

This module provides the detection API used by the status service,
combining station detection, train detection, and delay analysis.
"""

import re
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

# Import bunching and status config from centralized config
from lib.config import (
    BUNCHING_CLUSTER_DISTANCE,
    BUNCHING_THRESHOLD,
    BUNCHING_DEFAULT_ZONE_LENGTH,
    BUNCHING_EXCLUDED_STATIONS,
    BUNCHING_ZONE_LENGTH_UPPER,
    BUNCHING_ZONE_LENGTH_LOWER,
    MIN_TRAINS_OPERATING,
    HYSTERESIS_THRESHOLDS,
    PLATFORM_Y_WESTERN_UPPER_PCT,
    PLATFORM_Y_WESTERN_LOWER_PCT,
    PLATFORM_Y_EASTERN_UPPER_PCT,
    PLATFORM_Y_EASTERN_LOWER_PCT,
    REVENUE_SINGLE_SUFFIXES,
    REVENUE_DOUBLE_SUFFIXES,
)

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
        upper_y = int(img_height * PLATFORM_Y_WESTERN_UPPER_PCT)
        lower_y = int(img_height * PLATFORM_Y_WESTERN_LOWER_PCT)
    else:
        upper_y = int(img_height * PLATFORM_Y_EASTERN_UPPER_PCT)
        lower_y = int(img_height * PLATFORM_Y_EASTERN_LOWER_PCT)
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

    # Return 'red' if there are enough red pixels, regardless of cyan count
    # Any red section indicates track is disabled (safety-critical)
    # Use both absolute threshold and percentage to avoid false positives from UI elements
    min_pixel_threshold = 100
    min_percentage_threshold = 0.05  # 5% of segment must be red
    total_pixels = roi.shape[0] * roi.shape[1]
    red_percentage = red_pixels / total_pixels if total_pixels > 0 else 0

    if red_pixels >= min_pixel_threshold and red_percentage >= min_percentage_threshold:
        return 'red', red_pixels
    elif cyan_pixels >= min_pixel_threshold:
        return 'cyan', cyan_pixels
    else:
        return 'unknown', 0


def detect_train_bunching(trains, station_positions=None, threshold=BUNCHING_THRESHOLD, cluster_distance=BUNCHING_CLUSTER_DISTANCE):
    """
    Detect train bunching (multiple trains clustered close together).

    Args:
        trains: List of train dicts with 'x', 'track', 'id' keys.
        station_positions: Dict of station_code -> x_position. If None,
            uses STATION_X_POSITIONS from hardcoded fallback.
        threshold: Min trains to count as bunching.
        cluster_distance: Max pixels between trains in a cluster.

    Returns:
        list: List of bunching incidents [{station, track, direction, train_count}]
    """
    if station_positions is None:
        station_positions = STATION_X_POSITIONS

    def get_zone_length(station_code, track):
        if track == 'upper':
            return BUNCHING_ZONE_LENGTH_UPPER.get(station_code, BUNCHING_DEFAULT_ZONE_LENGTH)
        else:
            return BUNCHING_ZONE_LENGTH_LOWER.get(station_code, BUNCHING_DEFAULT_ZONE_LENGTH)

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
        for station_code, station_x in station_positions.items():
            if station_code in BUNCHING_EXCLUDED_STATIONS:
                continue
            # Find nearest station to cluster front (absolute distance)
            distance = abs(station_x - cluster_left_x)
            # Check against station-specific zone length
            max_zone = get_zone_length(station_code, 'upper')
            if distance < max_zone and distance < min_distance:
                min_distance = distance
                nearest_station = station_code
        if nearest_station:
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
        for station_code, station_x in station_positions.items():
            if station_code in BUNCHING_EXCLUDED_STATIONS:
                continue
            # Station must be at or to the right of the cluster front
            if station_x >= cluster_right_x:
                distance = station_x - cluster_right_x
                # Check against station-specific zone length
                max_zone = get_zone_length(station_code, 'lower')
                if distance < max_zone and distance < min_distance:
                    min_distance = distance
                    nearest_station = station_code
        if nearest_station:
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

    Only revenue trains (those with passenger-service route suffixes) count
    toward the operating threshold. Non-revenue trains (X suffix, *NNN* format)
    are detected but don't indicate the system is running.

    Returns:
        str: 'red', 'yellow', or 'green'
    """
    if bunching_incidents is None:
        bunching_incidents = []

    # Check how many trains have revenue route suffixes
    suffix_pattern = re.compile(r'\d{4}([A-Z]{1,2})$')
    revenue_trains = 0

    for train in trains:
        train_id = train.get('id', '')
        if train_id.startswith('UNKNOWN'):
            continue
        match = suffix_pattern.search(train_id)
        if match:
            suffix = match.group(1)
            if len(suffix) == 1 and suffix in REVENUE_SINGLE_SUFFIXES:
                revenue_trains += 1
            elif len(suffix) == 2 and suffix in REVENUE_DOUBLE_SUFFIXES:
                revenue_trains += 1
            # else: non-revenue suffix (X, etc.) — detected but not counted

    # Red: Fewer than MIN_TRAINS_OPERATING revenue trains (not operating)
    if revenue_trains < MIN_TRAINS_OPERATING:
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

    # Process red track segments - group adjacent segments by direction
    segments_by_direction = {}
    for segment in delays_segments:
        direction = segment['direction']
        if direction not in segments_by_direction:
            segments_by_direction[direction] = []
        segments_by_direction[direction].append(segment)

    for direction, segments in segments_by_direction.items():
        order = direction_orders.get(direction, [])

        # Build connections between stations from segments
        connections = {}
        all_stations = set()
        for seg in segments:
            from_st = seg['from']
            to_st = seg['to']
            all_stations.add(from_st)
            all_stations.add(to_st)
            if from_st not in connections:
                connections[from_st] = set()
            if to_st not in connections:
                connections[to_st] = set()
            connections[from_st].add(to_st)
            connections[to_st].add(from_st)

        # Find connected chains of stations (BFS)
        visited = set()
        chains = []

        for station in all_stations:
            if station in visited:
                continue
            chain = set()
            queue = [station]
            while queue:
                current = queue.pop(0)
                if current in visited:
                    continue
                visited.add(current)
                chain.add(current)
                for neighbor in connections.get(current, []):
                    if neighbor not in visited:
                        queue.append(neighbor)
            if chain:
                chains.append(chain)

        # Generate summary for each chain, ordered by direction of travel
        for chain in chains:
            sorted_stations = sorted(chain, key=lambda s: order.index(s) if s in order else 999)
            if len(sorted_stations) >= 2:
                first_name = STATION_NAMES.get(sorted_stations[0], sorted_stations[0])
                last_name = STATION_NAMES.get(sorted_stations[-1], sorted_stations[-1])
                summaries.append(f"{direction} service not running between {first_name} and {last_name}")

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

    # Get station detector and positions (auto-detect from image, hardcoded fallback)
    detector = _get_station_detector()
    positions = detector.get_positions(img, STATION_ORDER)

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
    suffix_pattern = re.compile(r'\d{4}([A-Z]{1,2})$')
    if train_detector is not None:
        raw_trains = train_detector.detect_trains(img)
        for t in raw_trains:
            # Extract route suffix for informational purposes
            route = None
            match = suffix_pattern.search(t['id'])
            if match:
                suffix = match.group(1)
                if suffix in REVENUE_SINGLE_SUFFIXES or suffix in REVENUE_DOUBLE_SUFFIXES:
                    route = suffix
            trains.append({
                'id': t['id'],
                'x': int(t['x']),
                'y': int(t['y']),
                'track': t['track'],
                'confidence': t.get('confidence', 'high'),
                'route': route,
            })

    # Detect train bunching using auto-detected station positions
    bunching_positions = {
        code: pos['center_x']
        for code, pos in positions['stations'].items()
    }
    bunching_incidents = detect_train_bunching(trains, station_positions=bunching_positions)

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


# Status priority for hysteresis calculation
STATUS_PRIORITY = {'green': 3, 'yellow': 2, 'red': 1}


def apply_status_hysteresis(best_status, reported_status, pending_status, pending_streak, timestamp=None):
    """
    Apply hysteresis to smooth status transitions.

    This prevents rapid status flips by requiring consistent agreement before
    changing the reported status. Different thresholds are used for different
    transitions, and overnight transition windows get extra smoothing.

    Args:
        best_status: The current best-of-3 status (dict with 'status' key)
        reported_status: The currently reported status (dict with 'status' key, or None)
        pending_status: The status we're considering changing to (string, or None)
        pending_streak: How many consecutive checks have had this pending status
        timestamp: ISO timestamp string (for time-of-day awareness)

    Returns:
        dict: {
            'reported_status': dict - the status to report (may be unchanged),
            'pending_status': str - the new pending status,
            'pending_streak': int - the new streak count,
            'status_changed': bool - whether reported status changed this check
        }
    """
    from datetime import datetime

    current_best = best_status['status']
    current_reported = reported_status['status'] if reported_status else None

    # Determine if we're in an overnight transition window
    # These are times when status naturally oscillates: 11pm-1am (night), 4am-6am (morning)
    in_transition_window = False
    if timestamp:
        try:
            dt = datetime.fromisoformat(timestamp)
            hour = dt.hour
            in_transition_window = (23 <= hour or hour < 1) or (4 <= hour < 6)
        except (ValueError, TypeError):
            pass

    # If no current reported status, just use best_status immediately
    if current_reported is None:
        return {
            'reported_status': best_status,
            'pending_status': None,
            'pending_streak': 0,
            'status_changed': True
        }

    # If best matches reported, use fresh data but mark as no status change
    if current_best == current_reported:
        return {
            'reported_status': best_status,  # Use fresh data (timestamp, trains, etc.)
            'pending_status': None,
            'pending_streak': 0,
            'status_changed': False
        }

    # Best differs from reported - track pending streak
    if pending_status == current_best:
        # Same pending status, increment streak
        new_streak = pending_streak + 1
    else:
        # Different pending status, start new streak
        new_streak = 1

    # Get threshold for this transition
    transition_key = (current_reported, current_best)
    base_threshold = HYSTERESIS_THRESHOLDS.get(transition_key, 2)

    # Add extra smoothing during overnight transition windows
    threshold = base_threshold + (1 if in_transition_window else 0)

    # Check if we've met the threshold
    if new_streak >= threshold:
        # Status change confirmed
        return {
            'reported_status': best_status,
            'pending_status': None,
            'pending_streak': 0,
            'status_changed': True
        }
    else:
        # Not yet - keep current reported status
        return {
            'reported_status': reported_status,
            'pending_status': current_best,
            'pending_streak': new_streak,
            'status_changed': False
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
