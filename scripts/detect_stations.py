#!/usr/bin/env python3
"""
Prototype: Detect station-level delays from Muni Metro display board.

Analyzes the sfmunicentral.com display board to determine which stations
have delays and in which direction, using computer vision.

Approach:
1. Find station labels (dark text on gray background) as anchors
2. Group into upper (westbound) and lower (eastbound) rows
3. Define track regions between adjacent stations
4. Detect yellow train icons (bowties) in the track corridor
5. Count icon density per segment — high density = delay

Usage:
    python scripts/detect_stations.py <image_path>
    python scripts/detect_stations.py <image_path> --debug    # saves annotated image
    python scripts/detect_stations.py <image_dir> --batch      # process all images in directory
"""

import cv2
import numpy as np
import json
import sys
from pathlib import Path
from typing import Optional

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import shared constants from lib/
from lib.station_constants import (
    STATION_ORDER,
    INTERNAL_STATIONS,
    SUBWAY_CODES,
    CENTRAL_CODES,
    PUBLIC_STATIONS,
    get_section_directions,
)


# Minimum train icons in a segment to flag as delayed
ICON_COUNT_THRESHOLD = 2

# Total icon count above which the image is flagged as delay regardless
# of per-segment clustering (catches spread-out delays)
TOTAL_ICON_THRESHOLD = 7

# Minimum red pixel ratio in track corridor to flag an outage
# Real outages have 30%+ red coverage; small signals are <3%
RED_TRACK_THRESHOLD = 0.05  # 5% of corridor pixels

# Cache path for station positions
CACHE_PATH = Path(__file__).parent.parent / "artifacts" / "runtime" / "cache" / "station_positions.json"

# Global detector instance (lazy-loaded)
_detector: Optional['StationDetector'] = None


def get_detector():
    """Get or create the station detector singleton."""
    global _detector
    if _detector is None:
        from lib.station_detector import StationDetector
        _detector = StationDetector(cache_path=str(CACHE_PATH))
    return _detector


def find_station_labels(gray):
    """Find station label bounding boxes by detecting dark text on gray background.

    The station labels (e.g., "WEL", "POL") are dark text (~0-80 intensity)
    on the gray background (~160-200 intensity). They appear at two specific
    y-bands: one for westbound (upper) and one for eastbound (lower).

    We filter aggressively by:
    - Y-position: labels appear in narrow horizontal bands
    - Size: 3-letter codes have consistent width (~25-45px) and height (~10-18px)
    - Spacing: adjacent stations are at least 50px apart

    Returns:
        List of (x, y, w, h) bounding boxes, sorted by x-position.
    """
    h_img = gray.shape[0]

    _, binary = cv2.threshold(gray, 90, 255, cv2.THRESH_BINARY_INV)

    # Dilate horizontally to merge individual characters into label blobs
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (6, 2))
    dilated = cv2.dilate(binary, kernel, iterations=2)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Station labels occupy two narrow y-bands:
    # Upper (westbound): ~42-50% of image height
    # Lower (eastbound): ~60-68% of image height
    wb_y_min = int(h_img * 0.42)
    wb_y_max = int(h_img * 0.50)
    eb_y_min = int(h_img * 0.60)
    eb_y_max = int(h_img * 0.68)

    labels = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cy = y + h // 2

        # Must be in a station label y-band
        in_wb_band = wb_y_min <= cy <= wb_y_max
        in_eb_band = eb_y_min <= cy <= eb_y_max
        if not (in_wb_band or in_eb_band):
            continue

        # Station labels are 3 chars: ~25-48px wide, ~10-18px tall
        if 22 < w < 50 and 8 < h < 20:
            labels.append((x, y, w, h))

    return sorted(labels, key=lambda b: b[0])


def _dedup_row(labels, min_spacing=30):
    """Remove duplicate labels within a row that are too close horizontally."""
    labels = sorted(labels, key=lambda b: b[0])
    filtered = []
    for label in labels:
        if not filtered or abs(label[0] - filtered[-1][0]) > min_spacing:
            filtered.append(label)
    return filtered


def split_into_rows(labels, img_height):
    """Split labels into upper (westbound) and lower (eastbound) rows.

    The display has westbound stations on top and eastbound on the bottom,
    separated roughly at the vertical midpoint. De-duplication happens
    after splitting so WB/EB labels at the same x-position don't remove
    each other.
    """
    mid_y = img_height // 2
    upper = sorted([l for l in labels if l[1] < mid_y], key=lambda b: b[0])
    lower = sorted([l for l in labels if l[1] >= mid_y], key=lambda b: b[0])
    return _dedup_row(upper), _dedup_row(lower)


def assign_station_codes(label_boxes, expected_codes):
    """Assign station codes to detected label boxes.

    If the count matches, assigns left-to-right. If there's a mismatch,
    uses spatial clustering to find the best match.

    Returns:
        List of (code, full_name, center_x, center_y) tuples.
        Also returns count of unmatched labels.
    """
    assigned = []

    if len(label_boxes) == len(expected_codes):
        # Perfect match - assign in order
        for i, (x, y, w, h) in enumerate(label_boxes):
            code, name = expected_codes[i]
            assigned.append((code, name, x + w // 2, y + h // 2))
        return assigned, 0

    # Count mismatch - do best effort by spacing
    # Use evenly-spaced assignment based on available labels
    if len(label_boxes) > 0:
        for i, (code, name) in enumerate(expected_codes):
            # Find the closest label box by proportional position
            target_frac = i / max(len(expected_codes) - 1, 1)
            best_idx = int(target_frac * (len(label_boxes) - 1))
            best_idx = min(best_idx, len(label_boxes) - 1)
            x, y, w, h = label_boxes[best_idx]
            assigned.append((code, name, x + w // 2, y + h // 2))

    unmatched = abs(len(label_boxes) - len(expected_codes))
    return assigned, unmatched


def detect_train_icons(hsv, img_height):
    """Detect yellow/amber train icons (bowties) in the track corridor.

    Train icons are bright yellow shapes (H~22-38, high saturation and value)
    that represent individual trains on the SCADA display. Their density
    per track segment indicates delays — bunched trains = delay.

    Args:
        hsv: HSV image array.
        img_height: Image height for corridor bounds.

    Returns:
        List of dicts with icon center, area, and bounding rect.
    """
    # Yellow/amber train icons: bright saturated yellow
    yellow_mask = cv2.inRange(hsv,
        np.array([22, 150, 180]),
        np.array([38, 255, 255]))

    # Restrict to track corridor (between the two label rows)
    corridor = np.zeros_like(yellow_mask)
    corridor[int(img_height * 0.32):int(img_height * 0.68), :] = 255
    yellow_in_corridor = yellow_mask & corridor

    # Clean up noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cleaned = cv2.morphologyEx(yellow_in_corridor, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    icons = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area >= 20:
            M = cv2.moments(cnt)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                x, y, w, h = cv2.boundingRect(cnt)
                icons.append({
                    'cx': cx, 'cy': cy,
                    'x': x, 'y': y, 'w': w, 'h': h,
                    'area': area,
                })

    return icons


def detect_route_codes(gray, icons):
    """Attempt to read 2-letter route codes displayed on/near train icons.

    Route codes (MM, KK, LL, NN, TT, SS, JJ) are displayed vertically on trains.
    This uses simple template matching against known letter patterns.

    Args:
        gray: Grayscale image array.
        icons: List of train icon dicts from detect_train_icons().

    Returns:
        Dict mapping icon index -> detected route code (or None if unreadable).
    """
    # Known Muni Metro route letters (displayed as pairs like MM, KK, etc.)
    ROUTE_LETTERS = {'M', 'K', 'L', 'N', 'T', 'S', 'J'}

    route_codes = {}

    for i, icon in enumerate(icons):
        # Look for text in a region above and below the icon
        # Route codes are typically displayed vertically near the train
        x, y, w, h = icon['x'], icon['y'], icon['w'], icon['h']

        # Expand search region vertically
        search_y_min = max(0, y - h)
        search_y_max = min(gray.shape[0], y + h * 2)
        search_x_min = max(0, x - 2)
        search_x_max = min(gray.shape[1], x + w + 2)

        if search_x_max <= search_x_min or search_y_max <= search_y_min:
            route_codes[i] = None
            continue

        roi = gray[search_y_min:search_y_max, search_x_min:search_x_max]

        # Look for dark text on the train icon (text is darker than yellow)
        # Threshold to find dark pixels that could be text
        _, text_mask = cv2.threshold(roi, 100, 255, cv2.THRESH_BINARY_INV)

        # Count dark pixels - if substantial, there's likely text
        dark_ratio = np.count_nonzero(text_mask) / text_mask.size if text_mask.size > 0 else 0

        # For now, just detect presence of route code (actual OCR would need tesseract)
        # We mark icons that appear to have text on them
        if dark_ratio > 0.05:  # At least 5% dark pixels suggests text
            # Could integrate tesseract here for actual letter recognition
            # For now, mark as "has_text" without identifying the specific code
            route_codes[i] = "detected"  # Placeholder - would be "MM", "KK", etc.
        else:
            route_codes[i] = None

    return route_codes


def detect_station_bunching(icons, stations, threshold_px=40):
    """Detect train bunching at station entrances.

    Bunching occurs when multiple trains are clustered near a station,
    indicating delays or congestion at that location.

    Args:
        icons: List of train icon dicts.
        stations: List of (code, name, cx, cy) station positions.
        threshold_px: Distance in pixels to consider a train "at" a station.

    Returns:
        List of dicts with station info and bunched train count.
    """
    bunched_stations = []

    for code, name, station_x, station_y in stations:
        # Skip internal stations
        if code in INTERNAL_STATIONS:
            continue

        # Count trains within threshold distance of this station
        nearby_trains = []
        for i, icon in enumerate(icons):
            dx = abs(icon['cx'] - station_x)
            dy = abs(icon['cy'] - station_y)
            distance = (dx**2 + dy**2) ** 0.5

            if distance <= threshold_px:
                nearby_trains.append({
                    'icon_idx': i,
                    'distance': distance,
                    'icon': icon,
                })

        # Bunching = 2+ trains at same station
        if len(nearby_trains) >= 2:
            bunched_stations.append({
                'station_code': code,
                'station_name': name,
                'station_x': station_x,
                'station_y': station_y,
                'train_count': len(nearby_trains),
                'trains': nearby_trains,
            })

    return bunched_stations


def detect_red_outage(hsv, img_height, regions):
    """Detect red track outage bars that replace the normal cyan track.

    During outages, large sections of track turn red instead of cyan.
    This is distinct from train icon clustering and indicates a shutdown.

    Args:
        hsv: HSV image array.
        img_height: Image height.
        regions: List of track region dicts to check.

    Returns:
        List of region dicts where red outage is detected.
    """
    # Red hue wraps around 0, check both ends
    red_mask1 = cv2.inRange(hsv,
        np.array([0, 80, 120]),
        np.array([8, 255, 255]))
    red_mask2 = cv2.inRange(hsv,
        np.array([170, 80, 120]),
        np.array([180, 255, 255]))
    red_mask = red_mask1 | red_mask2

    outage_regions = []
    for region in regions:
        roi = red_mask[
            region['y_min']:region['y_max'],
            region['x_min']:region['x_max']
        ]
        if roi.size == 0:
            continue

        ratio = np.count_nonzero(roi) / roi.size
        if ratio >= RED_TRACK_THRESHOLD:
            outage_regions.append(region)

    return outage_regions


def compute_red_ratios(hsv, regions):
    """Compute red pixel ratio for each track region (without thresholding).

    Args:
        hsv: HSV image array.
        regions: List of track region dicts.

    Returns:
        Dict mapping (from_code, to_code) -> red_ratio float.
    """
    red_mask1 = cv2.inRange(hsv,
        np.array([0, 80, 120]),
        np.array([8, 255, 255]))
    red_mask2 = cv2.inRange(hsv,
        np.array([170, 80, 120]),
        np.array([180, 255, 255]))
    red_mask = red_mask1 | red_mask2

    ratios = {}
    for region in regions:
        roi = red_mask[
            region['y_min']:region['y_max'],
            region['x_min']:region['x_max']
        ]
        key = (region['from_code'], region['to_code'])
        if roi.size == 0:
            ratios[key] = 0.0
        else:
            ratios[key] = np.count_nonzero(roi) / roi.size

    return ratios


def define_track_regions(stations, img_height, is_upper):
    """Define rectangular regions on the track between adjacent stations.

    Skips segments involving internal maintenance platforms. Each region
    includes its direction based on which section of the display it's in.

    Args:
        stations: List of (code, name, cx, cy) station positions.
        img_height: Image height for defining vertical bounds.
        is_upper: True for upper track row, False for lower track row.

    Returns:
        List of dicts defining each inter-station region.
    """
    # Track vertical bounds - narrow bands centered on the turquoise track lines
    # The display has two horizontal track lines between the upper and lower labels:
    #   - Upper track (WB/NB): around 50-54% of image height
    #   - Lower track (EB/SB): around 56-60% of image height
    # These narrow bands capture train icons positioned on/near the track lines
    if is_upper:
        track_y_min = int(img_height * 0.50)
        track_y_max = int(img_height * 0.54)
    else:
        track_y_min = int(img_height * 0.56)
        track_y_max = int(img_height * 0.60)

    regions = []
    for i in range(len(stations) - 1):
        code1, name1, x1, _ = stations[i]
        code2, name2, x2, _ = stations[i + 1]

        # Skip segments involving internal maintenance platforms
        if code1 in INTERNAL_STATIONS or code2 in INTERNAL_STATIONS:
            continue

        # Determine direction based on display section
        upper_dir, lower_dir = get_section_directions(code1, code2)
        direction = upper_dir if is_upper else lower_dir

        regions.append({
            'from_code': code1,
            'from_name': name1,
            'to_code': code2,
            'to_name': name2,
            'direction': direction,
            'x_min': x1,
            'x_max': x2,
            'y_min': track_y_min,
            'y_max': track_y_max,
        })

    return regions


def analyze_image(image_path, debug=False):
    """Analyze a Muni display board image for station-level delays.

    Uses train icon density per track segment to detect delays.
    Segments with ICON_COUNT_THRESHOLD or more yellow train icons
    are flagged as delayed.

    Args:
        image_path: Path to the display board image.
        debug: If True, saves an annotated debug image.

    Returns:
        Dict with delay information and detection metadata.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Step 1: Find station labels
    all_labels = find_station_labels(gray)
    upper_labels, lower_labels = split_into_rows(all_labels, h)

    # Step 2: Assign station codes
    upper_stations, upper_unmatched = assign_station_codes(upper_labels, STATION_ORDER)
    lower_stations, lower_unmatched = assign_station_codes(lower_labels, STATION_ORDER)

    # Step 3: Detect yellow train icons in the track corridor
    train_icons = detect_train_icons(hsv, h)

    # Step 4: Map icons to track segments and count per segment
    mid_y = h // 2
    upper_regions = define_track_regions(upper_stations, h, is_upper=True)
    lower_regions = define_track_regions(lower_stations, h, is_upper=False)

    # Count icons per segment
    segment_counts = {}  # seg_key -> {'count': N, 'region': region}

    for icon in train_icons:
        if icon['cy'] < mid_y:
            regions = upper_regions
        else:
            regions = lower_regions

        for region in regions:
            if (region['x_min'] <= icon['cx'] <= region['x_max'] and
                region['y_min'] <= icon['cy'] <= region['y_max']):
                seg_key = (region['from_code'], region['to_code'], region['direction'])
                if seg_key not in segment_counts:
                    segment_counts[seg_key] = {
                        'count': 0,
                        'region': region,
                    }
                segment_counts[seg_key]['count'] += 1
                break

    # Step 5a: Flag segments with high icon density as delayed
    delays = []
    seen_segments = set()

    for seg_key, info in segment_counts.items():
        if info['count'] >= ICON_COUNT_THRESHOLD:
            region = info['region']
            seen_segments.add(seg_key)
            delays.append({
                'segment': f"{region['from_name']} to {region['to_name']}",
                'from': region['from_name'],
                'to': region['to_name'],
                'direction': region['direction'],
                'icon_count': info['count'],
                'reason': 'icon_cluster',
            })

    # Step 5b: Check for red track outage bars
    all_regions = upper_regions + lower_regions
    outage_upper = detect_red_outage(hsv, h, upper_regions)
    outage_lower = detect_red_outage(hsv, h, lower_regions)

    for region in outage_upper + outage_lower:
        seg_key = (region['from_code'], region['to_code'], region['direction'])
        if seg_key not in seen_segments:
            seen_segments.add(seg_key)
            delays.append({
                'segment': f"{region['from_name']} to {region['to_name']}",
                'from': region['from_name'],
                'to': region['to_name'],
                'direction': region['direction'],
                'icon_count': segment_counts.get(seg_key, {}).get('count', 0),
                'reason': 'red_outage',
            })

    # Step 5c: If total icon count is high but no segment triggered,
    # flag as a spread-out delay (trains delayed across the system)
    spread_delay = (len(train_icons) >= TOTAL_ICON_THRESHOLD and len(delays) == 0)
    if spread_delay:
        delays.append({
            'segment': 'System-wide',
            'from': 'Multiple',
            'to': 'stations',
            'direction': 'Both',
            'icon_count': len(train_icons),
            'reason': 'high_total_count',
        })

    # Build result
    result = {
        'image': str(Path(image_path).name),
        'delays': delays,
        'summary': build_summary(delays),
        'detection_meta': {
            'upper_labels_found': len(upper_labels),
            'lower_labels_found': len(lower_labels),
            'expected_per_row': len(STATION_ORDER),
            'upper_unmatched': upper_unmatched,
            'lower_unmatched': lower_unmatched,
            'layout_ok': upper_unmatched == 0 and lower_unmatched == 0,
            'total_train_icons': len(train_icons),
            'outage_segments': len(outage_upper) + len(outage_lower),
        }
    }

    # Debug visualization
    if debug:
        debug_img = img.copy()

        # Draw station positions
        for code, name, cx, cy in upper_stations:
            cv2.circle(debug_img, (cx, cy), 5, (0, 255, 0), -1)
            cv2.putText(debug_img, code, (cx - 10, cy - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        for code, name, cx, cy in lower_stations:
            cv2.circle(debug_img, (cx, cy), 5, (0, 200, 0), -1)
            cv2.putText(debug_img, code, (cx - 10, cy + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 0), 1)

        # Draw detected train icons
        for icon in train_icons:
            cv2.rectangle(debug_img,
                         (icon['x'], icon['y']),
                         (icon['x'] + icon['w'], icon['y'] + icon['h']),
                         (0, 165, 255), 2)  # orange outline

        # Draw track regions
        for region in upper_regions + lower_regions:
            cv2.rectangle(debug_img,
                         (region['x_min'], region['y_min']),
                         (region['x_max'], region['y_max']),
                         (200, 200, 200), 1)

        # Highlight delayed segments (red for clusters, magenta for outage)
        for d in delays:
            if d['reason'] == 'high_total_count':
                continue
            # Find the matching region
            for region in upper_regions + lower_regions:
                if (region['from_name'] == d['from'] and
                    region['to_name'] == d['to'] and
                    region['direction'] == d['direction']):
                    color = (0, 0, 255) if d['reason'] == 'icon_cluster' else (255, 0, 255)
                    cv2.rectangle(debug_img,
                                 (region['x_min'], region['y_min']),
                                 (region['x_max'], region['y_max']),
                                 color, 2)
                    break

        debug_path = str(Path(image_path).with_suffix('.debug.jpg'))
        cv2.imwrite(debug_path, debug_img)
        result['debug_image'] = debug_path

    return result


def build_summary(delays):
    """Build a human-readable summary of detected delays."""
    if not delays:
        return "Normal - no delays detected"

    # Check for system-wide spread delay
    spread = [d for d in delays if d.get('reason') == 'high_total_count']
    if spread and len(delays) == 1:
        return f"System-wide delay ({spread[0]['icon_count']} trains visible)"

    # Group by direction (supports WB/EB for subway section, NB/SB for central section)
    directions = ['Westbound', 'Eastbound', 'Northbound', 'Southbound']
    parts = []

    for direction in directions:
        dir_delays = [d for d in delays if d['direction'] == direction]
        if not dir_delays:
            continue
        stations = set()
        has_outage = False
        for d in dir_delays:
            if d['from'] != 'Multiple':
                stations.add(d['from'])
                stations.add(d['to'])
            if d.get('reason') == 'red_outage':
                has_outage = True

        label = 'outage' if has_outage else 'delay'
        parts.append(f"{direction} {label}: {', '.join(sorted(stations))}")

    return "; ".join(parts)


def _get_stations_from_cache(img, h, w):
    """
    Try to get station positions from cache or hardcoded positions.

    Returns:
        Tuple of (upper_stations, lower_stations, upper_regions, lower_regions,
                  detection_method, cache_confidence) or None if unavailable.
    """
    try:
        detector = get_detector()
        detection_result = detector.detect_with_cache(img, STATION_ORDER, use_hardcoded=True)

        cached_stations = detection_result.get('stations', {})
        cached_segments = detection_result.get('track_segments', {})
        method = detection_result.get('detection_method', 'cache')
        confidence = detection_result.get('confidence', 0)

        if not cached_stations:
            return None

        # Convert cached format to the format used by the rest of the code
        upper_stations = []
        lower_stations = []

        for code, station in cached_stations.items():
            name = station.get('name', code)
            center_x = station.get('center_x', 0)

            # Use label Y positions from hardcoded data, or platform positions if available
            upper_p = station.get('upper_platform')
            lower_p = station.get('lower_platform')

            # Prefer explicit label Y, then platform cy, then fallback percentages
            upper_y = station.get('upper_label_y') or (upper_p['cy'] if upper_p else int(h * 0.375))
            lower_y = station.get('lower_label_y') or (lower_p['cy'] if lower_p else int(h * 0.50))

            upper_stations.append((code, name, center_x, upper_y))
            lower_stations.append((code, name, center_x, lower_y))

        # Sort by x-position
        upper_stations.sort(key=lambda s: s[2])
        lower_stations.sort(key=lambda s: s[2])

        # Convert cached segments to region format
        upper_regions = []
        lower_regions = []

        for seg_key, segment in cached_segments.items():
            bounds = segment.get('bounds', {})
            region = {
                'from_code': segment['from_code'],
                'to_code': segment['to_code'],
                'from_name': next(
                    (name for code, name in STATION_ORDER if code == segment['from_code']),
                    segment['from_code']
                ),
                'to_name': next(
                    (name for code, name in STATION_ORDER if code == segment['to_code']),
                    segment['to_code']
                ),
                'direction': segment['direction'],
                'x_min': bounds.get('x_min', 0),
                'x_max': bounds.get('x_max', 0),
                'y_min': bounds.get('y_min', 0),
                'y_max': bounds.get('y_max', 0),
            }

            if '_upper' in seg_key:
                upper_regions.append(region)
            else:
                lower_regions.append(region)

        # Sort regions by x_min
        upper_regions.sort(key=lambda r: r['x_min'])
        lower_regions.sort(key=lambda r: r['x_min'])

        return (
            upper_stations,
            lower_stations,
            upper_regions,
            lower_regions,
            method,
            confidence,
        )

    except Exception as e:
        # Fall back to text detection on any error
        print(f"Cache detection failed, falling back to text: {e}")
        return None


def analyze_image_detailed(image_path, use_cache=True):
    """Analyze image and return intermediate data for evaluation/fitting.

    Like analyze_image() but returns per-segment intermediate data (icon counts,
    red ratios) plus an annotated debug image as a numpy array. This allows the
    parameter fitter to re-apply different thresholds without re-running OpenCV.

    Args:
        image_path: Path to the display board image.
        use_cache: Whether to try using cached station positions (default True).

    Returns:
        Dict with:
          - 'result': the normal analyze_image() result dict
          - 'debug_img': numpy BGR array with debug annotations
          - 'segments': list of per-segment dicts with:
              'from_code', 'to_code', 'from_name', 'to_name', 'direction',
              'icon_count', 'red_ratio', 'predicted_delay', 'reason'
          - 'total_train_icons': int
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Try to use cached positions first
    detection_method = 'text'
    cache_confidence = 0.0
    cache_result = None

    if use_cache:
        cache_result = _get_stations_from_cache(img, h, w)

    # Track text detection label counts (for backward compatibility reporting)
    text_upper_labels_count = 0
    text_lower_labels_count = 0

    if cache_result is not None:
        (upper_stations, lower_stations, upper_regions, lower_regions,
         detection_method, cache_confidence) = cache_result
        upper_unmatched = 0
        lower_unmatched = 0
    else:
        # Fall back to text-based detection
        all_labels = find_station_labels(gray)
        upper_labels, lower_labels = split_into_rows(all_labels, h)
        text_upper_labels_count = len(upper_labels)
        text_lower_labels_count = len(lower_labels)
        upper_stations, upper_unmatched = assign_station_codes(upper_labels, STATION_ORDER)
        lower_stations, lower_unmatched = assign_station_codes(lower_labels, STATION_ORDER)
        upper_regions = define_track_regions(upper_stations, h, is_upper=True)
        lower_regions = define_track_regions(lower_stations, h, is_upper=False)

    # Detect train icons
    train_icons = detect_train_icons(hsv, h)

    # Detect route codes on train icons
    route_codes = detect_route_codes(gray, train_icons)

    # Detect station bunching (trains clustered at station entrances)
    all_stations = upper_stations + lower_stations
    bunched_stations = detect_station_bunching(train_icons, all_stations)

    # Track regions already defined above (from cache or text detection)
    mid_y = h // 2

    # Count icons per segment
    segment_counts = {}
    for icon in train_icons:
        if icon['cy'] < mid_y:
            regions = upper_regions
        else:
            regions = lower_regions

        for region in regions:
            if (region['x_min'] <= icon['cx'] <= region['x_max'] and
                region['y_min'] <= icon['cy'] <= region['y_max']):
                seg_key = (region['from_code'], region['to_code'], region['direction'])
                if seg_key not in segment_counts:
                    segment_counts[seg_key] = 0
                segment_counts[seg_key] += 1
                break

    # Compute red ratios per region (un-thresholded)
    upper_red_ratios = compute_red_ratios(hsv, upper_regions)
    lower_red_ratios = compute_red_ratios(hsv, lower_regions)

    # Build per-segment intermediate data
    segments = []
    for regions, red_ratios in [
        (upper_regions, upper_red_ratios),
        (lower_regions, lower_red_ratios),
    ]:
        for region in regions:
            seg_key = (region['from_code'], region['to_code'], region['direction'])
            icon_count = segment_counts.get(seg_key, 0)
            red_key = (region['from_code'], region['to_code'])
            red_ratio = red_ratios.get(red_key, 0.0)

            # Apply current thresholds to determine prediction
            predicted_delay = False
            reason = None
            if icon_count >= ICON_COUNT_THRESHOLD:
                predicted_delay = True
                reason = 'icon_cluster'
            elif red_ratio >= RED_TRACK_THRESHOLD:
                predicted_delay = True
                reason = 'red_outage'

            segments.append({
                'from_code': region['from_code'],
                'to_code': region['to_code'],
                'from_name': region['from_name'],
                'to_name': region['to_name'],
                'direction': region['direction'],
                'icon_count': icon_count,
                'red_ratio': red_ratio,
                'predicted_delay': predicted_delay,
                'reason': reason,
            })

    # Check for spread-out delay (high total but no segment triggered)
    cluster_or_outage_delays = [s for s in segments if s['predicted_delay']]
    if len(train_icons) >= TOTAL_ICON_THRESHOLD and not cluster_or_outage_delays:
        # Mark this as a system-wide spread delay — not per-segment
        pass  # handled in result below

    spread_delay = (len(train_icons) >= TOTAL_ICON_THRESHOLD and not cluster_or_outage_delays)

    # Build the normal result (same as analyze_image)
    delays = []
    for seg in segments:
        if seg['predicted_delay']:
            delays.append({
                'segment': f"{seg['from_name']} to {seg['to_name']}",
                'from': seg['from_name'],
                'to': seg['to_name'],
                'direction': seg['direction'],
                'icon_count': seg['icon_count'],
                'reason': seg['reason'],
            })

    if spread_delay:
        delays.append({
            'segment': 'System-wide',
            'from': 'Multiple',
            'to': 'stations',
            'direction': 'Both',
            'icon_count': len(train_icons),
            'reason': 'high_total_count',
        })

    # Calculate labels found (for backward compatibility)
    if detection_method in ('cache', 'color'):
        # When using cache, report stations found instead of text labels
        upper_labels_found = len(upper_stations)
        lower_labels_found = len(lower_stations)
        layout_ok = cache_confidence >= 0.7
    else:
        # Text detection - use actual label counts
        upper_labels_found = text_upper_labels_count
        lower_labels_found = text_lower_labels_count
        layout_ok = upper_unmatched == 0 and lower_unmatched == 0

    result = {
        'image': str(Path(image_path).name),
        'delays': delays,
        'summary': build_summary(delays),
        'detection_meta': {
            'upper_labels_found': upper_labels_found,
            'lower_labels_found': lower_labels_found,
            'expected_per_row': len(STATION_ORDER),
            'upper_unmatched': upper_unmatched,
            'lower_unmatched': lower_unmatched,
            'layout_ok': layout_ok,
            'total_train_icons': len(train_icons),
            'outage_segments': sum(1 for s in segments if s['reason'] == 'red_outage'),
            # New fields for cache-based detection
            'detection_method': detection_method,
            'cache_confidence': cache_confidence,
        }
    }

    # Build debug image
    debug_img = img.copy()

    for code, name, cx, cy in upper_stations:
        cv2.circle(debug_img, (cx, cy), 5, (0, 255, 0), -1)
        cv2.putText(debug_img, code, (cx - 10, cy - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    for code, name, cx, cy in lower_stations:
        cv2.circle(debug_img, (cx, cy), 5, (0, 200, 0), -1)
        cv2.putText(debug_img, code, (cx - 10, cy + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 0), 1)

    for icon in train_icons:
        cv2.rectangle(debug_img,
                     (icon['x'], icon['y']),
                     (icon['x'] + icon['w'], icon['y'] + icon['h']),
                     (0, 165, 255), 2)

    # Draw all track segment regions (gray = normal, colored = delay)
    # These are also the clickable regions
    for region in upper_regions + lower_regions:
        # Draw gray background rectangle for all segments
        cv2.rectangle(debug_img,
                     (region['x_min'], region['y_min']),
                     (region['x_max'], region['y_max']),
                     (200, 200, 200), 1)
        # Add small label showing segment codes (for debugging)
        label = f"{region['from_code']}-{region['to_code']}"
        label_y = region['y_min'] - 3 if region['y_min'] > 20 else region['y_max'] + 12
        cv2.putText(debug_img, label, (region['x_min'], label_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 150, 150), 1)

    # Highlight segments with predicted delays
    for seg in segments:
        if seg['predicted_delay']:
            # Find the matching region and highlight it
            matched = False
            for region in upper_regions + lower_regions:
                if (region['from_code'] == seg['from_code'] and
                    region['to_code'] == seg['to_code'] and
                    region['direction'] == seg['direction']):
                    color = (0, 0, 255) if seg['reason'] == 'icon_cluster' else (255, 0, 255)
                    cv2.rectangle(debug_img,
                                 (region['x_min'], region['y_min']),
                                 (region['x_max'], region['y_max']),
                                 color, 2)
                    # Draw segment name on the colored rectangle for debugging
                    cv2.putText(debug_img, f"{seg['from_name'][:3]}-{seg['to_name'][:3]}",
                               (region['x_min'] + 2, region['y_min'] + 12),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
                    matched = True
                    break
            if not matched:
                print(f"WARNING: No region found for delay segment: {seg['from_code']}->{seg['to_code']} ({seg['direction']})")

    # Draw circles around bunched stations (cyan with train count)
    for b in bunched_stations:
        cv2.circle(debug_img, (b['station_x'], b['station_y']), 35, (255, 255, 0), 2)
        cv2.putText(debug_img, f"{b['train_count']}",
                   (b['station_x'] - 5, b['station_y'] + 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

    return {
        'result': result,
        'debug_img': debug_img,
        'segments': segments,
        'total_train_icons': len(train_icons),
        'train_icons': train_icons,
        'route_codes': route_codes,
        'bunched_stations': bunched_stations,
        'upper_stations': upper_stations,
        'lower_stations': lower_stations,
        'upper_regions': upper_regions,
        'lower_regions': lower_regions,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Detect station-level delays from Muni Metro display board images"
    )
    parser.add_argument('path', help='Image path or directory (with --batch)')
    parser.add_argument('--debug', action='store_true',
                       help='Save annotated debug images')
    parser.add_argument('--batch', action='store_true',
                       help='Process all .jpg images in directory')
    parser.add_argument('--json', action='store_true',
                       help='Output as JSON')
    parser.add_argument('--recalibrate', action='store_true',
                       help='Force recalibration of station position cache')

    args = parser.parse_args()
    path = Path(args.path)

    # Handle recalibration
    if args.recalibrate:
        if path.is_file():
            print(f"Recalibrating station positions using: {path}")
            img = cv2.imread(str(path))
            if img is None:
                print(f"Error: Could not load image: {path}")
                sys.exit(1)

            detector = get_detector()
            cache_data = detector.calibrate(img, STATION_ORDER)

            print(f"\nCalibration complete!")
            print(f"  Confidence: {cache_data['confidence_score']:.1%}")
            print(f"  Stations detected: {len(cache_data['stations'])}")
            print(f"  Track segments: {len(cache_data['track_segments'])}")
            print(f"  Cache saved to: {CACHE_PATH}")

            if not args.batch:
                return
        else:
            print("Error: --recalibrate requires an image file path")
            sys.exit(1)

    if args.batch:
        if not path.is_dir():
            print(f"Error: {path} is not a directory")
            sys.exit(1)

        images = sorted(path.glob("*.jpg"))
        if not images:
            print(f"No .jpg images found in {path}")
            sys.exit(1)

        print(f"Processing {len(images)} images...\n")

        all_results = []
        for img_path in images:
            try:
                result = analyze_image(img_path, debug=args.debug)
                all_results.append(result)

                status = "DELAY" if result['delays'] else "OK"
                meta = result['detection_meta']
                layout_warn = "" if meta['layout_ok'] else " [LAYOUT?]"
                print(f"  [{status:5s}] {img_path.name}: {result['summary']}{layout_warn}")
            except Exception as e:
                print(f"  [ERROR] {img_path.name}: {e}")

        if args.json:
            print(json.dumps(all_results, indent=2))

    else:
        if not path.is_file():
            print(f"Error: {path} is not a file")
            sys.exit(1)

        result = analyze_image(path, debug=args.debug)

        if args.json:
            print(json.dumps(result, indent=2))
        else:
            meta = result['detection_meta']
            print(f"Image: {result['image']}")
            print(f"Layout OK: {meta['layout_ok']}")
            print(f"  Labels found: WB={meta['westbound_labels_found']}, "
                  f"EB={meta['eastbound_labels_found']} "
                  f"(expected {meta['expected_per_row']})")
            print(f"  Train icons detected: {meta['total_train_icons']}")
            print()

            if result['delays']:
                print("Delays detected:")
                for d in result['delays']:
                    reason = d.get('reason', '')
                    extra = f" [{reason}]" if reason != 'icon_cluster' else ''
                    print(f"  {d['direction']:10s}  {d['from']} → {d['to']}  "
                          f"({d['icon_count']} trains){extra}")
            else:
                print("No delays detected")

            print(f"\nSummary: {result['summary']}")

            if args.debug and 'debug_image' in result:
                print(f"\nDebug image saved: {result['debug_image']}")


if __name__ == '__main__':
    main()
