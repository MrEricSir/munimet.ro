#!/usr/bin/env python3
"""Investigate why each optional train goes undetected.

For each undetected optional train, inspects:
1. Whether OCR text columns exist at that position
2. What the dark pixel content looks like
3. Whether symbol contours exist (and why they're rejected)
4. What the actual pixel values are
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from lib.train_detector import TrainDetector

UPPER_TRAIN_BAND = (0.25, 0.48)
LOWER_TRAIN_BAND = (0.56, 0.80)


def load_baselines():
    baseline_path = PROJECT_ROOT / 'tests' / 'baseline_trains.json'
    return json.loads(baseline_path.read_text())


def get_undetected_optionals(data):
    """Get optional trains that are truly undetected (no other train within tolerance)."""
    tolerance = data.get('tolerance', 30)
    images_dir = PROJECT_ROOT / 'tests' / 'images'
    td = TrainDetector()
    undetected = []

    for image_name, baseline_entry in data['images_with_trains'].items():
        image_path = images_dir / image_name
        if not image_path.exists():
            continue

        img = cv2.imread(str(image_path))
        if img is None:
            continue

        # Run normal detection
        detected = td.detect_trains(img)
        detected_xs = {t['x'] for t in detected}

        for entry in baseline_entry:
            if len(entry) < 3 or not isinstance(entry[2], dict) or not entry[2].get('optional'):
                continue

            opt_id, opt_x = entry[0], entry[1]

            # Check if any detection is within tolerance
            matched = any(abs(dx - opt_x) <= tolerance for dx in detected_xs)
            if not matched:
                undetected.append((image_name, opt_id, opt_x))

    return undetected


def investigate_position(image_path, target_x, train_id):
    """Investigate what's at a specific x position in the image."""
    img = cv2.imread(str(image_path))
    if img is None:
        return None

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    result = {'train_id': train_id, 'x': target_x, 'image': Path(image_path).name}

    # Check for text columns near this position
    for track_name, (band_start, band_end) in [
        ('upper', UPPER_TRAIN_BAND),
        ('lower', LOWER_TRAIN_BAND)
    ]:
        y_min = int(h * band_start)
        y_max = int(h * band_end)
        band = gray[y_min:y_max, :]
        band_h = y_max - y_min

        # Dark pixel column sums near target x
        search_range = 20
        x_start = max(0, target_x - search_range)
        x_end = min(w, target_x + search_range)

        dark_mask = (band < 120).astype(np.uint8)
        col_sums = dark_mask.sum(axis=0)

        # Find peak dark column near target
        peak_col = x_start + np.argmax(col_sums[x_start:x_end])
        peak_val = col_sums[peak_col]

        # Also check with looser threshold (150 instead of 120)
        light_mask = (band < 150).astype(np.uint8)
        light_col_sums = light_mask.sum(axis=0)
        light_peak = x_start + np.argmax(light_col_sums[x_start:x_end])
        light_peak_val = light_col_sums[light_peak]

        result[f'{track_name}_dark_peak'] = int(peak_val)
        result[f'{track_name}_dark_peak_x'] = int(peak_col)
        result[f'{track_name}_light_peak'] = int(light_peak_val)
        result[f'{track_name}_light_peak_x'] = int(light_peak)

        # Check actual pixel values at text location
        # Sample a vertical strip at the target x
        strip = gray[y_min:y_max, max(0, target_x-2):min(w, target_x+3)]
        if strip.size > 0:
            result[f'{track_name}_strip_min'] = int(strip.min())
            result[f'{track_name}_strip_max'] = int(strip.max())
            result[f'{track_name}_strip_mean'] = float(strip.mean())

    # Check for symbol contours in the track band
    track_y_min = int(h * 0.48)
    track_y_max = int(h * 0.62)

    # Build candidate mask
    saturation = hsv[:, :, 1]
    track_band = np.zeros((h, w), dtype=np.uint8)
    track_band[track_y_min:track_y_max, :] = 255

    cyan_mask = cv2.inRange(hsv, np.array([85, 120, 140]), np.array([105, 255, 230]))
    red1 = cv2.inRange(hsv, np.array([0, 150, 100]), np.array([8, 255, 255]))
    red2 = cv2.inRange(hsv, np.array([172, 150, 100]), np.array([180, 255, 255]))
    red_mask = red1 | red2

    colored_mask = (saturation > 150).astype(np.uint8) * 255
    candidate_mask = cv2.bitwise_and(colored_mask, track_band)
    candidate_mask = cv2.bitwise_and(candidate_mask, cv2.bitwise_not(cyan_mask))
    candidate_mask = cv2.bitwise_and(candidate_mask, cv2.bitwise_not(red_mask))

    pre_close = candidate_mask.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    post_close = cv2.morphologyEx(candidate_mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(post_close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find contours near target x
    nearby_contours = []
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        cx = x + cw // 2
        if abs(cx - target_x) < 30:
            area = cv2.contourArea(cnt)
            bounding_area = cw * ch
            rectangularity = area / max(bounding_area, 1)
            aspect = cw / max(ch, 1)
            pre_pixels = cv2.countNonZero(pre_close[y:y+ch, x:x+cw])
            fill_ratio = pre_pixels / max(bounding_area, 1)
            nearby_contours.append({
                'x': x, 'y': y, 'w': cw, 'h': ch,
                'area': area, 'bounding_area': bounding_area,
                'rectangularity': round(rectangularity, 2),
                'aspect': round(aspect, 2),
                'fill': round(fill_ratio, 2),
                'center_x': cx
            })

    result['symbol_contours'] = nearby_contours

    # Check saturation in track band near target x
    track_strip = hsv[track_y_min:track_y_max, max(0, target_x-15):min(w, target_x+15)]
    if track_strip.size > 0:
        sat_values = track_strip[:, :, 1]
        result['track_sat_max'] = int(sat_values.max())
        result['track_sat_mean'] = float(sat_values.mean())
        result['track_high_sat_pixels'] = int((sat_values > 150).sum())

    # Check if this is near another detected train (pileup)
    # Check pixel values more carefully - what grayscale value is the text?
    for track_name, (band_start, band_end) in [
        ('upper', UPPER_TRAIN_BAND),
        ('lower', LOWER_TRAIN_BAND)
    ]:
        y_min = int(h * band_start)
        y_max = int(h * band_end)
        strip = gray[y_min:y_max, max(0, target_x-3):min(w, target_x+4)]
        if strip.size > 0:
            # Find the darkest vertical section (likely the text)
            row_mins = strip.min(axis=1)
            darkest_rows = np.where(row_mins < 160)[0]
            if len(darkest_rows) > 0:
                result[f'{track_name}_text_rows'] = len(darkest_rows)
                result[f'{track_name}_text_min_gray'] = int(row_mins[darkest_rows].min())
                result[f'{track_name}_text_y_range'] = f'{darkest_rows[0]}-{darkest_rows[-1]}'

    return result


def main():
    data = load_baselines()
    images_dir = PROJECT_ROOT / 'tests' / 'images'

    print('Finding undetected optional trains...')
    undetected = get_undetected_optionals(data)

    print(f'\nFound {len(undetected)} truly undetected optional trains:')
    print('=' * 90)

    # Group by failure category
    categories = {
        'FAINT_TEXT': [],      # Text exists but too light
        'PILEUP': [],          # Another train occupies same column
        'EDGE': [],            # Near image edge (x < 50)
        'NO_SYMBOL': [],       # No symbol contour found
        'SMALL_SYMBOL': [],    # Symbol contour exists but too small
        'UNKNOWN': [],
    }

    for image_name, train_id, target_x in sorted(undetected):
        image_path = images_dir / image_name
        info = investigate_position(str(image_path), target_x, train_id)
        if not info:
            continue

        # Classify the failure
        is_edge = target_x < 50
        has_symbol = len(info.get('symbol_contours', [])) > 0
        has_text_upper = info.get('upper_text_rows', 0) > 5
        has_text_lower = info.get('lower_text_rows', 0) > 5
        has_faint_text_upper = (info.get('upper_light_peak', 0) > 10 and
                                 info.get('upper_dark_peak', 0) < 5)
        has_faint_text_lower = (info.get('lower_light_peak', 0) > 10 and
                                 info.get('lower_dark_peak', 0) < 5)

        print(f'\n{"─" * 90}')
        print(f'{image_name}: {train_id}@{target_x}')

        if is_edge:
            category = 'EDGE'
        elif has_symbol:
            max_area = max(c['area'] for c in info['symbol_contours'])
            if max_area < 200:
                category = 'SMALL_SYMBOL'
            else:
                category = 'UNKNOWN'
        elif has_faint_text_upper or has_faint_text_lower:
            category = 'FAINT_TEXT'
        elif has_text_upper or has_text_lower:
            category = 'PILEUP'
        else:
            category = 'UNKNOWN'

        categories[category].append((image_name, train_id, target_x))
        print(f'  Category: {category}')

        # Text analysis
        for track in ['upper', 'lower']:
            dark_peak = info.get(f'{track}_dark_peak', 0)
            light_peak = info.get(f'{track}_light_peak', 0)
            text_rows = info.get(f'{track}_text_rows', 0)
            min_gray = info.get(f'{track}_text_min_gray', 255)

            if dark_peak > 3 or light_peak > 5 or text_rows > 3:
                print(f'  {track} band: dark_cols={dark_peak}, light_cols={light_peak}, '
                      f'text_rows={text_rows}, min_gray={min_gray}')

        # Symbol analysis
        if info.get('symbol_contours'):
            for c in info['symbol_contours']:
                print(f'  Symbol contour: x={c["x"]},y={c["y"]} '
                      f'size={c["w"]}x{c["h"]} area={c["area"]} '
                      f'rect={c["rectangularity"]} aspect={c["aspect"]} '
                      f'fill={c["fill"]}')

        # Track band analysis
        high_sat = info.get('track_high_sat_pixels', 0)
        max_sat = info.get('track_sat_max', 0)
        if high_sat > 0 or max_sat > 100:
            print(f'  Track band: max_sat={max_sat}, high_sat_pixels={high_sat}')

    # Summary
    print(f'\n{"=" * 90}')
    print('CATEGORY SUMMARY')
    print(f'{"=" * 90}')
    for cat, items in categories.items():
        if items:
            print(f'\n{cat} ({len(items)} trains):')
            for img, tid, tx in items:
                print(f'  {img}: {tid}@{tx}')


if __name__ == '__main__':
    main()
