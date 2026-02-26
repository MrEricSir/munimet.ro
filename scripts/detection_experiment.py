#!/usr/bin/env python3
"""Experiment with detection improvements to catch optional trains.

Tests the impact of:
1. Running symbol detection alongside OCR (not just as fallback)
2. Lowering SYMBOL_MIN_AREA threshold
3. Both combined

Reports which optional trains are newly detected and any new false positives.
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from lib.train_detector import TrainDetector


def load_baselines():
    baseline_path = PROJECT_ROOT / 'tests' / 'baseline_trains.json'
    data = json.loads(baseline_path.read_text())
    return data


def get_optional_trains(baseline_entry):
    """Extract optional trains from a baseline entry."""
    optional = []
    for entry in baseline_entry:
        if len(entry) >= 3 and isinstance(entry[2], dict) and entry[2].get('optional'):
            optional.append((entry[0], entry[1]))
    return optional


def get_required_trains(baseline_entry):
    """Extract required (non-optional) trains from a baseline entry."""
    required = []
    for entry in baseline_entry:
        is_optional = len(entry) >= 3 and isinstance(entry[2], dict) and entry[2].get('optional')
        if not is_optional:
            required.append((entry[0], entry[1]))
    return required


def run_detection(image_path, mode='normal', min_area=200):
    """Run train detection with specified mode.

    Modes:
    - 'normal': Standard detection (symbols only when OCR finds 0)
    - 'always_symbols': Always run symbols alongside OCR
    """
    img = cv2.imread(str(image_path))
    if img is None:
        return []

    td = TrainDetector()
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # OCR detection (always the same)
    ocr_trains = td._detect_by_ocr(gray, h, w, hsv)

    if mode == 'normal':
        if len(ocr_trains) == 0:
            symbols = detect_symbols_custom(td, hsv, h, w, gray, min_area)
            return td._merge(ocr_trains, symbols)
        return ocr_trains
    elif mode == 'always_symbols':
        symbols = detect_symbols_custom(td, hsv, h, w, gray, min_area)
        return td._merge(ocr_trains, symbols)

    return ocr_trains


def detect_symbols_custom(td, hsv, h, w, gray, min_area=200):
    """Run symbol detection with custom min_area threshold."""
    track_y_min = int(h * 0.48)
    track_y_max = int(h * 0.62)
    upper_track_y = int(h * 0.52)

    track_band = np.zeros((h, w), dtype=np.uint8)
    track_band[track_y_min:track_y_max, :] = 255

    cyan_mask = cv2.inRange(hsv, np.array([85, 120, 140]), np.array([105, 255, 230]))
    red1 = cv2.inRange(hsv, np.array([0, 150, 100]), np.array([8, 255, 255]))
    red2 = cv2.inRange(hsv, np.array([172, 150, 100]), np.array([180, 255, 255]))
    red_mask = red1 | red2

    saturation = hsv[:, :, 1]
    colored_mask = (saturation > 150).astype(np.uint8) * 255
    candidate_mask = cv2.bitwise_and(colored_mask, track_band)
    candidate_mask = cv2.bitwise_and(candidate_mask, cv2.bitwise_not(cyan_mask))
    candidate_mask = cv2.bitwise_and(candidate_mask, cv2.bitwise_not(red_mask))

    pre_close_mask = candidate_mask.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    candidate_mask = cv2.morphologyEx(candidate_mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(candidate_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    symbols = []
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)

        bounding_area = cw * ch
        rectangularity = area / max(bounding_area, 1)
        aspect = cw / max(ch, 1)
        pre_pixels = cv2.countNonZero(pre_close_mask[y:y+ch, x:x+cw])
        fill_ratio = pre_pixels / max(bounding_area, 1)

        # Apply filters with custom min_area
        if area < min_area:
            continue
        if bounding_area < min_area * 1.25:  # Scale rect area with min_area
            continue
        if cw < 12 or cw > 45 or ch < 6 or ch > 20:  # Relaxed size filters
            continue
        if rectangularity < 0.4:  # Slightly relaxed
            continue
        if aspect < 1.2 or aspect > 4.0:  # Wider aspect range
            continue
        if fill_ratio < 0.7:
            continue

        cx, cy = x + cw // 2, y + ch // 2
        track = 'upper' if cy < upper_track_y else 'lower'
        symbols.append({'x': cx, 'y': cy, 'track': track})

    # Add train-colored rectangles
    symbols.extend(td._detect_train_colors(hsv, h, w, track_y_min, track_y_max, upper_track_y))

    # Deduplicate
    symbols = sorted(symbols, key=lambda s: s['x'])
    unique = []
    for s in symbols:
        if not any(abs(s['x'] - u['x']) < 30 and s['track'] == u['track'] for u in unique):
            unique.append(s)

    return unique


def check_match(detected_trains, expected_id, expected_x, tolerance=30):
    """Check if a train was detected near expected position."""
    for t in detected_trains:
        if abs(t['x'] - expected_x) <= tolerance:
            return t
    return None


def main():
    data = load_baselines()
    tolerance = data.get('tolerance', 30)
    images_dir = PROJECT_ROOT / 'tests' / 'images'

    configs = [
        ('normal', 200, 'Current (symbols fallback only, area>=200)'),
        ('always_symbols', 200, 'Always symbols, area>=200'),
        ('always_symbols', 150, 'Always symbols, area>=150'),
        ('always_symbols', 100, 'Always symbols, area>=100'),
        ('always_symbols', 80, 'Always symbols, area>=80'),
        ('always_symbols', 50, 'Always symbols, area>=50'),
    ]

    # Track results
    results = {name: {'caught': [], 'false_positives': []} for _, _, name in configs}

    for image_name, baseline_entry in data['images_with_trains'].items():
        image_path = images_dir / image_name
        if not image_path.exists():
            continue

        optional = get_optional_trains(baseline_entry)
        required = get_required_trains(baseline_entry)
        all_trains = [(e[0], e[1]) for e in baseline_entry]

        if not optional:
            # Still check for false positives on images with no optional trains
            pass

        for mode, min_area, config_name in configs:
            detected = run_detection(image_path, mode=mode, min_area=min_area)

            # Check which optional trains are now caught
            for opt_id, opt_x in optional:
                match = check_match(detected, opt_id, opt_x, tolerance)
                if match:
                    results[config_name]['caught'].append(
                        (image_name, opt_id, opt_x, match['id'], match['x'])
                    )

            # Check for extra detections (not matching any baseline train)
            for t in detected:
                matched_any = False
                for tid, tx in all_trains:
                    if abs(t['x'] - tx) <= tolerance:
                        matched_any = True
                        break
                if not matched_any:
                    results[config_name]['false_positives'].append(
                        (image_name, t['id'], t['x'])
                    )

    # Also check train-free images for false positives
    for image_name, config in data.get('train_free_images', {}).items():
        image_path = images_dir / image_name
        if not image_path.exists():
            continue
        max_fp = config.get('max_false_positives', 0)

        for mode, min_area, config_name in configs:
            detected = run_detection(image_path, mode=mode, min_area=min_area)
            if len(detected) > max_fp:
                results[config_name]['false_positives'].append(
                    (image_name, f'{len(detected)} trains (max allowed: {max_fp})', 0)
                )

    # Print results
    print('=' * 80)
    print('DETECTION EXPERIMENT RESULTS')
    print('=' * 80)

    for _, _, config_name in configs:
        r = results[config_name]
        caught = r['caught']
        fps = r['false_positives']

        print(f'\n{"─" * 80}')
        print(f'Config: {config_name}')
        print(f'  Optional trains caught: {len(caught)}/24')
        print(f'  False positives: {len(fps)}')

        if caught:
            print(f'\n  CAUGHT:')
            for img, opt_id, opt_x, det_id, det_x in sorted(caught):
                print(f'    {img}: {opt_id}@{opt_x} → detected as {det_id}@{det_x}')

        if fps:
            print(f'\n  FALSE POSITIVES:')
            for img, det_id, det_x in sorted(fps):
                if det_x:
                    print(f'    {img}: {det_id}@{det_x}')
                else:
                    print(f'    {img}: {det_id}')

    # Summary comparison
    print(f'\n{"=" * 80}')
    print('SUMMARY')
    print(f'{"=" * 80}')
    print(f'{"Config":<50} {"Caught":>8} {"FP":>5} {"Net":>5}')
    print(f'{"─" * 70}')
    for _, _, config_name in configs:
        r = results[config_name]
        caught = len(r['caught'])
        fps = len(r['false_positives'])
        print(f'{config_name:<50} {caught:>8} {fps:>5} {caught - fps:>5}')


if __name__ == '__main__':
    main()
