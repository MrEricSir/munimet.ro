#!/usr/bin/env python3
"""Targeted experiments for specific detection improvements.

Tests:
1. Relaxed symbol detection for D2073J (fill ratio, area)
2. Faint text second-pass OCR (threshold 150 instead of 120)
3. Edge-of-image detection improvements
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


# ============================================================
# Experiment 1: Faint text detection
# ============================================================

def detect_faint_text(image_path, td):
    """Try to detect trains with gray text (threshold 150 instead of 120)."""
    img = cv2.imread(str(image_path))
    if img is None:
        return []

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # First run normal detection to know what's already found
    normal_trains = td.detect_trains(img)

    extra_trains = []

    for track, (band_start, band_end) in [
        ('upper', UPPER_TRAIN_BAND),
        ('lower', LOWER_TRAIN_BAND)
    ]:
        y_min = int(h * band_start)
        y_max = int(h * band_end)
        band = gray[y_min:y_max, :]
        band_h = y_max - y_min

        # Faint text detection: threshold 150 instead of 120
        faint_mask = (band < 150).astype(np.uint8)
        dark_mask = (band < 120).astype(np.uint8)
        col_sums_faint = faint_mask.sum(axis=0)
        col_sums_dark = dark_mask.sum(axis=0)

        # Find columns that have faint text but NOT dark text
        # These are the "new" columns we'd discover
        faint_only = np.where((col_sums_faint > 5) & (col_sums_dark < 3))[0]

        if len(faint_only) == 0:
            continue

        # Group into column regions
        groups = td._group_columns(faint_only)

        for x1, x2 in groups:
            width = x2 - x1
            if width < 5 or width > 30:
                continue

            center_x = (x1 + x2) // 2

            # Skip if a normal detection already exists nearby
            if any(abs(center_x - t['x']) < 40 for t in normal_trains):
                continue

            # Try OCR on this faint column
            # Use the faint mask for text extent
            col_mask = faint_mask[:, max(0, x1-3):min(w, x2+3)]
            row_sums = col_mask.sum(axis=1)
            text_rows = np.where(row_sums > 0)[0]
            if len(text_rows) < 5:
                continue

            y1 = max(0, text_rows[0] - 2)
            y2 = min(band_h, text_rows[-1] + 2)

            roi = band[y1:y2, max(0, x1-3):min(w, x2+3)]
            if roi.size == 0 or roi.shape[0] < 20:
                continue

            # OCR with enhanced contrast
            scale = 4
            roi_large = cv2.resize(roi, None, fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)

            # Use aggressive threshold for faint text
            _, roi_bin = cv2.threshold(roi_large, 155, 255, cv2.THRESH_BINARY)

            try:
                import pytesseract
                text = pytesseract.image_to_string(roi_bin, config='--oem 3 --psm 6')
                train_ids = td._extract_train_ids(text)
                if train_ids:
                    for tid in train_ids:
                        extra_trains.append({
                            'id': tid, 'x': center_x, 'track': track,
                            'method': 'faint_threshold'
                        })
                else:
                    # Try Otsu on the faint text
                    _, roi_bin2 = cv2.threshold(roi_large, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    text = pytesseract.image_to_string(roi_bin2, config='--oem 3 --psm 6')
                    train_ids = td._extract_train_ids(text)
                    if train_ids:
                        for tid in train_ids:
                            extra_trains.append({
                                'id': tid, 'x': center_x, 'track': track,
                                'method': 'faint_otsu'
                            })
                    else:
                        # Try CLAHE
                        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
                        enhanced = clahe.apply(roi_large)
                        _, roi_bin3 = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                        text = pytesseract.image_to_string(roi_bin3, config='--oem 3 --psm 6')
                        train_ids = td._extract_train_ids(text)
                        if train_ids:
                            for tid in train_ids:
                                extra_trains.append({
                                    'id': tid, 'x': center_x, 'track': track,
                                    'method': 'faint_clahe'
                                })
                        else:
                            extra_trains.append({
                                'id': f'FAINT_UNREAD@{center_x}',
                                'x': center_x, 'track': track,
                                'method': 'faint_unread',
                                'ocr_text': text.strip()[:50]
                            })
            except Exception:
                pass

    return extra_trains


# ============================================================
# Experiment 2: Relaxed symbol detection for D2073J
# ============================================================

def detect_symbols_relaxed(image_path, td, min_area=80, min_fill=0.55, min_rect=0.4):
    """Symbol detection with relaxed thresholds for D2073J-type contours."""
    img = cv2.imread(str(image_path))
    if img is None:
        return []

    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

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

    pre_close = candidate_mask.copy()
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
        pre_pixels = cv2.countNonZero(pre_close[y:y+ch, x:x+cw])
        fill_ratio = pre_pixels / max(bounding_area, 1)

        # Relaxed filters
        if area < min_area:
            continue
        if bounding_area < min_area * 1.2:
            continue
        if cw < 15 or cw > 45 or ch < 7 or ch > 20:
            continue
        if rectangularity < min_rect:
            continue
        if aspect < 1.5 or aspect > 3.5:
            continue
        if fill_ratio < min_fill:
            continue

        cx, cy = x + cw // 2, y + ch // 2
        track = 'upper' if cy < upper_track_y else 'lower'
        symbols.append({
            'x': cx, 'y': cy, 'track': track,
            'metrics': f'area={area:.0f} {cw}x{ch} aspect={aspect:.1f} fill={fill_ratio:.2f} rect={rectangularity:.2f}'
        })

    return symbols


def main():
    data = load_baselines()
    tolerance = data.get('tolerance', 30)
    images_dir = PROJECT_ROOT / 'tests' / 'images'
    td = TrainDetector()

    print('=' * 80)
    print('EXPERIMENT 1: Faint Text Detection (threshold 150)')
    print('=' * 80)

    for image_name in sorted(data['images_with_trains'].keys()):
        image_path = images_dir / image_name
        if not image_path.exists():
            continue

        faint = detect_faint_text(str(image_path), td)
        if faint:
            print(f'\n{image_name}:')
            for t in faint:
                ocr_text = t.get('ocr_text', '')
                extra = f' (OCR: "{ocr_text}")' if ocr_text else ''
                print(f'  {t["id"]} @ x={t["x"]} [{t["track"]}] via {t["method"]}{extra}')

    # Check train-free images too
    for image_name in sorted(data.get('train_free_images', {}).keys()):
        image_path = images_dir / image_name
        if not image_path.exists():
            continue
        faint = detect_faint_text(str(image_path), td)
        if faint:
            print(f'\n{image_name} (TRAIN-FREE - FALSE POSITIVE):')
            for t in faint:
                print(f'  {t["id"]} @ x={t["x"]} [{t["track"]}]')

    print(f'\n\n{"=" * 80}')
    print('EXPERIMENT 2: Relaxed Symbol Detection')
    print('=' * 80)

    # Test several fill ratio thresholds
    for min_fill in [0.70, 0.60, 0.55, 0.50, 0.45]:
        for min_area in [200, 100, 80]:
            caught_optional = 0
            false_positives = 0
            caught_details = []
            fp_details = []

            for image_name, baseline_entry in data['images_with_trains'].items():
                image_path = images_dir / image_name
                if not image_path.exists():
                    continue

                img = cv2.imread(str(image_path))
                normal = td.detect_trains(img)
                symbols = detect_symbols_relaxed(str(image_path), td, min_area=min_area, min_fill=min_fill)

                # Check which symbols are new (not near an existing detection)
                new_symbols = []
                for s in symbols:
                    if not any(abs(s['x'] - t['x']) < 50 for t in normal):
                        new_symbols.append(s)

                # Check against baselines
                all_trains = [(e[0], e[1]) for e in baseline_entry]
                optional_trains = [(e[0], e[1]) for e in baseline_entry
                                   if len(e) >= 3 and isinstance(e[2], dict) and e[2].get('optional')]

                for s in new_symbols:
                    matched_optional = any(abs(s['x'] - ox) <= tolerance for _, ox in optional_trains)
                    matched_any = any(abs(s['x'] - tx) <= tolerance for _, tx in all_trains)

                    if matched_optional:
                        caught_optional += 1
                        caught_details.append(f'  {image_name}: UNKNOWN@{s["x"]} ({s["metrics"]})')
                    elif not matched_any:
                        false_positives += 1
                        fp_details.append(f'  {image_name}: UNKNOWN@{s["x"]} ({s["metrics"]})')

            # Also check train-free images
            for image_name, config in data.get('train_free_images', {}).items():
                image_path = images_dir / image_name
                if not image_path.exists():
                    continue
                symbols = detect_symbols_relaxed(str(image_path), td, min_area=min_area, min_fill=min_fill)
                max_fp = config.get('max_false_positives', 0)
                if len(symbols) > max_fp:
                    false_positives += 1
                    fp_details.append(f'  {image_name}: {len(symbols)} symbols (max allowed: {max_fp})')

            if caught_optional > 0 or false_positives > 0:
                print(f'\n  fill>={min_fill}, area>={min_area}: caught={caught_optional} optional, fp={false_positives}')
                if caught_details:
                    for d in caught_details:
                        print(f'    CAUGHT: {d}')
                if fp_details:
                    for d in fp_details[:5]:
                        print(f'    FP: {d}')
                    if len(fp_details) > 5:
                        print(f'    ... and {len(fp_details) - 5} more FPs')

    # ============================================================
    # Experiment 3: What's at the recurring false positive positions?
    # ============================================================
    print(f'\n\n{"=" * 80}')
    print('EXPERIMENT 3: Investigating Recurring False Positive Positions')
    print('=' * 80)

    fp_positions = [882, 974, 1057, 1570, 1729, 1814]
    # Check what's there in a representative image
    test_image = str(images_dir / 'IMG_9791.jpg')
    img = cv2.imread(test_image)
    if img is not None:
        h, w = img.shape[:2]
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        for pos in fp_positions:
            track_y_min = int(h * 0.48)
            track_y_max = int(h * 0.62)
            strip = hsv[track_y_min:track_y_max, max(0, pos-5):min(w, pos+5)]
            if strip.size > 0:
                h_vals = strip[:, :, 0]
                s_vals = strip[:, :, 1]
                v_vals = strip[:, :, 2]
                high_sat = (s_vals > 150).sum()
                print(f'\n  x={pos}: H={h_vals.mean():.0f} S_mean={s_vals.mean():.0f} '
                      f'S_max={s_vals.max()} V_mean={v_vals.mean():.0f} '
                      f'high_sat_pixels={high_sat}')
                # What color is this?
                if h_vals.mean() < 15 or h_vals.mean() > 165:
                    color = 'RED'
                elif 15 <= h_vals.mean() <= 35:
                    color = 'YELLOW/ORANGE'
                elif 35 <= h_vals.mean() <= 85:
                    color = 'GREEN'
                elif 85 <= h_vals.mean() <= 105:
                    color = 'CYAN'
                elif 105 <= h_vals.mean() <= 130:
                    color = 'BLUE'
                else:
                    color = 'PURPLE'
                print(f'    Color: {color}')


if __name__ == '__main__':
    main()
