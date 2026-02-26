"""
Diagnostic orchestration for train detection analysis.

Wraps the detection pipeline to expose intermediate results, per-contour
metrics, pixel inspection, and mask overlays for the diagnostic webapp.
"""

import base64
import dataclasses
import json
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np

from lib.detection import detect_system_status
from lib.train_detector import TrainDetector, TESSERACT_AVAILABLE

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASELINE_TRAINS_PATH = PROJECT_ROOT / 'tests' / 'baseline_trains.json'


def run_diagnostic(image_path: str) -> dict:
    """Run full diagnostic analysis on an image.

    Returns a dict with detection results, pipeline diagnostics,
    per-contour symbol data, and baseline comparison.
    """
    img = cv2.imread(image_path)
    if img is None:
        return {'error': f'Could not read image: {image_path}'}

    h, w = img.shape[:2]

    # Run full system detection (status, delays, etc.)
    detection = detect_system_status(image_path)

    # Run train detection with diagnostics enabled
    td = TrainDetector()
    diag = td.detect_trains(img, diagnostics=True) if TESSERACT_AVAILABLE else None

    # Build pipeline summary
    pipeline = {
        'detection_method': diag.detection_method if diag else 'unavailable',
        'ocr_train_count': len(diag.ocr_trains) if diag else 0,
        'symbols_were_run': diag.symbols_were_run if diag else False,
        'merged_train_count': len(diag.merged_trains) if diag else 0,
    }

    # Build per-contour data from symbol diagnostics
    symbol_contours = []
    summary_stats = {'total': 0, 'accepted': 0, 'rejected_by': {}}

    if diag and diag.symbol_diagnostics:
        sd = diag.symbol_diagnostics
        pipeline['symbol_contour_count'] = len(sd.contours)
        pipeline['symbol_accepted_count'] = len(sd.accepted_symbols)
        pipeline['track_band'] = list(sd.track_band)

        for c in sd.contours:
            symbol_contours.append(dataclasses.asdict(c))
            summary_stats['total'] += 1
            if c.accepted:
                summary_stats['accepted'] += 1
            else:
                # Group by filter type (first word of rejection reason)
                filter_name = c.rejection_reason.split('(')[0] if c.rejection_reason else 'unknown'
                summary_stats['rejected_by'][filter_name] = summary_stats['rejected_by'].get(filter_name, 0) + 1

    # Baseline comparison
    baseline_comparison = _compare_baseline(image_path, detection.get('trains', []))

    return {
        'detection': detection,
        'pipeline': pipeline,
        'symbol_contours': symbol_contours,
        'summary_stats': summary_stats,
        'baseline_comparison': baseline_comparison,
        'image_dimensions': {'width': w, 'height': h},
    }


def get_pixel_info(image_path: str, x: int, y: int) -> dict:
    """Return HSV values and filter membership at a pixel coordinate."""
    img = cv2.imread(image_path)
    if img is None:
        return {'error': 'Could not read image'}

    h_img, w_img = img.shape[:2]
    if x < 0 or x >= w_img or y < 0 or y >= h_img:
        return {'error': f'Coordinates ({x}, {y}) out of bounds ({w_img}x{h_img})'}

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h_val, s_val, v_val = int(hsv[y, x, 0]), int(hsv[y, x, 1]), int(hsv[y, x, 2])
    b_val, g_val, r_val = int(img[y, x, 0]), int(img[y, x, 1]), int(img[y, x, 2])

    track_y_min = int(h_img * 0.48)
    track_y_max = int(h_img * 0.62)

    # Check filter membership
    in_track_band = track_y_min <= y < track_y_max
    passes_saturation = s_val > 150

    # Cyan exclusion range: H 85-105, S 120-255, V 140-230
    in_cyan_range = (85 <= h_val <= 105 and 120 <= s_val <= 255 and 140 <= v_val <= 230)

    # Red exclusion range: H 0-8 or 172-180, S 150-255, V 100-255
    in_red_range = (
        ((0 <= h_val <= 8) or (172 <= h_val <= 180))
        and 150 <= s_val <= 255
        and 100 <= v_val <= 255
    )

    return {
        'x': x, 'y': y,
        'hsv': {'h': h_val, 's': s_val, 'v': v_val},
        'rgb': {'r': r_val, 'g': g_val, 'b': b_val},
        'hex': f'#{r_val:02x}{g_val:02x}{b_val:02x}',
        'in_track_band': in_track_band,
        'passes_saturation': passes_saturation,
        'in_cyan_range': in_cyan_range,
        'in_red_range': in_red_range,
        'would_be_candidate': in_track_band and passes_saturation and not in_cyan_range and not in_red_range,
    }


def encode_mask_overlay(image_path: str, mask_name: str) -> str:
    """Generate a semi-transparent PNG overlay for a detection mask.

    Supported mask_name values:
        saturation  - pixels where S > 150
        pre_close   - candidate mask before morphological closing
        post_close  - candidate mask after morphological closing
        cyan        - cyan exclusion zone
        red         - red exclusion zone

    Returns base64-encoded PNG string.
    """
    img = cv2.imread(image_path)
    if img is None:
        return ''

    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    track_y_min = int(h * 0.48)
    track_y_max = int(h * 0.62)

    if mask_name == 'saturation':
        saturation = hsv[:, :, 1]
        mask = (saturation > 150).astype(np.uint8) * 255
        color = (0, 255, 255)  # yellow
    elif mask_name in ('pre_close', 'post_close'):
        # Rebuild the candidate mask pipeline
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

        if mask_name == 'pre_close':
            mask = candidate_mask
            color = (0, 200, 255)  # orange
        else:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            mask = cv2.morphologyEx(candidate_mask, cv2.MORPH_CLOSE, kernel)
            color = (255, 100, 100)  # blue
    elif mask_name == 'cyan':
        mask = cv2.inRange(hsv, np.array([85, 120, 140]), np.array([105, 255, 230]))
        color = (255, 255, 0)  # cyan
    elif mask_name == 'red':
        red1 = cv2.inRange(hsv, np.array([0, 150, 100]), np.array([8, 255, 255]))
        red2 = cv2.inRange(hsv, np.array([172, 150, 100]), np.array([180, 255, 255]))
        mask = red1 | red2
        color = (0, 0, 255)  # red
    else:
        return ''

    # Create RGBA overlay: colored where mask is active, transparent elsewhere
    overlay = np.zeros((h, w, 4), dtype=np.uint8)
    overlay[mask > 0] = [color[2], color[1], color[0], 128]  # RGBA, 50% opacity

    _, buffer = cv2.imencode('.png', overlay)
    return base64.b64encode(buffer).decode('utf-8')


def _compare_baseline(image_path: str, detected_trains: list) -> dict | None:
    """Compare detected trains against baseline if available."""
    if not BASELINE_TRAINS_PATH.exists():
        return None

    image_name = Path(image_path).name
    data = json.loads(BASELINE_TRAINS_PATH.read_text())
    tolerance = data.get('tolerance', 30)

    # Check images_with_trains
    if image_name in data.get('images_with_trains', {}):
        expected = data['images_with_trains'][image_name]
        return _match_trains(expected, detected_trains, tolerance, image_name)

    # Check train_free_images
    if image_name in data.get('train_free_images', {}):
        max_fp = data['train_free_images'][image_name].get('max_false_positives', 0)
        return {
            'type': 'train_free',
            'expected_max_false_positives': max_fp,
            'detected_count': len(detected_trains),
            'passes': len(detected_trains) <= max_fp,
        }

    return None


def _match_trains(expected: list, detected: list, tolerance: int, image_name: str) -> dict:
    """Match expected baseline trains against detected trains."""
    expected_set = [(e[0], e[1]) for e in expected]
    detected_pairs = [(t.get('id', ''), t.get('x', 0)) for t in detected]

    matched = []
    missing = []
    extra = list(detected_pairs)

    for exp_id, exp_x in expected_set:
        found = False
        for i, (det_id, det_x) in enumerate(extra):
            if abs(det_x - exp_x) <= tolerance:
                matched.append({
                    'expected_id': exp_id, 'expected_x': exp_x,
                    'detected_id': det_id, 'detected_x': det_x,
                    'x_diff': abs(det_x - exp_x),
                    'id_match': det_id == exp_id,
                })
                extra.pop(i)
                found = True
                break
        if not found:
            missing.append({'id': exp_id, 'x': exp_x})

    return {
        'type': 'trains_expected',
        'expected_count': len(expected_set),
        'detected_count': len(detected),
        'matched': matched,
        'missing': missing,
        'extra': [{'id': eid, 'x': ex} for eid, ex in extra],
    }
