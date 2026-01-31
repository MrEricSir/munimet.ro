"""
Train identifier detection using OCR.

Detects train IDs (e.g., "L1234M") from Muni status images using Tesseract OCR.
Optimized for speed with single preprocessing pass per text column.
"""

import cv2
import numpy as np
import re

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

# Train ID pattern
TRAIN_ID_PATTERN = re.compile(r'[A-Z]?\d{4}[A-Z*X]{1,2}')

# Y-bands for text labels
UPPER_TRAIN_BAND = (0.25, 0.48)
LOWER_TRAIN_BAND = (0.56, 0.80)

# Track Y positions
UPPER_TRACK_Y = (0.48, 0.54)
LOWER_TRACK_Y = (0.54, 0.62)


class TrainDetector:
    """Optimized train detector."""

    def __init__(self):
        if not TESSERACT_AVAILABLE:
            print("Warning: pytesseract not installed.")

        self.station_labels = {
            'USL', 'USR', 'YBL', 'YBR', 'CTL', 'CTR', 'TTL', 'TTR',
            'FPL', 'FPR', 'MNL', 'MNR', 'MOR', 'MOL', 'POR', 'POL',
            'CCL', 'CCR', 'VNL', 'VNR', 'CHR', 'CHL', 'CAR', 'CAL',
            'FHR', 'FHL', 'WER', 'WEL', 'EMR', 'EML', 'GDL', 'GDR',
            'WPL', 'WPR', 'BPL', 'BPR', 'FSL', 'FSR', 'GLL', 'GLR'
        }

    def detect_trains(self, image):
        """Detect trains in the image using OCR."""
        if not TESSERACT_AVAILABLE:
            return []

        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Step 1: OCR-based detection (primary method)
        ocr_trains = self._detect_by_ocr(gray, h, w, hsv)

        # Step 2: Symbol detection only if OCR found very few trains
        # This reduces false positives when OCR is working well
        if len(ocr_trains) < 3:
            symbols = self._detect_symbols(hsv, h, w, gray)
            trains = self._merge(ocr_trains, symbols)
        else:
            trains = ocr_trains

        return trains

    def _detect_by_ocr(self, gray, h, w, hsv):
        """Detect trains via OCR - single pass per column."""
        trains = []

        for track, (band_start, band_end) in [
            ('upper', UPPER_TRAIN_BAND),
            ('lower', LOWER_TRAIN_BAND)
        ]:
            y_min = int(h * band_start)
            y_max = int(h * band_end)
            band = gray[y_min:y_max, :]
            band_hsv = hsv[y_min:y_max, :]
            band_h = y_max - y_min

            # Find dark text columns
            dark_mask = (band < 100).astype(np.uint8)
            col_sums = dark_mask.sum(axis=0)
            text_cols = np.where(col_sums > 3)[0]

            if len(text_cols) > 0:
                # Group columns
                groups = self._group_columns(text_cols)

                for x1, x2 in groups:
                    width = x2 - x1
                    if width < 6:
                        continue

                    # Sub-column splitting for very wide groups only
                    # Train IDs are typically 15-25 pixels wide
                    if width > 35:
                        sub_cols = [(sub_x, min(sub_x + 20, x2))
                                    for sub_x in range(x1, x2, 18)]
                    else:
                        sub_cols = [(x1, x2)]

                    for sub_x1, sub_x2 in sub_cols:
                        train_id = self._ocr_column(band, dark_mask, sub_x1, sub_x2, band_h, track)
                        if train_id:
                            center_x = (sub_x1 + sub_x2) // 2
                            trains.append({
                                'id': train_id,
                                'x': center_x,
                                'y': y_min + band_h // 2,
                                'track': track,
                                'confidence': 'high'
                            })

            # Also detect colored text labels (yellow, green)
            colored_trains = self._detect_colored_labels(band, band_hsv, band_h, y_min, track)
            trains.extend(colored_trains)

        return self._deduplicate(trains)

    def _group_columns(self, columns):
        """Group adjacent columns."""
        if len(columns) == 0:
            return []
        groups = []
        start = columns[0]
        prev = start
        for x in columns[1:]:
            if x - prev > 10:
                groups.append((start, prev))
                start = x
            prev = x
        groups.append((start, prev))
        return groups

    def _ocr_column(self, band, dark_mask, x1, x2, band_h, track):
        """OCR a single column - ONE preprocessing pass."""
        pad = 3
        roi_x1 = max(0, x1 - pad)
        roi_x2 = min(band.shape[1], x2 + pad)
        col_mask = dark_mask[:, roi_x1:roi_x2]

        # Find vertical extent
        row_sums = col_mask.sum(axis=1)
        text_rows = np.where(row_sums > 0)[0]
        if len(text_rows) < 5:
            return None

        y1 = max(0, text_rows[0] - 2)
        y2 = min(band_h, text_rows[-1] + 2)

        # Skip station label regions
        if track == 'lower':
            station_cutoff = int(band_h * 0.15)
            if y1 < station_cutoff:
                y1 = station_cutoff
        elif track == 'upper':
            # Allow more of the band - station labels are filtered by name matching
            station_cutoff = int(band_h * 0.95)
            if y2 > station_cutoff:
                y2 = station_cutoff

        roi = band[y1:y2, roi_x1:roi_x2]
        if roi.size == 0 or roi.shape[0] < 20:
            return None

        # Single OCR pass with Otsu
        scale = 4
        roi_large = cv2.resize(roi, None, fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)
        _, roi_bin = cv2.threshold(roi_large, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        try:
            text = pytesseract.image_to_string(roi_bin, config='--psm 6')
            return self._extract_train_id(text)
        except Exception:
            return None

    def _detect_colored_labels(self, band_gray, band_hsv, band_h, y_offset, track):
        """Detect colored text labels (yellow, green train IDs)."""
        trains = []

        # Yellow text (H=15-40, high saturation and value)
        yellow = cv2.inRange(band_hsv, np.array([15, 80, 120]), np.array([40, 255, 255]))
        # Green text (H=35-85)
        green = cv2.inRange(band_hsv, np.array([35, 80, 120]), np.array([85, 255, 255]))

        colored_mask = yellow | green

        # Find columns with colored text
        col_sums = colored_mask.sum(axis=0)
        colored_cols = np.where(col_sums > 100)[0]

        if len(colored_cols) == 0:
            return trains

        # Group into regions
        groups = self._group_columns(colored_cols)

        for x1, x2 in groups:
            width = x2 - x1
            if width < 5 or width > 50:
                continue

            # Find vertical extent
            col_region = colored_mask[:, x1:x2+1]
            row_sums = col_region.sum(axis=1)
            text_rows = np.where(row_sums > 0)[0]
            if len(text_rows) < 10:
                continue

            y1 = max(0, text_rows[0] - 2)
            y2 = min(band_h, text_rows[-1] + 2)

            # Skip station label areas
            if track == 'lower':
                station_cutoff = int(band_h * 0.15)
                if y1 < station_cutoff:
                    y1 = station_cutoff
            elif track == 'upper':
                station_cutoff = int(band_h * 0.80)
                if y2 > station_cutoff:
                    y2 = station_cutoff

            if y2 - y1 < 20:
                continue

            # Extract and OCR the colored mask region
            pad = 3
            roi_x1 = max(0, x1 - pad)
            roi_x2 = min(band_gray.shape[1], x2 + pad)
            color_roi = colored_mask[y1:y2, roi_x1:roi_x2]

            if color_roi.size == 0:
                continue

            train_id = self._ocr_color_mask(color_roi)
            if train_id:
                center_x = (x1 + x2) // 2
                trains.append({
                    'id': train_id,
                    'x': center_x,
                    'y': y_offset + (y1 + y2) // 2,
                    'track': track,
                    'confidence': 'high'
                })

        return trains

    def _ocr_color_mask(self, mask):
        """Run OCR on a color mask."""
        # Scale up
        scale = 4
        mask_large = cv2.resize(mask, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

        # Clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        mask_clean = cv2.morphologyEx(mask_large, cv2.MORPH_CLOSE, kernel)

        # Single OCR attempt (vertical text as-is, tesseract handles it)
        try:
            text = pytesseract.image_to_string(mask_clean, config='--psm 6')
            train_id = self._extract_train_id(text)
            if train_id:
                return train_id

            # Try inverted
            text = pytesseract.image_to_string(255 - mask_clean, config='--psm 6')
            return self._extract_train_id(text)
        except Exception:
            return None

    def _detect_symbols(self, hsv, h, w, gray):
        """Detect train symbols on tracks."""
        track_y_min = int(h * 0.48)
        track_y_max = int(h * 0.62)
        upper_track_y = int(h * 0.52)

        # Track band mask
        track_band = np.zeros((h, w), dtype=np.uint8)
        track_band[track_y_min:track_y_max, :] = 255

        # Exclude cyan and red track pixels (widen cyan range to catch lighter cyan)
        cyan_mask = cv2.inRange(hsv, np.array([85, 120, 140]), np.array([105, 255, 230]))
        red1 = cv2.inRange(hsv, np.array([0, 150, 100]), np.array([8, 255, 255]))
        red2 = cv2.inRange(hsv, np.array([172, 150, 100]), np.array([180, 255, 255]))
        red_mask = red1 | red2

        # Find colored objects (high threshold to reduce false positives from track signals)
        saturation = hsv[:, :, 1]
        colored_mask = (saturation > 150).astype(np.uint8) * 255
        candidate_mask = cv2.bitwise_and(colored_mask, track_band)
        candidate_mask = cv2.bitwise_and(candidate_mask, cv2.bitwise_not(cyan_mask))
        candidate_mask = cv2.bitwise_and(candidate_mask, cv2.bitwise_not(red_mask))

        # Morphology
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        candidate_mask = cv2.morphologyEx(candidate_mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(candidate_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        symbols = []
        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            area = cv2.contourArea(cnt)

            # Strict filters to reduce false positives from track signals
            if area < 200 or cw * ch < 250:
                continue
            if cw < 18 or cw > 45 or ch < 10 or ch > 20:
                continue

            rectangularity = area / max(cw * ch, 1)
            if rectangularity < 0.5:
                continue

            aspect = cw / max(ch, 1)
            if aspect < 1.5 or aspect > 3.5:
                continue

            cx, cy = x + cw // 2, y + ch // 2
            track = 'upper' if cy < upper_track_y else 'lower'

            symbols.append({
                'x': cx,
                'y': cy,
                'track': track
            })

        # Add train-colored rectangles
        symbols.extend(self._detect_train_colors(hsv, h, w, track_y_min, track_y_max, upper_track_y))

        # Deduplicate
        symbols = sorted(symbols, key=lambda s: s['x'])
        unique = []
        for s in symbols:
            if not any(abs(s['x'] - u['x']) < 30 and s['track'] == u['track'] for u in unique):
                unique.append(s)

        return unique

    def _detect_train_colors(self, hsv, h, w, track_y_min, track_y_max, upper_track_y):
        """Detect train-colored rectangles."""
        symbols = []
        # Only detect yellow/gold train colors - exclude orange (used for track signals)
        train_colors = [
            (np.array([20, 100, 150]), np.array([35, 255, 255]), 45, 25),  # Yellow/gold trains
            (np.array([40, 80, 120]), np.array([65, 255, 255]), 45, 25),   # Green trains
        ]

        track_mask = np.zeros((h, w), dtype=np.uint8)
        track_mask[track_y_min:track_y_max, :] = 255

        for lower, upper, max_w, max_h in train_colors:
            color_mask = cv2.inRange(hsv, lower, upper)
            color_mask = cv2.bitwise_and(color_mask, track_mask)
            contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                x, y, cw, ch = cv2.boundingRect(cnt)
                area = cv2.contourArea(cnt)
                # Tighter filters: trains are wider than tall rectangles
                if cw < 15 or cw > max_w or ch < 8 or ch > max_h or area < 150:
                    continue
                aspect = cw / max(ch, 1)
                if aspect < 1.5 or aspect > 4.0:
                    continue
                cx, cy = x + cw // 2, y + ch // 2
                track = 'upper' if cy < upper_track_y else 'lower'
                symbols.append({'x': cx, 'y': cy, 'track': track})

        return symbols

    def _merge(self, ocr_trains, symbols):
        """Merge OCR and symbol detections."""
        trains = list(ocr_trains)

        for sym in symbols:
            has_match = any(
                abs(sym['x'] - t['x']) < 50 and sym['track'] == t['track']
                for t in trains
            )
            if not has_match:
                trains.append({
                    'id': f"UNKNOWN@{sym['x']}",
                    'x': sym['x'],
                    'y': sym['y'],
                    'track': sym['track'],
                    'confidence': 'low'
                })

        return trains

    def _extract_train_id(self, text):
        """Extract train ID from OCR text."""
        lines = [c.strip() for c in text.split('\n') if c.strip()]
        combined = ''.join(lines).upper()

        # OCR corrections
        combined = combined.replace('{', '7').replace('}', '7')
        combined = combined.replace('[', '7').replace(']', '7')
        combined = combined.replace('+', 'TT')
        combined = combined.replace('|', '1').replace('!', '1')
        combined = combined.replace('O', '0').replace('Q', '0')
        combined = combined.replace('Z', '7')
        combined = combined.replace('(', '').replace(')', '').replace(' ', '')

        # Fix duplicate leading letters (OCR sometimes doubles them in vertical text)
        if len(combined) >= 2 and combined[0].isalpha() and combined[0] == combined[1]:
            combined = combined[1:]

        if combined and combined[0].isdigit():
            combined = combined.replace('L', '1').replace('S', '5').replace('B', '8')

        for label in self.station_labels:
            combined = combined.replace(label, '')

        matches = TRAIN_ID_PATTERN.findall(combined)
        if matches:
            return matches[0]

        # Position-based 7/T fix for vertical text where 7 is often read as T
        if len(combined) >= 5:
            fixed = list(combined)
            # Find where digits start
            digit_start = 0
            for i, c in enumerate(combined):
                if c.isdigit() or c == 'T':  # T might be misread 7
                    digit_start = i
                    break

            for i in range(len(fixed)):
                if digit_start <= i < digit_start + 4:
                    if fixed[i] == 'T':
                        fixed[i] = '7'
                    elif fixed[i] == 'I':
                        fixed[i] = '1'
                elif i >= digit_start + 4:
                    if fixed[i] == '7':
                        fixed[i] = 'T'
                    elif fixed[i] == '1':
                        fixed[i] = 'L'
                    elif fixed[i] == '0':
                        fixed[i] = 'O'
            matches = TRAIN_ID_PATTERN.findall(''.join(fixed))
            if matches:
                return matches[0]

        return None

    def _deduplicate(self, trains):
        """Remove duplicates."""
        if not trains:
            return trains
        trains = sorted(trains, key=lambda t: (t['track'], t['x']))
        unique = []
        for train in trains:
            is_dup = False
            for existing in unique:
                if (existing['track'] == train['track'] and
                        abs(existing['x'] - train['x']) < 30):
                    if len(train['id']) > len(existing['id']):
                        unique.remove(existing)
                        unique.append(train)
                    is_dup = True
                    break
            if not is_dup:
                unique.append(train)
        return unique
