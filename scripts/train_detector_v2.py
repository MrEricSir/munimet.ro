"""
Train identifier detection using hybrid symbol + OCR approach.

This detector uses two methods:
1. Symbol detection: Find train markers on track lines
2. OCR detection: Read train ID text labels

The results are merged to maximize train detection while minimizing false positives.
"""

import cv2
import numpy as np
import re

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

# Train ID pattern: [Letter]?[4 digits][1-2 letters or *X]
TRAIN_ID_PATTERN = re.compile(r'[A-Z]?\d{4}[A-Z*X]{1,2}')

# Y-bands for text labels (percentage of image height)
UPPER_TRAIN_BAND = (0.25, 0.48)  # Above upper track
LOWER_TRAIN_BAND = (0.56, 0.80)  # Below lower track

# Track Y positions (percentage of image height)
UPPER_TRACK_Y = (0.49, 0.54)  # Upper track line
LOWER_TRACK_Y = (0.55, 0.60)  # Lower track line


class TrainDetectorV2:
    """Hybrid train detector using symbols + OCR."""

    def __init__(self):
        if not TESSERACT_AVAILABLE:
            print("Warning: pytesseract not installed. OCR detection disabled.")

    def detect_trains(self, image):
        """
        Detect all train identifiers in the image.

        Returns:
            List of dicts: [{'id': 'W2010L', 'x': 500, 'y': 300, 'track': 'upper', 'confidence': 'high'}, ...]
        """
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Method 1: Detect train symbols on tracks
        symbols = self._detect_symbols(image, hsv, h, w)

        # Method 2: OCR-based detection (if available)
        ocr_trains = []
        if TESSERACT_AVAILABLE:
            ocr_trains = self._detect_by_ocr(gray, hsv, h, w)

        # Merge results
        trains = self._merge_detections(symbols, ocr_trains, gray, h, w)

        return trains

    def _detect_symbols(self, image, hsv, h, w):
        """Detect train symbols on track lines using track mask and shape filtering."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Track Y band (more reliable than dilating cyan pixels)
        track_y_min = int(h * 0.48)
        track_y_max = int(h * 0.62)

        # Create track band mask
        track_band = np.zeros((h, w), dtype=np.uint8)
        track_band[track_y_min:track_y_max, :] = 255

        # Find cyan and red track pixels (to exclude)
        cyan_mask = cv2.inRange(hsv, np.array([85, 100, 150]), np.array([105, 255, 255]))
        red1 = cv2.inRange(hsv, np.array([0, 100, 100]), np.array([10, 255, 255]))
        red2 = cv2.inRange(hsv, np.array([170, 100, 100]), np.array([180, 255, 255]))
        red_mask = red1 | red2

        # Find colored (non-gray) objects - lower threshold to catch faded train symbols
        saturation = hsv[:, :, 1]
        colored_mask = (saturation > 30).astype(np.uint8) * 255

        # Candidates: colored objects in track band, excluding cyan and red track pixels
        candidate_mask = cv2.bitwise_and(colored_mask, track_band)
        candidate_mask = cv2.bitwise_and(candidate_mask, cv2.bitwise_not(cyan_mask))
        candidate_mask = cv2.bitwise_and(candidate_mask, cv2.bitwise_not(red_mask))

        # Find contours
        contours, _ = cv2.findContours(candidate_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        symbols = []
        upper_track_y = int(h * 0.52)  # Dividing line between upper and lower tracks

        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            area = cv2.contourArea(cnt)
            rect_area = cw * ch

            # Skip tiny noise
            if area < 40 or rect_area < 50:
                continue

            # Shape filtering: trains are rectangles
            # Rectangularity = how much of bounding box is filled
            rectangularity = area / max(rect_area, 1)

            # Aspect ratio (width/height)
            aspect = cw / max(ch, 1)

            # Train symbols: reasonably rectangular (>0.45) with normal aspect ratio
            # This filters out arrows (low rectangularity) and dots (very small)
            if rectangularity < 0.45:
                continue
            if aspect < 0.4 or aspect > 4.0:
                continue
            if cw < 8 or cw > 45 or ch < 5 or ch > 30:
                continue

            cx, cy = x + cw // 2, y + ch // 2

            # Color filtering: train symbols have lower saturation than track elements
            # Real trains: S=42-87, False positives (track elements): S=94+
            symbol_roi = hsv[y:y+ch, x:x+cw]
            avg_saturation = symbol_roi[:, :, 1].mean()
            if avg_saturation > 90:  # Filter high-saturation track elements
                continue

            # Determine which track based on Y position
            if cy < upper_track_y:
                track = 'upper'
            else:
                track = 'lower'

            # Check text in label region above/below this symbol
            nearby_text = self._get_nearby_text(gray, cx, h, w, track)

            symbols.append({
                'x': cx,
                'y': cy,
                'track': track,
                'type': 'symbol',
                'nearby_text': nearby_text
            })

        # Remove duplicate symbols (keep one per ~30 pixel cluster)
        symbols = sorted(symbols, key=lambda s: s['x'])
        unique = []
        for s in symbols:
            is_dup = False
            for u in unique:
                if abs(s['x'] - u['x']) < 30 and s['track'] == u['track']:
                    is_dup = True
                    break
            if not is_dup:
                unique.append(s)

        return unique

    def _get_nearby_text(self, gray, x, h, w, track):
        """Get raw OCR text from label region near a symbol position."""
        # Define search region for text label
        if track == 'upper':
            y_min = int(h * 0.25)
            y_max = int(h * 0.48)
        else:
            y_min = int(h * 0.56)
            y_max = int(h * 0.80)

        # Extract narrow column around x position
        x1 = max(0, x - 15)
        x2 = min(w, x + 15)
        roi = gray[y_min:y_max, x1:x2]

        # Check if there's any dark text
        dark_pixels = (roi < 100).sum()
        if dark_pixels < 50:
            return ''

        # Quick OCR (without extensive preprocessing)
        if TESSERACT_AVAILABLE:
            try:
                scale = 3
                roi_large = cv2.resize(roi, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
                _, roi_bin = cv2.threshold(roi_large, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                text = pytesseract.image_to_string(roi_bin, config='--psm 6')
                return ''.join(text.split()).upper()
            except Exception:
                pass
        return ''

    def _robust_ocr_for_symbol(self, gray, x, h, w, track):
        """Try multiple OCR methods to extract train ID for a symbol position."""
        if not TESSERACT_AVAILABLE:
            return None

        # Define search region - use wider column
        if track == 'upper':
            y_min = int(h * 0.25)
            y_max = int(h * 0.48)
        else:
            y_min = int(h * 0.56)
            y_max = int(h * 0.80)

        x1 = max(0, x - 25)
        x2 = min(w, x + 25)
        roi = gray[y_min:y_max, x1:x2]

        if roi.size == 0:
            return None

        scale = 4
        roi_large = cv2.resize(roi, None, fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)

        # Try multiple preprocessing methods
        methods = []

        # Method 1: Otsu threshold
        _, roi_otsu = cv2.threshold(roi_large, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        methods.append(roi_otsu)

        # Method 2: Inverted Otsu
        methods.append(cv2.bitwise_not(roi_otsu))

        # Method 3: Fixed threshold (for faint text)
        _, roi_fixed = cv2.threshold(roi_large, 100, 255, cv2.THRESH_BINARY)
        methods.append(roi_fixed)

        for preprocessed in methods:
            try:
                text = pytesseract.image_to_string(preprocessed, config='--psm 6')
                train_id = self._extract_train_id(text)
                if train_id:
                    return train_id
            except Exception:
                pass

        return None

    def _detect_by_ocr(self, gray, hsv, h, w):
        """Detect trains using OCR on text labels."""
        trains = []

        # Detect in upper and lower bands
        for track, (band_start, band_end) in [
            ('upper', UPPER_TRAIN_BAND),
            ('lower', LOWER_TRAIN_BAND)
        ]:
            y_min = int(h * band_start)
            y_max = int(h * band_end)
            band = gray[y_min:y_max, :]
            band_h = y_max - y_min

            # Find dark text columns
            dark_mask = (band < 100).astype(np.uint8)
            col_sums = dark_mask.sum(axis=0)
            text_columns = np.where(col_sums > 3)[0]

            if len(text_columns) == 0:
                continue

            # Group columns
            column_groups = []
            start = text_columns[0]
            prev = start
            for x in text_columns[1:]:
                if x - prev > 10:
                    column_groups.append((start, prev))
                    start = x
                prev = x
            column_groups.append((start, prev))

            # Process each column group
            for x1, x2 in column_groups:
                width = x2 - x1
                if width < 6:
                    continue

                # Sub-column splitting for wide groups
                if width > 20:
                    sub_cols = [(sub_x, min(sub_x + 15, x2))
                                for sub_x in range(x1, x2, 12)]
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
                            'type': 'ocr'
                        })

        # Deduplicate OCR trains
        trains = self._deduplicate(trains)

        return trains

    def _ocr_column(self, band, dark_mask, x1, x2, band_h, track):
        """Run OCR on a column and extract train ID."""
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

        # Skip station label areas
        if track == 'lower':
            station_cutoff = int(band_h * 0.15)
            if y1 < station_cutoff:
                y1 = station_cutoff
        elif track == 'upper':
            station_cutoff = int(band_h * 0.80)
            if y2 > station_cutoff:
                y2 = station_cutoff

        roi = band[y1:y2, roi_x1:roi_x2]
        if roi.size == 0 or roi.shape[0] < 20:
            return None

        # OCR
        scale = 4
        roi_large = cv2.resize(roi, None, fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)
        _, roi_bin = cv2.threshold(roi_large, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        try:
            text = pytesseract.image_to_string(roi_bin, config='--psm 6')
            return self._extract_train_id(text)
        except Exception:
            return None

    def _extract_train_id(self, text):
        """Clean OCR text and extract train ID."""
        lines = [c.strip() for c in text.split('\n') if c.strip()]
        combined = ''.join(lines).upper()

        # OCR corrections
        combined = combined.replace('{', '7').replace('}', '7')
        combined = combined.replace('[', '7').replace(']', '7')
        combined = combined.replace('+', 'TT')
        combined = combined.replace('|', '1').replace('!', '1')
        combined = combined.replace('O', '0').replace('Q', '0')
        combined = combined.replace('Z', '7')
        combined = combined.replace('(', '').replace(')', '')
        combined = combined.replace(' ', '')

        # Position-based corrections for leading digit
        if combined and combined[0].isdigit():
            combined = combined.replace('L', '1').replace('S', '5').replace('B', '8')

        # Remove station labels
        for label in ['USL', 'USR', 'YBL', 'YBR', 'CTL', 'CTR', 'TTL', 'TTR',
                      'FPL', 'FPR', 'MNL', 'MNR', 'MOR', 'MOL', 'POR', 'POL',
                      'CCL', 'CCR', 'VNL', 'VNR', 'CHR', 'CHL', 'CAR', 'CAL',
                      'FHR', 'FHL', 'WER', 'WEL', 'EMR', 'EML', 'GDL', 'GDR']:
            combined = combined.replace(label, '')

        matches = TRAIN_ID_PATTERN.findall(combined)
        if matches:
            return matches[0]

        # Try position-based 7/T fix
        if len(combined) >= 5:
            fixed = list(combined)
            digit_start = 1 if combined[0].isalpha() else 0
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

    def _merge_detections(self, symbols, ocr_trains, gray, h, w):
        """Merge symbol and OCR detections."""
        trains = []

        # Add all OCR trains with high confidence
        for t in ocr_trains:
            trains.append({
                'id': t['id'],
                'x': t['x'],
                'y': t['y'],
                'track': t['track'],
                'confidence': 'high'
            })

        # Station label patterns to filter out
        station_labels = {'USL', 'USR', 'YBL', 'YBR', 'CTL', 'CTR', 'TTL', 'TTR',
                         'FPL', 'FPR', 'MNL', 'MNR', 'MOR', 'MOL', 'POR', 'POL',
                         'CCL', 'CCR', 'VNL', 'VNR', 'CHR', 'CHL', 'CAR', 'CAL',
                         'FHR', 'FHL', 'WER', 'WEL', 'EMR', 'EML', 'GDL', 'GDR',
                         'WPL', 'WPR', 'BPL', 'BPR', 'FSL', 'FSR', 'GLL', 'GLR'}

        # Add symbols that don't match any OCR train
        for sym in symbols:
            has_match = False
            for t in ocr_trains:
                if abs(sym['x'] - t['x']) < 50 and sym['track'] == t['track']:
                    has_match = True
                    break

            if not has_match:
                # Check if this symbol position corresponds to a station label
                nearby_text = sym.get('nearby_text', '')
                is_station = any(label in nearby_text.upper() for label in station_labels)

                # Try to extract a train ID from nearby text first
                extracted_id = self._extract_train_id(nearby_text)

                # If no ID found and not a station, try robust OCR
                if not extracted_id and not is_station:
                    extracted_id = self._robust_ocr_for_symbol(gray, sym['x'], h, w, sym['track'])

                if is_station and not extracted_id:
                    # Station label without train ID - skip this symbol
                    continue

                if extracted_id:
                    # Found a train ID - use it
                    trains.append({
                        'id': extracted_id,
                        'x': sym['x'],
                        'y': sym['y'],
                        'track': sym['track'],
                        'confidence': 'medium'  # Symbol + recovered OCR
                    })
                else:
                    # Symbol without readable ID
                    trains.append({
                        'id': f"UNKNOWN@{sym['x']}",
                        'x': sym['x'],
                        'y': sym['y'],
                        'track': sym['track'],
                        'confidence': 'low'
                    })

        return trains

    def _deduplicate(self, trains):
        """Remove duplicate train detections."""
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
