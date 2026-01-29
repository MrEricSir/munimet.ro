"""
Train identifier detection using OCR.

Detects vertical train ID strings (e.g., "W2073LL") in track display images.

Train ID format: [TrainNumber][Route]
- TrainNumber: Letter + 4 digits (e.g., W2073, M2051) or just 4 digits
- Route: 1-2 letters (K, L, M, N, T, S, J, KK, LL, MM, NN, TT, SS)
         OR *, **, X, XX for out-of-service trains
"""

import cv2
import numpy as np
import re

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

# Pattern to find train IDs - matches letter + 4 digits + 1-2 letter suffix
# Also matches 4 digits + 1-2 letter suffix (some IDs don't have leading letter)
TRAIN_ID_PATTERN = re.compile(r'[A-Z]?\d{4}[A-Z*X]{1,2}')

# Y-bands where train IDs appear (percentage of image height)
UPPER_TRAIN_BAND = (0.25, 0.48)  # Above upper track
LOWER_TRAIN_BAND = (0.56, 0.80)  # Below lower track


class TrainDetector:
    """Detects train identifiers in mimic display images."""

    def __init__(self):
        if not TESSERACT_AVAILABLE:
            print("Warning: pytesseract not installed. Train detection disabled.")

    def detect_trains(self, image, hsv=None):
        """
        Detect all train identifiers in the image.

        Returns:
            List of dicts: [{'id': 'W2010L', 'x': 500, 'y': 300, 'track': 'upper'}, ...]
        """
        if not TESSERACT_AVAILABLE:
            return []

        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        trains = []

        # Detect in upper band
        upper_trains = self._detect_in_band(gray, w, h,
            int(h * UPPER_TRAIN_BAND[0]),
            int(h * UPPER_TRAIN_BAND[1]),
            'upper',
            hsv=hsv
        )
        trains.extend(upper_trains)

        # Detect in lower band
        lower_trains = self._detect_in_band(gray, w, h,
            int(h * LOWER_TRAIN_BAND[0]),
            int(h * LOWER_TRAIN_BAND[1]),
            'lower',
            hsv=hsv
        )
        trains.extend(lower_trains)

        # Deduplicate: remove trains with same ID or overlapping positions
        trains = self._deduplicate(trains)

        return trains

    def _deduplicate(self, trains):
        """Remove duplicate train detections."""
        if not trains:
            return trains

        # Sort by x position
        trains = sorted(trains, key=lambda t: (t['track'], t['x']))

        # Remove duplicates: same ID or very close positions
        unique = []
        for train in trains:
            is_dup = False
            for existing in unique:
                # Same track and close position (within 30 pixels)
                if (existing['track'] == train['track'] and
                    abs(existing['x'] - train['x']) < 30):
                    # Keep the one with longer ID or first one
                    if len(train['id']) > len(existing['id']):
                        unique.remove(existing)
                        unique.append(train)
                    is_dup = True
                    break
            if not is_dup:
                unique.append(train)

        return unique

    def _detect_in_band(self, gray, img_w, img_h, y_min, y_max, track, hsv=None):
        """Detect train IDs in a horizontal band."""
        trains = []
        band = gray[y_min:y_max, :]
        band_h = y_max - y_min

        # Find text column regions using multiple methods:
        # 1. Dark pixels (black text)
        dark_mask = (band < 100).astype(np.uint8)

        # 2. Colored text (high saturation) - yellow, green, red, cyan train IDs
        # Note: We use dark_mask for column grouping to avoid track lines filling gaps,
        # but use colored_mask for OCR to catch colored train IDs
        colored_mask = None
        if hsv is not None:
            band_hsv = hsv[y_min:y_max, :]
            # Yellow text (H=15-35, S>100, V>120) - higher thresholds to avoid track
            yellow = cv2.inRange(band_hsv, np.array([15, 100, 120]), np.array([35, 255, 255]))
            # Green text (H=35-85, S>100, V>120)
            green = cv2.inRange(band_hsv, np.array([35, 100, 120]), np.array([85, 255, 255]))
            # Red text (H=0-15 or 165-180, S>80, V>100)
            red1 = cv2.inRange(band_hsv, np.array([0, 80, 100]), np.array([15, 255, 255]))
            red2 = cv2.inRange(band_hsv, np.array([165, 80, 100]), np.array([180, 255, 255]))
            red = red1 | red2
            # Cyan text - exclude track cyan by requiring higher saturation and brightness
            # Track cyan is typically S=180-220, V=180-210; text cyan is brighter
            cyan = cv2.inRange(band_hsv, np.array([85, 150, 200]), np.array([115, 255, 255]))
            # Combine colored masks
            colored_mask = ((yellow | green | red | cyan) > 0).astype(np.uint8)

        # Use only dark pixels for column grouping (to avoid track lines merging columns)
        col_dark_count = dark_mask.sum(axis=0)

        # Find contiguous regions with dark pixels
        text_columns = np.where(col_dark_count > 3)[0]
        if len(text_columns) == 0:
            return trains

        # Group adjacent columns (gap > 10 pixels = separate group)
        # Use smaller gap to better separate adjacent train IDs
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
            if width < 6:  # Too narrow for train ID
                continue

            # For wide groups (>20px), split into sub-columns to handle
            # multiple train IDs that are side by side
            if width > 20:
                # Split into ~12px sub-columns (typical train ID character width)
                sub_cols = []
                for sub_x in range(x1, x2, 12):
                    sub_cols.append((sub_x, min(sub_x + 15, x2)))
            else:
                sub_cols = [(x1, x2)]

            for sub_x1, sub_x2 in sub_cols:
                # Extract column ROI with padding
                pad = 3
                roi_x1 = max(0, sub_x1 - pad)
                roi_x2 = min(img_w, sub_x2 + pad)
                col_roi = band[:, roi_x1:roi_x2]
                col_mask = dark_mask[:, roi_x1:roi_x2]

                # Find vertical extent of text in this column (using dark mask)
                row_text = col_mask.sum(axis=1)
                text_rows = np.where(row_text > 0)[0]
                if len(text_rows) < 5:  # Need some minimum height
                    continue

                y1 = max(0, text_rows[0] - 2)
                y2 = min(band_h, text_rows[-1] + 2)

                # Skip station label areas based on track:
                # - Lower band: station labels at TOP (skip y < 15% of band)
                # - Upper band: station labels at BOTTOM (skip y > 80% of band)
                if track == 'lower':
                    station_label_cutoff = int(band_h * 0.15)
                    if y1 < station_label_cutoff:
                        y1 = station_label_cutoff
                elif track == 'upper':
                    station_label_cutoff = int(band_h * 0.80)
                    if y2 > station_label_cutoff:
                        y2 = station_label_cutoff

                roi = col_roi[y1:y2, :]
                if roi.size == 0 or roi.shape[0] < 20:
                    continue

                # Check if this region has significant colored text
                roi_colored = colored_mask[y1:y2, roi_x1:roi_x2] if colored_mask is not None else None
                has_colored = roi_colored is not None and roi_colored.sum() > 30

                # OCR this region
                found_ids = self._ocr_column(roi, roi_colored if has_colored else None)

                # Add found train IDs with positions
                center_x = (roi_x1 + roi_x2) // 2
                center_y = y_min + (y1 + y2) // 2

                for i, train_id in enumerate(found_ids):
                    trains.append({
                        'id': train_id,
                        'x': int(center_x),
                        'y': int(center_y),
                        'track': track
                    })

        return trains

    def _ocr_column(self, roi, colored_mask=None):
        """Run OCR on a column ROI and extract all train IDs."""
        scale = 4
        found_ids = []

        # Method 1: Standard grayscale OCR
        roi_large = cv2.resize(roi, None, fx=scale, fy=scale,
                               interpolation=cv2.INTER_LANCZOS4)
        _, roi_bin = cv2.threshold(roi_large, 0, 255,
                                   cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        try:
            text = pytesseract.image_to_string(roi_bin, config='--psm 6')
            found_ids.extend(self._extract_train_ids(text))
        except Exception:
            pass

        # Method 2: If colored text detected and grayscale didn't find IDs, try color mask
        if colored_mask is not None and len(found_ids) == 0:
            # Use inverted color mask (black text on white background)
            mask_large = cv2.resize(colored_mask * 255, None, fx=scale, fy=scale,
                                   interpolation=cv2.INTER_NEAREST)
            # Apply morphological operations to clean up
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            mask_clean = cv2.morphologyEx(mask_large, cv2.MORPH_CLOSE, kernel)

            try:
                text = pytesseract.image_to_string(mask_clean, config='--psm 6')
                found_ids.extend(self._extract_train_ids(text))
            except Exception:
                pass

        return found_ids

    def _extract_train_ids(self, text):

        # Join lines and clean up
        lines = [c.strip() for c in text.split('\n') if c.strip()]
        combined = ''.join(lines)

        # Clean up common OCR errors
        combined = combined.upper()
        combined = combined.replace('{', '7').replace('}', '7')
        combined = combined.replace('[', '7').replace(']', '7')  # Brackets often misread as 7
        combined = combined.replace('+', 'TT')  # + is often TT misread as single character
        combined = combined.replace('|', '1').replace('!', '1').replace('L', '1') if combined and combined[0].isdigit() else combined.replace('|', '1').replace('!', '1')
        combined = combined.replace('O', '0').replace('Q', '0')
        combined = combined.replace('Z', '7')  # Z often misread as 7
        combined = combined.replace('S', '5') if combined and combined[0].isdigit() else combined
        combined = combined.replace('B', '8') if combined and combined[0].isdigit() else combined
        combined = combined.replace('(', '').replace(')', '')
        combined = combined.replace(' ', '')

        # Remove station label codes that might be concatenated with train IDs
        # Station labels are 2-3 letter codes like USL, YBL, EMR, FPL, etc.
        station_labels = ['USL', 'USR', 'YBL', 'YBR', 'CTL', 'CTR', 'TTL', 'TTR',
                         'FPL', 'FPR', 'MNL', 'MNR', 'MOR', 'MOL', 'POR', 'POL',
                         'CCL', 'CCR', 'VNL', 'VNR', 'CHR', 'CHL', 'CAR', 'CAL',
                         'FHR', 'FHL', 'WER', 'WEL', 'EMR', 'EML', 'GDL', 'GDR']
        for label in station_labels:
            combined = combined.replace(label, '')

        # Find all train IDs in the text
        found_ids = TRAIN_ID_PATTERN.findall(combined)

        # If no matches found, try fixing 7/T confusion based on position
        # Train ID format: [Letter]?[4 digits][1-2 letters]
        # In digit positions: T should be 7
        # In letter positions: 7 should be T
        if not found_ids and len(combined) >= 5:
            fixed = list(combined)
            # Find where digits should start (after optional leading letter)
            digit_start = 1 if combined[0].isalpha() else 0

            for i in range(len(fixed)):
                # In digit positions (digit_start to digit_start+3)
                if digit_start <= i < digit_start + 4:
                    if fixed[i] == 'T':
                        fixed[i] = '7'
                    elif fixed[i] == 'I':
                        fixed[i] = '1'
                # In letter positions (after digit_start+3)
                elif i >= digit_start + 4:
                    if fixed[i] == '7':
                        fixed[i] = 'T'
                    elif fixed[i] == '1':
                        fixed[i] = 'L'
                    elif fixed[i] == '0':
                        fixed[i] = 'O'

            fixed_str = ''.join(fixed)
            found_ids = TRAIN_ID_PATTERN.findall(fixed_str)

        return found_ids
