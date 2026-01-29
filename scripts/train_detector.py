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
        trains = []

        # Detect in upper band
        upper_trains = self._detect_in_band(gray, w, h,
            int(h * UPPER_TRAIN_BAND[0]),
            int(h * UPPER_TRAIN_BAND[1]),
            'upper'
        )
        trains.extend(upper_trains)

        # Detect in lower band
        lower_trains = self._detect_in_band(gray, w, h,
            int(h * LOWER_TRAIN_BAND[0]),
            int(h * LOWER_TRAIN_BAND[1]),
            'lower'
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

    def _detect_in_band(self, gray, img_w, img_h, y_min, y_max, track):
        """Detect train IDs in a horizontal band."""
        trains = []
        band = gray[y_min:y_max, :]
        band_h = y_max - y_min

        # Find text column regions (columns with dark pixels)
        dark_mask = (band < 100).astype(np.uint8)
        col_dark_count = dark_mask.sum(axis=0)

        # Find contiguous regions with dark pixels
        text_columns = np.where(col_dark_count > 3)[0]
        if len(text_columns) == 0:
            return trains

        # Group adjacent columns (gap > 15 pixels = separate group)
        column_groups = []
        start = text_columns[0]
        prev = start
        for x in text_columns[1:]:
            if x - prev > 15:
                column_groups.append((start, prev))
                start = x
            prev = x
        column_groups.append((start, prev))

        # Process each column group
        for x1, x2 in column_groups:
            width = x2 - x1
            if width < 6:  # Too narrow for train ID
                continue

            # For wide groups (>30px), split into sub-columns to handle
            # multiple train IDs that are side by side
            if width > 30:
                # Split into ~15px sub-columns
                sub_cols = []
                for sub_x in range(x1, x2, 15):
                    sub_cols.append((sub_x, min(sub_x + 18, x2)))
            else:
                sub_cols = [(x1, x2)]

            for sub_x1, sub_x2 in sub_cols:
                # Extract column ROI with padding
                pad = 3
                roi_x1 = max(0, sub_x1 - pad)
                roi_x2 = min(img_w, sub_x2 + pad)
                col_roi = band[:, roi_x1:roi_x2]

                # Find vertical extent of text in this column
                row_dark = (col_roi < 100).sum(axis=1)
                text_rows = np.where(row_dark > 0)[0]
                if len(text_rows) < 5:  # Need some minimum height
                    continue

                y1 = max(0, text_rows[0] - 2)
                y2 = min(band_h, text_rows[-1] + 2)

                # Skip if in station label area (top portion of lower band)
                if track == 'lower' and y1 < band_h * 0.08:
                    y1 = int(band_h * 0.08)

                roi = band[y1:y2, roi_x1:roi_x2]
                if roi.size == 0 or roi.shape[0] < 20:
                    continue

                # OCR this region
                found_ids = self._ocr_column(roi)

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

    def _ocr_column(self, roi):
        """Run OCR on a column ROI and extract all train IDs."""
        # Resize for better OCR
        scale = 4
        roi_large = cv2.resize(roi, None, fx=scale, fy=scale,
                               interpolation=cv2.INTER_LANCZOS4)

        # Binarize
        _, roi_bin = cv2.threshold(roi_large, 0, 255,
                                   cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Read vertically (PSM 6 = block of text)
        try:
            text = pytesseract.image_to_string(roi_bin, config='--psm 6')
        except Exception:
            return []

        # Join lines and clean up
        lines = [c.strip() for c in text.split('\n') if c.strip()]
        combined = ''.join(lines)

        # Clean up common OCR errors
        combined = combined.upper()
        combined = combined.replace('{', '7').replace('}', '7')
        combined = combined.replace('|', '1').replace('!', '1').replace('L', '1') if combined and combined[0].isdigit() else combined.replace('|', '1').replace('!', '1')
        combined = combined.replace('O', '0').replace('Q', '0')
        combined = combined.replace('Z', '7')  # Z often misread as 7
        combined = combined.replace('S', '5') if combined and combined[0].isdigit() else combined
        combined = combined.replace('B', '8') if combined and combined[0].isdigit() else combined
        combined = combined.replace('(', '').replace(')', '')
        combined = combined.replace('[', '').replace(']', '')
        combined = combined.replace(' ', '')

        # Find all train IDs in the text
        found_ids = TRAIN_ID_PATTERN.findall(combined)

        return found_ids
