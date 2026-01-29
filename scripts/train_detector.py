"""
Train identifier detection using OCR.

Detects vertical train ID strings (e.g., "W2073LL") in track corridors.
"""

import cv2
import numpy as np
import re

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

# Train ID pattern: Route letter + 3-4 digits + 1-2 letters
# Flexible to allow for OCR errors
TRAIN_ID_PATTERN = re.compile(r'^[A-Z][0-9]{3,4}[A-Z]{1,2}$')

# Pattern to find train ID within a longer string (e.g., when station label is included)
TRAIN_ID_SEARCH = re.compile(r'[WMBKJFDCNTSL][0-9]{3,4}[A-Z]{1,2}')

# Y-bands where train IDs appear (percentage of image height)
UPPER_TRAIN_BAND = (0.25, 0.48)  # Above upper track
LOWER_TRAIN_BAND = (0.58, 0.80)  # Below lower track


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
        upper_trains = self._detect_in_band(
            gray, w, h,
            int(h * UPPER_TRAIN_BAND[0]),
            int(h * UPPER_TRAIN_BAND[1]),
            'upper'
        )
        trains.extend(upper_trains)

        # Detect in lower band
        lower_trains = self._detect_in_band(
            gray, w, h,
            int(h * LOWER_TRAIN_BAND[0]),
            int(h * LOWER_TRAIN_BAND[1]),
            'lower'
        )
        trains.extend(lower_trains)

        return trains

    def _detect_in_band(self, gray, img_w, img_h, y_min, y_max, track):
        """Detect train IDs in a horizontal band by grouping vertical character columns."""
        trains = []

        # Extract band
        band = gray[y_min:y_max, :]
        band_height = y_max - y_min

        # Find dark text (threshold for dark text on gray background ~153)
        dark_mask = (band < 100).astype(np.uint8)

        # Find connected components (individual characters)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(dark_mask)

        # For lower band, skip station labels at the top of the band
        # Station labels are in the first ~15% of the band height
        y_skip = int(band_height * 0.15) if track == 'lower' else 0

        # Collect character bounding boxes
        chars = []
        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]
            # Filter for character-sized regions
            if 4 <= w <= 20 and 6 <= h <= 20 and area >= 15:
                # Skip characters in station label area for lower band
                if y < y_skip:
                    continue
                chars.append({
                    'x': x, 'y': y, 'w': w, 'h': h,
                    'cx': x + w // 2, 'cy': y + h // 2
                })

        # Group characters into vertical columns (train IDs)
        # Characters in the same train ID should have similar x-coordinates
        # and form a continuous vertical sequence
        chars.sort(key=lambda c: (c['cx'], c['cy']))

        groups = []
        used = set()

        for i, char in enumerate(chars):
            if i in used:
                continue

            # Start a new group with this character
            group = [char]
            used.add(i)

            # Find other characters vertically aligned with this one
            for j, other in enumerate(chars):
                if j in used:
                    continue
                # Check if horizontally close (within 10 pixels)
                if abs(other['cx'] - char['cx']) < 10:
                    # Check vertical distance - characters should be within ~20 pixels
                    # of at least one existing character in the group
                    min_y_dist = min(abs(other['cy'] - g['cy']) for g in group)
                    if min_y_dist < 25:  # Allow some gap between characters
                        group.append(other)
                        used.add(j)

            # Only keep groups with 5-9 characters (train ID length: route + 4 digits + 1-2 suffix)
            if 5 <= len(group) <= 9:
                groups.append(group)

        # Process each group
        for group in groups:
            # Get bounding box of the group
            x_min = min(c['x'] for c in group) - 2
            x_max = max(c['x'] + c['w'] for c in group) + 2
            y_min_g = min(c['y'] for c in group) - 2
            y_max_g = max(c['y'] + c['h'] for c in group) + 2

            x_min = max(0, x_min)
            x_max = min(img_w, x_max)
            y_min_g = max(0, y_min_g)
            y_max_g = min(y_max - y_min, y_max_g)

            # Extract ROI
            roi = band[y_min_g:y_max_g, x_min:x_max]
            if roi.size == 0:
                continue

            # Run OCR
            train_id = self._ocr_train_id(roi)

            if train_id:
                trains.append({
                    'id': train_id,
                    'x': int((x_min + x_max) // 2),
                    'y': int(y_min + (y_min_g + y_max_g) // 2),
                    'track': track
                })

        return trains

    def _ocr_train_id(self, roi):
        """Run OCR on a vertical text ROI and validate as train ID."""
        # Resize for better OCR
        scale = 5
        roi_large = cv2.resize(roi, None, fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)

        # Binarize
        _, roi_bin = cv2.threshold(roi_large, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Read vertically without rotation (PSM 6 = block of text)
        config = '--psm 6'
        try:
            text = pytesseract.image_to_string(roi_bin, config=config)
        except Exception:
            return None

        # Join lines to form train ID (each character is on its own line)
        chars = [c.strip() for c in text.split('\n') if c.strip()]
        train_id = ''.join(chars)

        # Clean up common OCR errors
        train_id = train_id.replace('{', '7').replace('}', '7')  # 7 often misread as brace
        train_id = train_id.replace('|', '1').replace('l', '1').replace('!', '1')
        train_id = train_id.replace('O', '0').replace('o', '0')
        train_id = train_id.replace(' ', '').replace('(', '').replace(')', '')
        train_id = train_id.upper()

        # First try exact match
        if TRAIN_ID_PATTERN.match(train_id):
            return train_id

        # Search for train ID within the string (handles station label prefix)
        match = TRAIN_ID_SEARCH.search(train_id)
        if match:
            return match.group(0)

        return None
