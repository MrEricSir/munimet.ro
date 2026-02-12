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

# Tesseract OCR configuration for consistent results across environments
# --oem 3: Use default OCR engine mode (LSTM + legacy, most compatible)
# --psm 6: Assume a single uniform block of text
# Note: Character whitelist removed as it can cause detection issues;
# invalid characters are filtered post-OCR in _extract_train_id()
OCR_CONFIG = '--oem 3 --psm 6'

# Train ID pattern
TRAIN_ID_PATTERN = re.compile(r'[A-Z]?\d{4}[A-Z*X]{1,2}')

# Pattern to detect invalid train IDs with all identical digits (e.g., 1111, 0000)
# Real Muni train numbers never have 4 identical digits
INVALID_REPEATED_DIGITS = re.compile(r'(\d)\1{3}')

# Valid Muni train line prefixes (single letter before 4-digit car number)
# Includes: line letters (M, W, B, C, D, F, I, J, K, L, N, S, T) and common OCR variants
# 'E' is included as a common OCR misread of 'F'
# 'A', 'G', 'H', 'O', 'P', 'Q', 'U', 'V', 'Y', 'Z' are NOT valid Muni prefixes
VALID_TRAIN_PREFIXES = set('MBCDEFIJKLNRSTW')

# Valid Muni train suffixes - be permissive but filter obvious OCR errors
# Valid single suffixes: line designators (J, K, L, M, N, S, T) plus X/* markers
# Valid double suffixes: most combinations are valid except unusual ones
VALID_SUFFIX_LETTERS = set('JKLMNSTX*')  # Common suffix characters

# Valid Muni train suffixes
# Single letter: line designators (J, K, L, M, N, S, T) plus X marker
# Double letters: doubled designators (JJ, KK, LL, MM, NN, SS, TT)
VALID_SINGLE_SUFFIXES = set('JKLMNSTX*')
VALID_DOUBLE_SUFFIXES = {'JJ', 'KK', 'LL', 'MM', 'NN', 'SS', 'TT'}


def _clean_suffix(train_id):
    """Clean up train ID suffix to fix common OCR contamination.

    Handles cases like:
    - D2073JL → D2073J (extra L from nearby text)
    - 7217OK → 7217K (O misread, should just be K)
    - W2015MM → W2015M (extra M from station label MNL)
    """
    # Find the suffix (1-3 chars after 4 digits)
    match = re.match(r'^([A-Z]?)(\d{4})([A-Z*X]{1,3})$', train_id)
    if not match:
        return train_id

    prefix = match.group(1)
    digits = match.group(2)
    suffix = match.group(3)

    # If suffix is already valid, return as-is
    if suffix in VALID_SINGLE_SUFFIXES or suffix in VALID_DOUBLE_SUFFIXES:
        return train_id

    # Try to clean up the suffix
    cleaned_suffix = suffix

    # Case 1: 3-char suffix - try to extract valid 1 or 2 char suffix
    if len(suffix) == 3:
        # Check if first 2 chars are valid double suffix
        if suffix[:2] in VALID_DOUBLE_SUFFIXES:
            cleaned_suffix = suffix[:2]
        # Check if first char is valid single suffix
        elif suffix[0] in VALID_SINGLE_SUFFIXES:
            cleaned_suffix = suffix[0]

    # Case 2: 2-char suffix that's not a valid double (like JJ, KK, etc.)
    elif len(suffix) == 2 and suffix not in VALID_DOUBLE_SUFFIXES:
        # If first char is O (likely OCR error for 0), use second char
        if suffix[0] == 'O' and suffix[1] in VALID_SINGLE_SUFFIXES:
            cleaned_suffix = suffix[1]
        # Mixed letters like JL, MN are contamination - train can't be on two routes
        # Keep only the first letter (assumes the second is from nearby text)
        elif suffix[0] in VALID_SINGLE_SUFFIXES:
            cleaned_suffix = suffix[0]

    if cleaned_suffix != suffix:
        return prefix + digits + cleaned_suffix
    return train_id


def _has_invalid_suffix(train_id):
    """Check if train ID has an invalid suffix that indicates OCR error."""
    # Find where suffix starts (after 4 digits)
    match = re.search(r'\d{4}([A-Z*X]{1,2})$', train_id)
    if not match:
        return False

    suffix = match.group(1)

    # Single character suffix - must be a valid line designator or X/*
    if len(suffix) == 1:
        return suffix not in VALID_SINGLE_SUFFIXES

    # Double character suffix - must be a valid double (JJ, KK, etc.)
    # or a combination of valid single suffixes
    if len(suffix) == 2:
        if suffix in VALID_DOUBLE_SUFFIXES:
            return False
        # Allow combinations like KL, JM, etc.
        if suffix[0] in VALID_SINGLE_SUFFIXES and suffix[1] in VALID_SINGLE_SUFFIXES:
            return False
        return True

    return False


def _has_suspicious_digits(train_id):
    """Check if train ID has suspicious digit patterns that indicate OCR error."""
    # Extract the 4-digit car number
    match = re.search(r'([A-Z]?)(\d{4})[A-Z*X]{1,2}$', train_id)
    if not match:
        return False

    prefix = match.group(1)
    digits = match.group(2)

    # Train IDs without prefix starting with 0 are suspicious
    # (real car numbers don't start with 0)
    if not prefix and digits[0] == '0':
        return True

    return False

# Pattern to detect invalid prefix (letter + 4 digits where letter is not valid)
def _has_invalid_prefix(train_id):
    """Check if train ID has an invalid line prefix."""
    if len(train_id) >= 5 and train_id[0].isalpha() and train_id[1].isdigit():
        return train_id[0] not in VALID_TRAIN_PREFIXES
    return False

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

        # Step 2: Symbol detection only if OCR found no trains
        # This reduces false positives from track signals and other visual elements
        if len(ocr_trains) == 0:
            symbols = self._detect_symbols(hsv, h, w, gray)
            trains = self._merge(ocr_trains, symbols)
        else:
            trains = ocr_trains

        return trains

    def _detect_by_ocr(self, gray, h, w, hsv):
        """Detect trains via OCR with selective component detection for pileups.

        Strategy:
        1. Run legacy column-based detection (proven stable)
        2. Find clusters where trains are bunched (3+ within 100px)
        3. Run component-based detection only in cluster regions (better isolation)
        4. Merge results
        """
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

            # Step 1: Legacy column-based detection (primary method)
            track_trains = []
            dark_mask = (band < 120).astype(np.uint8)
            col_sums = dark_mask.sum(axis=0)
            text_cols = np.where(col_sums > 3)[0]

            if len(text_cols) > 0:
                groups = self._group_columns(text_cols)
                for x1, x2 in groups:
                    width = x2 - x1
                    if width < 6:
                        continue

                    if width > 20:
                        sub_cols = self._split_wide_group(
                            x1, x2, col_sums, band, dark_mask, band_h, track
                        )
                    else:
                        sub_cols = [(x1, x2)]

                    found_from_splits = []
                    for sub_x1, sub_x2 in sub_cols:
                        train_ids = self._ocr_column(band, dark_mask, sub_x1, sub_x2, band_h, track)
                        if train_ids:
                            sub_width = sub_x2 - sub_x1
                            n_trains = len(train_ids)
                            for i, train_id in enumerate(train_ids):
                                if n_trains == 1:
                                    center_x = (sub_x1 + sub_x2) // 2
                                else:
                                    center_x = sub_x1 + int(sub_width * (i + 0.5) / n_trains)
                                found_from_splits.append({
                                    'id': train_id,
                                    'x': center_x,
                                    'y': y_min + band_h // 2,
                                    'track': track,
                                    'confidence': 'high'
                                })

                    # Fallback: if splits produced nothing, try the original unsplit range
                    if not found_from_splits and len(sub_cols) > 1:
                        train_ids = self._ocr_column(band, dark_mask, x1, x2, band_h, track)
                        if train_ids:
                            n_trains = len(train_ids)
                            for i, train_id in enumerate(train_ids):
                                if n_trains == 1:
                                    center_x = (x1 + x2) // 2
                                else:
                                    center_x = x1 + int(width * (i + 0.5) / n_trains)
                                found_from_splits.append({
                                    'id': train_id,
                                    'x': center_x,
                                    'y': y_min + band_h // 2,
                                    'track': track,
                                    'confidence': 'high'
                                })

                    track_trains.extend(found_from_splits)

            # Step 2: Find clusters (potential pileups) - 3+ trains within 200px
            clusters = self._find_train_clusters(track_trains, min_trains=3, max_spread=300)

            # Step 3: For each cluster, run component detection with relaxed filters
            for cluster_x_min, cluster_x_max in clusters:
                component_trains = self._detect_by_components_in_region(
                    band, y_min, band_h, track, cluster_x_min, cluster_x_max
                )
                track_trains.extend(component_trains)

            trains.extend(track_trains)

            # Also detect colored text labels (yellow, green)
            colored_trains = self._detect_colored_labels(band, band_hsv, band_h, y_min, track)
            trains.extend(colored_trains)

        return self._deduplicate(trains)

    def _find_train_clusters(self, trains, min_trains=3, max_spread=200):
        """Find clusters of trains that might indicate a pileup.

        Uses a sliding window approach: finds the largest cluster containing
        at least min_trains trains, where all trains are within max_spread
        of each other (measured from leftmost to rightmost train).

        Returns list of (x_min, x_max) tuples for each cluster region.
        """
        if len(trains) < min_trains:
            return []

        # Sort trains by x position
        sorted_trains = sorted(trains, key=lambda t: t['x'])
        clusters = []
        used = set()  # Track which trains are already in a cluster

        # Use a sliding window to find clusters
        # A cluster is a contiguous group where the spread (max_x - min_x) <= max_spread
        for start in range(len(sorted_trains)):
            if start in used:
                continue

            # Expand the window as far as possible while keeping spread <= max_spread
            end = start
            while end + 1 < len(sorted_trains):
                new_spread = sorted_trains[end + 1]['x'] - sorted_trains[start]['x']
                if new_spread <= max_spread:
                    end += 1
                else:
                    break

            cluster_size = end - start + 1
            if cluster_size >= min_trains:
                x_min = sorted_trains[start]['x']
                x_max = sorted_trains[end]['x']
                # Extend region to catch any missed trains at edges
                clusters.append((max(0, x_min - 40), x_max + 40))
                # Mark these trains as used
                for idx in range(start, end + 1):
                    used.add(idx)

        return clusters

    def _detect_by_components_in_region(self, band, y_offset, band_h, track, x_min, x_max):
        """Detect trains using component grouping in a specific x-region.

        This uses RELAXED filters compared to _detect_by_components because
        we only call this in regions where legacy detection already found
        a cluster of trains (high confidence there are real trains).
        """
        trains = []

        # Threshold to get dark text
        _, binary = cv2.threshold(band, 120, 255, cv2.THRESH_BINARY_INV)

        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

        # Group components by x-position with moderate tolerance
        columns = {}
        x_tolerance = 8  # Relaxed from 6

        for i in range(1, num_labels):
            x, y, bw, bh, area = stats[i]
            if area < 20:  # Relaxed from 30
                continue

            cx = x + bw // 2

            # Only consider components in the target region
            if cx < x_min or cx > x_max:
                continue

            # Find existing column within tolerance
            found = False
            for col_x in list(columns.keys()):
                if abs(col_x - cx) <= x_tolerance:
                    columns[col_x].append((x, y, bw, bh, area))
                    found = True
                    break

            if not found:
                columns[cx] = [(x, y, bw, bh, area)]

        # Process each column
        for col_x, components in columns.items():
            # Need at least 3 character fragments (relaxed from 4)
            if len(components) < 3:
                continue

            # Compute bounding box
            min_x = min(c[0] for c in components)
            min_y = min(c[1] for c in components)
            max_x = max(c[0] + c[2] for c in components)
            max_y = max(c[1] + c[3] for c in components)

            col_w = max_x - min_x
            col_h = max_y - min_y
            total_area = sum(c[4] for c in components)

            # Relaxed filters for cluster regions
            if col_w < 5 or col_w > 25:  # Relaxed from 20
                continue
            if col_h < 40 or col_h > 130:  # Relaxed from 50-120
                continue
            if total_area < 150:  # Relaxed from 200
                continue

            # Skip station label regions
            if track == 'lower':
                station_cutoff = int(band_h * 0.15)
                if min_y < station_cutoff:
                    min_y = station_cutoff
            else:
                station_cutoff = int(band_h * 0.95)
                if max_y > station_cutoff:
                    max_y = station_cutoff

            if max_y - min_y < 40:  # Relaxed from 50
                continue

            # Extract and OCR the column
            pad = 3
            roi_x1 = max(0, min_x - pad)
            roi_x2 = min(band.shape[1], max_x + pad)
            roi_y1 = max(0, min_y - pad)
            roi_y2 = min(band.shape[0], max_y + pad)

            roi = band[roi_y1:roi_y2, roi_x1:roi_x2]
            if roi.size == 0 or roi.shape[0] < 30:  # Relaxed from 40
                continue

            train_ids = self._ocr_roi(roi, track)
            if train_ids:
                for train_id in train_ids:
                    trains.append({
                        'id': train_id,
                        'x': col_x,
                        'y': y_offset + (min_y + max_y) // 2,
                        'track': track,
                        'confidence': 'high'
                    })

        return trains

    def _ocr_roi(self, roi, track):
        """Run OCR on a region of interest and extract train IDs."""
        # Scale up for better OCR
        scale = 4
        roi_large = cv2.resize(roi, None, fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)

        # Otsu threshold
        _, roi_bin = cv2.threshold(roi_large, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        try:
            text = pytesseract.image_to_string(roi_bin, config=OCR_CONFIG)
            return self._extract_train_ids(text)
        except Exception:
            return []

    def _detect_by_ocr_legacy(self, gray, h, w, hsv):
        """Legacy OCR detection method - kept for reference."""
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

            # Find dark text columns (threshold 120 to catch lighter text)
            dark_mask = (band < 120).astype(np.uint8)
            col_sums = dark_mask.sum(axis=0)
            text_cols = np.where(col_sums > 3)[0]

            if len(text_cols) > 0:
                # Group columns
                groups = self._group_columns(text_cols)

                for x1, x2 in groups:
                    width = x2 - x1
                    if width < 6:
                        continue

                    # Sub-column splitting for groups wider than a single train ID
                    # A typical train ID is 10-18 pixels wide
                    # Groups 20+ pixels wide may contain multiple trains
                    if width > 20:
                        # Try to find the best split point (largest internal gap)
                        sub_cols = self._split_wide_group(
                            x1, x2, col_sums, band, dark_mask, band_h, track
                        )
                    else:
                        sub_cols = [(x1, x2)]

                    found_from_splits = []
                    for sub_x1, sub_x2 in sub_cols:
                        train_ids = self._ocr_column(band, dark_mask, sub_x1, sub_x2, band_h, track)
                        if train_ids:
                            # Distribute multiple train IDs across the column width
                            sub_width = sub_x2 - sub_x1
                            n_trains = len(train_ids)
                            for i, train_id in enumerate(train_ids):
                                if n_trains == 1:
                                    center_x = (sub_x1 + sub_x2) // 2
                                else:
                                    # Space trains evenly across the column
                                    center_x = sub_x1 + int(sub_width * (i + 0.5) / n_trains)
                                found_from_splits.append({
                                    'id': train_id,
                                    'x': center_x,
                                    'y': y_min + band_h // 2,
                                    'track': track,
                                    'confidence': 'high'
                                })

                    # Fallback: if splits produced nothing, try the original unsplit range
                    if not found_from_splits and len(sub_cols) > 1:
                        train_ids = self._ocr_column(band, dark_mask, x1, x2, band_h, track)
                        if train_ids:
                            n_trains = len(train_ids)
                            for i, train_id in enumerate(train_ids):
                                if n_trains == 1:
                                    center_x = (x1 + x2) // 2
                                else:
                                    center_x = x1 + int(width * (i + 0.5) / n_trains)
                                found_from_splits.append({
                                    'id': train_id,
                                    'x': center_x,
                                    'y': y_min + band_h // 2,
                                    'track': track,
                                    'confidence': 'high'
                                })

                    trains.extend(found_from_splits)

            # Also detect colored text labels (yellow, green)
            colored_trains = self._detect_colored_labels(band, band_hsv, band_h, y_min, track)
            trains.extend(colored_trains)

        return self._deduplicate(trains)

    def _split_wide_group(self, x1, x2, col_sums, band=None, dark_mask=None, band_h=None, track=None):
        """Split a wide column group at the best split point.

        For groups that may contain multiple trains, find the column with
        the minimum sum (gap between trains) and split there.

        If band/dark_mask/band_h/track are provided, validates the split by
        comparing OCR results from split vs unsplit interpretations.
        """
        width = x2 - x1
        if width <= 20:
            return [(x1, x2)]

        # Find the minimum column sum in the middle portion of the group
        # (don't split at the edges)
        margin = max(6, width // 5)
        mid_start = x1 + margin
        mid_end = x2 - margin

        if mid_start >= mid_end:
            return [(x1, x2)]

        # Find the column with minimum sum (likely a gap between trains)
        min_sum = float('inf')
        split_x = None
        for x in range(mid_start, mid_end):
            if col_sums[x] < min_sum:
                min_sum = col_sums[x]
                split_x = x

        # Always split if we found a true gap (zero column sum)
        if split_x and min_sum == 0:
            return [(x1, split_x), (split_x + 1, x2)]

        # For non-zero gaps in wide groups, validate split with OCR
        # Only attempt if we have the necessary context for OCR validation
        if split_x and width > 30 and band is not None and dark_mask is not None:
            max_sum = max(col_sums[mid_start:mid_end])
            # Only consider splitting if there's a significant valley
            # (min is less than 20% of max - indicates potential gap between trains)
            if min_sum < max_sum * 0.20:
                split_result = self._validate_split_with_ocr(
                    x1, x2, split_x, band, dark_mask, band_h, track
                )
                if split_result is not None:
                    return split_result

        return [(x1, x2)]

    def _validate_split_with_ocr(self, x1, x2, split_x, band, dark_mask, band_h, track):
        """Validate a potential split by comparing OCR results.

        Returns split boundaries if splitting produces more valid train IDs,
        otherwise returns None to indicate the group should not be split.
        """
        # OCR the unsplit group
        unsplit_ids = self._ocr_column(band, dark_mask, x1, x2, band_h, track)

        # OCR both parts of the split
        left_ids = self._ocr_column(band, dark_mask, x1, split_x, band_h, track)
        right_ids = self._ocr_column(band, dark_mask, split_x + 1, x2, band_h, track)
        split_ids = left_ids + right_ids

        # Choose the interpretation that produces more valid train IDs
        if len(split_ids) > len(unsplit_ids):
            return [(x1, split_x), (split_x + 1, x2)]
        else:
            return None  # Don't split

    def _group_columns(self, columns, gap_threshold=6):
        """Group adjacent columns. Gap threshold determines when to split groups."""
        if len(columns) == 0:
            return []
        groups = []
        start = columns[0]
        prev = start
        for x in columns[1:]:
            if x - prev > gap_threshold:
                groups.append((start, prev))
                start = x
            prev = x
        groups.append((start, prev))
        return groups

    def _ocr_column(self, band, dark_mask, x1, x2, band_h, track):
        """OCR a single column - returns list of train IDs found."""
        pad = 3
        roi_x1 = max(0, x1 - pad)
        roi_x2 = min(band.shape[1], x2 + pad)
        col_mask = dark_mask[:, roi_x1:roi_x2]

        # Find vertical extent
        row_sums = col_mask.sum(axis=1)
        text_rows = np.where(row_sums > 0)[0]
        if len(text_rows) < 5:
            return []

        y1 = max(0, text_rows[0] - 2)
        y2 = min(band_h, text_rows[-1] + 2)

        # Skip station label regions
        if track == 'lower':
            # Lower track: station labels are at top of band
            station_cutoff = int(band_h * 0.15)
            if y1 < station_cutoff:
                y1 = station_cutoff
        else:
            # Upper track: station labels are at bottom of band
            # Use simple 95% cutoff - station name contamination is handled
            # by filtering in _extract_train_ids
            station_cutoff = int(band_h * 0.95)
            if y2 > station_cutoff:
                y2 = station_cutoff

        roi = band[y1:y2, roi_x1:roi_x2]
        if roi.size == 0 or roi.shape[0] < 20:
            return []

        # Single OCR pass with Otsu
        scale = 4
        roi_large = cv2.resize(roi, None, fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)
        _, roi_bin = cv2.threshold(roi_large, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        try:
            text = pytesseract.image_to_string(roi_bin, config=OCR_CONFIG)
            train_ids = self._extract_train_ids(text)
            if train_ids:
                return train_ids

            # Fallback 1: try CLAHE enhancement for faint text
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(roi_large)
            _, roi_bin_clahe = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            text = pytesseract.image_to_string(roi_bin_clahe, config=OCR_CONFIG)
            train_ids = self._extract_train_ids(text)
            if train_ids:
                return train_ids

            # Fallback 2: try gamma correction for low-contrast text
            gamma = 0.7
            lut = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)]).astype(np.uint8)
            gamma_corrected = cv2.LUT(roi_large, lut)
            _, roi_bin_gamma = cv2.threshold(gamma_corrected, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            text = pytesseract.image_to_string(roi_bin_gamma, config=OCR_CONFIG)
            return self._extract_train_ids(text)
        except Exception:
            return []

    def _detect_colored_labels(self, band_gray, band_hsv, band_h, y_offset, track):
        """Detect colored text labels (yellow, green train IDs)."""
        trains = []

        # Yellow text (H=15-40, high saturation and value)
        yellow = cv2.inRange(band_hsv, np.array([15, 80, 120]), np.array([40, 255, 255]))
        # Green text (H=35-85)
        green = cv2.inRange(band_hsv, np.array([35, 80, 120]), np.array([85, 255, 255]))

        # Process yellow/green together (they don't overlap with blue)
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

            # Skip station label areas (lower track only)
            # Upper track relies on station label filtering in _extract_train_id()
            if track == 'lower':
                station_cutoff = int(band_h * 0.15)
                if y1 < station_cutoff:
                    y1 = station_cutoff

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

        # Also detect blue text separately (blue station indicators can interfere with yellow/green)
        blue = cv2.inRange(band_hsv, np.array([100, 100, 80]), np.array([130, 255, 255]))
        blue_col_sums = blue.sum(axis=0)
        blue_cols = np.where(blue_col_sums > 100)[0]

        if len(blue_cols) > 0:
            blue_groups = self._group_columns(blue_cols)
            for x1, x2 in blue_groups:
                width = x2 - x1
                if width < 5 or width > 50:
                    continue

                col_region = blue[:, x1:x2+1]
                row_sums = col_region.sum(axis=1)
                text_rows = np.where(row_sums > 0)[0]
                if len(text_rows) < 10:
                    continue

                y1 = max(0, text_rows[0] - 2)
                y2 = min(band_h, text_rows[-1] + 2)

                if track == 'lower':
                    station_cutoff = int(band_h * 0.15)
                    if y1 < station_cutoff:
                        y1 = station_cutoff

                if y2 - y1 < 20:
                    continue

                pad = 3
                roi_x1 = max(0, x1 - pad)
                roi_x2 = min(band_gray.shape[1], x2 + pad)
                blue_roi = blue[y1:y2, roi_x1:roi_x2]

                if blue_roi.size == 0:
                    continue

                train_id = self._ocr_color_mask(blue_roi)
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
            text = pytesseract.image_to_string(mask_clean, config=OCR_CONFIG)
            train_id = self._extract_train_id(text)
            if train_id:
                return train_id

            # Try inverted
            text = pytesseract.image_to_string(255 - mask_clean, config=OCR_CONFIG)
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

    def _extract_train_ids(self, text):
        """Extract all train IDs from OCR text. Returns list of train IDs."""
        lines = [c.strip() for c in text.split('\n') if c.strip()]
        combined = ''.join(lines).upper()

        # Remove station labels FIRST (before O→0 conversion changes them)
        for label in self.station_labels:
            combined = combined.replace(label, '')

        # OCR corrections
        combined = combined.replace('{', '7').replace('}', '7')
        combined = combined.replace('[', '7').replace(']', '7')
        combined = combined.replace('+', 'TT')
        combined = combined.replace('|', '1').replace('!', '1')
        combined = combined.replace('O', '0').replace('Q', '0')
        combined = combined.replace('Z', '7')
        combined = combined.replace('(', '').replace(')', '').replace(' ', '')

        # OCR sometimes reads digits as 'RE' - could be 5 or 7
        # Try both interpretations and return all unique valid train IDs
        # Prefer 5 over 7 since RE is more commonly a misread 5
        if 'RE' in combined:
            combined_5 = combined.replace('RE', '5')
            combined_7 = combined.replace('RE', '7')
            matches_5 = TRAIN_ID_PATTERN.findall(combined_5)
            matches_7 = TRAIN_ID_PATTERN.findall(combined_7)
            # Return 5-versions first, then any unique 7-versions
            result = list(matches_5)
            for m in matches_7:
                if m not in result:
                    result.append(m)
            if result:
                return [_clean_suffix(m) for m in result]
            combined = combined_5  # Fall through if neither worked

        # Fix duplicate leading letters (OCR sometimes doubles them in vertical text)
        if len(combined) >= 2 and combined[0].isalpha() and combined[0] == combined[1]:
            combined = combined[1:]

        # First try to match without aggressive corrections
        matches = TRAIN_ID_PATTERN.findall(combined)
        if matches:
            # Clean suffixes first, then filter out invalid train IDs
            # (cleaning must happen before validation since contamination like
            # "W2131KE" should clean to "W2131K" before checking suffix validity)
            cleaned_matches = [_clean_suffix(m) for m in matches]
            # Filter out invalid train IDs with:
            # - repeated digits (e.g., 1111II, 0000MM)
            # - invalid prefixes (e.g., A4111I - 'A' is not a valid Muni line)
            # - invalid suffixes (e.g., 2117XJ, 2111FX - X not used in combinations)
            # - suspicious digit patterns (e.g., 8566MK - starts with 8)
            cleaned_matches = [m for m in cleaned_matches if not INVALID_REPEATED_DIGITS.search(m)
                               and not _has_invalid_prefix(m)
                               and not _has_invalid_suffix(m)
                               and not _has_suspicious_digits(m)]
            if cleaned_matches:
                return cleaned_matches

        # Position-based corrections for vertical text OCR errors
        # Find where digits likely start (first digit or letter that looks like a digit)
        if len(combined) >= 5:
            fixed = list(combined)
            digit_start = 0
            for i, c in enumerate(combined):
                if c.isdigit() or c in 'TIFLS':  # Characters that might be misread digits
                    digit_start = i
                    break

            for i in range(len(fixed)):
                if digit_start <= i < digit_start + 4:
                    # In digit positions: convert letters that look like digits
                    # Note: B, M, W, etc. are valid train line prefixes, don't convert
                    if fixed[i] == 'T':
                        fixed[i] = '7'
                    elif fixed[i] == 'I':
                        fixed[i] = '1'
                    elif fixed[i] == 'F':
                        fixed[i] = '7'  # F and 7 look similar
                    elif fixed[i] == 'L':
                        fixed[i] = '1'
                    elif fixed[i] == 'S':
                        fixed[i] = '5'
                elif i >= digit_start + 4:
                    # In suffix positions: convert digits that look like letters
                    if fixed[i] == '7':
                        fixed[i] = 'T'
                    elif fixed[i] == '1':
                        fixed[i] = 'L'
                    elif fixed[i] == '0':
                        fixed[i] = 'O'
            matches = TRAIN_ID_PATTERN.findall(''.join(fixed))
            if matches:
                # Filter out invalid train IDs with repeated digits, invalid prefixes, invalid suffixes, and suspicious digits
                matches = [m for m in matches if not INVALID_REPEATED_DIGITS.search(m)
                           and not _has_invalid_prefix(m)
                           and not _has_invalid_suffix(m)
                           and not _has_suspicious_digits(m)]
                if matches:
                    return [_clean_suffix(m) for m in matches]

        return []

    def _extract_train_id(self, text):
        """Extract first train ID from OCR text (for backward compatibility)."""
        matches = self._extract_train_ids(text)
        return matches[0] if matches else None

    def _deduplicate(self, trains):
        """Remove duplicates (same train detected twice), but keep bunched trains."""
        if not trains:
            return trains
        trains = sorted(trains, key=lambda t: (t['track'], t['x']))
        unique = []
        for train in trains:
            is_dup = False
            for existing in unique:
                if (existing['track'] == train['track'] and
                        abs(existing['x'] - train['x']) < 40):
                    # Only consider duplicates if IDs are similar
                    # (one is prefix of other, or differ by at most 2 chars)
                    if self._ids_are_similar(existing['id'], train['id']):
                        if len(train['id']) > len(existing['id']):
                            unique.remove(existing)
                            unique.append(train)
                        is_dup = True
                        break
            if not is_dup:
                unique.append(train)
        return unique

    def _ids_are_similar(self, id1, id2):
        """Check if two train IDs are likely the same train detected twice."""
        # One is a prefix of the other (e.g., "M2034L" and "M2034LL")
        if id1.startswith(id2) or id2.startswith(id1):
            return True
        # Same length and differ by at most 2 characters
        if len(id1) == len(id2):
            diff = sum(c1 != c2 for c1, c2 in zip(id1, id2))
            if diff <= 2:
                return True
        return False
