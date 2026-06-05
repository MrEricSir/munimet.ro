#!/usr/bin/env python3
"""
Inspect raw OCR output for a specific x position in a SCADA image.

Useful for debugging train ID detection failures without running the full test
suite. Shows raw Tesseract output and the IDs extracted by _extract_train_ids.

Usage:
    .venv/bin/python scripts/inspect-ocr.py <image_path> <x_position>

Example:
    .venv/bin/python scripts/inspect-ocr.py tests/images/foo.jpg 804
"""
import sys
import cv2
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.train_detector import TrainDetector, TESSERACT_AVAILABLE
from lib.config import UPPER_TRAIN_BAND, LOWER_TRAIN_BAND, OCR_CONFIG

if not TESSERACT_AVAILABLE:
    print("Error: Tesseract not installed. Install with: brew install tesseract")
    sys.exit(1)

import pytesseract


def ocr_region(gray, y1, y2, x1, x2):
    """Run OCR on a region using the same preprocessing as _ocr_column."""
    roi = gray[y1:y2, x1:x2]
    if roi.size == 0:
        return ""
    scale = 4
    roi_large = cv2.resize(roi, None, fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)
    _, roi_bin = cv2.threshold(roi_large, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return pytesseract.image_to_string(roi_bin, config=OCR_CONFIG)


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    img_path = sys.argv[1]
    x = int(sys.argv[2])
    half_width = int(sys.argv[3]) if len(sys.argv) > 3 else 20

    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: Could not read {img_path}")
        sys.exit(1)

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    x1, x2 = max(0, x - half_width), min(w, x + half_width)

    td = TrainDetector()

    for band_name, band_pct in [("upper", UPPER_TRAIN_BAND), ("lower", LOWER_TRAIN_BAND)]:
        y1, y2 = int(h * band_pct[0]), int(h * band_pct[1])
        raw = ocr_region(gray, y1, y2, x1, x2)
        ids = td._extract_train_ids(raw)

        print(f"\n--- {band_name} band  y={y1}-{y2}  x={x1}-{x2} ---")
        print(f"Raw OCR : {repr(raw)}")
        print(f"IDs     : {ids if ids else '(none)'}")


if __name__ == "__main__":
    main()
