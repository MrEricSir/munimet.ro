"""
Tests for train detection to ensure we don't regress from current baseline.

Baseline data is stored in baseline_trains.json.
Test images are in tests/images/.
"""

import json
import pytest
import cv2
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.train_detector_v2 import TrainDetectorV2, TESSERACT_AVAILABLE

# Paths
TESTS_DIR = Path(__file__).parent
IMAGES_DIR = TESTS_DIR / "images"
BASELINE_FILE = TESTS_DIR / "baseline_trains.json"

# Load baseline data
with open(BASELINE_FILE) as f:
    BASELINE_DATA = json.load(f)

TOLERANCE = BASELINE_DATA["tolerance"]
IMAGES_WITH_TRAINS = [
    (IMAGES_DIR / name, trains)
    for name, trains in BASELINE_DATA["images_with_trains"].items()
]
TRAIN_FREE_IMAGES = [
    (IMAGES_DIR / name, data["max_false_positives"])
    for name, data in BASELINE_DATA["train_free_images"].items()
]


@pytest.fixture
def detector():
    """Create a train detector instance."""
    return TrainDetectorV2()


class TestTrainDetectionBaseline:
    """Tests to ensure we don't regress from current detection baseline."""

    @pytest.mark.skipif(not TESSERACT_AVAILABLE, reason="Tesseract not installed")
    @pytest.mark.parametrize("img_path,baseline", IMAGES_WITH_TRAINS)
    def test_baseline_trains_detected(self, detector, img_path, baseline):
        """Ensure all baseline trains are detected."""
        if not img_path.exists():
            pytest.skip(f"Test image not found: {img_path}")

        img = cv2.imread(str(img_path))
        trains = detector.detect_trains(img)
        detected_ids = {t["id"] for t in trains}

        missing = [train_id for train_id, _ in baseline if train_id not in detected_ids]
        assert not missing, f"Missing baseline trains in {img_path.name}: {missing}"

    @pytest.mark.skipif(not TESSERACT_AVAILABLE, reason="Tesseract not installed")
    @pytest.mark.parametrize("img_path,baseline", IMAGES_WITH_TRAINS)
    def test_baseline_train_positions(self, detector, img_path, baseline):
        """Ensure baseline trains are detected at approximately correct positions."""
        if not img_path.exists():
            pytest.skip(f"Test image not found: {img_path}")

        img = cv2.imread(str(img_path))
        trains = detector.detect_trains(img)
        trains_by_id = {t["id"]: t for t in trains}

        position_errors = []
        for train_id, expected_x in baseline:
            if train_id in trains_by_id:
                actual_x = trains_by_id[train_id]["x"]
                if abs(actual_x - expected_x) > TOLERANCE:
                    position_errors.append(
                        f"{train_id}: expected x≈{expected_x}, got x={actual_x}"
                    )

        assert not position_errors, f"Position errors in {img_path.name}: {position_errors}"

    @pytest.mark.skipif(not TESSERACT_AVAILABLE, reason="Tesseract not installed")
    def test_train_data_structure(self, detector):
        """Ensure train detections have required fields."""
        img_path = IMAGES_WITH_TRAINS[0][0]
        if not img_path.exists():
            pytest.skip(f"Test image not found: {img_path}")

        img = cv2.imread(str(img_path))
        trains = detector.detect_trains(img)

        required_fields = ["id", "x", "y", "track", "confidence"]

        for train in trains:
            for field in required_fields:
                assert field in train, f"Train missing field '{field}': {train}"

            assert train["track"] in ("upper", "lower"), (
                f"Invalid track value: {train['track']}"
            )
            assert train["confidence"] in ("high", "medium", "low"), (
                f"Invalid confidence value: {train['confidence']}"
            )


class TestFalsePositives:
    """Tests to ensure false positives don't increase."""

    @pytest.mark.skipif(not TESSERACT_AVAILABLE, reason="Tesseract not installed")
    @pytest.mark.parametrize("img_path,max_allowed", TRAIN_FREE_IMAGES)
    def test_false_positives_bounded(self, detector, img_path, max_allowed):
        """Ensure false positives in train-free images stay within bounds."""
        if not img_path.exists():
            pytest.skip(f"Test image not found: {img_path}")

        img = cv2.imread(str(img_path))
        trains = detector.detect_trains(img)

        assert len(trains) <= max_allowed, (
            f"Too many false positives in {img_path.name}: "
            f"got {len(trains)}, max allowed {max_allowed}"
        )


class TestDetectorRobustness:
    """Tests for detector robustness and edge cases."""

    def test_detector_without_tesseract(self):
        """Ensure detector handles missing Tesseract gracefully."""
        detector = TrainDetectorV2()
        assert detector is not None

    @pytest.mark.skipif(not TESSERACT_AVAILABLE, reason="Tesseract not installed")
    def test_empty_image(self, detector):
        """Ensure detector handles empty/blank images."""
        import numpy as np
        blank = np.zeros((800, 1860, 3), dtype=np.uint8)
        trains = detector.detect_trains(blank)
        assert isinstance(trains, list)

    @pytest.mark.skipif(not TESSERACT_AVAILABLE, reason="Tesseract not installed")
    def test_small_image(self, detector):
        """Ensure detector handles small images without crashing."""
        import numpy as np
        small = np.zeros((100, 100, 3), dtype=np.uint8)
        trains = detector.detect_trains(small)
        assert isinstance(trains, list)

    @pytest.mark.skipif(not TESSERACT_AVAILABLE, reason="Tesseract not installed")
    def test_no_duplicate_detections(self, detector):
        """Ensure no duplicate train IDs at same position."""
        img_path = IMAGES_WITH_TRAINS[0][0]
        if not img_path.exists():
            pytest.skip(f"Test image not found: {img_path}")
        img = cv2.imread(str(img_path))
        trains = detector.detect_trains(img)

        for i, t1 in enumerate(trains):
            for t2 in trains[i+1:]:
                if t1["track"] == t2["track"] and abs(t1["x"] - t2["x"]) < 30:
                    assert t1["id"] != t2["id"] or t1["id"].startswith("UNKNOWN"), (
                        f"Duplicate detection: {t1} and {t2}"
                    )


class TestConfidenceLevels:
    """Tests for confidence level assignment."""

    @pytest.mark.skipif(not TESSERACT_AVAILABLE, reason="Tesseract not installed")
    def test_ocr_trains_are_high_or_medium_confidence(self, detector):
        """Trains with valid OCR IDs should be high or medium confidence."""
        img_path = IMAGES_WITH_TRAINS[0][0]
        if not img_path.exists():
            pytest.skip(f"Test image not found: {img_path}")
        img = cv2.imread(str(img_path))
        trains = detector.detect_trains(img)

        import re
        train_id_pattern = re.compile(r'^[A-Z]?\d{4}[A-Z*X]{1,2}$')

        for train in trains:
            if train_id_pattern.match(train["id"]):
                # Valid IDs can be high (direct OCR) or medium (symbol + recovered OCR)
                assert train["confidence"] in ("high", "medium"), (
                    f"Train with valid ID should be high/medium confidence: {train}"
                )

    @pytest.mark.skipif(not TESSERACT_AVAILABLE, reason="Tesseract not installed")
    def test_unknown_trains_are_low_confidence(self, detector):
        """Trains without readable IDs should be low confidence."""
        img_path = IMAGES_WITH_TRAINS[0][0]
        if not img_path.exists():
            pytest.skip(f"Test image not found: {img_path}")
        img = cv2.imread(str(img_path))
        trains = detector.detect_trains(img)

        for train in trains:
            if train["id"].startswith("UNKNOWN"):
                assert train["confidence"] == "low", (
                    f"UNKNOWN train should be low confidence: {train}"
                )


if __name__ == "__main__":
    # Print current detections for updating baseline
    detector = TrainDetectorV2()
    for img_path, baseline in IMAGES_WITH_TRAINS:
        print(f"\n=== {img_path.name} ===")
        img = cv2.imread(str(img_path))
        trains = detector.detect_trains(img)
        high_conf = [t for t in trains if t["confidence"] == "high"]
        print(f"Baseline expects: {len(baseline)} trains")
        print(f"Currently detecting: {len(high_conf)} high-confidence trains")
        for t in sorted(high_conf, key=lambda x: x["x"]):
            print(f'  ["{t["id"]}", {t["x"]}],')
