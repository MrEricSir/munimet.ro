"""
Tests for train detection to ensure we don't regress from current baseline.

Current baseline (2026-01-29):
- IMG_9792.jpg: 12 high-confidence trains detected via OCR
- Train-free images: Some false positives expected (tracked as upper bound)
"""

import pytest
import cv2
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.train_detector_v2 import TrainDetectorV2, TESSERACT_AVAILABLE

# Test image paths
IMAGES_DIR = Path(__file__).parent.parent / "artifacts" / "training_data" / "images"
TRAIN_FREE_IMAGES = [
    IMAGES_DIR / "muni_snapshot_20260124_015604.jpg",
    IMAGES_DIR / "muni_snapshot_20260124_041839.jpg",
]

# Images with trains and their expected baseline detections
# Format: (image_path, [(train_id, approximate_x, tolerance), ...])
IMAGES_WITH_BASELINE = [
    (IMAGES_DIR / "IMG_9792.jpg", [
        ("M2062LL", 145, 30),
        ("W2032MM", 260, 30),
        ("I2089MM", 356, 30),
        ("H2162LL", 478, 30),
        ("2037XX", 672, 30),
        ("2011NN", 800, 30),
        ("D2099J", 882, 30),
        ("W2023LL", 961, 30),
        ("2043LL", 1047, 30),
        ("M2066KK", 1175, 30),
        ("B2175TT", 1562, 30),
        ("C2109R", 1855, 30),
    ]),
    (IMAGES_DIR / "IMG_9791.jpg", [
        ("M2089MM", 279, 30),
        ("A2162LL", 352, 30),
        ("W2010LL", 352, 30),
        ("H2032MM", 478, 30),
        ("2011NN", 800, 30),
        ("D2099J", 882, 30),
        ("M2066KK", 967, 30),
        ("W2124MM", 974, 30),
        ("2043LL", 1047, 30),
        ("F2164SS", 1174, 30),
        ("2178NN", 1175, 30),
        ("B2181TT", 1723, 30),
        ("C2175TT", 1726, 30),
    ]),
]

# Maximum allowed false positives in train-free images
# Current baseline: 4 in first image, 16 in second
MAX_FALSE_POSITIVES = {
    "muni_snapshot_20260124_015604.jpg": 10,  # Allow some margin
    "muni_snapshot_20260124_041839.jpg": 20,  # Allow some margin
}


@pytest.fixture
def detector():
    """Create a train detector instance."""
    return TrainDetectorV2()


class TestTrainDetectionBaseline:
    """Tests to ensure we don't regress from current detection baseline."""

    @pytest.mark.skipif(not TESSERACT_AVAILABLE, reason="Tesseract not installed")
    @pytest.mark.parametrize("img_path,baseline", IMAGES_WITH_BASELINE)
    def test_baseline_trains_detected(self, detector, img_path, baseline):
        """Ensure all baseline trains are detected."""
        if not img_path.exists():
            pytest.skip(f"Test image not found: {img_path}")

        img = cv2.imread(str(img_path))
        trains = detector.detect_trains(img)
        detected_ids = {t["id"] for t in trains}

        missing = []
        for train_id, expected_x, tolerance in baseline:
            if train_id not in detected_ids:
                missing.append(train_id)

        assert not missing, f"Missing baseline trains in {img_path.name}: {missing}"

    @pytest.mark.skipif(not TESSERACT_AVAILABLE, reason="Tesseract not installed")
    @pytest.mark.parametrize("img_path,baseline", IMAGES_WITH_BASELINE)
    def test_baseline_train_positions(self, detector, img_path, baseline):
        """Ensure baseline trains are detected at approximately correct positions."""
        if not img_path.exists():
            pytest.skip(f"Test image not found: {img_path}")

        img = cv2.imread(str(img_path))
        trains = detector.detect_trains(img)
        trains_by_id = {t["id"]: t for t in trains}

        position_errors = []
        for train_id, expected_x, tolerance in baseline:
            if train_id in trains_by_id:
                actual_x = trains_by_id[train_id]["x"]
                if abs(actual_x - expected_x) > tolerance:
                    position_errors.append(
                        f"{train_id}: expected x≈{expected_x}, got x={actual_x}"
                    )

        assert not position_errors, f"Position errors in {img_path.name}: {position_errors}"

    @pytest.mark.skipif(not TESSERACT_AVAILABLE, reason="Tesseract not installed")
    def test_train_data_structure(self, detector):
        """Ensure train detections have required fields."""
        img_path = IMAGES_WITH_BASELINE[0][0]
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
    @pytest.mark.parametrize("image_name", [
        "muni_snapshot_20260124_015604.jpg",
        "muni_snapshot_20260124_041839.jpg",
    ])
    def test_false_positives_bounded(self, detector, image_name):
        """Ensure false positives in train-free images stay within bounds."""
        img_path = IMAGES_DIR / image_name
        if not img_path.exists():
            pytest.skip(f"Test image not found: {img_path}")

        img = cv2.imread(str(img_path))
        trains = detector.detect_trains(img)

        max_allowed = MAX_FALSE_POSITIVES.get(image_name, 20)

        assert len(trains) <= max_allowed, (
            f"Too many false positives in {image_name}: "
            f"got {len(trains)}, max allowed {max_allowed}"
        )


class TestDetectorRobustness:
    """Tests for detector robustness and edge cases."""

    def test_detector_without_tesseract(self):
        """Ensure detector handles missing Tesseract gracefully."""
        detector = TrainDetectorV2()
        # Should not raise even if Tesseract is missing
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
        img_path = IMAGES_WITH_BASELINE[0][0]
        if not img_path.exists():
            pytest.skip(f"Test image not found: {img_path}")
        img = cv2.imread(str(img_path))
        trains = detector.detect_trains(img)

        # Check for duplicates at same x position (within 30 pixels)
        for i, t1 in enumerate(trains):
            for t2 in trains[i+1:]:
                if t1["track"] == t2["track"] and abs(t1["x"] - t2["x"]) < 30:
                    # Same track, same position - should not happen
                    assert t1["id"] != t2["id"] or t1["id"].startswith("UNKNOWN"), (
                        f"Duplicate detection: {t1} and {t2}"
                    )


class TestConfidenceLevels:
    """Tests for confidence level assignment."""

    @pytest.mark.skipif(not TESSERACT_AVAILABLE, reason="Tesseract not installed")
    def test_ocr_trains_are_high_confidence(self, detector):
        """Trains with valid OCR IDs should be high confidence."""
        img_path = IMAGES_WITH_BASELINE[0][0]
        if not img_path.exists():
            pytest.skip(f"Test image not found: {img_path}")
        img = cv2.imread(str(img_path))
        trains = detector.detect_trains(img)

        import re
        train_id_pattern = re.compile(r'^[A-Z]?\d{4}[A-Z*X]{1,2}$')

        for train in trains:
            if train_id_pattern.match(train["id"]):
                assert train["confidence"] == "high", (
                    f"Train with valid ID should be high confidence: {train}"
                )

    @pytest.mark.skipif(not TESSERACT_AVAILABLE, reason="Tesseract not installed")
    def test_unknown_trains_are_low_confidence(self, detector):
        """Trains without readable IDs should be low confidence."""
        img_path = IMAGES_WITH_BASELINE[0][0]
        if not img_path.exists():
            pytest.skip(f"Test image not found: {img_path}")
        img = cv2.imread(str(img_path))
        trains = detector.detect_trains(img)

        for train in trains:
            if train["id"].startswith("UNKNOWN"):
                assert train["confidence"] == "low", (
                    f"UNKNOWN train should be low confidence: {train}"
                )


# Utility function to update baseline if detection improves
def print_current_detections():
    """Print current detections for updating baseline."""
    detector = TrainDetectorV2()

    for img_path, _ in IMAGES_WITH_BASELINE:
        print(f"\n=== {img_path.name} ===")
        img = cv2.imread(str(img_path))
        trains = detector.detect_trains(img)

        print("High-confidence detections:")
        print("    [")
        for t in sorted(trains, key=lambda x: x["x"]):
            if t["confidence"] == "high":
                print(f'        ("{t["id"]}", {t["x"]}, 30),')
        print("    ],")

        high_conf = [t for t in trains if t["confidence"] == "high"]
        low_conf = [t for t in trains if t["confidence"] == "low"]
        print(f"Total: {len(high_conf)} high, {len(low_conf)} low confidence")


if __name__ == "__main__":
    # Run this to see current detections for updating baseline
    print_current_detections()
