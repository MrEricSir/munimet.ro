"""
Tests for system status prediction (red/yellow/green).

Compares OpenCV-based detection against known correct statuses.
"""

import pytest
import cv2
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.station_viewer import calculate_system_status, get_detection_data

TESTS_DIR = Path(__file__).parent
IMAGES_DIR = TESTS_DIR.parent / "artifacts" / "training_data" / "images"


# Known correct statuses for specific images
# Format: (image_name, expected_status, notes)
KNOWN_STATUSES = [
    # Normal operation (green)
    ("muni_snapshot_20251209_000943.jpg", "green", "Normal late night operation"),
    ("muni_snapshot_20251206_184417.jpg", "green", "Normal evening operation"),
    ("muni_snapshot_20251206_144619.jpg", "green", "Normal afternoon operation"),
    ("muni_snapshot_20251207_221313.jpg", "green", "Normal operation"),
    ("muni_snapshot_20251228_071410.jpg", "green", "Normal early morning operation"),

    # Delays detected (yellow)
    ("muni_snapshot_20251207_092107.jpg", "yellow", "2 platforms in hold"),
    ("muni_snapshot_20251216_190936.jpg", "yellow", "2 platforms in hold"),
    ("muni_snapshot_20251207_122227.jpg", "yellow", "2 platforms in hold"),
    ("IMG_9794.jpg", "yellow", "Multiple platforms in hold"),

    # Not operating (red)
    ("muni_snapshot_20251207_021407.jpg", "red", "Late night - not operating"),
    ("muni_snapshot_20251207_005046.jpg", "red", "Late night - not operating"),
    ("muni_snapshot_20251208_002133.jpg", "red", "Late night - not operating"),
    # These show trains but system is actually not operating (overnight maintenance display)
    ("muni_snapshot_20251212_002127.jpg", "red", "Not operating - maintenance display"),
    ("muni_snapshot_20251212_002115.jpg", "red", "Not operating - maintenance display"),
    ("muni_snapshot_20251212_002351.jpg", "red", "Not operating - maintenance display"),
    ("muni_snapshot_20260124_020104.jpg", "red", "Not operating - overnight"),
]


class TestSystemStatus:
    """Tests for system status prediction accuracy."""

    @pytest.mark.parametrize("image_name,expected_status,notes", KNOWN_STATUSES)
    def test_known_status(self, image_name, expected_status, notes):
        """Test that known images produce correct status."""
        img_path = IMAGES_DIR / image_name
        if not img_path.exists():
            pytest.skip(f"Test image not found: {img_path}")

        detection = get_detection_data(img_path)
        assert detection is not None, f"Failed to process image: {image_name}"

        actual_status = detection['system_status']

        assert actual_status == expected_status, (
            f"{image_name}: expected '{expected_status}' but got '{actual_status}'. "
            f"Notes: {notes}. "
            f"Detection details: trains={len(detection['trains'])}, "
            f"holds={len(detection['delays_platforms'])}, "
            f"disabled={len(detection['delays_segments'])}"
        )

    def test_status_precedence(self):
        """Test that red > yellow > green precedence is maintained."""
        from scripts.station_viewer import calculate_system_status

        # Mock data for testing precedence
        # Need at least 2 trains with routes for green
        trains_with_routes = [
            {'id': 'W2010LL', 'confidence': 'high'},
            {'id': 'M2089MM', 'confidence': 'high'},
        ]
        trains_no_routes = [{'id': 'UNKNOWN@500', 'confidence': 'low'}]
        # Only 1 valid train - should be red
        trains_one_valid = [
            {'id': 'W2010LL', 'confidence': 'high'},
            {'id': 'UNKNOWN@500', 'confidence': 'low'},
            {'id': 'UNKNOWN@600', 'confidence': 'low'},
        ]
        two_holds = [{'station': 'CT'}, {'station': 'US'}]
        one_hold = [{'station': 'CT'}]
        track_disabled = [{'from': 'CT', 'to': 'US'}]

        # Green: normal operation (>= 2 valid routes, no major delays)
        assert calculate_system_status(trains_with_routes, [], []) == 'green'
        assert calculate_system_status(trains_with_routes, one_hold, []) == 'green'  # 1 hold is OK

        # Yellow: 2+ holds OR disabled tracks
        assert calculate_system_status(trains_with_routes, two_holds, []) == 'yellow'
        assert calculate_system_status(trains_with_routes, [], track_disabled) == 'yellow'
        assert calculate_system_status(trains_with_routes, two_holds, track_disabled) == 'yellow'

        # Red: < 2 trains with route suffixes
        assert calculate_system_status(trains_no_routes, [], []) == 'red'
        assert calculate_system_status(trains_one_valid, [], []) == 'red'
        # Red takes precedence over yellow
        assert calculate_system_status(trains_no_routes, two_holds, track_disabled) == 'red'
        assert calculate_system_status(trains_one_valid, two_holds, track_disabled) == 'red'


class TestStatusCalculationLogic:
    """Tests for the status calculation helper function."""

    def test_empty_trains_is_red(self):
        """With no trains or insufficient valid trains, should be red."""
        from scripts.station_viewer import calculate_system_status

        # No trains -> red (< 2 valid routes)
        assert calculate_system_status([], [], []) == 'red'

        # No trains with delays -> still red (red takes precedence over yellow)
        two_holds = [{'station': 'CT'}, {'station': 'US'}]
        assert calculate_system_status([], two_holds, []) == 'red'

    def test_route_suffix_detection(self):
        """Test that route suffix detection works correctly."""
        from scripts.station_viewer import calculate_system_status

        # Single train with valid suffix - red (need at least 2)
        assert calculate_system_status([{'id': 'W2010LL'}], [], []) == 'red'
        assert calculate_system_status([{'id': 'M2089MM'}], [], []) == 'red'
        assert calculate_system_status([{'id': '2011NN'}], [], []) == 'red'  # No leading letter
        assert calculate_system_status([{'id': 'D2099J'}], [], []) == 'red'  # Single letter suffix

        # Two valid trains - green
        assert calculate_system_status([{'id': 'W2010LL'}, {'id': 'M2089MM'}], [], []) == 'green'

        # 2+ valid trains even with some UNKNOWN - green
        trains_with_unknown = [
            {'id': 'W2010LL'},
            {'id': 'M2089MM'},
            {'id': 'UNKNOWN@500'},
        ]  # 2 valid trains >= 2 threshold
        assert calculate_system_status(trains_with_unknown, [], []) == 'green'

        # Invalid/missing suffixes only (should be red)
        assert calculate_system_status([{'id': 'UNKNOWN@500'}], [], []) == 'red'
        assert calculate_system_status([{'id': 'UNKNOWN@123'}, {'id': 'UNKNOWN@456'}], [], []) == 'red'

        # Only 1 valid train among many UNKNOWN - red
        trains_one_valid = [
            {'id': 'W2010LL'},  # 1 valid
            {'id': 'UNKNOWN@500'},
            {'id': 'UNKNOWN@600'},
            {'id': 'UNKNOWN@700'},
        ]  # Only 1 valid train < 2 threshold
        assert calculate_system_status(trains_one_valid, [], []) == 'red'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
