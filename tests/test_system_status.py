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

from lib.detection import calculate_system_status, detect_system_status, detect_train_bunching

TESTS_DIR = Path(__file__).parent
IMAGES_DIR = TESTS_DIR / "images"


# Known correct statuses for specific images
# Format: (image_name, expected_status, notes)
KNOWN_STATUSES = [
    # Normal operation (green)
    ("muni_snapshot_20251209_000943.jpg", "green", "Normal late night operation"),
    ("muni_snapshot_20251206_184417.jpg", "green", "Normal evening operation"),
    ("muni_snapshot_20251206_144619.jpg", "green", "Normal afternoon operation"),
    ("muni_snapshot_20251207_221313.jpg", "green", "Normal operation"),
    ("muni_snapshot_20251228_071410.jpg", "green", "Normal early morning operation"),
    ("muni_snapshot_20260130_233157.jpg", "green", "Normal operation - 8 trains (3 upper, 5 lower)"),

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

        detection = detect_system_status(str(img_path))
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
        from lib.detection import calculate_system_status

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
        from lib.detection import calculate_system_status

        # No trains -> red (< 2 valid routes)
        assert calculate_system_status([], [], []) == 'red'

        # No trains with delays -> still red (red takes precedence over yellow)
        two_holds = [{'station': 'CT'}, {'station': 'US'}]
        assert calculate_system_status([], two_holds, []) == 'red'

    def test_route_suffix_detection(self):
        """Test that route suffix detection works correctly."""
        from lib.detection import calculate_system_status

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


class TestTrainBunchingDetection:
    """Tests for train bunching detection feature."""

    def test_no_bunching_with_evenly_spaced_trains(self):
        """Evenly spaced trains (>100px apart) should not trigger bunching."""
        from lib.detection import detect_train_bunching

        # Trains evenly spaced across the system (x positions >100px apart)
        trains = [
            {'id': 'W2010LL', 'x': 200, 'track': 'upper'},
            {'id': 'M2089MM', 'x': 500, 'track': 'upper'},
            {'id': 'D2099J', 'x': 900, 'track': 'upper'},
            {'id': 'F2164SS', 'x': 1200, 'track': 'lower'},
            {'id': 'B2181TT', 'x': 800, 'track': 'lower'},
        ]

        bunching = detect_train_bunching(trains)
        assert bunching == [], f"Should detect no bunching, got: {bunching}"

    def test_bunching_detected_upper_track(self):
        """4+ trains clustered close together on upper track should trigger bunching."""
        from lib.detection import detect_train_bunching

        # Powell (PO) is at x=971
        # 4 trains clustered (within 70px of each other) to the right of Powell
        # Cluster is at x=1000-1150, approaching Powell from the east (westbound track)
        trains = [
            {'id': 'W2010LL', 'x': 1000, 'track': 'upper'},
            {'id': 'M2089MM', 'x': 1050, 'track': 'upper'},  # 50px gap
            {'id': 'D2099J', 'x': 1100, 'track': 'upper'},   # 50px gap
            {'id': 'F2164SS', 'x': 1150, 'track': 'upper'},  # 50px gap
        ]

        bunching = detect_train_bunching(trains)
        assert len(bunching) > 0, f"Should detect bunching, got: {bunching}"
        # Cluster is approaching PO (x=971) from the right
        assert any(b['station'] == 'PO' for b in bunching), f"Should detect bunching approaching PO, got: {bunching}"

    def test_bunching_detected_lower_track(self):
        """4+ trains clustered close together on lower track should trigger bunching."""
        from lib.detection import detect_train_bunching

        # Powell (PO) is at x=971
        # 4 trains clustered (within 70px of each other) to the left of Powell
        # Cluster is at x=750-900, approaching Powell from the west (eastbound track)
        trains = [
            {'id': 'W2010LL', 'x': 750, 'track': 'lower'},
            {'id': 'M2089MM', 'x': 800, 'track': 'lower'},  # 50px gap
            {'id': 'D2099J', 'x': 850, 'track': 'lower'},   # 50px gap
            {'id': 'F2164SS', 'x': 900, 'track': 'lower'},  # 50px gap
        ]

        bunching = detect_train_bunching(trains)
        assert len(bunching) > 0, f"Should detect bunching, got: {bunching}"
        # Cluster is approaching PO (x=971) from the left
        assert any(b['station'] == 'PO' for b in bunching), f"Should detect bunching approaching PO, got: {bunching}"

    def test_excluded_stations_ignored(self):
        """Track-specific excluded stations should not report bunching.

        Exclusions by track:
        - Upper track (westbound): CT (Chinatown) is the northern terminus
        - Lower track (eastbound): EM (Embarcadero), MO (Montgomery) are turnaround area
        """
        from lib.detection import detect_train_bunching

        # CT is at x=1564 - excluded for upper track (terminus)
        # Bunch trains right at CT (upper track)
        upper_trains = [
            {'id': 'W2010LL', 'x': 1580, 'track': 'upper'},
            {'id': 'M2089MM', 'x': 1630, 'track': 'upper'},  # 50px gap
            {'id': 'D2099J', 'x': 1680, 'track': 'upper'},   # 50px gap
            {'id': 'F2164SS', 'x': 1730, 'track': 'upper'},  # 50px gap
        ]

        bunching = detect_train_bunching(upper_trains)
        assert not any(b['station'] == 'CT' for b in bunching), f"CT should be excluded for upper track, got: {bunching}"

        # EM is at x=1182 - excluded for lower track (turnaround)
        # Bunch trains right at EM (lower track, approaching from west)
        lower_trains = [
            {'id': 'W2010LL', 'x': 1100, 'track': 'lower'},
            {'id': 'M2089MM', 'x': 1150, 'track': 'lower'},  # 50px gap
            {'id': 'D2099J', 'x': 1200, 'track': 'lower'},   # 50px gap - at EM
            {'id': 'F2164SS', 'x': 1250, 'track': 'lower'},  # 50px gap - past EM
        ]

        bunching = detect_train_bunching(lower_trains)
        assert not any(b['station'] == 'EM' for b in bunching), f"EM should be excluded for lower track, got: {bunching}"
        assert not any(b['station'] == 'MO' for b in bunching), f"MO should be excluded for lower track, got: {bunching}"

    def test_three_trains_not_bunching(self):
        """3 trains clustered should NOT trigger bunching (threshold is 4)."""
        from lib.detection import detect_train_bunching

        # 3 trains clustered near Powell - below threshold
        trains = [
            {'id': 'W2010LL', 'x': 1000, 'track': 'upper'},
            {'id': 'M2089MM', 'x': 1050, 'track': 'upper'},  # 50px gap
            {'id': 'D2099J', 'x': 1100, 'track': 'upper'},   # 50px gap
        ]

        bunching = detect_train_bunching(trains)
        assert bunching == [], f"3 trains should not trigger bunching, got: {bunching}"

    def test_four_trains_spread_out_not_bunching(self):
        """4 trains spread apart (>70px between each) should NOT trigger bunching."""
        from lib.detection import detect_train_bunching

        # 4 trains but spread apart - not clustered (gaps > 70px)
        trains = [
            {'id': 'W2010LL', 'x': 1000, 'track': 'upper'},
            {'id': 'M2089MM', 'x': 1100, 'track': 'upper'},  # 100px gap
            {'id': 'D2099J', 'x': 1200, 'track': 'upper'},   # 100px gap
            {'id': 'F2164SS', 'x': 1300, 'track': 'upper'},  # 100px gap
        ]

        bunching = detect_train_bunching(trains)
        assert bunching == [], f"Spread out trains should not trigger bunching, got: {bunching}"

    def test_bunching_triggers_yellow_status(self):
        """Bunching incidents should trigger yellow system status."""
        from lib.detection import calculate_system_status

        trains_with_routes = [
            {'id': 'W2010LL', 'x': 1000, 'track': 'upper'},
            {'id': 'M2089MM', 'x': 1050, 'track': 'upper'},
            {'id': 'D2099J', 'x': 1100, 'track': 'upper'},
            {'id': 'F2164SS', 'x': 1150, 'track': 'upper'},
        ]

        # With bunching incidents, should be yellow
        bunching = [{'station': 'PO', 'track': 'upper', 'train_count': 4}]
        status = calculate_system_status(trains_with_routes, [], [], bunching)
        assert status == 'yellow', f"Expected yellow with bunching, got {status}"

    def test_bunching_at_embarcadero_upper_track(self):
        """Upper track bunching near Embarcadero should report EM, not Montgomery.

        EM is at x=1182. Trains clustered around x=1167-1325 are at/near Embarcadero,
        not approaching Montgomery (x=1054). Uses absolute distance for upper track.
        """
        from lib.detection import detect_train_bunching

        # Real-world scenario: 5 trains clustered near Embarcadero on upper track
        # EM is at x=1182, MO is at x=1054
        trains = [
            {'id': 'W2174KK', 'x': 1167, 'track': 'upper'},
            {'id': '5109ML', 'x': 1199, 'track': 'upper'},   # 32px gap
            {'id': 'W2148MM', 'x': 1254, 'track': 'upper'},  # 55px gap
            {'id': 'D2054J', 'x': 1273, 'track': 'upper'},   # 19px gap
            {'id': 'W2004KK', 'x': 1325, 'track': 'upper'},  # 52px gap
        ]

        bunching = detect_train_bunching(trains)
        assert len(bunching) == 1, f"Should detect one bunching incident, got: {bunching}"
        assert bunching[0]['station'] == 'EM', f"Should report bunching at EM (nearest to cluster), got: {bunching[0]['station']}"
        assert bunching[0]['train_count'] == 5
        assert bunching[0]['direction'] == 'Westbound'

    def test_bunching_real_image(self):
        """Test bunching detection on real image with visible bunching at Embarcadero."""
        result = detect_system_status(str(IMAGES_DIR / "muni_snapshot_20260203_165648.jpg"))

        # This image shows bunching on upper track near Embarcadero
        bunching = result.get('delays_bunching', [])
        assert len(bunching) > 0, f"Should detect bunching in this image, got none"

        # Should report bunching at Embarcadero (not Montgomery)
        em_bunching = [b for b in bunching if b['station'] == 'EM']
        assert len(em_bunching) > 0, f"Should detect bunching at EM, got: {bunching}"
        assert em_bunching[0]['track'] == 'upper'
        assert em_bunching[0]['direction'] == 'Westbound'


class TestOCRImprovementCandidates:
    """Tests for images where OCR improvements could detect more delays.

    These tests document known cases where bunching or delays are visible
    but not detected due to OCR limitations. They are marked as xfail
    and will pass once OCR is improved.
    """

    @pytest.mark.xfail(reason="Environment-sensitive: passes with some Tesseract versions but not others")
    def test_civic_center_pileup(self):
        """Westbound pileup at Civic Center should be detected as yellow.

        Image shows 4+ trains clustered on the upper track near Civic Center
        (around x=800-970). Selective component-based detection correctly
        identifies trains in this dense pileup area, triggering bunching detection.

        Note: This test is environment-sensitive - OCR results vary between
        Tesseract versions. Passes locally but may fail in Docker.
        """
        result = detect_system_status(str(IMAGES_DIR / "muni-pileup-westbound-at-civic-center.jpg"))

        # Currently detects 19 trains but no route info (all '?')
        # Should detect bunching and return yellow
        assert result['system_status'] == 'yellow', (
            f"Expected yellow due to bunching at Civic Center, got {result['system_status']}. "
            f"Trains detected: {len(result['trains'])}, "
            f"Bunching: {result.get('delays_bunching', [])}"
        )

        # Should have bunching incident near Civic Center (CC) or Van Ness (VN)
        bunching = result.get('delays_bunching', [])
        assert len(bunching) > 0, "Should detect train bunching"
        assert any(b['station'] in ('CC', 'VN', 'PO') for b in bunching), (
            f"Should detect bunching near Civic Center area, got: {bunching}"
        )


class TestDelaySummaryGeneration:
    """Unit tests for delay summary generation logic."""

    def test_single_station_hold(self):
        """Single station hold produces 'delay at X' message."""
        from lib.detection import generate_delay_summaries

        delays = [{'station': 'US', 'name': 'Union Square', 'track': 'upper', 'direction': 'Northbound'}]
        summaries = generate_delay_summaries(delays, [], [])

        assert summaries == ["Northbound delay at Union Square"]

    def test_consecutive_delays_grouped(self):
        """Consecutive station holds produce 'from X to Y' message."""
        from lib.detection import generate_delay_summaries

        delays = [
            {'station': 'CH', 'name': 'Church', 'track': 'upper', 'direction': 'Westbound'},
            {'station': 'CA', 'name': 'Castro', 'track': 'upper', 'direction': 'Westbound'},
            {'station': 'FH', 'name': 'Forest Hill', 'track': 'upper', 'direction': 'Westbound'},
            {'station': 'WE', 'name': 'West Portal', 'track': 'upper', 'direction': 'Westbound'},
        ]
        summaries = generate_delay_summaries(delays, [], [])

        assert summaries == ["Westbound delay from Church to West Portal"]

    def test_red_track_segment(self):
        """Red track segment produces 'service not running' message."""
        from lib.detection import generate_delay_summaries

        segments = [{'from': 'PO', 'to': 'MO', 'direction': 'Eastbound', 'key': 'PO_MO_lower'}]
        summaries = generate_delay_summaries([], segments, [])

        assert summaries == ["Eastbound service not running between Powell and Montgomery"]

    def test_train_bunching(self):
        """Train bunching produces 'backup at X' message."""
        from lib.detection import generate_delay_summaries

        bunching = [{'station': 'PO', 'track': 'upper', 'direction': 'Westbound', 'train_count': 4}]
        summaries = generate_delay_summaries([], [], bunching)

        assert summaries == ["Westbound backup at Powell (4 trains)"]

    def test_empty_delays(self):
        """No delays produces empty list."""
        from lib.detection import generate_delay_summaries

        assert generate_delay_summaries([], [], []) == []


class TestDelaySummaryBaseline:
    """Regression tests for delay summaries against known images.

    Expected results are in tests/baseline_delay_summaries.json
    """

    @pytest.fixture
    def baseline(self):
        """Load baseline from JSON file."""
        import json
        with open(TESTS_DIR / "baseline_delay_summaries.json") as f:
            return json.load(f)["images"]

    def _get_baseline_images(self):
        """Get list of images that have delay summaries (yellow status with delays)."""
        import json
        with open(TESTS_DIR / "baseline_delay_summaries.json") as f:
            data = json.load(f)["images"]
        # Return images that have non-empty delay_summaries
        return [(name, info) for name, info in data.items() if info["delay_summaries"]]

    @pytest.mark.parametrize("image_name,expected", [
        ("IMG_9791.jpg", ["Eastbound delay from Forest Hill to Church", "Eastbound delay at Embarcadero", "Westbound delay from Embarcadero to Van Ness"]),
        ("IMG_9794.jpg", ["Westbound delay from Embarcadero to Van Ness", "Eastbound delay from Montgomery to Embarcadero"]),
        ("muni_snapshot_20251207_092107.jpg", ["Eastbound delay from Civic Center to Powell"]),
        ("muni_snapshot_20251207_122227.jpg", ["Eastbound delay from Castro to Church"]),
        ("muni_snapshot_20251216_190936.jpg", ["Eastbound delay from Powell to Montgomery"]),
    ])
    def test_delay_summaries_match_baseline(self, image_name, expected):
        """Test that delay summaries match baseline for images with delays.

        Note: Bunching summaries (containing 'backup') are excluded from comparison
        because bunching detection depends on OCR accuracy which varies between
        environments (macOS vs Ubuntu CI).
        """
        from lib.detection import detect_system_status

        img_path = IMAGES_DIR / image_name
        if not img_path.exists():
            pytest.skip(f"Test image not found: {img_path}")

        result = detect_system_status(str(img_path))
        actual = result.get('delay_summaries', [])

        # Filter out bunching summaries - these are OCR-dependent and vary between environments
        actual_no_bunching = [s for s in actual if 'backup' not in s]

        assert len(actual_no_bunching) == len(expected), (
            f"{image_name}: expected {len(expected)} summaries, got {len(actual_no_bunching)}. "
            f"Expected: {expected}, Got: {actual_no_bunching}"
        )
        for exp in expected:
            assert exp in actual_no_bunching, (
                f"{image_name}: expected '{exp}' in summaries. Got: {actual_no_bunching}"
            )

    @pytest.mark.parametrize("image_name", [
        "muni_snapshot_20251206_144619.jpg",
        "muni_snapshot_20251206_184417.jpg",
        "muni_snapshot_20251207_221313.jpg",
        "muni_snapshot_20251209_000943.jpg",
        "muni_snapshot_20251228_071410.jpg",
        "muni_snapshot_20260130_233157.jpg",
    ])
    def test_green_status_no_delays(self, image_name):
        """Test that green status images have no delay summaries."""
        from lib.detection import detect_system_status

        img_path = IMAGES_DIR / image_name
        if not img_path.exists():
            pytest.skip(f"Test image not found: {img_path}")

        result = detect_system_status(str(img_path))

        assert result['system_status'] == 'green', f"{image_name}: expected green status"
        assert result.get('delay_summaries', []) == [], f"{image_name}: expected no delay summaries"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
