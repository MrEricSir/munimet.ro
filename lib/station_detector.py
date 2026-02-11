#!/usr/bin/env python3
"""
Robust station and track detection using color-based detection with caching.

This module provides a more reliable alternative to text-based station label
detection by identifying station platforms and track lines by their distinctive
colors (blue/yellow platforms, cyan/red tracks).

Positions are cached since the display layout is stable, with fallback to
text detection when color detection fails.
"""

import cv2
import numpy as np
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# Import shared constants
from lib.station_constants import get_section_directions, INTERNAL_STATIONS

# Import from centralized config and re-export for backward compatibility
from lib.config import (
    # Image dimensions
    REFERENCE_WIDTH as REFERENCE_IMAGE_WIDTH,
    REFERENCE_HEIGHT as REFERENCE_IMAGE_HEIGHT,
    # Y-position percentages
    UPPER_LABEL_Y_PCT,
    LOWER_LABEL_Y_PCT,
    UPPER_TRACK_Y_PCT,
    LOWER_TRACK_Y_PCT,
    TRACK_HEIGHT_PCT,
    Y_BANDS,
    # HSV color ranges
    HSV_RANGES,
    # Detection thresholds
    PLATFORM_MIN_AREA,
    PLATFORM_MAX_AREA,
    PLATFORM_MIN_WIDTH,
    PLATFORM_MAX_WIDTH,
    PLATFORM_MIN_HEIGHT,
    PLATFORM_MAX_HEIGHT,
    STATION_CLUSTER_THRESHOLD,
)

# Station X-positions (center of each station) at reference resolution
# These are specific to the Muni display layout and don't belong in generic config
STATION_X_POSITIONS = {
    'WE': 36,
    'FH': 178,
    'CA': 362,
    'CH': 485,
    'VN': 802,   # Moved from 540 (was maintenance platform)
    'CC': 879,
    'PO': 971,
    'MO': 1054,
    'EM': 1182,
    'MN': 1280,  # internal
    'FP': 1380,  # internal
    'TT': 1470,  # internal
    'CT': 1564,
    'US': 1732,
    'YB': 1814,
}

# Segments that should NOT be created (tracks not connected)
DISCONNECTED_SEGMENTS = {('EM', 'CT')}


class ColorDetector:
    """HSV color-based detection of platforms and tracks."""

    def detect_platforms_in_band(
        self,
        hsv: np.ndarray,
        y_min: int,
        y_max: int,
        img_width: int
    ) -> List[Dict]:
        """
        Find blue and yellow platform rectangles in a Y-band.

        Args:
            hsv: HSV image array.
            y_min: Top of search band (pixels).
            y_max: Bottom of search band (pixels).
            img_width: Image width for bounds checking.

        Returns:
            List of platform dicts with x, y, w, h, cx, state ('blue'/'yellow').
        """
        # Extract the Y-band region
        band = hsv[y_min:y_max, :]

        # Detect blue platforms
        blue_mask = cv2.inRange(
            band,
            HSV_RANGES['platform_blue']['lower'],
            HSV_RANGES['platform_blue']['upper']
        )

        # Detect yellow platforms
        yellow_mask = cv2.inRange(
            band,
            HSV_RANGES['platform_yellow']['lower'],
            HSV_RANGES['platform_yellow']['upper']
        )

        platforms = []

        # Process blue platforms
        blue_platforms = self._find_platform_contours(blue_mask, y_min, 'blue')
        platforms.extend(blue_platforms)

        # Process yellow platforms
        yellow_platforms = self._find_platform_contours(yellow_mask, y_min, 'yellow')
        platforms.extend(yellow_platforms)

        # Sort by x-position
        platforms.sort(key=lambda p: p['cx'])

        return platforms

    def _find_platform_contours(
        self,
        mask: np.ndarray,
        y_offset: int,
        state: str
    ) -> List[Dict]:
        """Find platform rectangles in a binary mask."""
        # Clean up noise
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(
            cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        platforms = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if not (PLATFORM_MIN_AREA <= area <= PLATFORM_MAX_AREA):
                continue

            x, y, w, h = cv2.boundingRect(cnt)

            # Filter by size
            if not (PLATFORM_MIN_WIDTH <= w <= PLATFORM_MAX_WIDTH):
                continue
            if not (PLATFORM_MIN_HEIGHT <= h <= PLATFORM_MAX_HEIGHT):
                continue

            platforms.append({
                'x': x,
                'y': y + y_offset,  # Adjust for band offset
                'w': w,
                'h': h,
                'cx': x + w // 2,
                'cy': y + y_offset + h // 2,
                'area': area,
                'state': state,
            })

        return platforms

    def detect_track_lines(
        self,
        hsv: np.ndarray,
        y_min: int,
        y_max: int
    ) -> Dict:
        """
        Detect track line positions and colors in a Y-band.

        Args:
            hsv: HSV image array.
            y_min: Top of search band.
            y_max: Bottom of search band.

        Returns:
            Dict with 'cyan_mask', 'red_mask', 'track_y' (detected track Y-position).
        """
        band = hsv[y_min:y_max, :]

        # Detect cyan track
        cyan_mask = cv2.inRange(
            band,
            HSV_RANGES['track_cyan']['lower'],
            HSV_RANGES['track_cyan']['upper']
        )

        # Detect red track (combine both hue ranges)
        red_mask_low = cv2.inRange(
            band,
            HSV_RANGES['track_red_low']['lower'],
            HSV_RANGES['track_red_low']['upper']
        )
        red_mask_high = cv2.inRange(
            band,
            HSV_RANGES['track_red_high']['lower'],
            HSV_RANGES['track_red_high']['upper']
        )
        red_mask = red_mask_low | red_mask_high

        # Find track Y-position by looking for horizontal line
        combined_mask = cyan_mask | red_mask
        row_sums = np.sum(combined_mask, axis=1)
        if row_sums.max() > 0:
            track_row = np.argmax(row_sums)
            track_y = y_min + track_row
        else:
            track_y = (y_min + y_max) // 2  # Default to center of band

        return {
            'cyan_mask': cyan_mask,
            'red_mask': red_mask,
            'track_y': track_y,
            'y_min': y_min,
            'y_max': y_max,
        }

    def classify_platform_state(
        self,
        hsv: np.ndarray,
        rect: Dict
    ) -> str:
        """
        Classify a platform rectangle by dominant color.

        Args:
            hsv: HSV image array.
            rect: Dict with x, y, w, h.

        Returns:
            'normal' (blue), 'holding' (yellow), or 'unknown'.
        """
        if rect is None:
            return 'unknown'

        x, y, w, h = rect['x'], rect['y'], rect['w'], rect['h']

        # Bounds check
        img_h, img_w = hsv.shape[:2]
        if x < 0 or y < 0 or x + w > img_w or y + h > img_h:
            return 'unknown'

        roi = hsv[y:y+h, x:x+w]
        if roi.size == 0:
            return 'unknown'

        # Count yellow pixels
        yellow_mask = cv2.inRange(
            roi,
            HSV_RANGES['platform_yellow']['lower'],
            HSV_RANGES['platform_yellow']['upper']
        )
        yellow_ratio = np.count_nonzero(yellow_mask) / roi.size

        # Count blue pixels
        blue_mask = cv2.inRange(
            roi,
            HSV_RANGES['platform_blue']['lower'],
            HSV_RANGES['platform_blue']['upper']
        )
        blue_ratio = np.count_nonzero(blue_mask) / roi.size

        if yellow_ratio > 0.3:
            return 'holding'
        elif blue_ratio > 0.3:
            return 'normal'
        else:
            return 'unknown'

    def detect_track_segment_color(
        self,
        hsv: np.ndarray,
        bounds: Dict
    ) -> Dict:
        """
        Detect color state of a track segment.

        Args:
            hsv: HSV image array.
            bounds: Dict with x_min, x_max, y_min, y_max.

        Returns:
            Dict with color ('cyan', 'red', 'mixed'), red_ratio, cyan_ratio.
        """
        x_min = bounds['x_min']
        x_max = bounds['x_max']
        y_min = bounds['y_min']
        y_max = bounds['y_max']

        # Bounds check
        img_h, img_w = hsv.shape[:2]
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(img_w, x_max)
        y_max = min(img_h, y_max)

        if x_max <= x_min or y_max <= y_min:
            return {'color': 'unknown', 'red_ratio': 0.0, 'cyan_ratio': 0.0}

        roi = hsv[y_min:y_max, x_min:x_max]

        # Detect cyan
        cyan_mask = cv2.inRange(
            roi,
            HSV_RANGES['track_cyan']['lower'],
            HSV_RANGES['track_cyan']['upper']
        )
        cyan_ratio = np.count_nonzero(cyan_mask) / max(roi.size, 1)

        # Detect red
        red_mask_low = cv2.inRange(
            roi,
            HSV_RANGES['track_red_low']['lower'],
            HSV_RANGES['track_red_low']['upper']
        )
        red_mask_high = cv2.inRange(
            roi,
            HSV_RANGES['track_red_high']['lower'],
            HSV_RANGES['track_red_high']['upper']
        )
        red_mask = red_mask_low | red_mask_high
        red_ratio = np.count_nonzero(red_mask) / max(roi.size, 1)

        # Classify
        if red_ratio > 0.15:
            color = 'red'
        elif cyan_ratio > 0.15:
            color = 'cyan'
        else:
            color = 'mixed'

        return {
            'color': color,
            'red_ratio': red_ratio,
            'cyan_ratio': cyan_ratio,
        }


class PositionCache:
    """Manages the station position cache file."""

    def __init__(self, cache_path: str):
        """
        Initialize the position cache.

        Args:
            cache_path: Path to the JSON cache file.
        """
        self.cache_path = Path(cache_path)
        self.data: Optional[Dict] = None

    def load(self) -> bool:
        """
        Load cache from disk.

        Returns:
            True if loaded successfully, False otherwise.
        """
        if not self.cache_path.exists():
            return False

        try:
            with open(self.cache_path, 'r') as f:
                self.data = json.load(f)
            return True
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not load cache: {e}")
            return False

    def save(self) -> bool:
        """
        Save cache to disk.

        Returns:
            True if saved successfully, False otherwise.
        """
        if self.data is None:
            return False

        try:
            # Ensure directory exists
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)

            # Convert numpy types to Python native types for JSON serialization
            serializable_data = self._make_serializable(self.data)

            with open(self.cache_path, 'w') as f:
                json.dump(serializable_data, f, indent=2)
            return True
        except IOError as e:
            print(f"Warning: Could not save cache: {e}")
            return False

    def _make_serializable(self, obj):
        """Convert numpy types to Python native types for JSON serialization."""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

    def is_valid(self, width: int, height: int) -> bool:
        """
        Check if cache is valid for given image dimensions.

        Args:
            width: Image width.
            height: Image height.

        Returns:
            True if cache is valid, False otherwise.
        """
        if self.data is None:
            return False

        dims = self.data.get('image_dimensions', {})
        if dims.get('width') != width or dims.get('height') != height:
            return False

        # Check confidence
        if self.data.get('confidence_score', 0) < 0.7:
            return False

        return True

    def get_station(self, code: str) -> Optional[Dict]:
        """
        Get cached position for a station.

        Args:
            code: Station code (e.g., 'WE', 'FH').

        Returns:
            Station position dict or None if not found.
        """
        if self.data is None:
            return None

        stations = self.data.get('stations', {})
        return stations.get(code)

    def get_all_stations(self) -> Dict:
        """Get all cached station positions."""
        if self.data is None:
            return {}
        return self.data.get('stations', {})

    def get_track_segments(self) -> Dict:
        """Get all cached track segment bounds."""
        if self.data is None:
            return {}
        return self.data.get('track_segments', {})

    def update(self, calibration_data: Dict):
        """Update cache with new calibration data."""
        self.data = calibration_data


class StationDetector:
    """Main detector combining color detection and caching."""

    def __init__(self, cache_path: Optional[str] = None):
        """
        Initialize the station detector.

        Args:
            cache_path: Path to cache file. If None, uses default location.
        """
        if cache_path is None:
            # Default cache location
            cache_path = str(
                Path(__file__).parent.parent /
                "artifacts" / "runtime" / "cache" / "station_positions.json"
            )

        self.color_detector = ColorDetector()
        self.cache = PositionCache(cache_path)
        self._cache_loaded = False

    def get_hardcoded_positions(
        self,
        img_width: int,
        img_height: int,
        station_order: List[Tuple[str, str]]
    ) -> Dict:
        """
        Generate station positions from hardcoded reference values.

        Scales positions proportionally if image size differs from reference.

        Args:
            img_width: Actual image width.
            img_height: Actual image height.
            station_order: List of (code, name) tuples.

        Returns:
            Dict with stations and track_segments in cache format.
        """
        # Scale factors for different image sizes
        x_scale = img_width / REFERENCE_IMAGE_WIDTH
        y_scale = img_height / REFERENCE_IMAGE_HEIGHT

        # Calculate track Y positions
        upper_track_y = int(img_height * UPPER_TRACK_Y_PCT)
        lower_track_y = int(img_height * LOWER_TRACK_Y_PCT)
        track_height = int(img_height * TRACK_HEIGHT_PCT)

        # Calculate label Y positions (for green station markers)
        upper_label_y = int(img_height * UPPER_LABEL_Y_PCT)
        lower_label_y = int(img_height * LOWER_LABEL_Y_PCT)

        # Build stations dict
        stations = {}
        for code, name in station_order:
            base_x = STATION_X_POSITIONS.get(code, 0)
            scaled_x = int(base_x * x_scale)

            stations[code] = {
                'code': code,
                'name': name,
                'center_x': scaled_x,
                'upper_label_y': upper_label_y,
                'lower_label_y': lower_label_y,
                'upper_platform': None,
                'lower_platform': None,
            }

        # Build track segments
        track_segments = {}

        # Filter to only visible stations (exclude internal)
        visible_stations = [
            (code, name) for code, name in station_order
            if code not in INTERNAL_STATIONS
        ]

        # Track spans nearly the full image width (measured: x=18 to x=1855 at 1860px)
        track_start_x = int(18 * x_scale)
        track_end_x = int(1855 * x_scale)

        for i in range(len(visible_stations) - 1):
            from_code, _ = visible_stations[i]
            to_code, _ = visible_stations[i + 1]

            # Skip disconnected segments (tracks not physically connected)
            if (from_code, to_code) in DISCONNECTED_SEGMENTS:
                continue

            from_x = stations[from_code]['center_x']
            to_x = stations[to_code]['center_x']

            if from_x >= to_x:
                continue

            # Segment boundaries: from station center to next station center
            # Extend first segment left to track start, last segment right to track end
            seg_x_min = track_start_x if i == 0 else from_x
            seg_x_max = track_end_x if i == len(visible_stations) - 2 else to_x

            upper_dir, lower_dir = get_section_directions(from_code, to_code)

            # Upper track segment
            upper_key = f"{from_code}_{to_code}_upper"
            track_segments[upper_key] = {
                'from_code': from_code,
                'to_code': to_code,
                'direction': upper_dir,
                'bounds': {
                    'x_min': seg_x_min,
                    'x_max': seg_x_max,
                    'y_min': upper_track_y - track_height // 2,
                    'y_max': upper_track_y + track_height // 2,
                },
            }

            # Lower track segment
            lower_key = f"{from_code}_{to_code}_lower"
            track_segments[lower_key] = {
                'from_code': from_code,
                'to_code': to_code,
                'direction': lower_dir,
                'bounds': {
                    'x_min': seg_x_min,
                    'x_max': seg_x_max,
                    'y_min': lower_track_y - track_height // 2,
                    'y_max': lower_track_y + track_height // 2,
                },
            }

        return {
            'version': '1.0',
            'image_dimensions': {'width': img_width, 'height': img_height},
            'calibrated_at': datetime.now().isoformat(),
            'confidence_score': 1.0,  # Hardcoded positions are 100% confident
            'stations': stations,
            'track_segments': track_segments,
            'detection_parameters': {
                'source': 'hardcoded',
                'upper_track_y': upper_track_y,
                'lower_track_y': lower_track_y,
            },
        }

    def _ensure_cache_loaded(self):
        """Load cache if not already loaded."""
        if not self._cache_loaded:
            self.cache.load()
            self._cache_loaded = True

    def calibrate(
        self,
        image: np.ndarray,
        station_order: List[Tuple[str, str]]
    ) -> Dict:
        """
        Run full calibration to detect all station positions.

        Args:
            image: BGR image array.
            station_order: List of (code, name) tuples in left-to-right order.

        Returns:
            Calibration data dict suitable for caching.
        """
        h, w = image.shape[:2]
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Calculate Y-band pixel positions
        upper_platform_y = (
            int(h * Y_BANDS['upper_platform'][0]),
            int(h * Y_BANDS['upper_platform'][1])
        )
        lower_platform_y = (
            int(h * Y_BANDS['lower_platform'][0]),
            int(h * Y_BANDS['lower_platform'][1])
        )
        upper_track_y = (
            int(h * Y_BANDS['upper_track'][0]),
            int(h * Y_BANDS['upper_track'][1])
        )
        lower_track_y = (
            int(h * Y_BANDS['lower_track'][0]),
            int(h * Y_BANDS['lower_track'][1])
        )

        # Detect platforms in each band
        upper_platforms = self.color_detector.detect_platforms_in_band(
            hsv, upper_platform_y[0], upper_platform_y[1], w
        )
        lower_platforms = self.color_detector.detect_platforms_in_band(
            hsv, lower_platform_y[0], lower_platform_y[1], w
        )

        # Detect track lines
        upper_track = self.color_detector.detect_track_lines(
            hsv, upper_track_y[0], upper_track_y[1]
        )
        lower_track = self.color_detector.detect_track_lines(
            hsv, lower_track_y[0], lower_track_y[1]
        )

        # Cluster platforms by X-position to identify station columns
        upper_clusters = self._cluster_platforms(upper_platforms)
        lower_clusters = self._cluster_platforms(lower_platforms)

        # Match clusters to station order
        stations = self._match_clusters_to_stations(
            upper_clusters, lower_clusters, station_order, w
        )

        # Calculate confidence
        expected = len(station_order)
        upper_found = sum(
            1 for s in stations.values()
            if s.get('upper_platform') is not None
        )
        lower_found = sum(
            1 for s in stations.values()
            if s.get('lower_platform') is not None
        )
        confidence = (upper_found + lower_found) / (expected * 2)

        # Derive track segments from station positions
        track_segments = self._derive_track_segments(
            stations, station_order, upper_track_y, lower_track_y
        )

        # Build cache data
        cache_data = {
            'version': '1.0',
            'image_dimensions': {'width': w, 'height': h},
            'calibrated_at': datetime.now().isoformat(),
            'confidence_score': confidence,
            'stations': stations,
            'track_segments': track_segments,
            'detection_parameters': {
                'upper_platform_y_band': list(Y_BANDS['upper_platform']),
                'lower_platform_y_band': list(Y_BANDS['lower_platform']),
                'upper_track_y_band': list(Y_BANDS['upper_track']),
                'lower_track_y_band': list(Y_BANDS['lower_track']),
                'upper_track_y_actual': upper_track['track_y'],
                'lower_track_y_actual': lower_track['track_y'],
            },
        }

        # Update cache
        self.cache.update(cache_data)
        self.cache.save()

        return cache_data

    def _cluster_platforms(
        self,
        platforms: List[Dict]
    ) -> List[List[Dict]]:
        """
        Cluster platforms by X-position.

        Platforms within STATION_CLUSTER_THRESHOLD pixels are grouped together.
        """
        if not platforms:
            return []

        clusters = []
        current_cluster = [platforms[0]]

        for platform in platforms[1:]:
            if platform['cx'] - current_cluster[-1]['cx'] <= STATION_CLUSTER_THRESHOLD:
                current_cluster.append(platform)
            else:
                clusters.append(current_cluster)
                current_cluster = [platform]

        clusters.append(current_cluster)
        return clusters

    def _match_clusters_to_stations(
        self,
        upper_clusters: List[List[Dict]],
        lower_clusters: List[List[Dict]],
        station_order: List[Tuple[str, str]],
        img_width: int
    ) -> Dict:
        """
        Match detected platform clusters to known station order.

        Uses proportional positioning when cluster count doesn't match.
        """
        stations = {}

        # Calculate expected X-positions based on station order
        num_stations = len(station_order)

        # Get cluster center X-positions
        upper_xs = [
            sum(p['cx'] for p in c) / len(c)
            for c in upper_clusters
        ] if upper_clusters else []
        lower_xs = [
            sum(p['cx'] for p in c) / len(c)
            for c in lower_clusters
        ] if lower_clusters else []

        for i, (code, name) in enumerate(station_order):
            # Calculate expected X-position (proportional)
            expected_x = int((i + 0.5) * img_width / num_stations)

            # Find closest upper cluster
            upper_platform = None
            if upper_xs:
                distances = [abs(x - expected_x) for x in upper_xs]
                min_idx = distances.index(min(distances))
                if distances[min_idx] < img_width / num_stations:
                    # Use the first platform in the cluster
                    upper_platform = upper_clusters[min_idx][0].copy()

            # Find closest lower cluster
            lower_platform = None
            if lower_xs:
                distances = [abs(x - expected_x) for x in lower_xs]
                min_idx = distances.index(min(distances))
                if distances[min_idx] < img_width / num_stations:
                    lower_platform = lower_clusters[min_idx][0].copy()

            # Calculate station center X
            if upper_platform and lower_platform:
                center_x = (upper_platform['cx'] + lower_platform['cx']) // 2
            elif upper_platform:
                center_x = upper_platform['cx']
            elif lower_platform:
                center_x = lower_platform['cx']
            else:
                center_x = expected_x

            stations[code] = {
                'code': code,
                'name': name,
                'center_x': center_x,
                'upper_platform': upper_platform,
                'lower_platform': lower_platform,
            }

        return stations

    def _derive_track_segments(
        self,
        stations: Dict,
        station_order: List[Tuple[str, str]],
        upper_track_y: Tuple[int, int],
        lower_track_y: Tuple[int, int]
    ) -> Dict:
        """Derive track segment bounds from station positions."""
        segments = {}
        codes = [code for code, _ in station_order]

        for i in range(len(codes) - 1):
            from_code = codes[i]
            to_code = codes[i + 1]

            # Skip internal stations
            if from_code in INTERNAL_STATIONS or to_code in INTERNAL_STATIONS:
                continue

            from_station = stations.get(from_code, {})
            to_station = stations.get(to_code, {})

            x_min = from_station.get('center_x', 0)
            x_max = to_station.get('center_x', 0)

            if x_min >= x_max:
                continue

            upper_dir, lower_dir = get_section_directions(from_code, to_code)

            # Upper track segment
            upper_key = f"{from_code}_{to_code}_upper"
            segments[upper_key] = {
                'from_code': from_code,
                'to_code': to_code,
                'direction': upper_dir,
                'bounds': {
                    'x_min': x_min,
                    'x_max': x_max,
                    'y_min': upper_track_y[0],
                    'y_max': upper_track_y[1],
                },
            }

            # Lower track segment
            lower_key = f"{from_code}_{to_code}_lower"
            segments[lower_key] = {
                'from_code': from_code,
                'to_code': to_code,
                'direction': lower_dir,
                'bounds': {
                    'x_min': x_min,
                    'x_max': x_max,
                    'y_min': lower_track_y[0],
                    'y_max': lower_track_y[1],
                },
            }

        return segments

    def should_recalibrate(
        self,
        image: np.ndarray
    ) -> Tuple[bool, str]:
        """
        Determine if cache should be rebuilt.

        Returns:
            (should_recalibrate, reason)
        """
        self._ensure_cache_loaded()

        if self.cache.data is None:
            return True, "No cache file exists"

        h, w = image.shape[:2]
        if not self.cache.is_valid(w, h):
            dims = self.cache.data.get('image_dimensions', {})
            if dims.get('width') != w or dims.get('height') != h:
                return True, f"Image dimensions changed: {w}x{h}"
            return True, "Low confidence score in cache"

        return False, "Cache is valid"

    def detect_with_cache(
        self,
        image: np.ndarray,
        station_order: List[Tuple[str, str]],
        force_recalibrate: bool = False,
        use_hardcoded: bool = True
    ) -> Dict:
        """
        Detect stations using cache with fallback to hardcoded positions.

        Args:
            image: BGR image array.
            station_order: List of (code, name) tuples.
            force_recalibrate: Force cache rebuild via color detection.
            use_hardcoded: Use hardcoded positions (default True, most reliable).

        Returns:
            Detection result with station positions and track segments.
        """
        h, w = image.shape[:2]

        # Option 1: Use hardcoded positions (most reliable)
        if use_hardcoded and not force_recalibrate:
            hardcoded = self.get_hardcoded_positions(w, h, station_order)
            return {
                'stations': hardcoded['stations'],
                'track_segments': hardcoded['track_segments'],
                'confidence': 1.0,
                'cache_used': False,
                'detection_method': 'hardcoded',
            }

        # Option 2: Try color-based calibration
        should_recal, reason = self.should_recalibrate(image)

        if force_recalibrate or should_recal:
            print(f"Calibrating: {reason}")
            self.calibrate(image, station_order)

        self._ensure_cache_loaded()

        # Check if cache has good confidence
        cache_confidence = self.cache.data.get('confidence_score', 0) if self.cache.data else 0

        if cache_confidence >= 0.7:
            # Use cached data
            return {
                'stations': self.cache.get_all_stations(),
                'track_segments': self.cache.get_track_segments(),
                'confidence': cache_confidence,
                'cache_used': True,
                'detection_method': 'cache',
            }
        else:
            # Fall back to hardcoded positions
            hardcoded = self.get_hardcoded_positions(w, h, station_order)
            return {
                'stations': hardcoded['stations'],
                'track_segments': hardcoded['track_segments'],
                'confidence': 1.0,
                'cache_used': False,
                'detection_method': 'hardcoded',
            }

    def detect_platform_states(
        self,
        hsv: np.ndarray
    ) -> Dict[str, Dict[str, str]]:
        """
        Detect current state of each platform.

        Returns:
            Dict of station_code -> {'upper': state, 'lower': state}
            where state is 'normal', 'holding', or 'unknown'.
        """
        self._ensure_cache_loaded()
        stations = self.cache.get_all_stations()

        states = {}
        for code, station in stations.items():
            upper_state = 'unknown'
            lower_state = 'unknown'

            upper_platform = station.get('upper_platform')
            if upper_platform:
                upper_state = self.color_detector.classify_platform_state(
                    hsv, upper_platform
                )

            lower_platform = station.get('lower_platform')
            if lower_platform:
                lower_state = self.color_detector.classify_platform_state(
                    hsv, lower_platform
                )

            states[code] = {
                'upper': upper_state,
                'lower': lower_state,
            }

        return states

    def detect_track_colors(
        self,
        hsv: np.ndarray
    ) -> Dict[str, Dict]:
        """
        Detect color state of each track segment.

        Returns:
            Dict of segment_key -> {'color': str, 'red_ratio': float, 'cyan_ratio': float}
        """
        self._ensure_cache_loaded()
        segments = self.cache.get_track_segments()

        results = {}
        for seg_key, segment in segments.items():
            bounds = segment.get('bounds', {})
            color_info = self.color_detector.detect_track_segment_color(hsv, bounds)
            results[seg_key] = {
                **color_info,
                'from_code': segment.get('from_code'),
                'to_code': segment.get('to_code'),
                'direction': segment.get('direction'),
            }

        return results
