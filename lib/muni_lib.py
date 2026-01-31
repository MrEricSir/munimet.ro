#!/usr/bin/env python3
"""
Shared library for SF Muni Central image processing.

Contains reusable functions for:
- Downloading and validating Muni subway status images
- Detecting status from images using OpenCV-based analysis
- Caching status data locally or in Google Cloud Storage
- Posting status updates to Bluesky
"""

import os
import requests
import json
from datetime import datetime
from pathlib import Path
from PIL import Image

# Load .env file for local development (not on Cloud Run)
if not os.getenv('CLOUD_RUN'):
    try:
        from dotenv import load_dotenv
        _env_path = Path(__file__).resolve().parent.parent / '.env'
        if _env_path.exists():
            load_dotenv(_env_path)
    except ImportError:
        pass  # python-dotenv not installed, skip


# Path resolution - get absolute paths relative to project root
# This works regardless of where the script is run from
LIB_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = LIB_DIR.parent

# Configuration constants
WEBPAGE_URL = "http://sfmunicentral.com/Enterprise/MetroLO.htm"
IMAGE_ID = "snapshotImage"
WAIT_TIME = 10
EXPECTED_WIDTH = 1860
EXPECTED_HEIGHT = 800


def get_cache_path():
    """
    Get cache file path based on environment.

    Returns local path for development, Google Cloud Storage path for production.

    Returns:
        str: Cache file path (local file path or gs:// URL)
    """
    if os.getenv('CLOUD_RUN'):
        # Cloud Run environment - use Cloud Storage
        bucket = os.getenv('GCS_BUCKET', 'munimetro-cache')
        return f'gs://{bucket}/latest_status.json'
    else:
        # Local development - use local file
        return str(PROJECT_ROOT / "artifacts" / "runtime" / "cache" / "latest_status.json")


def read_cache():
    """
    Read cached status from local file or Cloud Storage.

    Returns:
        dict: Cached status data, or None if not found
    """
    cache_path = get_cache_path()

    try:
        if cache_path.startswith('gs://'):
            # Read from Cloud Storage
            from google.cloud import storage

            # Parse gs://bucket/path
            parts = cache_path[5:].split('/', 1)
            bucket_name = parts[0]
            blob_name = parts[1] if len(parts) > 1 else 'latest_status.json'

            client = storage.Client()
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(blob_name)

            if not blob.exists():
                return None

            content = blob.download_as_string()
            return json.loads(content)
        else:
            # Read from local file
            if not os.path.exists(cache_path):
                return None

            with open(cache_path, 'r') as f:
                return json.load(f)
    except Exception as e:
        print(f"Error reading cache: {e}")
        return None


def write_cache(data):
    """
    Write status data to local file or Cloud Storage.

    Args:
        data: Dict containing status data to cache

    Returns:
        bool: True if successful, False otherwise
    """
    cache_path = get_cache_path()

    try:
        if cache_path.startswith('gs://'):
            # Write to Cloud Storage
            from google.cloud import storage

            # Parse gs://bucket/path
            parts = cache_path[5:].split('/', 1)
            bucket_name = parts[0]
            blob_name = parts[1] if len(parts) > 1 else 'latest_status.json'

            client = storage.Client()
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(blob_name)

            # Upload with content type
            blob.upload_from_string(
                json.dumps(data, indent=2),
                content_type='application/json'
            )
            return True
        else:
            # Write to local file
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, 'w') as f:
                json.dump(data, f, indent=2)
            return True
    except Exception as e:
        print(f"Error writing cache: {e}")
        return False


def write_cached_image(image_path):
    """
    Write the analyzed image to cache (local file or Cloud Storage).

    Args:
        image_path: Path to the local image file

    Returns:
        bool: True if successful, False otherwise
    """
    cache_path = get_cache_path()

    try:
        if cache_path.startswith('gs://'):
            # Write to Cloud Storage
            from google.cloud import storage

            # Parse gs://bucket/path to get bucket name
            parts = cache_path[5:].split('/', 1)
            bucket_name = parts[0]

            client = storage.Client()
            bucket = client.bucket(bucket_name)
            blob = bucket.blob('latest_image.jpg')

            # Upload image
            blob.upload_from_filename(image_path, content_type='image/jpeg')
            return True
        else:
            # Local development - just keep the file where it is
            return True
    except Exception as e:
        print(f"Error writing cached image: {e}")
        return False


def read_cached_image():
    """
    Read the cached image from local file or Cloud Storage.

    Returns:
        bytes: Image data, or None if not found
    """
    cache_path = get_cache_path()

    try:
        if cache_path.startswith('gs://'):
            # Read from Cloud Storage
            from google.cloud import storage

            # Parse gs://bucket/path to get bucket name
            parts = cache_path[5:].split('/', 1)
            bucket_name = parts[0]

            client = storage.Client()
            bucket = client.bucket(bucket_name)
            blob = bucket.blob('latest_image.jpg')

            if not blob.exists():
                return None

            return blob.download_as_bytes()
        else:
            # Local development - read from cache directory
            cache_dir = os.path.dirname(cache_path)
            image_path = os.path.join(cache_dir, 'latest_image.jpg')
            if os.path.exists(image_path):
                with open(image_path, 'rb') as f:
                    return f.read()
            return None
    except Exception as e:
        print(f"Error reading cached image: {e}")
        return None


# Status priority for "best of N" calculation
# Higher number = more optimistic status
STATUS_PRIORITY = {'green': 3, 'yellow': 2, 'red': 1}


def calculate_best_status(statuses, window_size=3):
    """
    Calculate the best (most optimistic) status from a list of recent statuses.

    This implements the "best of N" smoothing to filter out brief transient issues.
    The webapp, RSS feed, and Bluesky all use this to ensure consistent status reporting.

    Args:
        statuses: List of status dicts, most recent first. Each must have 'status' key.
        window_size: Number of recent statuses to consider (default 3)

    Returns:
        dict: The status entry with the most optimistic status within the window,
              preferring the most recent if there are ties. Returns None if empty.
    """
    if not statuses:
        return None

    # Only consider the most recent N statuses
    recent = statuses[:window_size]

    # Find the best (most optimistic) status value
    best_value = max(
        [s['status'] for s in recent],
        key=lambda x: STATUS_PRIORITY.get(x, 0)
    )

    # Return the most recent entry with that status
    # This ensures we use the most recent delay info if status is yellow
    for s in recent:
        if s['status'] == best_value:
            return s

    # Fallback (shouldn't happen)
    return recent[0]


# Backward compatibility: import post_to_bluesky from notifiers module
# This allows existing code to continue using: from lib.muni_lib import post_to_bluesky
from lib.notifiers import post_to_bluesky


def download_muni_image(output_folder="muni_snapshots", validate_dimensions=True):
    """
    Download a single Muni subway status image.

    The SF Muni Central page uses JavaScript to update the image every 5 seconds,
    but we can access the actual image URL directly.

    Args:
        output_folder: Directory to save the image
        validate_dimensions: If True, verify image is 1860x800 and delete if not

    Returns:
        dict: {
            'success': bool,
            'filepath': str or None,
            'width': int or None,
            'height': int or None,
            'error': str or None
        }
    """
    import random

    # Direct image URL (extracted from obfuscated JavaScript)
    IMAGE_URL = "http://sfmunicentral.com/sfmunicentral_Snapshot_Objects/Mimic1_A7SE582P.jpg"

    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    try:
        # Download with cache-busting parameter (mimics JavaScript behavior)
        params = {'nocache': random.randint(0, 999)}
        response = requests.get(IMAGE_URL, params=params, timeout=10)
        response.raise_for_status()

        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"muni_snapshot_{timestamp}.jpg"
        filepath = os.path.join(output_folder, filename)

        # Save the image
        with open(filepath, 'wb') as f:
            f.write(response.content)

        # Verify image dimensions if requested
        img = Image.open(filepath)
        width, height = img.size

        if validate_dimensions and (width != EXPECTED_WIDTH or height != EXPECTED_HEIGHT):
            os.remove(filepath)
            return {
                'success': False,
                'filepath': None,
                'width': width,
                'height': height,
                'error': f"Invalid dimensions: {width}x{height}, expected {EXPECTED_WIDTH}x{EXPECTED_HEIGHT}"
            }

        return {
            'success': True,
            'filepath': filepath,
            'width': width,
            'height': height,
            'error': None
        }

    except Exception as e:
        return {
            'success': False,
            'filepath': None,
            'width': None,
            'height': None,
            'error': str(e)
        }


def detect_muni_status(image_path):
    """
    Detect subway status from an image using OpenCV-based analysis.

    Uses a deterministic computer vision approach that analyzes:
    - Train positions and IDs via OCR
    - Platform hold status (yellow vs blue)
    - Track segment status (red vs cyan)
    - Train bunching/clustering

    Args:
        image_path: Path to the image file

    Returns:
        dict: {
            'status': str ('green', 'yellow', 'red'),
            'status_confidence': float (always 1.0 for deterministic detection),
            'description': str (human-readable status description),
            'probabilities': {
                'green': float,
                'yellow': float,
                'red': float
            },
            'detection': {
                'stations': [...],
                'segments': [...],
                'trains': [...],
                'delays_platforms': [...],
                'delays_segments': [...],
                'delays_bunching': [...]
            }
        }

    Raises:
        ValueError: If image cannot be loaded
    """
    from lib.detection import detect_system_status

    # Run OpenCV-based detection
    result = detect_system_status(image_path)

    # Build backwards-compatible response format
    status = result['system_status']

    return {
        'status': status,
        'status_confidence': result['confidence'],
        'description': result['description'],
        'probabilities': {
            'green': 1.0 if status == 'green' else 0.0,
            'yellow': 1.0 if status == 'yellow' else 0.0,
            'red': 1.0 if status == 'red' else 0.0,
        },
        'detection': {
            'system_status': result['system_status'],
            'stations': result['stations'],
            'segments': result['segments'],
            'trains': result['trains'],
            'delays_platforms': result['delays_platforms'],
            'delays_segments': result['delays_segments'],
            'delays_bunching': result['delays_bunching'],
            'delay_summaries': result.get('delay_summaries', []),
            'image_dimensions': result['image_dimensions'],
        }
    }
