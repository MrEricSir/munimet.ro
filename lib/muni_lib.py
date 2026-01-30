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


# Status messages matching the frontend (api/html/index.html)
STATUS_MESSAGES = {
    'green': '🟢 All aboard: Muni is on track',
    'yellow': "🟡 Uh oh: Muni's not feeling well",
    'red': '🔴 Muni is taking a nap',
}


def post_to_bluesky(status, previous_status=None):
    """
    Post a status update to Bluesky.

    Requires environment variables:
    - BLUESKY_HANDLE: The account handle (e.g., 'munimetro.bsky.social')
    - BLUESKY_APP_PASSWORD: An app password for the account

    Args:
        status: Current status ('green', 'yellow', 'red')
        previous_status: Previous status for context (optional)

    Returns:
        dict: {'success': bool, 'uri': str or None, 'error': str or None}
    """
    handle = os.getenv('BLUESKY_HANDLE')
    app_password = os.getenv('BLUESKY_APP_PASSWORD')

    if not handle or not app_password:
        return {
            'success': False,
            'uri': None,
            'error': 'BLUESKY_HANDLE and BLUESKY_APP_PASSWORD environment variables required'
        }

    message = STATUS_MESSAGES.get(status, f'Status: {status}')

    try:
        from atproto import Client

        client = Client()
        client.login(handle, app_password)

        post = client.send_post(text=message)

        return {
            'success': True,
            'uri': post.uri,
            'error': None
        }
    except Exception as e:
        return {
            'success': False,
            'uri': None,
            'error': str(e)
        }


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

    This replaces the previous BLIP model-based prediction with a deterministic
    computer vision approach that analyzes:
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
            'image_dimensions': result['image_dimensions'],
        }
    }
