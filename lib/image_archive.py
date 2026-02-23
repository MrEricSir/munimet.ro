"""
Image archival to Google Cloud Storage for debugging and auditing.

Archives snapshot images on key occasions (status transitions, hysteresis
overrides, periodic baselines) to a dedicated GCS bucket with 30-day
lifecycle. Cloud Run only — no-op in local development.
"""

import logging
import os
from datetime import datetime

from lib.config import DEFAULT_ARCHIVE_BUCKET, BASELINE_ARCHIVE_INTERVAL
from lib.gcs_utils import gcs_upload_from_file

logger = logging.getLogger(__name__)


def _build_archive_path(timestamp_str: str, reason: str, raw_status: str = None) -> str:
    """
    Build the GCS object path for an archived image.

    Args:
        timestamp_str: ISO format timestamp (e.g., '2026-02-21T08:30:00')
        reason: Archive reason ('transition', 'override', or 'baseline')
        raw_status: For overrides, the raw detection status before hysteresis
                    (e.g., 'yellow'). Encoded in filename as '_rawYellow'.

    Returns:
        GCS object path like '2026/02/21/muni_snapshot_20260221_083000_override_rawYellow.jpg'
    """
    dt = datetime.fromisoformat(timestamp_str)
    date_prefix = dt.strftime('%Y/%m/%d')
    suffix = reason
    if reason == 'override' and raw_status:
        suffix = f'{reason}_raw{raw_status.capitalize()}'
    filename = dt.strftime(f'muni_snapshot_%Y%m%d_%H%M%S_{suffix}.jpg')
    return f'{date_prefix}/{filename}'


def archive_image(image_path: str, timestamp_str: str, reason: str,
                  raw_status: str = None) -> bool:
    """
    Upload an image to the GCS archive bucket.

    Cloud Run only — returns False immediately in local development.
    Best-effort: catches all exceptions and logs warnings without crashing.

    Args:
        image_path: Local path to the image file
        timestamp_str: ISO format timestamp for the image
        reason: Archive reason ('transition', 'override', or 'baseline')
        raw_status: For overrides, the raw detection status before hysteresis

    Returns:
        True if uploaded successfully, False otherwise
    """
    if not os.getenv('CLOUD_RUN'):
        return False

    try:
        bucket = os.getenv('GCS_ARCHIVE_BUCKET', DEFAULT_ARCHIVE_BUCKET)
        blob_name = _build_archive_path(timestamp_str, reason, raw_status=raw_status)
        gcs_upload_from_file(bucket, blob_name, image_path, content_type='image/jpeg')
        logger.info(f"Archived image: gs://{bucket}/{blob_name}")
        return True
    except Exception as e:
        logger.warning(f"Image archival failed ({reason}): {e}")
        return False


def should_archive_baseline(cache_data: dict, current_status: str) -> bool:
    """
    Determine whether a baseline image should be archived.

    Returns True only when status is green and more than 1 hour has elapsed
    since the last baseline archive. Uses cache_data for persistence across
    Cloud Run cold starts.

    Args:
        cache_data: The JSON dict persisted to GCS cache
        current_status: Current reported status string

    Returns:
        True if a baseline archive should be taken
    """
    if current_status != 'green':
        return False

    last_archive = cache_data.get('last_baseline_archive')
    if not last_archive:
        return True

    try:
        last_dt = datetime.fromisoformat(last_archive)
        elapsed = (datetime.now() - last_dt).total_seconds()
        return elapsed >= BASELINE_ARCHIVE_INTERVAL
    except (ValueError, TypeError):
        return True
