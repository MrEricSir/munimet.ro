"""
Google Cloud Storage utilities with retry logic.

Provides wrapper functions for common GCS operations with:
- Exponential backoff retry for transient errors
- Consistent error handling and logging
- Support for both local development and Cloud Run environments
"""

import json
import logging
import os
import time
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Optional, TypeVar

# Configure logging
logger = logging.getLogger(__name__)

# Retry configuration
DEFAULT_MAX_RETRIES = 3
DEFAULT_INITIAL_DELAY = 1.0  # seconds
DEFAULT_MAX_DELAY = 30.0  # seconds
DEFAULT_EXPONENTIAL_BASE = 2


# Type variable for generic return types
T = TypeVar('T')


def retry_with_backoff(
    max_retries: int = DEFAULT_MAX_RETRIES,
    initial_delay: float = DEFAULT_INITIAL_DELAY,
    max_delay: float = DEFAULT_MAX_DELAY,
    exponential_base: float = DEFAULT_EXPONENTIAL_BASE,
    retryable_exceptions: tuple = None,
) -> Callable:
    """
    Decorator that retries a function with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay between retries in seconds
        max_delay: Maximum delay between retries in seconds
        exponential_base: Base for exponential backoff calculation
        retryable_exceptions: Tuple of exception types to retry on.
            If None, retries on all exceptions.

    Returns:
        Decorated function that will retry on failure
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    # Check if this exception type should be retried
                    if retryable_exceptions and not isinstance(e, retryable_exceptions):
                        raise

                    last_exception = e

                    if attempt < max_retries:
                        delay = min(
                            initial_delay * (exponential_base ** attempt),
                            max_delay
                        )
                        logger.warning(
                            f"Retry {attempt + 1}/{max_retries} for {func.__name__}: "
                            f"{type(e).__name__}: {e}. Waiting {delay:.1f}s..."
                        )
                        time.sleep(delay)
                    else:
                        logger.error(
                            f"Failed after {max_retries + 1} attempts for {func.__name__}: "
                            f"{type(e).__name__}: {e}"
                        )

            raise last_exception

        return wrapper
    return decorator


def get_gcs_client():
    """
    Get a GCS client instance.

    Returns:
        google.cloud.storage.Client or None if not available
    """
    try:
        from google.cloud import storage
        return storage.Client()
    except ImportError:
        logger.warning("google-cloud-storage not installed")
        return None
    except Exception as e:
        logger.error(f"Failed to create GCS client: {e}")
        return None


def is_gcs_path(path: str) -> bool:
    """Check if a path is a GCS path."""
    return path.startswith('gs://')


def parse_gcs_path(gcs_path: str) -> tuple:
    """
    Parse a GCS path into bucket and blob name.

    Args:
        gcs_path: Path in format gs://bucket/path/to/object

    Returns:
        Tuple of (bucket_name, blob_name)
    """
    if not gcs_path.startswith('gs://'):
        raise ValueError(f"Invalid GCS path: {gcs_path}")

    path = gcs_path[5:]  # Remove 'gs://'
    parts = path.split('/', 1)
    bucket_name = parts[0]
    blob_name = parts[1] if len(parts) > 1 else ''
    return bucket_name, blob_name


# Define retryable GCS exceptions
def _get_retryable_exceptions():
    """Get tuple of retryable GCS exceptions."""
    exceptions = [
        ConnectionError,
        TimeoutError,
        OSError,
    ]

    try:
        from google.api_core import exceptions as gcs_exceptions
        exceptions.extend([
            gcs_exceptions.ServiceUnavailable,
            gcs_exceptions.InternalServerError,
            gcs_exceptions.TooManyRequests,
            gcs_exceptions.GatewayTimeout,
            gcs_exceptions.DeadlineExceeded,
        ])
    except ImportError:
        pass

    return tuple(exceptions)


RETRYABLE_EXCEPTIONS = _get_retryable_exceptions()


@retry_with_backoff(retryable_exceptions=RETRYABLE_EXCEPTIONS)
def gcs_download_as_string(bucket_name: str, blob_name: str) -> Optional[bytes]:
    """
    Download a blob's contents as a string with retry.

    Args:
        bucket_name: Name of the GCS bucket
        blob_name: Name of the blob to download

    Returns:
        Blob contents as bytes, or None if blob doesn't exist
    """
    from google.cloud import storage

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    if not blob.exists():
        return None

    return blob.download_as_string()


@retry_with_backoff(retryable_exceptions=RETRYABLE_EXCEPTIONS)
def gcs_download_as_bytes(bucket_name: str, blob_name: str) -> Optional[bytes]:
    """
    Download a blob's contents as bytes with retry.

    Args:
        bucket_name: Name of the GCS bucket
        blob_name: Name of the blob to download

    Returns:
        Blob contents as bytes, or None if blob doesn't exist
    """
    from google.cloud import storage

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    if not blob.exists():
        return None

    return blob.download_as_bytes()


@retry_with_backoff(retryable_exceptions=RETRYABLE_EXCEPTIONS)
def gcs_download_to_file(bucket_name: str, blob_name: str, local_path: str) -> bool:
    """
    Download a blob to a local file with retry.

    Args:
        bucket_name: Name of the GCS bucket
        blob_name: Name of the blob to download
        local_path: Local file path to save to

    Returns:
        True if successful, False if blob doesn't exist
    """
    from google.cloud import storage

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    if not blob.exists():
        return False

    blob.download_to_filename(local_path)
    return True


@retry_with_backoff(retryable_exceptions=RETRYABLE_EXCEPTIONS)
def gcs_upload_from_string(
    bucket_name: str,
    blob_name: str,
    content: str,
    content_type: str = 'application/octet-stream'
) -> bool:
    """
    Upload a string to a GCS blob with retry.

    Args:
        bucket_name: Name of the GCS bucket
        blob_name: Name of the blob to create/update
        content: String content to upload
        content_type: MIME type for the content

    Returns:
        True if successful
    """
    from google.cloud import storage

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    blob.upload_from_string(content, content_type=content_type)
    return True


@retry_with_backoff(retryable_exceptions=RETRYABLE_EXCEPTIONS)
def gcs_upload_from_file(
    bucket_name: str,
    blob_name: str,
    local_path: str,
    content_type: str = None
) -> bool:
    """
    Upload a local file to a GCS blob with retry.

    Args:
        bucket_name: Name of the GCS bucket
        blob_name: Name of the blob to create/update
        local_path: Local file path to upload
        content_type: MIME type for the content (optional)

    Returns:
        True if successful
    """
    from google.cloud import storage

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    if content_type:
        blob.upload_from_filename(local_path, content_type=content_type)
    else:
        blob.upload_from_filename(local_path)
    return True


@retry_with_backoff(retryable_exceptions=RETRYABLE_EXCEPTIONS)
def gcs_blob_exists(bucket_name: str, blob_name: str) -> bool:
    """
    Check if a blob exists with retry.

    Args:
        bucket_name: Name of the GCS bucket
        blob_name: Name of the blob to check

    Returns:
        True if blob exists, False otherwise
    """
    from google.cloud import storage

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    return blob.exists()


@retry_with_backoff(retryable_exceptions=RETRYABLE_EXCEPTIONS)
def gcs_get_blob_size(bucket_name: str, blob_name: str) -> Optional[int]:
    """
    Get the size of a blob with retry.

    Args:
        bucket_name: Name of the GCS bucket
        blob_name: Name of the blob to check

    Returns:
        Size in bytes, or None if blob doesn't exist
    """
    from google.cloud import storage

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    blob.reload()  # Fetch metadata
    return blob.size


@retry_with_backoff(retryable_exceptions=RETRYABLE_EXCEPTIONS)
def gcs_list_blobs(bucket_name: str, prefix: str = '') -> list[str]:
    """
    List blob names in a GCS bucket with an optional prefix filter.

    Args:
        bucket_name: Name of the GCS bucket
        prefix: Optional prefix to filter blobs (e.g., '2026/02/21/')

    Returns:
        List of blob name strings
    """
    from google.cloud import storage

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=prefix)

    return [blob.name for blob in blobs]
