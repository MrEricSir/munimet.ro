"""
Tests for GCS utilities with retry logic.

Tests cover:
- Retry decorator behavior
- Exponential backoff timing
- Exception handling
"""

import pytest
import time
from unittest.mock import patch, MagicMock

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.gcs_utils import (
    retry_with_backoff,
    is_gcs_path,
    parse_gcs_path,
)


class TestRetryDecorator:
    """Tests for the retry_with_backoff decorator."""

    def test_succeeds_without_retry(self):
        """Function succeeds on first try."""
        call_count = 0

        @retry_with_backoff(max_retries=3)
        def success_func():
            nonlocal call_count
            call_count += 1
            return "success"

        result = success_func()
        assert result == "success"
        assert call_count == 1

    def test_retries_on_failure(self):
        """Function retries after failure."""
        call_count = 0

        @retry_with_backoff(max_retries=3, initial_delay=0.01)
        def fail_then_succeed():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Transient error")
            return "success"

        result = fail_then_succeed()
        assert result == "success"
        assert call_count == 3

    def test_raises_after_max_retries(self):
        """Function raises after exhausting retries."""
        call_count = 0

        @retry_with_backoff(max_retries=2, initial_delay=0.01)
        def always_fail():
            nonlocal call_count
            call_count += 1
            raise ConnectionError("Permanent error")

        with pytest.raises(ConnectionError) as exc_info:
            always_fail()

        assert "Permanent error" in str(exc_info.value)
        assert call_count == 3  # Initial + 2 retries

    def test_only_retries_specified_exceptions(self):
        """Function only retries on specified exception types."""
        call_count = 0

        @retry_with_backoff(
            max_retries=3,
            initial_delay=0.01,
            retryable_exceptions=(ConnectionError,)
        )
        def fail_with_value_error():
            nonlocal call_count
            call_count += 1
            raise ValueError("Not retryable")

        with pytest.raises(ValueError):
            fail_with_value_error()

        assert call_count == 1  # No retries for ValueError

    def test_exponential_backoff_timing(self):
        """Verify exponential backoff increases delay."""
        call_times = []

        @retry_with_backoff(
            max_retries=3,
            initial_delay=0.05,
            exponential_base=2,
            max_delay=10.0
        )
        def track_timing():
            call_times.append(time.time())
            if len(call_times) < 4:
                raise ConnectionError("Fail")
            return "success"

        result = track_timing()
        assert result == "success"
        assert len(call_times) == 4

        # Check delays are increasing (approximately)
        delays = [call_times[i+1] - call_times[i] for i in range(len(call_times)-1)]
        # First delay ~0.05s, second ~0.1s, third ~0.2s
        assert delays[0] < delays[1] < delays[2]

    def test_max_delay_caps_backoff(self):
        """Verify max_delay caps the exponential growth."""
        call_times = []

        @retry_with_backoff(
            max_retries=5,
            initial_delay=0.1,
            exponential_base=10,
            max_delay=0.15
        )
        def track_timing():
            call_times.append(time.time())
            if len(call_times) < 3:
                raise ConnectionError("Fail")
            return "success"

        result = track_timing()
        assert result == "success"

        delays = [call_times[i+1] - call_times[i] for i in range(len(call_times)-1)]
        # Both delays should be capped at ~0.15s (max_delay)
        for delay in delays:
            assert delay < 0.25  # Allow some timing variance


class TestGCSPathHelpers:
    """Tests for GCS path parsing utilities."""

    def test_is_gcs_path_true(self):
        """Correctly identifies GCS paths."""
        assert is_gcs_path("gs://bucket/path/to/file")
        assert is_gcs_path("gs://bucket")
        assert is_gcs_path("gs://my-bucket/file.json")

    def test_is_gcs_path_false(self):
        """Correctly rejects non-GCS paths."""
        assert not is_gcs_path("/local/path/file")
        assert not is_gcs_path("s3://bucket/file")
        assert not is_gcs_path("http://example.com/file")
        assert not is_gcs_path("")

    def test_parse_gcs_path_with_blob(self):
        """Parses bucket and blob from GCS path."""
        bucket, blob = parse_gcs_path("gs://my-bucket/path/to/file.json")
        assert bucket == "my-bucket"
        assert blob == "path/to/file.json"

    def test_parse_gcs_path_bucket_only(self):
        """Parses bucket-only GCS path."""
        bucket, blob = parse_gcs_path("gs://my-bucket")
        assert bucket == "my-bucket"
        assert blob == ""

    def test_parse_gcs_path_invalid(self):
        """Raises error for invalid GCS path."""
        with pytest.raises(ValueError) as exc_info:
            parse_gcs_path("/local/path")
        assert "Invalid GCS path" in str(exc_info.value)


class TestGCSOperationsIntegration:
    """Integration tests verifying retry decorator is applied to GCS functions."""

    def test_gcs_functions_have_retry_decorator(self):
        """Verify GCS functions are wrapped with retry decorator."""
        from lib.gcs_utils import (
            gcs_download_as_string,
            gcs_download_as_bytes,
            gcs_download_to_file,
            gcs_upload_from_string,
            gcs_upload_from_file,
            gcs_blob_exists,
            gcs_get_blob_size,
        )

        # Check that functions are wrapped (have __wrapped__ attribute)
        # This indicates the retry decorator was applied
        functions = [
            gcs_download_as_string,
            gcs_download_as_bytes,
            gcs_download_to_file,
            gcs_upload_from_string,
            gcs_upload_from_file,
            gcs_blob_exists,
            gcs_get_blob_size,
        ]

        for func in functions:
            assert hasattr(func, '__wrapped__'), f"{func.__name__} should have retry decorator"
