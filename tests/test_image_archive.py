"""
Tests for the image archival module.

Tests cover:
- Archive path construction for each reason type
- Baseline archive gating (green-only, 1-hour interval, edge cases)
- archive_image behavior (local no-op, cloud upload, failure handling)
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.image_archive import _build_archive_path, should_archive_baseline, archive_image
from lib.config import BASELINE_ARCHIVE_INTERVAL


class TestBuildArchivePath:
    """Tests for _build_archive_path."""

    def test_transition_reason(self):
        path = _build_archive_path('2026-02-21T08:30:00', 'transition')
        assert path == '2026/02/21/muni_snapshot_20260221_083000_transition.jpg'

    def test_override_reason_without_raw_status(self):
        path = _build_archive_path('2026-02-21T14:05:30', 'override')
        assert path == '2026/02/21/muni_snapshot_20260221_140530_override.jpg'

    def test_override_reason_with_raw_status(self):
        path = _build_archive_path('2026-02-21T14:05:30', 'override', raw_status='yellow')
        assert path == '2026/02/21/muni_snapshot_20260221_140530_override_rawYellow.jpg'

    def test_override_reason_with_raw_red(self):
        path = _build_archive_path('2026-02-21T14:05:30', 'override', raw_status='red')
        assert path == '2026/02/21/muni_snapshot_20260221_140530_override_rawRed.jpg'

    def test_non_override_ignores_raw_status(self):
        """raw_status is only encoded for override reason."""
        path = _build_archive_path('2026-02-21T08:30:00', 'transition', raw_status='yellow')
        assert path == '2026/02/21/muni_snapshot_20260221_083000_transition.jpg'

    def test_baseline_reason(self):
        path = _build_archive_path('2026-01-01T00:00:00', 'baseline')
        assert path == '2026/01/01/muni_snapshot_20260101_000000_baseline.jpg'

    def test_midnight_boundary(self):
        path = _build_archive_path('2026-12-31T23:59:59', 'transition')
        assert path == '2026/12/31/muni_snapshot_20261231_235959_transition.jpg'


class TestShouldArchiveBaseline:
    """Tests for should_archive_baseline."""

    def test_returns_false_for_yellow(self):
        assert should_archive_baseline({}, 'yellow') is False

    def test_returns_false_for_red(self):
        assert should_archive_baseline({}, 'red') is False

    def test_returns_true_for_green_no_previous(self):
        """First baseline ever — no last_baseline_archive in cache."""
        assert should_archive_baseline({}, 'green') is True

    def test_returns_true_when_over_one_hour(self):
        two_hours_ago = (datetime.now() - timedelta(hours=2)).isoformat()
        cache = {'last_baseline_archive': two_hours_ago}
        assert should_archive_baseline(cache, 'green') is True

    def test_returns_false_when_under_one_hour(self):
        ten_minutes_ago = (datetime.now() - timedelta(minutes=10)).isoformat()
        cache = {'last_baseline_archive': ten_minutes_ago}
        assert should_archive_baseline(cache, 'green') is False

    def test_returns_true_exactly_at_one_hour(self):
        exactly_one_hour = (datetime.now() - timedelta(seconds=BASELINE_ARCHIVE_INTERVAL)).isoformat()
        cache = {'last_baseline_archive': exactly_one_hour}
        assert should_archive_baseline(cache, 'green') is True

    def test_returns_true_for_invalid_timestamp(self):
        """Invalid timestamp in cache should trigger a new baseline."""
        cache = {'last_baseline_archive': 'not-a-timestamp'}
        assert should_archive_baseline(cache, 'green') is True

    def test_returns_true_for_none_timestamp(self):
        cache = {'last_baseline_archive': None}
        assert should_archive_baseline(cache, 'green') is True


class TestArchiveImage:
    """Tests for archive_image."""

    def test_skips_locally(self):
        """Should return False when not on Cloud Run."""
        with patch.dict('os.environ', {}, clear=True):
            result = archive_image('/tmp/test.jpg', '2026-02-21T08:30:00', 'transition')
        assert result is False

    def test_uploads_on_cloud_run(self):
        """Should call gcs_upload_from_file when on Cloud Run."""
        with patch.dict('os.environ', {'CLOUD_RUN': 'true'}):
            with patch('lib.image_archive.gcs_upload_from_file') as mock_upload:
                mock_upload.return_value = True
                result = archive_image('/tmp/test.jpg', '2026-02-21T08:30:00', 'transition')

        assert result is True
        mock_upload.assert_called_once_with(
            'munimetro-image-archive',
            '2026/02/21/muni_snapshot_20260221_083000_transition.jpg',
            '/tmp/test.jpg',
            content_type='image/jpeg'
        )

    def test_override_with_raw_status_encodes_in_path(self):
        """Override with raw_status should encode it in the GCS path."""
        with patch.dict('os.environ', {'CLOUD_RUN': 'true'}):
            with patch('lib.image_archive.gcs_upload_from_file') as mock_upload:
                mock_upload.return_value = True
                result = archive_image('/tmp/test.jpg', '2026-02-21T08:30:00', 'override',
                                       raw_status='yellow')

        assert result is True
        mock_upload.assert_called_once_with(
            'munimetro-image-archive',
            '2026/02/21/muni_snapshot_20260221_083000_override_rawYellow.jpg',
            '/tmp/test.jpg',
            content_type='image/jpeg'
        )

    def test_uses_custom_bucket_env(self):
        """Should use GCS_ARCHIVE_BUCKET env var if set."""
        with patch.dict('os.environ', {'CLOUD_RUN': 'true', 'GCS_ARCHIVE_BUCKET': 'my-bucket'}):
            with patch('lib.image_archive.gcs_upload_from_file') as mock_upload:
                mock_upload.return_value = True
                archive_image('/tmp/test.jpg', '2026-02-21T08:30:00', 'baseline')

        assert mock_upload.call_args[0][0] == 'my-bucket'

    def test_handles_upload_failure_gracefully(self):
        """Should catch exceptions and return False without raising."""
        with patch.dict('os.environ', {'CLOUD_RUN': 'true'}):
            with patch('lib.image_archive.gcs_upload_from_file', side_effect=Exception('GCS error')):
                result = archive_image('/tmp/test.jpg', '2026-02-21T08:30:00', 'override')

        assert result is False
