"""
Tests for check_status interval handling.

Tests interval detection, environment-specific behavior, and configuration.
"""

import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestIntervalConfiguration:
    """Tests for interval configuration values."""

    def test_config_intervals_match_expected_values(self):
        """Test that config intervals match expected production values."""
        from lib.config import DEFAULT_CHECK_INTERVAL, CLOUD_CHECK_INTERVAL

        # Local development: 30 seconds
        assert DEFAULT_CHECK_INTERVAL == 30

        # Cloud Run: 180 seconds (3 minutes, matches Cloud Scheduler)
        assert CLOUD_CHECK_INTERVAL == 180

    def test_cloud_interval_is_longer_than_default(self):
        """Test that Cloud Run interval is longer than local interval."""
        from lib.config import DEFAULT_CHECK_INTERVAL, CLOUD_CHECK_INTERVAL

        assert CLOUD_CHECK_INTERVAL > DEFAULT_CHECK_INTERVAL

    def test_cloud_interval_matches_scheduler(self):
        """Test that Cloud Run interval matches the 3-minute Cloud Scheduler."""
        from lib.config import CLOUD_CHECK_INTERVAL

        # Cloud Scheduler runs every 3 minutes = 180 seconds
        assert CLOUD_CHECK_INTERVAL == 180


class TestGetCheckIntervalLogic:
    """Tests for interval detection logic (without full module import)."""

    def test_local_environment_detection(self):
        """Test that local environment is detected correctly."""
        # Simulate the logic from _get_check_interval()
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop('CLOUD_RUN', None)
            is_cloud_run = os.getenv('CLOUD_RUN') is not None

        assert is_cloud_run is False

    def test_cloud_run_environment_detection(self):
        """Test that Cloud Run environment is detected correctly."""
        with patch.dict(os.environ, {'CLOUD_RUN': 'true'}):
            is_cloud_run = os.getenv('CLOUD_RUN') is not None

        assert is_cloud_run is True

    def test_cloud_run_any_value_is_detected(self):
        """Test that any CLOUD_RUN value triggers cloud detection."""
        for value in ['true', 'TRUE', '1', 'yes', 'anything']:
            with patch.dict(os.environ, {'CLOUD_RUN': value}):
                is_cloud_run = os.getenv('CLOUD_RUN') is not None
            assert is_cloud_run is True, f"Failed for CLOUD_RUN={value}"

    def test_interval_selection_logic(self):
        """Test the interval selection logic matches expected behavior."""
        from lib.config import DEFAULT_CHECK_INTERVAL, CLOUD_CHECK_INTERVAL

        # Simulate _get_check_interval() logic
        def get_interval_for_env(cloud_run_value):
            with patch.dict(os.environ, {'CLOUD_RUN': cloud_run_value} if cloud_run_value else {}, clear=True):
                if cloud_run_value is None:
                    os.environ.pop('CLOUD_RUN', None)
                if os.getenv('CLOUD_RUN'):
                    return CLOUD_CHECK_INTERVAL
                return DEFAULT_CHECK_INTERVAL

        # Local environment
        assert get_interval_for_env(None) == 30

        # Cloud Run environment
        assert get_interval_for_env('true') == 180


class TestAnalyticsIntervalIntegration:
    """Tests verifying analytics correctly uses interval values."""

    def test_log_status_check_accepts_interval(self):
        """Test that log_status_check accepts interval_seconds parameter."""
        from lib import analytics
        import inspect

        sig = inspect.signature(analytics.log_status_check)
        params = sig.parameters

        assert 'interval_seconds' in params
        # Should default to None
        assert params['interval_seconds'].default is None

    def test_log_status_check_stores_custom_interval(self, tmp_path):
        """Test that custom interval is stored in database."""
        from lib import analytics
        import sqlite3
        from datetime import datetime

        db_path = tmp_path / "test_analytics.db"

        with patch.object(analytics, 'LOCAL_DB_PATH', db_path):
            with patch.object(analytics, '_is_cloud_run', return_value=False):
                with patch.object(analytics, 'backup_db_to_gcs', return_value=False):
                    analytics.init_db()

                    # Log with Cloud Run interval
                    check_id = analytics.log_status_check(
                        status='green',
                        best_status='green',
                        detection_data={
                            'trains': [],
                            'delays_platforms': [],
                            'delays_segments': [],
                            'delays_bunching': []
                        },
                        timestamp=datetime.now().isoformat(),
                        interval_seconds=180  # Cloud Run interval
                    )

        # Verify in database
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute('SELECT interval_seconds FROM status_checks WHERE id = ?', (check_id,))
        row = cursor.fetchone()
        conn.close()

        assert row['interval_seconds'] == 180

    def test_frequency_calculation_respects_interval(self, tmp_path):
        """Test that delay frequency uses actual intervals, not counts."""
        from lib import analytics
        from datetime import datetime, timedelta

        db_path = tmp_path / "test_analytics.db"

        with patch.object(analytics, 'LOCAL_DB_PATH', db_path):
            with patch.object(analytics, '_is_cloud_run', return_value=False):
                with patch.object(analytics, 'backup_db_to_gcs', return_value=False):
                    analytics.init_db()

                    base_time = datetime.now() - timedelta(hours=1)

                    # 1 check at 30 seconds
                    analytics.log_status_check(
                        status='green',
                        best_status='green',
                        detection_data={'trains': [], 'delays_platforms': [], 'delays_segments': [], 'delays_bunching': []},
                        timestamp=base_time.isoformat(),
                        interval_seconds=30
                    )

                    # 1 check at 180 seconds
                    analytics.log_status_check(
                        status='green',
                        best_status='green',
                        detection_data={'trains': [], 'delays_platforms': [], 'delays_segments': [], 'delays_bunching': []},
                        timestamp=(base_time + timedelta(minutes=5)).isoformat(),
                        interval_seconds=180
                    )

                    result = analytics.get_delay_frequency(days=1)

        # Should be 30/60 + 180/60 = 0.5 + 3 = 3.5 minutes
        # NOT 2 checks * some_default
        assert result['total_minutes'] == 3.5


class TestNotificationIdempotency:
    """Tests that unconfigured notification channels don't block last_notified_status.

    Regression test for a bug where adding webhooks to the dispatcher caused
    every check to re-trigger notifications. The webhook channel returned
    success=False with "Not configured" when WEBHOOK_URLS was unset, which
    prevented last_notified_status from being saved. On subsequent runs, the
    checker saw last_notified_status != current_reported and entered the
    "recovering missed notification" path, spamming all channels every 3 minutes.
    """

    def _make_status(self, status, timestamp='2026-03-01T12:00:00'):
        """Helper to build a status dict."""
        return {
            'status': status,
            'description': f'{status} description',
            'confidence': 0.99,
            'probabilities': {'green': 0.99, 'yellow': 0.005, 'red': 0.005},
            'detection': {'trains': [{'id': 'TT', 'x': 500}], 'delays_platforms': [],
                          'delays_segments': [], 'delays_bunching': [], 'delay_summaries': []},
            'image_path': '/tmp/test.jpg',
            'image_dimensions': {'width': 1860, 'height': 800},
            'timestamp': timestamp,
        }

    def _run_check_status(self, cache_before, notify_results, detection_status='red'):
        """Run check_status with mocked dependencies and return the final cache.

        Sets up a scenario where the reported status is already `detection_status`
        (hysteresis has already transitioned), and the new detection also returns
        `detection_status`, so the notification path can be exercised via the
        "recovering missed notification" logic.
        """
        from api import check_status as cs_module

        saved_caches = []

        def fake_write_cache(data):
            saved_caches.append(data.copy())
            return True

        download_result = {
            'success': True,
            'filepath': '/tmp/test.jpg',
            'width': 1860,
            'height': 800,
        }
        detection_result = {
            'status': detection_status,
            'description': 'Not operating',
            'status_confidence': 0.99,
            'probabilities': {'green': 0.005, 'yellow': 0.005, 'red': 0.99},
            'detection': {'trains': [], 'delays_platforms': [],
                          'delays_segments': [], 'delays_bunching': [], 'delay_summaries': []},
        }

        with patch.object(cs_module, 'download_muni_image', return_value=download_result), \
             patch.object(cs_module, 'detect_muni_status', return_value=detection_result), \
             patch.object(cs_module, 'read_cache', return_value=cache_before), \
             patch.object(cs_module, 'write_cache', side_effect=fake_write_cache), \
             patch.object(cs_module, 'write_cached_image', return_value=True), \
             patch.object(cs_module, 'write_cached_badge', return_value=True), \
             patch.object(cs_module, 'notify_status_change', return_value=notify_results), \
             patch.object(cs_module, 'log_status_check', return_value=1), \
             patch('lib.analytics.check_database_health', return_value={'exists': True, 'has_data': True, 'check_count': 1}), \
             patch.object(cs_module, 'archive_image', return_value=False), \
             patch.object(cs_module, 'should_archive_baseline', return_value=False):
            cs_module.check_status(should_write_cache=True)

        return saved_caches

    def _make_cache_with_missed_notification(self):
        """Build a cache simulating a missed notification.

        The reported status has already transitioned to red (via hysteresis),
        but last_notified_status is still 'green' because the previous
        notification run failed to save it. This triggers the "recovering
        missed notification" path on the next check.
        """
        red_status = self._make_status('red', '2026-03-01T12:06:00')
        return {
            # 3 consecutive red statuses — hysteresis has already transitioned
            'statuses': [
                self._make_status('red', '2026-03-01T12:06:00'),
                self._make_status('red', '2026-03-01T12:03:00'),
                self._make_status('red', '2026-03-01T12:00:00'),
            ],
            'reported_status': red_status,
            'best_status': red_status,
            'pending_status': None,
            'pending_streak': 0,
            'cached_at': '2026-03-01T12:06:00',
            'last_successful_check': '2026-03-01T12:06:00',
            'consecutive_failures': 0,
            'last_error': None,
            # KEY: last_notified_status is still 'green' — the notification was missed
            'last_notified_status': 'green',
        }

    def test_unconfigured_channel_does_not_block_last_notified(self):
        """Unconfigured channels should NOT prevent last_notified_status from being saved.

        This is the core regression test. When a channel returns success=False
        with "Not configured", it should be treated as a no-op, not a failure.
        If last_notified_status is not saved, every subsequent check will
        re-trigger notifications via the "recovering missed notification" path.
        """
        cache_before = self._make_cache_with_missed_notification()

        # Dispatcher returns: RSS succeeded, others "Not configured"
        notify_results = {
            'rss': {'success': True},
            'bluesky': {'success': False, 'error': 'Not configured: BLUESKY_HANDLE'},
            'mastodon': {'success': False, 'error': 'Not configured: MASTODON_ACCESS_TOKEN'},
            'webhooks': {'success': False, 'error': 'Not configured: no WEBHOOK_URLS'},
        }

        saved_caches = self._run_check_status(cache_before, notify_results)

        # The final cache write should have last_notified_status updated to 'red'
        # Bug behavior: last_notified_status stays 'green' because all_succeeded=False
        final_cache = saved_caches[-1]
        assert final_cache['last_notified_status'] == 'red', \
            "last_notified_status should be updated even when unconfigured channels return success=False"

    def test_real_failure_blocks_last_notified(self):
        """A genuine notification failure SHOULD prevent last_notified_status update.

        This ensures we didn't over-correct — if a configured channel actually
        fails (network error, auth error, etc.), we should NOT update
        last_notified_status so recovery is attempted on the next run.
        """
        cache_before = self._make_cache_with_missed_notification()

        # Dispatcher returns: RSS succeeded, but Bluesky had a real failure
        notify_results = {
            'rss': {'success': True},
            'bluesky': {'success': False, 'error': 'HTTP 500: Internal Server Error'},
            'webhooks': {'success': False, 'error': 'Not configured: no WEBHOOK_URLS'},
        }

        saved_caches = self._run_check_status(cache_before, notify_results)

        # last_notified_status should NOT be updated because Bluesky actually failed
        final_cache = saved_caches[-1]
        assert final_cache.get('last_notified_status') != 'red', \
            "last_notified_status should NOT be updated when a real channel failure occurs"
