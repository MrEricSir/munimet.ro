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
