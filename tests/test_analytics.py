"""
Tests for the analytics module (SQLite logging and queries).

Tests database operations, status check logging, delay incident tracking,
and analytics query functions.
"""

import json
import os
import pytest
import sqlite3
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lib import analytics


class TestDatabaseInitialization:
    """Tests for database initialization and schema management."""

    def test_init_db_creates_tables(self, tmp_path):
        """Test that init_db creates all required tables."""
        db_path = tmp_path / "test_analytics.db"

        with patch.object(analytics, 'LOCAL_DB_PATH', db_path):
            with patch.object(analytics, '_is_cloud_run', return_value=False):
                analytics.init_db()

        # Verify database was created
        assert db_path.exists()

        # Verify tables exist
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cursor.fetchall()}

        assert 'status_checks' in tables
        assert 'delay_incidents' in tables
        assert 'schema_version' in tables

        conn.close()

    def test_init_db_creates_indexes(self, tmp_path):
        """Test that init_db creates performance indexes."""
        db_path = tmp_path / "test_analytics.db"

        with patch.object(analytics, 'LOCAL_DB_PATH', db_path):
            with patch.object(analytics, '_is_cloud_run', return_value=False):
                analytics.init_db()

        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        cursor.execute("SELECT name FROM sqlite_master WHERE type='index'")
        indexes = {row[0] for row in cursor.fetchall()}

        # Check expected indexes exist
        expected_indexes = [
            'idx_checks_timestamp',
            'idx_checks_status',
            'idx_incidents_station',
            'idx_incidents_type',
            'idx_incidents_timestamp',
        ]

        for idx in expected_indexes:
            assert idx in indexes, f"Missing index: {idx}"

        conn.close()

    def test_schema_version_tracking(self, tmp_path):
        """Test that schema version is tracked correctly."""
        db_path = tmp_path / "test_analytics.db"

        with patch.object(analytics, 'LOCAL_DB_PATH', db_path):
            with patch.object(analytics, '_is_cloud_run', return_value=False):
                analytics.init_db()

        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute('SELECT version FROM schema_version')
        row = cursor.fetchone()
        assert row is not None
        assert row['version'] == analytics.SCHEMA_VERSION

        conn.close()

    def test_init_db_idempotent(self, tmp_path):
        """Test that init_db can be called multiple times safely."""
        db_path = tmp_path / "test_analytics.db"

        with patch.object(analytics, 'LOCAL_DB_PATH', db_path):
            with patch.object(analytics, '_is_cloud_run', return_value=False):
                # Call multiple times
                analytics.init_db()
                analytics.init_db()
                analytics.init_db()

        # Should still work and have correct schema
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute('SELECT version FROM schema_version')
        assert cursor.fetchone()['version'] == analytics.SCHEMA_VERSION

        conn.close()


class TestStatusCheckLogging:
    """Tests for logging status checks."""

    @pytest.fixture
    def test_db(self, tmp_path):
        """Create a fresh test database."""
        db_path = tmp_path / "test_analytics.db"

        with patch.object(analytics, 'LOCAL_DB_PATH', db_path):
            with patch.object(analytics, '_is_cloud_run', return_value=False):
                with patch.object(analytics, 'backup_db_to_gcs', return_value=False):
                    analytics.init_db()
                    yield db_path

    def test_log_green_status(self, test_db):
        """Test logging a green status check."""
        with patch.object(analytics, 'LOCAL_DB_PATH', test_db):
            with patch.object(analytics, '_is_cloud_run', return_value=False):
                with patch.object(analytics, 'backup_db_to_gcs', return_value=False):
                    timestamp = datetime.now().isoformat()
                    detection_data = {
                        'trains': [{'id': 'W2010LL'}, {'id': 'M2089MM'}],
                        'delays_platforms': [],
                        'delays_segments': [],
                        'delays_bunching': [],
                    }

                    check_id = analytics.log_status_check(
                        status='green',
                        best_status='green',
                        detection_data=detection_data,
                        timestamp=timestamp
                    )

                    assert check_id is not None
                    assert check_id > 0

        # Verify the record was created
        conn = sqlite3.connect(str(test_db))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute('SELECT * FROM status_checks WHERE id = ?', (check_id,))
        row = cursor.fetchone()

        assert row is not None
        assert row['status'] == 'green'
        assert row['best_status'] == 'green'
        assert row['train_count'] == 2

        # Green status should not create delay incidents
        cursor.execute('SELECT COUNT(*) as count FROM delay_incidents WHERE check_id = ?', (check_id,))
        assert cursor.fetchone()['count'] == 0

        conn.close()

    def test_log_yellow_status_with_platform_holds(self, test_db):
        """Test logging a yellow status with platform hold delays."""
        with patch.object(analytics, 'LOCAL_DB_PATH', test_db):
            with patch.object(analytics, '_is_cloud_run', return_value=False):
                with patch.object(analytics, 'backup_db_to_gcs', return_value=False):
                    timestamp = datetime.now().isoformat()
                    detection_data = {
                        'trains': [{'id': 'W2010LL'}, {'id': 'M2089MM'}],
                        'delays_platforms': [
                            {'station': 'PO', 'name': 'Powell', 'direction': 'Westbound'},
                            {'station': 'MO', 'name': 'Montgomery', 'direction': 'Eastbound'},
                        ],
                        'delays_segments': [],
                        'delays_bunching': [],
                    }

                    check_id = analytics.log_status_check(
                        status='yellow',
                        best_status='yellow',
                        detection_data=detection_data,
                        timestamp=timestamp
                    )

        # Verify delay incidents were created
        conn = sqlite3.connect(str(test_db))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute('''
            SELECT * FROM delay_incidents
            WHERE check_id = ?
            ORDER BY station
        ''', (check_id,))
        incidents = cursor.fetchall()

        assert len(incidents) == 2

        # Check Montgomery incident
        mo_incident = next(i for i in incidents if i['station'] == 'MO')
        assert mo_incident['type'] == 'platform_hold'
        assert mo_incident['station_name'] == 'Montgomery'
        assert mo_incident['direction'] == 'Eastbound'

        # Check Powell incident
        po_incident = next(i for i in incidents if i['station'] == 'PO')
        assert po_incident['type'] == 'platform_hold'
        assert po_incident['station_name'] == 'Powell'
        assert po_incident['direction'] == 'Westbound'

        conn.close()

    def test_log_yellow_status_with_red_segments(self, test_db):
        """Test logging a yellow status with red track segments."""
        with patch.object(analytics, 'LOCAL_DB_PATH', test_db):
            with patch.object(analytics, '_is_cloud_run', return_value=False):
                with patch.object(analytics, 'backup_db_to_gcs', return_value=False):
                    timestamp = datetime.now().isoformat()
                    detection_data = {
                        'trains': [{'id': 'W2010LL'}, {'id': 'M2089MM'}],
                        'delays_platforms': [],
                        'delays_segments': [
                            {'from': 'CA', 'to': 'CH', 'direction': 'Westbound'},
                        ],
                        'delays_bunching': [],
                    }

                    check_id = analytics.log_status_check(
                        status='yellow',
                        best_status='yellow',
                        detection_data=detection_data,
                        timestamp=timestamp
                    )

        conn = sqlite3.connect(str(test_db))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute('SELECT * FROM delay_incidents WHERE check_id = ?', (check_id,))
        incidents = cursor.fetchall()

        assert len(incidents) == 1

        incident = incidents[0]
        assert incident['type'] == 'red_segment'
        assert incident['station'] == 'CA'
        assert incident['station_name'] == 'Castro'

        # Check details JSON
        details = json.loads(incident['details'])
        assert details['from'] == 'CA'
        assert details['to'] == 'CH'

        conn.close()

    def test_log_yellow_status_with_bunching(self, test_db):
        """Test logging a yellow status with train bunching."""
        with patch.object(analytics, 'LOCAL_DB_PATH', test_db):
            with patch.object(analytics, '_is_cloud_run', return_value=False):
                with patch.object(analytics, 'backup_db_to_gcs', return_value=False):
                    timestamp = datetime.now().isoformat()
                    detection_data = {
                        'trains': [{'id': f'W{i}' for i in range(5)}],
                        'delays_platforms': [],
                        'delays_segments': [],
                        'delays_bunching': [
                            {'station': 'EM', 'direction': 'Lower', 'train_count': 5},
                        ],
                    }

                    check_id = analytics.log_status_check(
                        status='yellow',
                        best_status='yellow',
                        detection_data=detection_data,
                        timestamp=timestamp
                    )

        conn = sqlite3.connect(str(test_db))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute('SELECT * FROM delay_incidents WHERE check_id = ?', (check_id,))
        incidents = cursor.fetchall()

        assert len(incidents) == 1

        incident = incidents[0]
        assert incident['type'] == 'bunching'
        assert incident['station'] == 'EM'
        assert incident['station_name'] == 'Embarcadero'

        details = json.loads(incident['details'])
        assert details['train_count'] == 5

        conn.close()

    def test_red_status_no_incidents(self, test_db):
        """Test that red status (not operating) doesn't log delay incidents."""
        with patch.object(analytics, 'LOCAL_DB_PATH', test_db):
            with patch.object(analytics, '_is_cloud_run', return_value=False):
                with patch.object(analytics, 'backup_db_to_gcs', return_value=False):
                    timestamp = datetime.now().isoformat()
                    detection_data = {
                        'trains': [],
                        'delays_platforms': [],
                        'delays_segments': [],
                        'delays_bunching': [],
                    }

                    check_id = analytics.log_status_check(
                        status='red',
                        best_status='red',
                        detection_data=detection_data,
                        timestamp=timestamp
                    )

        conn = sqlite3.connect(str(test_db))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Verify status check was logged
        cursor.execute('SELECT * FROM status_checks WHERE id = ?', (check_id,))
        assert cursor.fetchone() is not None

        # Verify no delay incidents were created
        cursor.execute('SELECT COUNT(*) as count FROM delay_incidents WHERE check_id = ?', (check_id,))
        assert cursor.fetchone()['count'] == 0

        conn.close()


class TestDelayFrequencyQuery:
    """Tests for get_delay_frequency query function."""

    @pytest.fixture
    def populated_db(self, tmp_path):
        """Create a database with test data."""
        db_path = tmp_path / "test_analytics.db"

        with patch.object(analytics, 'LOCAL_DB_PATH', db_path):
            with patch.object(analytics, '_is_cloud_run', return_value=False):
                with patch.object(analytics, 'backup_db_to_gcs', return_value=False):
                    analytics.init_db()

                    # Log various status checks with 30-second intervals
                    base_time = datetime.now() - timedelta(days=3)

                    for i in range(100):
                        ts = (base_time + timedelta(seconds=i * 30)).isoformat()

                        if i < 70:  # 70% green
                            status = 'green'
                        elif i < 90:  # 20% yellow
                            status = 'yellow'
                        else:  # 10% red
                            status = 'red'

                        detection_data = {
                            'trains': [{'id': 'W1'}] if status != 'red' else [],
                            'delays_platforms': [{'station': 'PO', 'name': 'Powell', 'direction': 'WB'}] if status == 'yellow' else [],
                            'delays_segments': [],
                            'delays_bunching': [],
                        }

                        analytics.log_status_check(
                            status=status,
                            best_status=status,
                            detection_data=detection_data,
                            timestamp=ts,
                            interval_seconds=30  # 30 second intervals
                        )

                    yield db_path

    def test_get_delay_frequency_basic(self, populated_db):
        """Test basic delay frequency calculation (time-based)."""
        with patch.object(analytics, 'LOCAL_DB_PATH', populated_db):
            with patch.object(analytics, '_is_cloud_run', return_value=False):
                result = analytics.get_delay_frequency(days=7)

        # 100 records at 30s intervals = 99 inter-record gaps * 30s = 49.5 min
        # Plus gap from cutoff (7 days ago) to first record (3 days ago), capped at 5 min
        # Plus trailing gap from last record (3 days ago) to now, capped at 5 min
        assert result['total_minutes'] == 59.5
        # 20 yellow records contribute 20 gaps * 30 seconds = 10 minutes delayed
        assert result['delayed_minutes'] == 10.0
        assert 0.15 <= result['delay_rate'] <= 0.18  # ~17%

        # Time per status in minutes
        # First record (green) gets capped 5-min gap from cutoff, records 1-69 get 69 * 30s
        # Last record (red) gets capped 5-min trailing gap to now
        assert result['by_status']['green'] == 39.5  # (69 gaps * 30s) + 5 min cutoff gap
        assert result['by_status']['yellow'] == 10.0  # 20 gaps * 30s = 10 min
        assert result['by_status']['red'] == 10.0  # (10 gaps * 30s) + 5 min trailing gap

    def test_get_delay_frequency_empty_db(self, tmp_path):
        """Test delay frequency with empty database."""
        db_path = tmp_path / "empty_analytics.db"

        with patch.object(analytics, 'LOCAL_DB_PATH', db_path):
            with patch.object(analytics, '_is_cloud_run', return_value=False):
                analytics.init_db()
                result = analytics.get_delay_frequency(days=7)

        assert result['total_minutes'] == 0.0
        assert result['delayed_minutes'] == 0.0
        assert result['delay_rate'] == 0.0
        assert result['by_status'] == {'green': 0.0, 'yellow': 0.0, 'red': 0.0}


class TestDelaysByStationQuery:
    """Tests for get_delays_by_station query function."""

    @pytest.fixture
    def station_db(self, tmp_path):
        """Create database with station-specific delay data."""
        db_path = tmp_path / "test_analytics.db"

        with patch.object(analytics, 'LOCAL_DB_PATH', db_path):
            with patch.object(analytics, '_is_cloud_run', return_value=False):
                with patch.object(analytics, 'backup_db_to_gcs', return_value=False):
                    analytics.init_db()

                    # Log delays at different stations with 60-second intervals
                    base_time = datetime.now() - timedelta(days=1)

                    station_delays = [
                        ('PO', 'Powell', 5),      # 5 checks at Powell = 5 min
                        ('MO', 'Montgomery', 3),  # 3 checks at Montgomery = 3 min
                        ('EM', 'Embarcadero', 2), # 2 checks at Embarcadero = 2 min
                    ]

                    for station, name, count in station_delays:
                        for i in range(count):
                            ts = (base_time + timedelta(hours=i)).isoformat()
                            detection_data = {
                                'trains': [{'id': 'W1'}],
                                'delays_platforms': [{'station': station, 'name': name, 'direction': 'WB'}],
                                'delays_segments': [],
                                'delays_bunching': [],
                            }
                            analytics.log_status_check(
                                status='yellow',
                                best_status='yellow',
                                detection_data=detection_data,
                                timestamp=ts,
                                interval_seconds=60  # 60 second intervals for easy math
                            )

                    yield db_path

    def test_get_delays_by_station_ordering(self, station_db):
        """Test that stations are ordered by delay time descending."""
        with patch.object(analytics, 'LOCAL_DB_PATH', station_db):
            with patch.object(analytics, '_is_cloud_run', return_value=False):
                result = analytics.get_delays_by_station(days=7)

        assert len(result) == 3

        # Should be ordered by minutes descending
        assert result[0]['station'] == 'PO'
        assert result[0]['minutes'] == 5.0  # 5 checks * 60s = 5 min
        assert result[0]['name'] == 'Powell'

        assert result[1]['station'] == 'MO'
        assert result[1]['minutes'] == 3.0  # 3 checks * 60s = 3 min

        assert result[2]['station'] == 'EM'
        assert result[2]['minutes'] == 2.0  # 2 checks * 60s = 2 min

    def test_get_delays_by_station_types(self, station_db):
        """Test that delay types are tracked per station (in minutes)."""
        with patch.object(analytics, 'LOCAL_DB_PATH', station_db):
            with patch.object(analytics, '_is_cloud_run', return_value=False):
                result = analytics.get_delays_by_station(days=7)

        # All delays in this fixture are platform_hold type
        for station in result:
            assert 'platform_hold' in station['types']
            assert station['types']['platform_hold'] == station['minutes']


class TestDelaysByTimeQuery:
    """Tests for get_delays_by_time query function."""

    @pytest.fixture
    def time_db(self, tmp_path):
        """Create database with time-distributed delay data."""
        db_path = tmp_path / "test_analytics.db"

        with patch.object(analytics, 'LOCAL_DB_PATH', db_path):
            with patch.object(analytics, '_is_cloud_run', return_value=False):
                with patch.object(analytics, 'backup_db_to_gcs', return_value=False):
                    analytics.init_db()

                    # Log delays at specific hours with 60-second intervals
                    # Monday at 8am, 9am, 5pm, 6pm (rush hours)
                    # Find a recent Monday
                    today = datetime.now()
                    days_since_monday = today.weekday()
                    last_monday = today - timedelta(days=days_since_monday)

                    rush_hours = [8, 9, 17, 18]
                    for hour in rush_hours:
                        ts = last_monday.replace(hour=hour, minute=0, second=0).isoformat()
                        detection_data = {
                            'trains': [{'id': 'W1'}],
                            'delays_platforms': [{'station': 'PO', 'name': 'Powell', 'direction': 'WB'}],
                            'delays_segments': [],
                            'delays_bunching': [],
                        }
                        analytics.log_status_check(
                            status='yellow',
                            best_status='yellow',
                            detection_data=detection_data,
                            timestamp=ts,
                            interval_seconds=60  # 60 second intervals = 1 minute each
                        )

                    yield db_path

    def test_get_delays_by_time_hours(self, time_db):
        """Test delay distribution by hour of day (in minutes)."""
        with patch.object(analytics, 'LOCAL_DB_PATH', time_db):
            with patch.object(analytics, '_is_cloud_run', return_value=False):
                result = analytics.get_delays_by_time(days=7)

        by_hour = result['by_hour']

        # Rush hours should have 1 minute of delay each
        assert by_hour[8] == 1.0
        assert by_hour[9] == 1.0
        assert by_hour[17] == 1.0
        assert by_hour[18] == 1.0

        # Non-rush hours should be empty
        assert by_hour[3] == 0.0
        assert by_hour[14] == 0.0

    def test_get_delays_by_time_days(self, time_db):
        """Test delay distribution by day of week (in minutes)."""
        with patch.object(analytics, 'LOCAL_DB_PATH', time_db):
            with patch.object(analytics, '_is_cloud_run', return_value=False):
                result = analytics.get_delays_by_time(days=7)

        by_day = result['by_day']

        # All delays are on Monday (weekday 0) = 4 minutes total
        assert by_day[0] == 4.0

        # Other days should be empty
        for day in range(1, 7):
            assert by_day[day] == 0.0


class TestRecentIncidentsQuery:
    """Tests for get_recent_incidents query function."""

    def test_get_recent_incidents_ordering(self, tmp_path):
        """Test that incidents are ordered by timestamp descending."""
        db_path = tmp_path / "test_analytics.db"

        with patch.object(analytics, 'LOCAL_DB_PATH', db_path):
            with patch.object(analytics, '_is_cloud_run', return_value=False):
                with patch.object(analytics, 'backup_db_to_gcs', return_value=False):
                    analytics.init_db()

                    # Log incidents at different times
                    base_time = datetime.now() - timedelta(hours=5)
                    stations = ['PO', 'MO', 'EM', 'CC', 'VN']

                    for i, station in enumerate(stations):
                        ts = (base_time + timedelta(hours=i)).isoformat()
                        detection_data = {
                            'trains': [{'id': 'W1'}],
                            'delays_platforms': [{'station': station, 'name': analytics.STATION_NAMES[station], 'direction': 'WB'}],
                            'delays_segments': [],
                            'delays_bunching': [],
                        }
                        analytics.log_status_check(
                            status='yellow',
                            best_status='yellow',
                            detection_data=detection_data,
                            timestamp=ts
                        )

                    result = analytics.get_recent_incidents(limit=10)

        # Should be ordered newest first
        assert len(result) == 5
        assert result[0]['station'] == 'VN'  # Most recent
        assert result[4]['station'] == 'PO'  # Oldest

    def test_get_recent_incidents_limit(self, tmp_path):
        """Test that limit parameter works correctly."""
        db_path = tmp_path / "test_analytics.db"

        with patch.object(analytics, 'LOCAL_DB_PATH', db_path):
            with patch.object(analytics, '_is_cloud_run', return_value=False):
                with patch.object(analytics, 'backup_db_to_gcs', return_value=False):
                    analytics.init_db()

                    # Log 10 incidents
                    base_time = datetime.now() - timedelta(hours=10)
                    for i in range(10):
                        ts = (base_time + timedelta(hours=i)).isoformat()
                        detection_data = {
                            'trains': [{'id': 'W1'}],
                            'delays_platforms': [{'station': 'PO', 'name': 'Powell', 'direction': 'WB'}],
                            'delays_segments': [],
                            'delays_bunching': [],
                        }
                        analytics.log_status_check(
                            status='yellow',
                            best_status='yellow',
                            detection_data=detection_data,
                            timestamp=ts
                        )

                    result = analytics.get_recent_incidents(limit=3)

        assert len(result) == 3


class TestReportGeneration:
    """Tests for analytics report generation and caching."""

    def test_generate_report_structure(self, tmp_path):
        """Test that generated report has correct structure."""
        db_path = tmp_path / "test_analytics.db"
        cache_dir = tmp_path / "cache"

        with patch.object(analytics, 'LOCAL_DB_PATH', db_path):
            with patch.object(analytics, 'LOCAL_CACHE_DIR', cache_dir):
                with patch.object(analytics, '_is_cloud_run', return_value=False):
                    with patch.object(analytics, 'backup_db_to_gcs', return_value=False):
                        analytics.init_db()

                        # Log some data
                        analytics.log_status_check(
                            status='green',
                            best_status='green',
                            detection_data={'trains': [], 'delays_platforms': [], 'delays_segments': [], 'delays_bunching': []},
                            timestamp=datetime.now().isoformat()
                        )

                        report = analytics.generate_report(days=7)

        assert report is not None
        assert 'period_days' in report
        assert report['period_days'] == 7

        assert 'frequency' in report
        assert 'total_minutes' in report['frequency']
        assert 'delay_rate' in report['frequency']

        assert 'by_station' in report
        assert isinstance(report['by_station'], list)

        assert 'by_time' in report
        assert 'by_hour' in report['by_time']
        assert 'by_day' in report['by_time']

        assert 'generated_at' in report
        assert 'cached_at' in report

    def test_empty_report_structure(self):
        """Test that empty report has correct structure."""
        report = analytics._empty_report(7)

        assert report['period_days'] == 7
        assert report['frequency']['total_minutes'] == 0.0
        assert report['frequency']['delay_rate'] == 0.0
        assert report['by_station'] == []
        assert report['no_data'] is True


class TestDatabaseHealthCheck:
    """Tests for database health check functionality."""

    def test_health_check_no_database(self, tmp_path):
        """Test health check when database doesn't exist."""
        db_path = tmp_path / "nonexistent.db"

        with patch.object(analytics, 'LOCAL_DB_PATH', db_path):
            with patch.object(analytics, '_is_cloud_run', return_value=False):
                result = analytics.check_database_health()

        assert result['exists'] is False
        assert result['has_data'] is False

    def test_health_check_empty_database(self, tmp_path):
        """Test health check with empty database."""
        db_path = tmp_path / "test_analytics.db"

        with patch.object(analytics, 'LOCAL_DB_PATH', db_path):
            with patch.object(analytics, '_is_cloud_run', return_value=False):
                analytics.init_db()
                result = analytics.check_database_health()

        assert result['exists'] is True
        assert result['has_data'] is False
        assert result['check_count'] == 0

    def test_health_check_with_data(self, tmp_path):
        """Test health check with data in database."""
        db_path = tmp_path / "test_analytics.db"

        with patch.object(analytics, 'LOCAL_DB_PATH', db_path):
            with patch.object(analytics, '_is_cloud_run', return_value=False):
                with patch.object(analytics, 'backup_db_to_gcs', return_value=False):
                    analytics.init_db()

                    # Log some checks
                    for _ in range(5):
                        analytics.log_status_check(
                            status='green',
                            best_status='green',
                            detection_data={'trains': [], 'delays_platforms': [], 'delays_segments': [], 'delays_bunching': []},
                            timestamp=datetime.now().isoformat()
                        )

                    result = analytics.check_database_health()

        assert result['exists'] is True
        assert result['has_data'] is True
        assert result['check_count'] == 5


class TestIntervalHandling:
    """Tests for check interval handling in time-based analytics."""

    @pytest.fixture
    def test_db(self, tmp_path):
        """Create a fresh test database."""
        db_path = tmp_path / "test_analytics.db"

        with patch.object(analytics, 'LOCAL_DB_PATH', db_path):
            with patch.object(analytics, '_is_cloud_run', return_value=False):
                with patch.object(analytics, 'backup_db_to_gcs', return_value=False):
                    analytics.init_db()
                    yield db_path

    def test_interval_stored_correctly(self, test_db):
        """Test that interval_seconds is stored in the database."""
        with patch.object(analytics, 'LOCAL_DB_PATH', test_db):
            with patch.object(analytics, '_is_cloud_run', return_value=False):
                with patch.object(analytics, 'backup_db_to_gcs', return_value=False):
                    check_id = analytics.log_status_check(
                        status='green',
                        best_status='green',
                        detection_data={'trains': [], 'delays_platforms': [], 'delays_segments': [], 'delays_bunching': []},
                        timestamp=datetime.now().isoformat(),
                        interval_seconds=180
                    )

        # Verify interval was stored
        conn = sqlite3.connect(str(test_db))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute('SELECT interval_seconds FROM status_checks WHERE id = ?', (check_id,))
        row = cursor.fetchone()
        conn.close()

        assert row['interval_seconds'] == 180

    def test_default_interval_used_when_not_specified(self, test_db):
        """Test that DEFAULT_CHECK_INTERVAL is used when interval not specified."""
        from lib.config import DEFAULT_CHECK_INTERVAL

        with patch.object(analytics, 'LOCAL_DB_PATH', test_db):
            with patch.object(analytics, '_is_cloud_run', return_value=False):
                with patch.object(analytics, 'backup_db_to_gcs', return_value=False):
                    check_id = analytics.log_status_check(
                        status='green',
                        best_status='green',
                        detection_data={'trains': [], 'delays_platforms': [], 'delays_segments': [], 'delays_bunching': []},
                        timestamp=datetime.now().isoformat()
                        # No interval_seconds specified
                    )

        conn = sqlite3.connect(str(test_db))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute('SELECT interval_seconds FROM status_checks WHERE id = ?', (check_id,))
        row = cursor.fetchone()
        conn.close()

        assert row['interval_seconds'] == DEFAULT_CHECK_INTERVAL

    def test_different_intervals_calculate_correctly(self, test_db):
        """Test that checks with different intervals calculate correct total time."""
        with patch.object(analytics, 'LOCAL_DB_PATH', test_db):
            with patch.object(analytics, '_is_cloud_run', return_value=False):
                with patch.object(analytics, 'backup_db_to_gcs', return_value=False):
                    base_time = datetime.now() - timedelta(hours=1)

                    # 2 checks at 30 seconds = 1 minute
                    for i in range(2):
                        ts = (base_time + timedelta(minutes=i)).isoformat()
                        analytics.log_status_check(
                            status='green',
                            best_status='green',
                            detection_data={'trains': [], 'delays_platforms': [], 'delays_segments': [], 'delays_bunching': []},
                            timestamp=ts,
                            interval_seconds=30
                        )

                    # 2 checks at 180 seconds = 6 minutes
                    for i in range(2):
                        ts = (base_time + timedelta(minutes=10 + i)).isoformat()
                        analytics.log_status_check(
                            status='green',
                            best_status='green',
                            detection_data={'trains': [], 'delays_platforms': [], 'delays_segments': [], 'delays_bunching': []},
                            timestamp=ts,
                            interval_seconds=180
                        )

                    result = analytics.get_delay_frequency(days=1)

        # Inter-record gaps: 1 + 6 = 7 minutes
        # Plus gap from cutoff (1 day ago) to first record (1 hour ago), capped at 5 min
        # Plus trailing gap from last record (50 min ago) to now, capped at 5 min
        assert result['total_minutes'] == 17.0

    def test_mixed_intervals_in_station_delays(self, test_db):
        """Test that station delay minutes are calculated correctly.

        Note: Station delays use a fixed 60-second estimate per incident
        rather than stored intervals (which may be inaccurate).
        """
        with patch.object(analytics, 'LOCAL_DB_PATH', test_db):
            with patch.object(analytics, '_is_cloud_run', return_value=False):
                with patch.object(analytics, 'backup_db_to_gcs', return_value=False):
                    base_time = datetime.now() - timedelta(hours=1)

                    # 2 delays at Powell (60 seconds each = 2 minutes total)
                    analytics.log_status_check(
                        status='yellow',
                        best_status='yellow',
                        detection_data={
                            'trains': [{'id': 'W1'}],
                            'delays_platforms': [{'station': 'PO', 'name': 'Powell', 'direction': 'WB'}],
                            'delays_segments': [],
                            'delays_bunching': []
                        },
                        timestamp=base_time.isoformat(),
                        interval_seconds=30
                    )

                    analytics.log_status_check(
                        status='yellow',
                        best_status='yellow',
                        detection_data={
                            'trains': [{'id': 'W1'}],
                            'delays_platforms': [{'station': 'PO', 'name': 'Powell', 'direction': 'WB'}],
                            'delays_segments': [],
                            'delays_bunching': []
                        },
                        timestamp=(base_time + timedelta(minutes=5)).isoformat(),
                        interval_seconds=180
                    )

                    result = analytics.get_delays_by_station(days=1)

        # Powell should have 2 incidents × 60 seconds = 2 minutes of delay
        powell = next(s for s in result if s['station'] == 'PO')
        assert powell['minutes'] == 2.0

    def test_null_interval_uses_default_in_queries(self, test_db):
        """Test that timestamp-based calculation handles a single record.

        A single record gets the gap from cutoff to its timestamp,
        capped at MAX_GAP_SECONDS (5 min).
        """
        # Manually insert a record with NULL interval (simulating pre-migration data)
        conn = sqlite3.connect(str(test_db))
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO status_checks (timestamp, status, best_status, train_count, interval_seconds)
            VALUES (?, 'green', 'green', 0, NULL)
        ''', (datetime.now().isoformat(),))
        conn.commit()
        conn.close()

        with patch.object(analytics, 'LOCAL_DB_PATH', test_db):
            with patch.object(analytics, '_is_cloud_run', return_value=False):
                result = analytics.get_delay_frequency(days=1)

        # Single record gets capped gap from cutoff (1 day ago) = 5 min
        assert result['total_minutes'] == 5.0

    def test_cloud_run_interval_constant_exists(self):
        """Test that CLOUD_CHECK_INTERVAL is defined in config."""
        from lib.config import CLOUD_CHECK_INTERVAL, DEFAULT_CHECK_INTERVAL

        # Cloud Run interval should be 180 seconds (3 minutes)
        assert CLOUD_CHECK_INTERVAL == 180
        # Local interval should be 30 seconds
        assert DEFAULT_CHECK_INTERVAL == 30
        # Cloud Run interval should be longer than local
        assert CLOUD_CHECK_INTERVAL > DEFAULT_CHECK_INTERVAL
