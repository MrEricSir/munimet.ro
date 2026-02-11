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

                    # Log various status checks
                    base_time = datetime.now() - timedelta(days=3)

                    for i in range(100):
                        ts = (base_time + timedelta(minutes=i * 30)).isoformat()

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
                            timestamp=ts
                        )

                    yield db_path

    def test_get_delay_frequency_basic(self, populated_db):
        """Test basic delay frequency calculation."""
        with patch.object(analytics, 'LOCAL_DB_PATH', populated_db):
            with patch.object(analytics, '_is_cloud_run', return_value=False):
                result = analytics.get_delay_frequency(days=7)

        assert result['total_checks'] == 100
        assert result['delayed_checks'] == 20  # 20% yellow
        assert 0.19 <= result['delay_rate'] <= 0.21  # ~20%

        assert result['by_status']['green'] == 70
        assert result['by_status']['yellow'] == 20
        assert result['by_status']['red'] == 10

    def test_get_delay_frequency_empty_db(self, tmp_path):
        """Test delay frequency with empty database."""
        db_path = tmp_path / "empty_analytics.db"

        with patch.object(analytics, 'LOCAL_DB_PATH', db_path):
            with patch.object(analytics, '_is_cloud_run', return_value=False):
                analytics.init_db()
                result = analytics.get_delay_frequency(days=7)

        assert result['total_checks'] == 0
        assert result['delayed_checks'] == 0
        assert result['delay_rate'] == 0.0
        assert result['by_status'] == {'green': 0, 'yellow': 0, 'red': 0}


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

                    # Log delays at different stations
                    base_time = datetime.now() - timedelta(days=1)

                    station_delays = [
                        ('PO', 'Powell', 5),      # 5 delays at Powell
                        ('MO', 'Montgomery', 3),  # 3 delays at Montgomery
                        ('EM', 'Embarcadero', 2), # 2 delays at Embarcadero
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
                                timestamp=ts
                            )

                    yield db_path

    def test_get_delays_by_station_ordering(self, station_db):
        """Test that stations are ordered by delay count descending."""
        with patch.object(analytics, 'LOCAL_DB_PATH', station_db):
            with patch.object(analytics, '_is_cloud_run', return_value=False):
                result = analytics.get_delays_by_station(days=7)

        assert len(result) == 3

        # Should be ordered by count descending
        assert result[0]['station'] == 'PO'
        assert result[0]['count'] == 5
        assert result[0]['name'] == 'Powell'

        assert result[1]['station'] == 'MO'
        assert result[1]['count'] == 3

        assert result[2]['station'] == 'EM'
        assert result[2]['count'] == 2

    def test_get_delays_by_station_types(self, station_db):
        """Test that delay types are tracked per station."""
        with patch.object(analytics, 'LOCAL_DB_PATH', station_db):
            with patch.object(analytics, '_is_cloud_run', return_value=False):
                result = analytics.get_delays_by_station(days=7)

        # All delays in this fixture are platform_hold type
        for station in result:
            assert 'platform_hold' in station['types']
            assert station['types']['platform_hold'] == station['count']


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

                    # Log delays at specific hours
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
                            timestamp=ts
                        )

                    yield db_path

    def test_get_delays_by_time_hours(self, time_db):
        """Test delay distribution by hour of day."""
        with patch.object(analytics, 'LOCAL_DB_PATH', time_db):
            with patch.object(analytics, '_is_cloud_run', return_value=False):
                result = analytics.get_delays_by_time(days=7)

        by_hour = result['by_hour']

        # Rush hours should have delays
        assert by_hour[8] >= 1
        assert by_hour[9] >= 1
        assert by_hour[17] >= 1
        assert by_hour[18] >= 1

        # Non-rush hours should be empty
        assert by_hour[3] == 0
        assert by_hour[14] == 0

    def test_get_delays_by_time_days(self, time_db):
        """Test delay distribution by day of week."""
        with patch.object(analytics, 'LOCAL_DB_PATH', time_db):
            with patch.object(analytics, '_is_cloud_run', return_value=False):
                result = analytics.get_delays_by_time(days=7)

        by_day = result['by_day']

        # All delays are on Monday (weekday 0)
        assert by_day[0] == 4

        # Other days should be empty
        for day in range(1, 7):
            assert by_day[day] == 0


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
        assert 'total_checks' in report['frequency']
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
        assert report['frequency']['total_checks'] == 0
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
