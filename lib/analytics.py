#!/usr/bin/env python3
"""
Analytics module for tracking Muni delay data.

Provides SQLite-based logging of status checks and delay incidents,
with query functions for analytics display.

On Cloud Run, the SQLite database and cache files are backed up to GCS
to persist across container restarts.
"""

import json
import os
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

# Path resolution
LIB_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = LIB_DIR.parent

# Local paths (used directly in development, as staging area in production)
LOCAL_DB_PATH = PROJECT_ROOT / "artifacts" / "runtime" / "analytics.db"
LOCAL_CACHE_DIR = PROJECT_ROOT / "artifacts" / "runtime" / "cache"

# Cache configuration
REPORT_CACHE_MAX_AGE = 86400  # 24 hours - reports regenerate daily

# Track if we've restored from GCS this session
_gcs_restored = False


def _is_cloud_run():
    """Check if running on Cloud Run."""
    return os.getenv('CLOUD_RUN') is not None


def _get_gcs_bucket():
    """Get the GCS bucket name."""
    return os.getenv('GCS_BUCKET', 'munimetro-cache')


def _get_gcs_client():
    """Get a GCS client (lazy import to avoid issues in local dev)."""
    from google.cloud import storage
    return storage.Client()


def restore_db_from_gcs():
    """
    Restore the analytics database from GCS if running on Cloud Run.
    Called once at startup to restore persisted data.

    Returns:
        bool: True if restored, False if no backup exists or not on Cloud Run
    """
    global _gcs_restored

    if not _is_cloud_run():
        return False

    if _gcs_restored:
        return True  # Already restored this session

    try:
        client = _get_gcs_client()
        bucket = client.bucket(_get_gcs_bucket())
        blob = bucket.blob('analytics/analytics.db')

        if not blob.exists():
            print("No analytics DB backup found in GCS")
            _gcs_restored = True
            return False

        # Ensure local directory exists
        os.makedirs(LOCAL_DB_PATH.parent, exist_ok=True)

        # Download the database
        blob.download_to_filename(str(LOCAL_DB_PATH))
        print(f"Restored analytics DB from GCS ({blob.size} bytes)")
        _gcs_restored = True
        return True
    except Exception as e:
        print(f"Error restoring analytics DB from GCS: {e}")
        _gcs_restored = True  # Don't retry on error
        return False


def backup_db_to_gcs():
    """
    Backup the analytics database to GCS if running on Cloud Run.
    Should be called periodically (e.g., after each status check).

    Returns:
        bool: True if backed up successfully
    """
    if not _is_cloud_run():
        return False

    if not LOCAL_DB_PATH.exists():
        return False

    try:
        client = _get_gcs_client()
        bucket = client.bucket(_get_gcs_bucket())
        blob = bucket.blob('analytics/analytics.db')

        blob.upload_from_filename(str(LOCAL_DB_PATH))
        return True
    except Exception as e:
        print(f"Error backing up analytics DB to GCS: {e}")
        return False


def _read_gcs_json(blob_name):
    """Read a JSON file from GCS."""
    try:
        client = _get_gcs_client()
        bucket = client.bucket(_get_gcs_bucket())
        blob = bucket.blob(blob_name)

        if not blob.exists():
            return None

        content = blob.download_as_string()
        return json.loads(content)
    except Exception as e:
        print(f"Error reading {blob_name} from GCS: {e}")
        return None


def _write_gcs_json(blob_name, data):
    """Write a JSON file to GCS."""
    try:
        client = _get_gcs_client()
        bucket = client.bucket(_get_gcs_bucket())
        blob = bucket.blob(blob_name)

        blob.upload_from_string(
            json.dumps(data),
            content_type='application/json'
        )
        return True
    except Exception as e:
        print(f"Error writing {blob_name} to GCS: {e}")
        return False


def get_db_connection():
    """Get a connection to the analytics database."""
    # On Cloud Run, restore from GCS if not already done
    if _is_cloud_run():
        restore_db_from_gcs()

    os.makedirs(LOCAL_DB_PATH.parent, exist_ok=True)
    conn = sqlite3.connect(str(LOCAL_DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


# Current schema version - increment when adding migrations
SCHEMA_VERSION = 1


def get_schema_version(cursor):
    """Get current schema version from database."""
    cursor.execute('''
        SELECT name FROM sqlite_master
        WHERE type='table' AND name='schema_version'
    ''')
    if not cursor.fetchone():
        return 0
    cursor.execute('SELECT version FROM schema_version')
    row = cursor.fetchone()
    return row['version'] if row else 0


def run_migrations(cursor, current_version):
    """
    Run database migrations to bring schema up to date.

    Add new migrations here as elif blocks when schema changes.
    Each migration should increment the version number.
    """
    if current_version < 1:
        # Initial schema (version 1)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS schema_version (
                version INTEGER PRIMARY KEY
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS status_checks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                status TEXT NOT NULL,
                best_status TEXT NOT NULL,
                train_count INTEGER,
                check_duration_ms INTEGER
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS delay_incidents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                check_id INTEGER NOT NULL,
                timestamp TEXT NOT NULL,
                type TEXT NOT NULL,
                station TEXT,
                station_name TEXT,
                direction TEXT,
                details TEXT,
                FOREIGN KEY (check_id) REFERENCES status_checks(id)
            )
        ''')

        cursor.execute('CREATE INDEX IF NOT EXISTS idx_checks_timestamp ON status_checks(timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_checks_status ON status_checks(status)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_incidents_station ON delay_incidents(station)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_incidents_type ON delay_incidents(type)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_incidents_timestamp ON delay_incidents(timestamp)')

        cursor.execute('INSERT OR REPLACE INTO schema_version (version) VALUES (1)')

    # Add future migrations here:
    # if current_version < 2:
    #     cursor.execute('ALTER TABLE ...')
    #     cursor.execute('UPDATE schema_version SET version = 2')


def init_db():
    """Initialize database and run any pending migrations."""
    conn = get_db_connection()
    cursor = conn.cursor()

    current_version = get_schema_version(cursor)

    if current_version < SCHEMA_VERSION:
        run_migrations(cursor, current_version)
        conn.commit()

    conn.close()


def log_status_check(status, best_status, detection_data, timestamp):
    """
    Log a status check and its delay incidents.

    Args:
        status: Raw detected status ('green', 'yellow', 'red')
        best_status: Smoothed best status
        detection_data: Dict with trains, delays_platforms, delays_segments, delays_bunching
        timestamp: ISO8601 timestamp string

    Returns:
        int: The check_id of the inserted record
    """
    # Ensure database exists
    init_db()

    conn = get_db_connection()
    cursor = conn.cursor()

    # Count trains
    trains = detection_data.get('trains', [])
    train_count = len(trains)

    # Insert status check
    cursor.execute('''
        INSERT INTO status_checks (timestamp, status, best_status, train_count)
        VALUES (?, ?, ?, ?)
    ''', (timestamp, status, best_status, train_count))

    check_id = cursor.lastrowid

    # Only log delay incidents for yellow status (actual service delays)
    # Skip red status - that's after-hours maintenance, not delays
    if status != 'yellow':
        conn.commit()
        conn.close()
        return check_id

    # Insert delay incidents
    delays_platforms = detection_data.get('delays_platforms', [])
    delays_segments = detection_data.get('delays_segments', [])
    delays_bunching = detection_data.get('delays_bunching', [])

    # Platform holds
    for delay in delays_platforms:
        cursor.execute('''
            INSERT INTO delay_incidents (check_id, timestamp, type, station, station_name, direction)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            check_id,
            timestamp,
            'platform_hold',
            delay.get('code'),
            delay.get('name'),
            delay.get('direction')
        ))

    # Red segments
    for delay in delays_segments:
        details = json.dumps({
            'from': delay.get('from'),
            'to': delay.get('to')
        })
        # Use the 'from' station as the primary station
        cursor.execute('''
            INSERT INTO delay_incidents (check_id, timestamp, type, station, station_name, direction, details)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            check_id,
            timestamp,
            'red_segment',
            delay.get('from_code'),
            delay.get('from'),
            delay.get('direction'),
            details
        ))

    # Train bunching
    for delay in delays_bunching:
        details = json.dumps({
            'train_count': delay.get('train_count')
        })
        cursor.execute('''
            INSERT INTO delay_incidents (check_id, timestamp, type, station, station_name, direction, details)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            check_id,
            timestamp,
            'bunching',
            delay.get('station_code'),
            delay.get('station'),
            delay.get('direction'),
            details
        ))

    conn.commit()
    conn.close()

    # Backup to GCS after logging (if on Cloud Run)
    backup_db_to_gcs()

    return check_id


def get_delay_frequency(days=7):
    """
    Get delay frequency statistics for the specified time period.

    Args:
        days: Number of days to look back

    Returns:
        dict: {
            'total_checks': int,
            'delayed_checks': int,
            'delay_rate': float,
            'by_status': {'green': int, 'yellow': int, 'red': int}
        }
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    cutoff = (datetime.now() - timedelta(days=days)).isoformat()

    # Total checks
    cursor.execute('''
        SELECT COUNT(*) as total FROM status_checks WHERE timestamp >= ?
    ''', (cutoff,))
    total_checks = cursor.fetchone()['total']

    # Checks by status
    cursor.execute('''
        SELECT status, COUNT(*) as count
        FROM status_checks
        WHERE timestamp >= ?
        GROUP BY status
    ''', (cutoff,))

    by_status = {'green': 0, 'yellow': 0, 'red': 0}
    for row in cursor.fetchall():
        by_status[row['status']] = row['count']

    conn.close()

    # Only count yellow as delayed - red is after-hours maintenance
    delayed_checks = by_status['yellow']
    delay_rate = delayed_checks / total_checks if total_checks > 0 else 0.0

    return {
        'total_checks': total_checks,
        'delayed_checks': delayed_checks,
        'delay_rate': round(delay_rate, 4),
        'by_status': by_status
    }


def get_delays_by_station(days=7):
    """
    Get delay counts grouped by station.

    Args:
        days: Number of days to look back

    Returns:
        list: [
            {'station': 'PO', 'name': 'Powell', 'count': 45, 'types': {...}},
            ...
        ]
        Sorted by count descending.
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    cutoff = (datetime.now() - timedelta(days=days)).isoformat()

    # Get counts by station and type
    cursor.execute('''
        SELECT station, station_name, type, COUNT(*) as count
        FROM delay_incidents
        WHERE timestamp >= ? AND station IS NOT NULL
        GROUP BY station, type
        ORDER BY station
    ''', (cutoff,))

    # Aggregate by station
    stations = {}
    for row in cursor.fetchall():
        station = row['station']
        if station not in stations:
            stations[station] = {
                'station': station,
                'name': row['station_name'],
                'count': 0,
                'types': {}
            }
        stations[station]['count'] += row['count']
        stations[station]['types'][row['type']] = row['count']

    conn.close()

    # Sort by count descending
    result = sorted(stations.values(), key=lambda x: x['count'], reverse=True)
    return result


def get_delays_by_time(days=7):
    """
    Get delay patterns by hour of day and day of week.

    Args:
        days: Number of days to look back

    Returns:
        dict: {
            'by_hour': {0: count, 1: count, ..., 23: count},
            'by_day': {0: count, ..., 6: count}  # 0=Monday
        }
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    cutoff = (datetime.now() - timedelta(days=days)).isoformat()

    # Get all delay incidents in the period
    cursor.execute('''
        SELECT timestamp FROM delay_incidents WHERE timestamp >= ?
    ''', (cutoff,))

    by_hour = {h: 0 for h in range(24)}
    by_day = {d: 0 for d in range(7)}

    for row in cursor.fetchall():
        try:
            dt = datetime.fromisoformat(row['timestamp'])
            by_hour[dt.hour] += 1
            by_day[dt.weekday()] += 1
        except ValueError:
            continue

    conn.close()

    return {
        'by_hour': by_hour,
        'by_day': by_day
    }


def get_recent_incidents(limit=50):
    """
    Get most recent delay incidents.

    Args:
        limit: Maximum number of incidents to return

    Returns:
        list: Recent delay incidents with full details
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute('''
        SELECT
            di.id,
            di.timestamp,
            di.type,
            di.station,
            di.station_name,
            di.direction,
            di.details,
            sc.status,
            sc.best_status
        FROM delay_incidents di
        JOIN status_checks sc ON di.check_id = sc.id
        ORDER BY di.timestamp DESC
        LIMIT ?
    ''', (limit,))

    incidents = []
    for row in cursor.fetchall():
        incident = {
            'id': row['id'],
            'timestamp': row['timestamp'],
            'type': row['type'],
            'station': row['station'],
            'station_name': row['station_name'],
            'direction': row['direction'],
            'status': row['status'],
            'best_status': row['best_status']
        }
        if row['details']:
            try:
                incident['details'] = json.loads(row['details'])
            except json.JSONDecodeError:
                incident['details'] = None
        else:
            incident['details'] = None
        incidents.append(incident)

    conn.close()
    return incidents


def _get_report_cache_path(days):
    """Get the local cache file path for a report with the given time period."""
    return LOCAL_CACHE_DIR / f"analytics_report_{days}d.json"


def _get_report_gcs_path(days):
    """Get the GCS blob name for a report with the given time period."""
    return f"analytics/analytics_report_{days}d.json"


def get_analytics_report(days=7):
    """
    Get a full analytics report, using cache if available and fresh.

    This is the main entry point for the /analytics endpoint. It caches
    reports to avoid expensive queries on every request.

    Args:
        days: Number of days to analyze

    Returns:
        dict: Full analytics report with frequency, by_station, by_time, etc.
    """
    # Ensure database and tables exist
    init_db()

    cached = None

    # Try to read from cache (GCS on Cloud Run, local file otherwise)
    if _is_cloud_run():
        cached = _read_gcs_json(_get_report_gcs_path(days))
    else:
        cache_path = _get_report_cache_path(days)
        if cache_path.exists():
            try:
                with open(cache_path, 'r') as f:
                    cached = json.load(f)
            except (json.JSONDecodeError, ValueError):
                cached = None

    # Check if cache is fresh
    if cached:
        try:
            cached_at = datetime.fromisoformat(cached.get('cached_at', ''))
            cache_age = (datetime.now() - cached_at).total_seconds()

            if cache_age < REPORT_CACHE_MAX_AGE:
                # Return cached report with cache info
                cached['from_cache'] = True
                cached['cache_age'] = round(cache_age, 1)
                return cached
        except (ValueError, KeyError):
            pass  # Cache corrupted or missing timestamp

    # No fresh cache - return empty report
    # Reports are generated by scheduled task, not on-demand
    return _empty_report(days)


def _empty_report(days):
    """Return an empty report structure."""
    return {
        'period_days': days,
        'frequency': {
            'total_checks': 0,
            'delayed_checks': 0,
            'delay_rate': 0.0,
            'by_status': {'green': 0, 'yellow': 0, 'red': 0}
        },
        'by_station': [],
        'by_time': {
            'by_hour': {h: 0 for h in range(24)},
            'by_day': {d: 0 for d in range(7)}
        },
        'generated_at': None,
        'cached_at': None,
        'from_cache': False,
        'no_data': True
    }


def generate_report(days):
    """
    Generate and cache an analytics report for the specified period.

    This is called by scheduled tasks, not by API requests.

    Args:
        days: Number of days to analyze

    Returns:
        dict: The generated report
    """
    init_db()

    try:
        report = {
            'period_days': days,
            'frequency': get_delay_frequency(days),
            'by_station': get_delays_by_station(days),
            'by_time': get_delays_by_time(days),
            'generated_at': datetime.now().isoformat(),
            'cached_at': datetime.now().isoformat(),
            'from_cache': False
        }

        # Cache the report (GCS on Cloud Run, local file otherwise)
        if _is_cloud_run():
            _write_gcs_json(_get_report_gcs_path(days), report)
        else:
            cache_path = _get_report_cache_path(days)
            os.makedirs(LOCAL_CACHE_DIR, exist_ok=True)
            with open(cache_path, 'w') as f:
                json.dump(report, f)

        return report
    except Exception as e:
        print(f"Error generating {days}-day report: {e}")
        return None


def generate_all_reports():
    """
    Generate and cache reports for all standard time periods.

    Called by scheduled task (e.g., daily at midnight).
    Returns summary of what was generated.
    """
    periods = [1, 7, 30, 365]
    results = {}

    for days in periods:
        report = generate_report(days)
        if report:
            results[days] = {
                'success': True,
                'total_checks': report['frequency']['total_checks'],
                'delayed_checks': report['frequency']['delayed_checks']
            }
        else:
            results[days] = {'success': False}

    return results
