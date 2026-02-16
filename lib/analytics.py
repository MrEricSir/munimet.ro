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
REPORT_CACHE_MAX_AGE = 1800  # 30 minutes - reports regenerate frequently for fresh data

# Station code to name mapping (for incidents that only have codes)
STATION_NAMES = {
    'WE': 'West Portal',
    'FH': 'Forest Hill',
    'CA': 'Castro',
    'CH': 'Church',
    'VN': 'Van Ness',
    'CC': 'Civic Center',
    'PO': 'Powell',
    'MO': 'Montgomery',
    'EM': 'Embarcadero',
    'MN': 'Main',
    'FP': 'Folsom',
    'TT': 'Temporary Terminal',
    'CT': 'Chinatown',
    'US': 'Union Square',
    'YB': 'Yerba Buena',
}

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
    from lib.gcs_utils import get_gcs_client
    return get_gcs_client()


def restore_db_from_gcs():
    """
    Restore the analytics database from GCS if running on Cloud Run.
    Called once at startup to restore persisted data.

    Uses retry with exponential backoff for transient GCS errors.

    Returns:
        bool: True if restored, False if no backup exists or not on Cloud Run
    """
    global _gcs_restored

    if not _is_cloud_run():
        return False

    if _gcs_restored:
        return True  # Already restored this session

    bucket_name = _get_gcs_bucket()
    blob_path = 'analytics/analytics.db'

    try:
        from lib.gcs_utils import gcs_blob_exists, gcs_get_blob_size, gcs_download_to_file

        print(f"Restoring analytics DB from gs://{bucket_name}/{blob_path}...")

        if not gcs_blob_exists(bucket_name, blob_path):
            print(f"No analytics DB backup found in GCS (gs://{bucket_name}/{blob_path})")
            _gcs_restored = True
            return False

        # Ensure local directory exists
        os.makedirs(LOCAL_DB_PATH.parent, exist_ok=True)

        # Get blob size before download
        gcs_size = gcs_get_blob_size(bucket_name, blob_path) or 0

        # Download the database with retry
        gcs_download_to_file(bucket_name, blob_path, str(LOCAL_DB_PATH))
        local_size = LOCAL_DB_PATH.stat().st_size
        print(f"Restored analytics DB from GCS ({gcs_size} bytes downloaded, {local_size} bytes on disk)")
        _gcs_restored = True
        return True
    except Exception as e:
        print(f"Error restoring analytics DB from gs://{bucket_name}/{blob_path}: {e}")
        import traceback
        traceback.print_exc()
        _gcs_restored = True  # Don't retry on error (but backup safety check will prevent overwrite)
        return False


def backup_db_to_gcs():
    """
    Backup the analytics database to GCS if running on Cloud Run.
    Should be called periodically (e.g., after each status check).

    Uses retry with exponential backoff for transient GCS errors.

    Safety: Will not overwrite a larger GCS backup with a smaller local file,
    which could indicate data loss from a failed restore.

    Returns:
        bool: True if backed up successfully
    """
    import sys

    print("  Backup: Starting...", flush=True)

    if not _is_cloud_run():
        print("  Backup: Skipped (not on Cloud Run)", flush=True)
        return False

    if not LOCAL_DB_PATH.exists():
        print(f"  Backup: Skipped (local DB doesn't exist at {LOCAL_DB_PATH})", flush=True)
        return False

    try:
        from lib.gcs_utils import gcs_blob_exists, gcs_get_blob_size, gcs_upload_from_file

        local_size = LOCAL_DB_PATH.stat().st_size
        print(f"  Backup: Local DB size = {local_size} bytes", flush=True)

        bucket_name = _get_gcs_bucket()
        blob_path = 'analytics/analytics.db'

        # Safety check: don't overwrite a larger backup with smaller data
        # This prevents data loss if restore failed and we have a fresh DB
        if gcs_blob_exists(bucket_name, blob_path):
            gcs_size = gcs_get_blob_size(bucket_name, blob_path) or 0
            print(f"  Backup: GCS size = {gcs_size} bytes", flush=True)
            if local_size < gcs_size * 0.9:  # Allow 10% variance for normal fluctuation
                print(f"  Backup: WARNING - Local ({local_size}) much smaller than GCS ({gcs_size})", flush=True)
                print("  Backup: Skipping to prevent data loss - investigate restore failure", flush=True)
                return False
            print(f"  Backup: Safety check passed ({local_size} >= {gcs_size * 0.9:.0f})", flush=True)
        else:
            print("  Backup: No existing GCS backup found", flush=True)

        print("  Backup: Uploading...", flush=True)
        gcs_upload_from_file(bucket_name, blob_path, str(LOCAL_DB_PATH))
        print(f"  Backup: Success - uploaded {local_size} bytes to GCS", flush=True)
        sys.stdout.flush()
        return True
    except Exception as e:
        print(f"  Backup: ERROR - {e}", flush=True)
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        return False


def _read_gcs_json(blob_name):
    """
    Read a JSON file from GCS with retry.

    Uses exponential backoff for transient GCS errors.
    """
    try:
        from lib.gcs_utils import gcs_download_as_string

        content = gcs_download_as_string(_get_gcs_bucket(), blob_name)
        if content is None:
            return None

        return json.loads(content)
    except Exception as e:
        print(f"Error reading {blob_name} from GCS: {e}")
        return None


def _write_gcs_json(blob_name, data):
    """
    Write a JSON file to GCS with retry.

    Uses exponential backoff for transient GCS errors.
    """
    try:
        from lib.gcs_utils import gcs_upload_from_string

        gcs_upload_from_string(
            _get_gcs_bucket(),
            blob_name,
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
        restored = restore_db_from_gcs()
        # If restore failed and no local DB exists, we have a problem
        # But we still proceed - the backup will only happen if we have
        # meaningful data (checked in backup_db_to_gcs)
        if not restored and not LOCAL_DB_PATH.exists():
            print("Warning: No GCS backup found and no local DB - starting fresh")

    os.makedirs(LOCAL_DB_PATH.parent, exist_ok=True)
    conn = sqlite3.connect(str(LOCAL_DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


# Current schema version - increment when adding migrations
SCHEMA_VERSION = 2


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

    if current_version < 2:
        # Version 2: Add interval_seconds column for time-based analytics
        # This stores the check interval so we can calculate durations instead of counts
        cursor.execute('ALTER TABLE status_checks ADD COLUMN interval_seconds INTEGER DEFAULT 30')
        cursor.execute('UPDATE schema_version SET version = 2')


def init_db():
    """Initialize database and run any pending migrations."""
    conn = get_db_connection()
    cursor = conn.cursor()

    current_version = get_schema_version(cursor)

    if current_version < SCHEMA_VERSION:
        run_migrations(cursor, current_version)
        conn.commit()

    conn.close()


def log_status_check(status, best_status, detection_data, timestamp, interval_seconds=None):
    """
    Log a status check and its delay incidents.

    Args:
        status: Raw detected status ('green', 'yellow', 'red')
        best_status: Smoothed best status
        detection_data: Dict with trains, delays_platforms, delays_segments, delays_bunching
        timestamp: ISO8601 timestamp string
        interval_seconds: Check interval in seconds (default: from config)

    Returns:
        int: The check_id of the inserted record
    """
    from lib.config import DEFAULT_CHECK_INTERVAL

    # Ensure database exists
    init_db()

    conn = get_db_connection()
    cursor = conn.cursor()

    # Use default interval if not specified
    if interval_seconds is None:
        interval_seconds = DEFAULT_CHECK_INTERVAL

    # Count trains
    trains = detection_data.get('trains', [])
    train_count = len(trains)

    # Insert status check with interval
    cursor.execute('''
        INSERT INTO status_checks (timestamp, status, best_status, train_count, interval_seconds)
        VALUES (?, ?, ?, ?, ?)
    ''', (timestamp, status, best_status, train_count, interval_seconds))

    check_id = cursor.lastrowid

    # Only log delay incidents for yellow status (actual service delays)
    # Skip red status - that's after-hours maintenance, not delays
    if status != 'yellow':
        conn.commit()
        conn.close()
        # Backup to GCS after logging (if on Cloud Run)
        backup_db_to_gcs()
        return check_id

    # Insert delay incidents
    delays_platforms = detection_data.get('delays_platforms', [])
    delays_segments = detection_data.get('delays_segments', [])
    delays_bunching = detection_data.get('delays_bunching', [])

    # Platform holds
    # detection.py uses 'station' for code and 'name' for station name
    for delay in delays_platforms:
        cursor.execute('''
            INSERT INTO delay_incidents (check_id, timestamp, type, station, station_name, direction)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            check_id,
            timestamp,
            'platform_hold',
            delay.get('station'),  # station code like 'PO', 'MO'
            delay.get('name'),     # station name like 'Powell', 'Montgomery'
            delay.get('direction')
        ))

    # Red segments
    # detection.py uses 'from' and 'to' for station codes (not 'from_code')
    for delay in delays_segments:
        from_code = delay.get('from')
        details = json.dumps({
            'from': from_code,
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
            from_code,
            STATION_NAMES.get(from_code, from_code),  # lookup name, fallback to code
            delay.get('direction'),
            details
        ))

    # Train bunching
    # detection.py uses 'station' for station code (no separate 'station_code' field)
    for delay in delays_bunching:
        station_code = delay.get('station')
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
            station_code,
            STATION_NAMES.get(station_code, station_code),  # lookup name, fallback to code
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

    Returns time-based metrics (in minutes) instead of check counts,
    making the data independent of check interval.

    Uses actual timestamp differences to calculate time, which is more
    accurate than stored interval_seconds (which may be misconfigured).

    Args:
        days: Number of days to look back

    Returns:
        dict: {
            'total_minutes': float,      # Total monitored time
            'delayed_minutes': float,    # Time spent in delayed state
            'delay_rate': float,         # Percentage of time delayed
            'by_status': {'green': float, 'yellow': float, 'red': float}  # Minutes per status
        }
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    cutoff = (datetime.now() - timedelta(days=days)).isoformat()

    # Calculate actual time coverage using timestamp differences
    # This is more accurate than stored interval_seconds which may be wrong
    # Cap individual gaps at 5 minutes to handle overnight/restart gaps
    MAX_GAP_SECONDS = 300  # 5 minutes - any gap larger is considered a break in monitoring

    cursor.execute('''
        SELECT timestamp, best_status, status
        FROM status_checks
        WHERE timestamp >= ?
        ORDER BY timestamp
    ''', (cutoff,))

    rows = cursor.fetchall()
    conn.close()

    if not rows:
        return {
            'total_minutes': 0.0,
            'delayed_minutes': 0.0,
            'delay_rate': 0.0,
            'by_status': {'green': 0.0, 'yellow': 0.0, 'red': 0.0}
        }

    total_seconds = 0.0
    delayed_seconds = 0.0
    by_status_seconds = {'green': 0.0, 'yellow': 0.0, 'red': 0.0}

    prev_timestamp = None
    for row in rows:
        current_timestamp = datetime.fromisoformat(row['timestamp'])

        if prev_timestamp is not None:
            gap = (current_timestamp - prev_timestamp).total_seconds()
            # Cap the gap to handle breaks in monitoring
            effective_gap = min(gap, MAX_GAP_SECONDS)

            total_seconds += effective_gap

            # Attribute time to the status that was active during this period
            best_status = row['best_status']
            if best_status in by_status_seconds:
                by_status_seconds[best_status] += effective_gap

            # Track actual delay detections (raw status)
            if row['status'] == 'yellow':
                delayed_seconds += effective_gap

        prev_timestamp = current_timestamp

    total_minutes = total_seconds / 60.0
    delayed_minutes = delayed_seconds / 60.0

    by_status = {s: round(secs / 60.0, 1) for s, secs in by_status_seconds.items()}

    delay_rate = delayed_minutes / total_minutes if total_minutes > 0 else 0.0

    return {
        'total_minutes': round(total_minutes, 1),
        'delayed_minutes': round(delayed_minutes, 1),
        'delay_rate': round(delay_rate, 4),
        'by_status': by_status
    }


def get_delays_by_station(days=7):
    """
    Get delay time grouped by station.

    Returns time-based metrics (in minutes) instead of incident counts.

    Args:
        days: Number of days to look back

    Returns:
        list: [
            {'station': 'PO', 'name': 'Powell', 'minutes': 45.5, 'types': {...}},
            ...
        ]
        Sorted by minutes descending.
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    cutoff = (datetime.now() - timedelta(days=days)).isoformat()

    # Use 60 seconds as default - this matches actual observed check intervals
    # (stored interval_seconds may be incorrect)
    ACTUAL_CHECK_INTERVAL = 60

    # Join with status_checks to get interval for each incident
    # Each incident represents one check interval of delay at that station
    cursor.execute('''
        SELECT
            di.station,
            di.station_name,
            di.type,
            COUNT(*) as count
        FROM delay_incidents di
        JOIN status_checks sc ON di.check_id = sc.id
        WHERE di.timestamp >= ? AND di.station IS NOT NULL
        GROUP BY di.station, di.type
        ORDER BY di.station
    ''', (cutoff,))

    # Aggregate by station - multiply count by actual interval
    stations = {}
    for row in cursor.fetchall():
        station = row['station']
        # Each incident represents one check interval
        minutes = (row['count'] * ACTUAL_CHECK_INTERVAL) / 60.0
        if station not in stations:
            stations[station] = {
                'station': station,
                'name': row['station_name'],
                'minutes': 0.0,
                'types': {}
            }
        stations[station]['minutes'] += minutes
        stations[station]['types'][row['type']] = round(minutes, 1)

    conn.close()

    # Round totals and sort by minutes descending
    for s in stations.values():
        s['minutes'] = round(s['minutes'], 1)

    result = sorted(stations.values(), key=lambda x: x['minutes'], reverse=True)
    return result


def get_delays_by_time(days=7):
    """
    Get delay patterns by hour of day and day of week.

    Returns time-based metrics (in minutes) instead of incident counts.

    Args:
        days: Number of days to look back

    Returns:
        dict: {
            'by_hour': {0: minutes, 1: minutes, ..., 23: minutes},
            'by_day': {0: minutes, ..., 6: minutes}  # 0=Monday
        }
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    cutoff = (datetime.now() - timedelta(days=days)).isoformat()

    # Use 60 seconds as default - this matches actual observed check intervals
    ACTUAL_CHECK_INTERVAL = 60

    # Get delay incidents with timestamps
    cursor.execute('''
        SELECT di.timestamp
        FROM delay_incidents di
        JOIN status_checks sc ON di.check_id = sc.id
        WHERE di.timestamp >= ?
    ''', (cutoff,))

    by_hour = {h: 0.0 for h in range(24)}
    by_day = {d: 0.0 for d in range(7)}

    # Each incident represents one check interval of delay
    minutes_per_incident = ACTUAL_CHECK_INTERVAL / 60.0

    for row in cursor.fetchall():
        try:
            dt = datetime.fromisoformat(row['timestamp'])
            by_hour[dt.hour] += minutes_per_incident
            by_day[dt.weekday()] += minutes_per_incident
        except ValueError:
            continue

    conn.close()

    # Round all values
    by_hour = {h: round(m, 1) for h, m in by_hour.items()}
    by_day = {d: round(m, 1) for d, m in by_day.items()}

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

    # Always return cached data if it exists (never return empty if we have data)
    if cached:
        cached['from_cache'] = True
        try:
            cached_at = datetime.fromisoformat(cached.get('cached_at', ''))
            cache_age = (datetime.now() - cached_at).total_seconds()
            cached['cache_age'] = round(cache_age, 1)
            cached['cache_stale'] = cache_age >= REPORT_CACHE_MAX_AGE
        except (ValueError, KeyError):
            cached['cache_age'] = None
            cached['cache_stale'] = True
        return cached

    # No cache at all - return empty report
    return _empty_report(days)


def _empty_report(days):
    """Return an empty report structure."""
    return {
        'period_days': days,
        'frequency': {
            'total_minutes': 0.0,
            'delayed_minutes': 0.0,
            'delay_rate': 0.0,
            'by_status': {'green': 0.0, 'yellow': 0.0, 'red': 0.0}
        },
        'by_station': [],
        'by_time': {
            'by_hour': {h: 0.0 for h in range(24)},
            'by_day': {d: 0.0 for d in range(7)}
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
        frequency = get_delay_frequency(days)

        # Log status breakdown for debugging
        by_status = frequency.get('by_status', {})
        print(f"Generating {days}-day report: total={frequency['total_minutes']:.1f}min, "
              f"green={by_status.get('green', 0):.1f}min, yellow={by_status.get('yellow', 0):.1f}min, red={by_status.get('red', 0):.1f}min")

        report = {
            'period_days': days,
            'frequency': frequency,
            'by_station': get_delays_by_station(days),
            'by_time': get_delays_by_time(days),
            'generated_at': datetime.now().isoformat(),
            'cached_at': datetime.now().isoformat(),
            'from_cache': False
        }

        # Cache the report (GCS on Cloud Run, local file otherwise)
        if _is_cloud_run():
            success = _write_gcs_json(_get_report_gcs_path(days), report)
            if success:
                print(f"  Cached to GCS: {_get_report_gcs_path(days)}")
            else:
                print(f"  WARNING: Failed to cache to GCS")
        else:
            cache_path = _get_report_cache_path(days)
            os.makedirs(LOCAL_CACHE_DIR, exist_ok=True)
            with open(cache_path, 'w') as f:
                json.dump(report, f)
            print(f"  Cached to local: {cache_path}")

        return report
    except Exception as e:
        print(f"Error generating {days}-day report: {e}")
        import traceback
        traceback.print_exc()
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
                'total_minutes': report['frequency']['total_minutes'],
                'delayed_minutes': report['frequency']['delayed_minutes']
            }
        else:
            results[days] = {'success': False}

    return results


def check_database_health():
    """
    Check if the analytics database exists and has data.

    Returns:
        dict: {
            'exists': bool,           # Database file exists locally
            'restored_from_gcs': bool,  # Was restored from GCS (Cloud Run only)
            'has_data': bool,         # Has any status checks recorded
            'check_count': int,       # Number of status checks
            'error': str or None      # Error message if any
        }
    """
    result = {
        'exists': False,
        'restored_from_gcs': False,
        'has_data': False,
        'check_count': 0,
        'error': None
    }

    try:
        # On Cloud Run, try to restore from GCS first
        if _is_cloud_run():
            result['restored_from_gcs'] = restore_db_from_gcs()

        # Check if database file exists
        result['exists'] = LOCAL_DB_PATH.exists()

        if not result['exists']:
            return result

        # Check if database has any data
        conn = get_db_connection()
        cursor = conn.cursor()

        # Check if tables exist
        cursor.execute('''
            SELECT name FROM sqlite_master
            WHERE type='table' AND name='status_checks'
        ''')
        if not cursor.fetchone():
            conn.close()
            result['error'] = 'status_checks table does not exist'
            return result

        # Count status checks
        cursor.execute('SELECT COUNT(*) as count FROM status_checks')
        count = cursor.fetchone()['count']
        conn.close()

        result['check_count'] = count
        result['has_data'] = count > 0

    except Exception as e:
        result['error'] = str(e)

    return result
