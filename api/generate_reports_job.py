#!/usr/bin/env python3
"""
Cloud Run Job script for MuniMetro analytics report generation.
This script is executed by Cloud Scheduler via Cloud Run Jobs (daily at midnight).
"""

import logging
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
API_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = API_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from lib.analytics import generate_all_reports, init_db, check_database_health

# Configure logging for Cloud Logging
# On Cloud Run, logs at ERROR level will trigger alerts
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


def print_diagnostics():
    """Print diagnostic information."""
    from datetime import datetime
    print("=" * 60)
    print("DIAGNOSTIC INFO")
    print("=" * 60)
    print(f"Timestamp (UTC): {datetime.utcnow().isoformat()}")
    print(f"Timestamp (local): {datetime.now().isoformat()}")
    print(f"CLOUD_RUN={os.getenv('CLOUD_RUN')}")
    print(f"GCS_BUCKET={os.getenv('GCS_BUCKET')}")
    print("=" * 60)
    print()


def main():
    """Generate all analytics reports."""
    try:
        print_diagnostics()

        print("Starting MuniMetro analytics report generation...")
        print("-" * 60)

        # Check database health first
        health = check_database_health()
        is_cloud_run = os.getenv('CLOUD_RUN') is not None

        print(f"Database health check:")
        print(f"  exists: {health['exists']}")
        if is_cloud_run:
            print(f"  restored_from_gcs: {health['restored_from_gcs']}")
        print(f"  has_data: {health['has_data']}")
        print(f"  check_count: {health['check_count']}")
        if health['error']:
            print(f"  error: {health['error']}")

        # Check for critical issues that should trigger alerts
        if is_cloud_run and not health['restored_from_gcs'] and not health['exists']:
            # On Cloud Run with no database - this is a problem
            logger.error(
                "ANALYTICS_NO_DATABASE: No analytics database found. "
                "Database was not restored from GCS and does not exist locally. "
                "This may indicate a GCS backup issue or first-time deployment."
            )
            # Continue to generate empty reports, but log the error

        if not health['has_data']:
            if health['check_count'] == 0:
                logger.warning(
                    "ANALYTICS_NO_DATA: Analytics database has no status check records. "
                    "Reports will be empty until status checks are logged."
                )

        # Ensure database schema exists
        init_db()

        # Check status distribution in database before generating
        try:
            from lib.analytics import get_db_connection
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute('SELECT best_status, COUNT(*) FROM status_checks GROUP BY best_status')
            status_counts = {row[0]: row[1] for row in cursor.fetchall()}
            conn.close()
            print(f"\nStatus distribution in DB: green={status_counts.get('green', 0)}, yellow={status_counts.get('yellow', 0)}, red={status_counts.get('red', 0)}")
        except Exception as e:
            print(f"Could not query status distribution: {e}")

        # Generate reports for all time periods
        results = generate_all_reports()

        # Print results
        all_success = True
        for days, result in results.items():
            if result['success']:
                print(f"  {days}-day report: {result['total_checks']} checks, {result['delayed_checks']} delays")
            else:
                print(f"  {days}-day report: FAILED")
                all_success = False

        if all_success:
            print("\n✓ All reports generated successfully")
            sys.exit(0)
        else:
            logger.error("ANALYTICS_REPORT_FAILED: Some reports failed to generate")
            print("\n⚠️ Some reports failed to generate", file=sys.stderr)
            sys.exit(1)

    except Exception as e:
        logger.error(f"ANALYTICS_ERROR: Exception during report generation: {e}")
        print(f"\n❌ Error during report generation: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
