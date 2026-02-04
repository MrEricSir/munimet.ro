#!/usr/bin/env python3
"""
Cloud Run Job script for MuniMetro status checker.
This script is executed by Cloud Scheduler via Cloud Run Jobs.
"""

import os
import sys
from datetime import datetime
from pathlib import Path

# Add parent directory to path for imports
API_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = API_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from api.check_status import check_status


def print_diagnostics():
    """Print diagnostic information about the environment."""
    print("=" * 60)
    print("DIAGNOSTIC INFO")
    print("=" * 60)
    print(f"Timestamp (UTC): {datetime.utcnow().isoformat()}")
    print(f"Timestamp (local): {datetime.now().isoformat()}")
    print(f"Python version: {sys.version}")
    print(f"Working directory: {os.getcwd()}")
    print()

    # Environment variables relevant to Cloud Run
    env_vars = [
        'CLOUD_RUN',
        'GCS_BUCKET',
        'K_SERVICE',
        'K_REVISION',
        'CLOUD_RUN_JOB',
        'CLOUD_RUN_EXECUTION',
        'CLOUD_RUN_TASK_INDEX',
        'CLOUD_RUN_TASK_COUNT',
    ]
    print("Environment:")
    for var in env_vars:
        value = os.getenv(var)
        if value:
            print(f"  {var}={value}")
        else:
            print(f"  {var}=(not set)")
    print()

    # Check GCS connectivity
    if os.getenv('CLOUD_RUN'):
        print("GCS Connectivity Check:")
        try:
            from google.cloud import storage
            client = storage.Client()
            bucket_name = os.getenv('GCS_BUCKET', 'munimetro-cache')
            bucket = client.bucket(bucket_name)

            # Check analytics DB
            analytics_blob = bucket.blob('analytics/analytics.db')
            if analytics_blob.exists():
                analytics_blob.reload()
                print(f"  analytics/analytics.db: {analytics_blob.size} bytes, updated {analytics_blob.updated}")
            else:
                print(f"  analytics/analytics.db: NOT FOUND")

            # Check status cache (stored as latest_status.json)
            status_blob = bucket.blob('latest_status.json')
            if status_blob.exists():
                status_blob.reload()
                print(f"  latest_status.json: {status_blob.size} bytes, updated {status_blob.updated}")
            else:
                print(f"  latest_status.json: NOT FOUND")

            # Check report cache
            report_blob = bucket.blob('analytics/analytics_report_7d.json')
            if report_blob.exists():
                report_blob.reload()
                print(f"  analytics_report_7d.json: {report_blob.size} bytes, updated {report_blob.updated}")
            else:
                print(f"  analytics_report_7d.json: NOT FOUND")

        except Exception as e:
            print(f"  ERROR: {e}")
    else:
        print("GCS Check: Skipped (not on Cloud Run)")

    print("=" * 60)
    print()


def main():
    """Run status check and write to cache."""
    start_time = datetime.now()

    try:
        print_diagnostics()

        print("Starting MuniMetro status check...")
        print("-" * 60)
        success = check_status(should_write_cache=True)

        elapsed = (datetime.now() - start_time).total_seconds()

        if success:
            print(f"\n✓ Status check completed successfully in {elapsed:.1f}s")
            sys.exit(0)
        else:
            print(f"\n❌ Status check failed after {elapsed:.1f}s - see logs above", file=sys.stderr)
            sys.exit(1)
    except Exception as e:
        elapsed = (datetime.now() - start_time).total_seconds()
        print(f"\n❌ Error during status check after {elapsed:.1f}s: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
