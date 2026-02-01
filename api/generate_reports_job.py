#!/usr/bin/env python3
"""
Cloud Run Job script for MuniMetro analytics report generation.
This script is executed by Cloud Scheduler via Cloud Run Jobs (daily at midnight).
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
API_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = API_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from lib.analytics import generate_all_reports, init_db


def main():
    """Generate all analytics reports."""
    try:
        print("Starting MuniMetro analytics report generation...")
        print("-" * 60)

        # Ensure database exists
        init_db()

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
            print("\n⚠️ Some reports failed to generate", file=sys.stderr)
            sys.exit(1)

    except Exception as e:
        print(f"\n❌ Error during report generation: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
