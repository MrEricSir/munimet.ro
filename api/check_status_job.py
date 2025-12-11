#!/usr/bin/env python3
"""
Cloud Run Job script for MuniMetro status checker.
This script is executed by Cloud Scheduler via Cloud Run Jobs.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
API_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = API_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from api.check_status import check_status


def main():
    """Run status check and write to cache."""
    try:
        print("Starting MuniMetro status check...")
        check_status(should_write_cache=True)
        print("\n✓ Status check completed successfully")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error during status check: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
