#!/usr/bin/env python3
"""
Script to download SF Muni Central snapshot images continuously.
Waits for the page to refresh before downloading the actual image.
"""

import sys
import time
from datetime import datetime
from pathlib import Path

# Add parent directory to path for lib imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from lib.muni_lib import download_muni_image

# Configuration
OUTPUT_FOLDER = "../data/muni_snapshots"
SLEEP_INTERVAL = 300  # Seconds between downloads


if __name__ == "__main__":
    print("=" * 60)
    print("SF Muni Central Snapshot Downloader")
    print(f"Download interval: {SLEEP_INTERVAL} seconds")
    print("Press Ctrl+C to stop")
    print("=" * 60)
    print()

    count = 0
    successful = 0
    failed = 0

    try:
        while True:
            count += 1
            print(f"\n[Run #{count}] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("-" * 60)

            # Download image using shared library
            result = download_muni_image(output_folder=OUTPUT_FOLDER, validate_dimensions=True)

            if result['success']:
                print(f"Downloaded image to: {result['filepath']}")
                print(f"Image dimensions: {result['width']} × {result['height']}")
                print(f"✓ Image dimensions verified")
                successful += 1
            else:
                print(f"Download failed: {result['error']}")
                if result['width'] and result['height']:
                    print(f"Image dimensions: {result['width']} × {result['height']}")
                failed += 1

            print(f"\nStats: {successful} successful, {failed} failed")

            if count > 1:
                print(f"Waiting {SLEEP_INTERVAL} seconds until next download...")

            time.sleep(SLEEP_INTERVAL)

    except KeyboardInterrupt:
        print("\n\n" + "=" * 60)
        print("Script stopped by user")
        print(f"Total runs: {count}")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        print("=" * 60)
