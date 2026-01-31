#!/usr/bin/env python3
"""
Download latest Muni snapshot and detect its status.

Usage:
    python check_status.py                    # Single check
    python check_status.py --continuous       # Keep checking every 30 seconds
    python check_status.py --write-cache      # Single check, write to cache
    python check_status.py --continuous --write-cache --interval 60  # Cache mode with custom interval
"""

import sys
import time
from datetime import datetime
from pathlib import Path

# Path resolution - get absolute paths relative to project root
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

# Add parent directory to path for lib imports
sys.path.insert(0, str(PROJECT_ROOT))
from lib.muni_lib import download_muni_image, detect_muni_status, read_cache, write_cache, write_cached_image, post_to_bluesky

# Configuration
SNAPSHOT_DIR = str(PROJECT_ROOT / "artifacts" / "runtime" / "downloads")
DEFAULT_INTERVAL = 30  # seconds


def check_status(should_write_cache=False):
    """Download image and detect status.

    Args:
        should_write_cache: If True, write result to cache file

    Returns:
        bool: True if successful, False otherwise
    """
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]")
    print("-" * 60)

    # Download image
    print("Downloading latest snapshot...")
    result = download_muni_image(output_folder=SNAPSHOT_DIR, validate_dimensions=True)

    if not result['success']:
        print(f"Download failed: {result['error']}")
        return False

    print(f"Downloaded: {result['filepath']}")
    print(f"  Dimensions: {result['width']} x {result['height']}")
    print()

    # Detect status using OpenCV
    print("Analyzing image...")
    try:
        detection = detect_muni_status(result['filepath'])
    except Exception as e:
        print(f"Detection failed: {e}")
        return False

    # Display results
    status_emoji = {
        'green': '[GREEN]',
        'yellow': '[YELLOW]',
        'red': '[RED]'
    }
    emoji = status_emoji.get(detection['status'], '[?]')

    print()
    print(f"Status: {emoji} {detection['status'].upper()}")
    print(f"Description: {detection['description']}")
    print()

    # Display detection details
    det = detection.get('detection', {})
    trains = det.get('trains', [])
    delays_platforms = det.get('delays_platforms', [])
    delays_segments = det.get('delays_segments', [])
    delays_bunching = det.get('delays_bunching', [])

    print(f"Trains detected: {len(trains)}")
    if delays_platforms:
        print(f"Platforms in hold: {len(delays_platforms)}")
        for d in delays_platforms:
            print(f"  - {d['name']} ({d['direction']})")
    if delays_segments:
        print(f"Track segments disabled: {len(delays_segments)}")
        for d in delays_segments:
            print(f"  - {d['from']} to {d['to']} ({d['direction']})")
    if delays_bunching:
        print(f"Train bunching: {len(delays_bunching)}")
        for d in delays_bunching:
            print(f"  - {d['train_count']} trains at {d['station']} ({d['direction']})")

    # Write to cache if requested
    if should_write_cache:
        # Create new status entry
        new_status = {
            'status': detection['status'],
            'description': detection['description'],
            'confidence': detection['status_confidence'],
            'probabilities': detection['probabilities'],
            'detection': detection.get('detection', {}),
            'image_path': result['filepath'],
            'image_dimensions': {
                'width': result['width'],
                'height': result['height']
            },
            'timestamp': datetime.now().isoformat()
        }

        # Read existing cache to get previous status
        statuses = []
        previous_status = None
        cache_data = read_cache()
        if cache_data:
            # Get the current status from previous cache (becomes previous)
            if 'statuses' in cache_data and len(cache_data['statuses']) > 0:
                previous_status = cache_data['statuses'][0]['status']
                statuses.append(cache_data['statuses'][0])

        # Add new status at the front
        statuses.insert(0, new_status)

        # Keep only last 2 statuses
        statuses = statuses[:2]

        # Determine best status (most optimistic)
        # Priority: green > yellow > red
        status_priority = {'green': 3, 'yellow': 2, 'red': 1}
        best_status_value = max([s['status'] for s in statuses], key=lambda x: status_priority.get(x, 0))

        # Find the most recent entry with the best status
        # This ensures we use the most recent delay info if status is yellow
        best_status = None
        for s in statuses:  # statuses[0] is most recent
            if s['status'] == best_status_value:
                best_status = s
                break

        # Fallback to most recent if somehow not found
        if best_status is None:
            best_status = statuses[0]

        # Write cache with status history
        cache_data = {
            'statuses': statuses,
            'best_status': best_status,
            'cached_at': datetime.now().isoformat()
        }

        if write_cache(cache_data):
            print(f"\nCache updated")
            if len(statuses) > 1:
                print(f"  Current: {statuses[0]['status']}, Previous: {statuses[1]['status']}, Best: {best_status['status']}")
            # Also cache the image for the dashboard
            if write_cached_image(result['filepath']):
                print(f"  Image cached")
            else:
                print(f"  Image cache failed")
        else:
            print(f"\nCache write failed")

        # Post to Bluesky if status changed
        current_status = detection['status']
        if previous_status is not None and current_status != previous_status:
            print(f"\nStatus changed: {previous_status} -> {current_status}")
            delay_summaries = detection.get('detection', {}).get('delay_summaries', [])
            bluesky_result = post_to_bluesky(current_status, previous_status, delay_summaries)
            if bluesky_result['success']:
                print(f"Posted to Bluesky: {bluesky_result['uri']}")
            else:
                print(f"Bluesky post failed: {bluesky_result['error']}")

    return True


def main():
    # Parse arguments
    continuous = '--continuous' in sys.argv or '-c' in sys.argv
    should_write_cache = '--write-cache' in sys.argv

    # Parse interval
    interval = DEFAULT_INTERVAL
    if '--interval' in sys.argv:
        try:
            idx = sys.argv.index('--interval')
            interval = int(sys.argv[idx + 1])
        except (IndexError, ValueError):
            print("Warning: Invalid --interval value, using default (30 seconds)")

    print("=" * 60)
    print("Muni Status Checker (OpenCV Detection)")
    if continuous:
        print("Mode: Continuous (Ctrl+C to stop)")
        print(f"Interval: {interval} seconds")
    else:
        print("Mode: Single check")
    if should_write_cache:
        from lib.muni_lib import get_cache_path
        print(f"Cache: Enabled ({get_cache_path()})")
    print("=" * 60)

    if not continuous:
        # Single check
        check_status(should_write_cache=should_write_cache)
    else:
        # Continuous checking
        count = 0
        successful = 0
        failed = 0

        try:
            while True:
                count += 1
                print(f"\n{'='*60}")
                print(f"Check #{count}")
                print(f"{'='*60}")

                if check_status(should_write_cache=should_write_cache):
                    successful += 1
                else:
                    failed += 1

                print(f"\nStats: {successful} successful, {failed} failed")

                if count > 1:
                    print(f"\nWaiting {interval} seconds until next check...")

                time.sleep(interval)

        except KeyboardInterrupt:
            print("\n\n" + "=" * 60)
            print("Stopped by user")
            print(f"Total checks: {count}")
            print(f"Successful: {successful}")
            print(f"Failed: {failed}")
            print("=" * 60)


if __name__ == "__main__":
    main()
