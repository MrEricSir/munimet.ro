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
from lib.muni_lib import download_muni_image, detect_muni_status, read_cache, write_cache, write_cached_image, calculate_best_status
from lib.notifiers import notify_status_change

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

        # Read existing cache to get previous statuses and best_status
        statuses = []
        previous_best_status = None
        cache_data = read_cache()
        if cache_data:
            if 'statuses' in cache_data:
                statuses = cache_data['statuses'][:]
            if 'best_status' in cache_data:
                previous_best_status = cache_data['best_status']

        # Add new status at the front
        statuses.insert(0, new_status)

        # Calculate best status using shared function
        # This ensures webapp, RSS, and Bluesky all show the same status
        best_status = calculate_best_status(statuses, window_size=3)

        # Keep only last 3 statuses (~1.5 min window at 30s intervals)
        statuses = statuses[:3]

        # Write cache with status history
        cache_data = {
            'statuses': statuses,
            'best_status': best_status,
            'cached_at': datetime.now().isoformat()
        }

        if write_cache(cache_data):
            print(f"\nCache updated")
            if len(statuses) > 1:
                history = ' -> '.join(s['status'] for s in statuses)
                print(f"  History: [{history}], Best: {best_status['status']}")
            # Also cache the image for the dashboard
            if write_cached_image(result['filepath']):
                print(f"  Image cached")
            else:
                print(f"  Image cache failed")
        else:
            print(f"\nCache write failed")

        # Notify all channels if BEST status changed
        # This ensures notifications match what the webapp shows
        current_best = best_status['status']
        previous_best = previous_best_status['status'] if previous_best_status else None
        if previous_best is not None and current_best != previous_best:
            print(f"\nBest status changed: {previous_best} -> {current_best}")
            delay_summaries = best_status.get('detection', {}).get('delay_summaries', [])
            notify_results = notify_status_change(
                status=current_best,
                previous_status=previous_best,
                delay_summaries=delay_summaries,
                timestamp=best_status['timestamp']
            )
            for channel, result in notify_results.items():
                if result['success']:
                    print(f"  {channel}: OK")
                else:
                    print(f"  {channel}: Failed - {result.get('error', 'Unknown error')}")

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
