#!/usr/bin/env python3
"""
Download latest Muni snapshot and predict its status.

Usage:
    python check_status.py                    # Single check
    python check_status.py --continuous       # Keep checking every 30 seconds
    python check_status.py --write-cache      # Single check, write to cache
    python check_status.py --continuous --write-cache --interval 60  # Cache mode with custom interval
"""

import sys
import os
import json
import time
from datetime import datetime
from pathlib import Path

# Path resolution - get absolute paths relative to project root
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

# Add parent directory to path for lib imports
sys.path.insert(0, str(PROJECT_ROOT))
from lib.muni_lib import download_muni_image, predict_muni_status

# Configuration
CACHE_DIR = str(PROJECT_ROOT / "artifacts" / "runtime" / "cache")
CACHE_FILE = str(PROJECT_ROOT / "artifacts" / "runtime" / "cache" / "latest_status.json")
SNAPSHOT_DIR = str(PROJECT_ROOT / "artifacts" / "runtime" / "downloads")
DEFAULT_INTERVAL = 30  # seconds


def check_status(write_cache=False, model=None, processor=None, label_to_status=None, device=None):
    """Download image and predict status.

    Args:
        write_cache: If True, write result to cache file
        model: Pre-loaded model (optional, will load if not provided)
        processor: Pre-loaded processor (optional, will load if not provided)
        label_to_status: Pre-loaded label mapping (optional, will load if not provided)
        device: Pre-loaded device (optional, will load if not provided)

    Returns:
        bool: True if successful, False otherwise
    """
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]")
    print("-" * 60)

    # Download image
    print("Downloading latest snapshot...")
    result = download_muni_image(output_folder=SNAPSHOT_DIR, validate_dimensions=True)

    if not result['success']:
        print(f"❌ Download failed: {result['error']}")
        return False

    print(f"✓ Downloaded: {result['filepath']}")
    print(f"  Dimensions: {result['width']} × {result['height']}")
    print()

    # Predict status
    print("Analyzing image...")
    try:
        prediction = predict_muni_status(
            result['filepath'],
            model=model,
            processor=processor,
            label_to_status=label_to_status,
            device=device
        )
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        return False

    # Display results
    status_emoji = {
        'green': '🟢',
        'yellow': '🟡',
        'red': '🔴'
    }
    emoji = status_emoji.get(prediction['status'], '⚪')

    print()
    print(f"Status: {emoji} {prediction['status'].upper()}")
    print(f"Confidence: {prediction['status_confidence']:.1%}")
    print(f"Description: {prediction['description']}")
    print()
    print("Probabilities:")
    print(f"  🟢 Green:  {prediction['probabilities']['green']:.1%}")
    print(f"  🟡 Yellow: {prediction['probabilities']['yellow']:.1%}")
    print(f"  🔴 Red:    {prediction['probabilities']['red']:.1%}")

    # Write to cache if requested
    if write_cache:
        os.makedirs(CACHE_DIR, exist_ok=True)

        # Create new status entry
        new_status = {
            'status': prediction['status'],
            'description': prediction['description'],
            'confidence': prediction['status_confidence'],
            'probabilities': prediction['probabilities'],
            'image_path': result['filepath'],
            'image_dimensions': {
                'width': result['width'],
                'height': result['height']
            },
            'timestamp': datetime.now().isoformat()
        }

        # Read existing cache to get previous status
        statuses = []
        if os.path.exists(CACHE_FILE):
            try:
                with open(CACHE_FILE, 'r') as f:
                    cache_data = json.load(f)
                    # Get the current status from previous cache (becomes previous)
                    if 'statuses' in cache_data and len(cache_data['statuses']) > 0:
                        statuses.append(cache_data['statuses'][0])
            except (json.JSONDecodeError, KeyError):
                pass  # Start fresh if cache is corrupted

        # Add new status at the front
        statuses.insert(0, new_status)

        # Keep only last 2 statuses
        statuses = statuses[:2]

        # Determine best status (most optimistic)
        # Priority: green > yellow > red
        status_priority = {'green': 3, 'yellow': 2, 'red': 1}
        best_status = max(statuses, key=lambda s: status_priority.get(s['status'], 0))

        # Write cache with status history
        cache_data = {
            'statuses': statuses,
            'best_status': best_status,
            'cached_at': datetime.now().isoformat()
        }

        with open(CACHE_FILE, 'w') as f:
            json.dump(cache_data, f, indent=2)

        print(f"\n✓ Cache updated: {CACHE_FILE}")
        if len(statuses) > 1:
            print(f"  Current: {statuses[0]['status']}, Previous: {statuses[1]['status']}, Best: {best_status['status']}")

    return True


def main():
    from lib.muni_lib import load_muni_model

    # Parse arguments
    continuous = '--continuous' in sys.argv or '-c' in sys.argv
    write_cache = '--write-cache' in sys.argv

    # Parse interval
    interval = DEFAULT_INTERVAL
    if '--interval' in sys.argv:
        try:
            idx = sys.argv.index('--interval')
            interval = int(sys.argv[idx + 1])
        except (IndexError, ValueError):
            print("Warning: Invalid --interval value, using default (30 seconds)")

    print("=" * 60)
    print("Muni Status Checker")
    if continuous:
        print("Mode: Continuous (Ctrl+C to stop)")
        print(f"Interval: {interval} seconds")
    else:
        print("Mode: Single check")
    if write_cache:
        print(f"Cache: Enabled ({CACHE_FILE})")
    print("=" * 60)

    # Pre-load model for continuous mode to avoid reloading on every iteration
    model = processor = label_to_status = device = None
    if continuous:
        print("\nLoading ML model (one-time setup)...")
        model, processor, label_to_status, device = load_muni_model()
        print(f"✓ Model loaded on {device}")

    if not continuous:
        # Single check
        check_status(write_cache=write_cache)
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

                if check_status(
                    write_cache=write_cache,
                    model=model,
                    processor=processor,
                    label_to_status=label_to_status,
                    device=device
                ):
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
