#!/usr/bin/env python3
"""
Inference script to detect Muni subway status from images.

Usage:
    python predict_status.py <image_path>
    python predict_status.py muni_snapshots/muni_snapshot_20251206_134756.jpg
"""

import sys
import os
from pathlib import Path

# Add parent directory to path for lib imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from lib.muni_lib import detect_muni_status


def main():
    if len(sys.argv) < 2:
        print("Usage: python predict_status.py <image_path>")
        print("\nExample:")
        print("  python predict_status.py muni_snapshots/muni_snapshot_20251206_134756.jpg")
        sys.exit(1)

    image_path = sys.argv[1]

    if not os.path.exists(image_path):
        print(f"Error: Image not found: {image_path}")
        sys.exit(1)

    # Display header
    print("=" * 60)
    print("Muni Subway Status Detector")
    print("=" * 60)
    print()

    # Make detection using shared library
    print(f"Analyzing image: {image_path}")
    try:
        result = detect_muni_status(image_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    print()

    # Display results
    print("DETECTION RESULTS")
    print("=" * 60)
    print()

    # Status with color
    status_emoji = {
        'green': '[GREEN]',
        'yellow': '[YELLOW]',
        'red': '[RED]'
    }
    emoji = status_emoji.get(result['status'], '[?]')

    print(f"Status: {emoji} {result['status'].upper()}")
    print(f"Description: {result['description']}")
    print()

    # Detection details
    detection = result.get('detection', {})
    trains = detection.get('trains', [])
    delays_platforms = detection.get('delays_platforms', [])
    delays_segments = detection.get('delays_segments', [])
    delays_bunching = detection.get('delays_bunching', [])

    print(f"Trains detected: {len(trains)}")

    if delays_platforms:
        print(f"\nPlatforms in hold ({len(delays_platforms)}):")
        for d in delays_platforms:
            print(f"  - {d['name']} ({d['direction']})")

    if delays_segments:
        print(f"\nTrack segments disabled ({len(delays_segments)}):")
        for d in delays_segments:
            print(f"  - {d['from']} -> {d['to']} ({d['direction']})")

    if delays_bunching:
        print(f"\nTrain bunching ({len(delays_bunching)}):")
        for d in delays_bunching:
            print(f"  - {d['train_count']} trains at {d['station']} ({d['direction']})")

    print()
    print("=" * 60)


if __name__ == "__main__":
    main()
