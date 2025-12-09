#!/usr/bin/env python3
"""
Inference script to predict Muni subway status from images.

Usage:
    python predict_status.py <image_path>
    python predict_status.py muni_snapshots/muni_snapshot_20251206_134756.jpg
"""

import sys
import os
from pathlib import Path

# Add parent directory to path for lib imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from lib.muni_lib import predict_muni_status


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
    print("Muni Subway Status Predictor")
    print("=" * 60)
    print()

    # Make prediction using shared library
    print(f"Loading model...")
    try:
        result = predict_muni_status(image_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    print(f"Analyzing image: {image_path}")
    print()

    # Display results
    print("PREDICTION RESULTS")
    print("=" * 60)
    print()

    # Status with color
    status_emoji = {
        'green': '🟢',
        'yellow': '🟡',
        'red': '🔴'
    }
    emoji = status_emoji.get(result['status'], '⚪')

    print(f"Status: {emoji} {result['status'].upper()}")
    print(f"Confidence: {result['status_confidence']:.1%}")
    print()

    print("Status Probabilities:")
    print(f"  🟢 Green:  {result['probabilities']['green']:.1%}")
    print(f"  🟡 Yellow: {result['probabilities']['yellow']:.1%}")
    print(f"  🔴 Red:    {result['probabilities']['red']:.1%}")
    print()

    print("Description:")
    print(f"  {result['description']}")
    print()
    print("=" * 60)


if __name__ == "__main__":
    main()
