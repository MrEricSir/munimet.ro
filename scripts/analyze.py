#!/usr/bin/env python3
"""
Analyze a Muni Metro status image from the command line.

Usage:
    python scripts/analyze.py <image_path>
    python scripts/analyze.py <image_path> --json     # Output as JSON
    python scripts/analyze.py <image_path> --verbose  # Show all detection details

Examples:
    python scripts/analyze.py artifacts/runtime/downloads/muni_snapshot_20260130_154523.jpg
    python scripts/analyze.py ~/Downloads/screenshot.jpg --json
"""

import argparse
import json
import sys
from pathlib import Path

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.detection import detect_system_status


STATUS_ICONS = {
    'green': '\033[92m●\033[0m',   # Green circle
    'yellow': '\033[93m●\033[0m',  # Yellow circle
    'red': '\033[91m●\033[0m',     # Red circle
}

STATUS_LABELS = {
    'green': 'Normal Operation',
    'yellow': 'Delays Detected',
    'red': 'Not Operating',
}


def main():
    parser = argparse.ArgumentParser(
        description='Analyze a Muni Metro status image',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('image', help='Path to the image file')
    parser.add_argument('--json', '-j', action='store_true', help='Output as JSON')
    parser.add_argument('--verbose', '-v', action='store_true', help='Show all detection details')

    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Error: Image not found: {image_path}", file=sys.stderr)
        sys.exit(1)

    try:
        result = detect_system_status(str(image_path))
    except Exception as e:
        print(f"Error analyzing image: {e}", file=sys.stderr)
        sys.exit(1)

    if args.json:
        print(json.dumps(result, indent=2))
        return

    # Pretty print results
    status = result['system_status']
    icon = STATUS_ICONS.get(status, '●')
    label = STATUS_LABELS.get(status, status.upper())

    print()
    print(f"  {icon} {label}")
    print(f"  {'─' * 40}")

    # Show delay summaries for yellow status
    summaries = result.get('delay_summaries', [])
    if summaries:
        print()
        for s in summaries:
            print(f"  • {s}")

    # Show train count
    trains = result.get('trains', [])
    print()
    print(f"  Trains detected: {len(trains)}")

    if args.verbose:
        # Show platform status
        stations = result.get('stations', [])
        holds = [s for s in stations if s.get('upper_color') == 'yellow' or s.get('lower_color') == 'yellow']
        if holds:
            print()
            print(f"  Platforms in hold ({len(holds)}):")
            for s in holds:
                upper = '⬤' if s.get('upper_color') == 'yellow' else '○'
                lower = '⬤' if s.get('lower_color') == 'yellow' else '○'
                print(f"    {s['name']}: upper={upper} lower={lower}")

        # Show track segments
        segments = result.get('delays_segments', [])
        if segments:
            print()
            print(f"  Track segments disabled ({len(segments)}):")
            for s in segments:
                print(f"    {s['from']} → {s['to']} ({s['direction']})")

        # Show bunching
        bunching = result.get('delays_bunching', [])
        if bunching:
            print()
            print(f"  Train bunching ({len(bunching)}):")
            for b in bunching:
                print(f"    {b['station']} ({b['direction']}): {b['train_count']} trains")

        # Show all trains
        if trains:
            print()
            print(f"  All trains:")
            for t in sorted(trains, key=lambda x: x['x']):
                conf = f" [{t['confidence']}]" if t.get('confidence') != 'high' else ''
                print(f"    {t['id']}{conf} @ x={t['x']} ({t['track']})")

    print()


if __name__ == '__main__':
    main()
