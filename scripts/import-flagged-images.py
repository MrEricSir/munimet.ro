#!/usr/bin/env python3
"""
Import flagged images from a review-archive.py export into the test suite.

For each flagged image:
1. Copies it to tests/images/ (strips archive reason suffix)
2. Runs detection and shows results
3. Updates tests/baseline_trains.json
4. Appends to KNOWN_STATUSES in tests/test_system_status.py
5. Updates tests/baseline_delay_summaries.json

Usage:
    python scripts/import-flagged-images.py flagged-images.json
    python scripts/import-flagged-images.py flagged-images.json --yes  # Skip confirmation prompts
"""

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from lib.detection import detect_system_status

TESTS_DIR = PROJECT_ROOT / 'tests'
IMAGES_DIR = TESTS_DIR / 'images'
BASELINE_TRAINS_PATH = TESTS_DIR / 'baseline_trains.json'
BASELINE_DELAYS_PATH = TESTS_DIR / 'baseline_delay_summaries.json'
KNOWN_STATUSES_PATH = TESTS_DIR / 'test_system_status.py'

FLAG_DESCRIPTIONS = {
    'missing_trains': 'Flagged: missing trains',
    'extra_trains': 'Flagged: extra/false trains',
    'wrong_status': 'Flagged: wrong status',
}

# Strips reason (and optional raw status) suffix from archive filenames
# e.g. muni_snapshot_20260221_083000_override_rawYellow.jpg -> muni_snapshot_20260221_083000.jpg
ARCHIVE_SUFFIX_PATTERN = re.compile(
    r'(muni_snapshot_\d{8}_\d{6})_\w+\.jpg'
)


def clean_image_name(archive_filename: str) -> str:
    """Strip archive reason suffix to get a clean test image name."""
    m = ARCHIVE_SUFFIX_PATTERN.match(archive_filename)
    if m:
        return m.group(1) + '.jpg'
    return archive_filename


def run_detection(image_path: str) -> dict:
    """Run detection and return results."""
    return detect_system_status(image_path)


def display_detection(det: dict, image_name: str):
    """Print a human-readable summary of detection results."""
    status = det['system_status']
    trains = det.get('trains', [])
    summaries = det.get('delay_summaries', [])

    upper = [t for t in trains if t.get('track') == 'upper']
    lower = [t for t in trains if t.get('track') == 'lower']

    print(f'\n  Status: {status.upper()}')
    print(f'  Trains: {len(trains)} ({len(upper)} upper, {len(lower)} lower)')

    if summaries:
        for s in summaries:
            print(f'    - {s}')

    if trains:
        for track_name, track_trains in [('Upper', upper), ('Lower', lower)]:
            if track_trains:
                ids = ', '.join(t['id'] for t in sorted(track_trains, key=lambda t: t['x']))
                print(f'  {track_name}: {ids}')


def update_baseline_trains(image_name: str, trains: list[dict]):
    """Add or update an entry in baseline_trains.json."""
    data = json.loads(BASELINE_TRAINS_PATH.read_text())

    train_entries = [[t['id'], t['x']] for t in sorted(trains, key=lambda t: t['x'])]
    data['images_with_trains'][image_name] = train_entries

    BASELINE_TRAINS_PATH.write_text(json.dumps(data, indent=2) + '\n')
    print(f'    Updated baseline_trains.json ({len(train_entries)} trains)')


def update_baseline_delays(image_name: str, status: str, summaries: list[str]):
    """Add or update an entry in baseline_delay_summaries.json."""
    data = json.loads(BASELINE_DELAYS_PATH.read_text())

    data['images'][image_name] = {
        'status': status,
        'delay_summaries': summaries,
    }

    BASELINE_DELAYS_PATH.write_text(json.dumps(data, indent=2) + '\n')
    print(f'    Updated baseline_delay_summaries.json')


def update_known_statuses(image_name: str, status: str, notes: str):
    """Append an entry to KNOWN_STATUSES in test_system_status.py."""
    content = KNOWN_STATUSES_PATH.read_text()

    entry = f'    ("{image_name}", "{status}", "{notes}"),\n'

    # Find the closing bracket of KNOWN_STATUSES
    # Insert before it
    marker = '\n]\n'
    pos = content.find(marker)
    if pos == -1:
        print(f'    WARNING: Could not find KNOWN_STATUSES closing bracket')
        print(f'    Add manually: {entry.strip()}')
        return

    new_content = content[:pos] + '\n' + entry + content[pos:]
    KNOWN_STATUSES_PATH.write_text(new_content)
    print(f'    Updated KNOWN_STATUSES in test_system_status.py')


def process_image(entry: dict, auto_yes: bool) -> bool:
    """Process a single flagged image. Returns True if imported."""
    filename = entry['filename']
    source_dir = Path(entry['source_dir'])
    flag = entry['flag']
    notes = entry.get('notes', '')

    image_name = clean_image_name(filename)
    source_path = source_dir / filename
    dest_path = IMAGES_DIR / image_name

    print(f'\n{"="*60}')
    print(f'  File: {filename}')
    print(f'  Flag: {FLAG_DESCRIPTIONS.get(flag, flag)}')
    if notes:
        print(f'  Notes: {notes}')
    print(f'  Test name: {image_name}')

    if not source_path.exists():
        print(f'  ERROR: Source file not found: {source_path}')
        return False

    if dest_path.exists():
        print(f'  WARNING: {image_name} already exists in tests/images/')

    # Run detection
    print(f'\n  Running detection...')
    det = run_detection(str(source_path))
    display_detection(det, image_name)

    status = det['system_status']
    trains = det.get('trains', [])
    summaries = det.get('delay_summaries', [])

    # Build description for KNOWN_STATUSES
    desc_parts = [FLAG_DESCRIPTIONS.get(flag, flag)]
    if notes:
        desc_parts.append(notes)
    description = ' - '.join(desc_parts)
    # Escape any double quotes in the description
    description = description.replace('"', '\\"')

    # Confirm
    if not auto_yes:
        print(f'\n  Will import as:')
        print(f'    Image: tests/images/{image_name}')
        print(f'    Status: {status}')
        print(f'    Trains: {len(trains)}')
        print(f'    Description: {description}')

        answer = input('\n  Import this image? [Y/n/s(kip)] ').strip().lower()
        if answer == 's':
            print('  Skipped.')
            return False
        if answer and answer != 'y':
            # Allow overriding the status
            if answer in ('green', 'yellow', 'red'):
                status = answer
                print(f'  Status overridden to: {status}')
            else:
                print('  Skipped.')
                return False

    # Copy image
    shutil.copy2(source_path, dest_path)
    print(f'\n    Copied to tests/images/{image_name}')

    # Update test files
    if trains:
        update_baseline_trains(image_name, trains)
    update_baseline_delays(image_name, status, summaries)
    update_known_statuses(image_name, status, description)

    return True


def main():
    parser = argparse.ArgumentParser(
        description='Import flagged images from review-archive.py export into the test suite.'
    )
    parser.add_argument('manifest', help='Path to flagged-images.json exported from the review report')
    parser.add_argument('--yes', '-y', action='store_true',
                        help='Skip confirmation prompts (auto-accept all)')

    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(f'Error: File not found: {manifest_path}')
        sys.exit(1)

    data = json.loads(manifest_path.read_text())
    images = data.get('images', [])

    if not images:
        print('No flagged images in manifest.')
        sys.exit(0)

    print(f'Found {len(images)} flagged image(s)')
    print(f'Exported at: {data.get("exported_at", "unknown")}')

    imported = 0
    for entry in images:
        if process_image(entry, auto_yes=args.yes):
            imported += 1

    print(f'\n{"="*60}')
    print(f'Imported {imported}/{len(images)} images')
    if imported > 0:
        print(f'\nNext steps:')
        print(f'  1. Review the changes: git diff tests/')
        print(f'  2. Run tests: python -m pytest tests/test_train_detection.py tests/test_system_status.py -v')
        print(f'  3. Fix any detection issues, then re-run tests')


if __name__ == '__main__':
    main()
