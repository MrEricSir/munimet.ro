#!/usr/bin/env python3
"""Regenerate tests/baseline_trains.json from current detection output.

Runs TrainDetector.detect_trains() on every test image and updates baseline
entries while preserving metadata (ocr_override, optional flags) where
the underlying detection still matches by position.

Usage:
    # Dry run — print what would change
    python scripts/regenerate_baselines.py

    # Write changes to baseline_trains.json
    python scripts/regenerate_baselines.py --write
"""

import json
import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
from lib.train_detector import TrainDetector
from lib.baseline_writer import _compact_json

TESTS_DIR = PROJECT_ROOT / 'tests'
IMAGES_DIR = TESTS_DIR / 'images'
BASELINE_PATH = TESTS_DIR / 'baseline_trains.json'
TOLERANCE = 30


def load_baseline():
    return json.loads(BASELINE_PATH.read_text())


def detect_all(images_with_trains, train_free_images):
    """Run detection on all baseline images, return {filename: [trains]}."""
    detector = TrainDetector()
    results = {}

    all_images = list(images_with_trains.keys()) + list(train_free_images.keys())
    for name in sorted(set(all_images)):
        img_path = IMAGES_DIR / name
        if not img_path.exists():
            print(f"  WARNING: {name} not found, skipping")
            continue
        img = cv2.imread(str(img_path))
        trains = detector.detect_trains(img)
        results[name] = trains
    return results


def match_baselines(old_entries, detected_trains):
    """Match old baseline entries to new detections, preserving metadata.

    Returns new list of baseline entries.
    """
    new_entries = []
    used_detected = set()  # indices into detected_trains already matched

    # Sort old entries by x position
    old_sorted = sorted(old_entries, key=lambda e: e[1])

    for entry in old_sorted:
        old_id = entry[0]
        old_x = entry[1]
        old_meta = entry[2] if len(entry) > 2 else None

        # Find closest detected train within tolerance on same approximate x
        best_idx = None
        best_dist = TOLERANCE + 1
        for i, det in enumerate(detected_trains):
            if i in used_detected:
                continue
            dist = abs(det['x'] - old_x)
            if dist < best_dist:
                best_dist = dist
                best_idx = i

        if best_idx is not None and best_dist <= TOLERANCE:
            det = detected_trains[best_idx]
            used_detected.add(best_idx)
            det_id = det['id']
            det_x = det['x']

            det_x = int(det_x)  # numpy int64 -> Python int

            if det_id == old_id:
                # Same ID, update x position, preserve metadata
                if old_meta:
                    new_entries.append([old_id, det_x, old_meta])
                else:
                    new_entries.append([old_id, det_x])
            elif old_meta and old_meta.get('ocr_override') == det_id:
                # Detection matches the ocr_override — keep entry as-is with updated x
                new_entries.append([old_id, det_x, old_meta])
            else:
                # Different ID detected at same position
                # Use detected ID as primary, keep old primary as ocr_override
                new_entries.append([det_id, det_x, {'ocr_override': old_id}])
        else:
            # Old baseline entry not found in detections — mark optional
            if old_meta and old_meta.get('optional'):
                new_entries.append(entry)  # Already optional, keep as-is
            else:
                meta = dict(old_meta) if old_meta else {}
                meta['optional'] = True
                new_entries.append([old_id, old_x, meta])

    # Add newly detected trains not matched to any old entry
    for i, det in enumerate(detected_trains):
        if i not in used_detected:
            det_id = det['id']
            # Skip UNKNOWN detections as new entries — they're noise
            if det_id.startswith('UNKNOWN'):
                continue
            new_entries.append([det_id, int(det['x'])])

    # Sort by x position
    new_entries.sort(key=lambda e: e[1])
    return new_entries


def format_entry(entry):
    """Format a single baseline entry for display."""
    if len(entry) > 2:
        return f'["{entry[0]}", {entry[1]}, {json.dumps(entry[2])}]'
    return f'["{entry[0]}", {entry[1]}]'


def main():
    write = '--write' in sys.argv

    baseline = load_baseline()
    images_with_trains = baseline['images_with_trains']
    train_free_images = baseline.get('train_free_images', {})

    print(f"Detecting trains in {len(images_with_trains)} images...")
    detections = detect_all(images_with_trains, train_free_images)

    updated = {}
    changes = 0

    for name, old_entries in images_with_trains.items():
        detected = detections.get(name, [])
        new_entries = match_baselines(old_entries, detected)
        updated[name] = new_entries

        # Check for changes
        if json.dumps(new_entries) != json.dumps(old_entries):
            changes += 1
            print(f"\n=== {name} ===")
            print(f"  Old: {len(old_entries)} entries, New: {len(new_entries)} entries")

            # Show removed entries
            old_ids = {(e[0], e[1]) for e in old_entries}
            new_ids = {(e[0], e[1]) for e in new_entries}

            for entry in old_entries:
                key = (entry[0], entry[1])
                if key not in new_ids:
                    # Find if it was updated (same approx x, different id)
                    matched_new = [e for e in new_entries if abs(e[1] - entry[1]) <= TOLERANCE]
                    if matched_new:
                        print(f"  CHANGED: {format_entry(entry)} -> {format_entry(matched_new[0])}")
                    else:
                        print(f"  REMOVED: {format_entry(entry)}")

            for entry in new_entries:
                key = (entry[0], entry[1])
                if key not in old_ids:
                    matched_old = [e for e in old_entries if abs(e[1] - entry[1]) <= TOLERANCE]
                    if not matched_old:
                        print(f"  ADDED: {format_entry(entry)}")

    if changes == 0:
        print("\nNo changes detected. Baselines are up to date.")
        return

    print(f"\n{changes} images had baseline changes.")

    if write:
        new_baseline = {
            'description': baseline['description'],
            'tolerance': baseline['tolerance'],
            'images_with_trains': updated,
            'train_free_images': train_free_images,
        }
        BASELINE_PATH.write_text(_compact_json(json.dumps(new_baseline, indent=2)) + '\n')
        print(f"Written to {BASELINE_PATH}")
    else:
        print("\nDry run. Use --write to update baseline_trains.json")


if __name__ == '__main__':
    main()
