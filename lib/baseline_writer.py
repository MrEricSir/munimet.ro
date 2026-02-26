"""
Shared functions for updating test baseline files.

Used by both scripts/import-flagged-images.py (batch import) and
scripts/diagnostic_viewer.py (interactive corrections).
"""

import json
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TESTS_DIR = PROJECT_ROOT / 'tests'
BASELINE_TRAINS_PATH = TESTS_DIR / 'baseline_trains.json'
BASELINE_DELAYS_PATH = TESTS_DIR / 'baseline_delay_summaries.json'


def _compact_json(json_str: str) -> str:
    """Collapse small multi-line structures back to single lines.

    json.dumps with indent=2 expands ["M2031KK", 176] and
    {"max_false_positives": 0} across multiple lines. This collapses
    them back to compact single-line format.
    """
    # Collapse [id, x] train pairs
    json_str = re.sub(
        r'\[\s*\n\s*"([^"]+)",\s*\n\s*(\d+)\s*\n\s*\]',
        r'["\1", \2]',
        json_str,
    )
    # Collapse [id, x, {"ocr_override": "..."}]
    json_str = re.sub(
        r'\[\s*\n\s*"([^"]+)",\s*\n\s*(\d+),\s*\n\s*\{\s*\n\s*"ocr_override":\s*"([^"]+)"\s*\n\s*\}\s*\n\s*\]',
        r'["\1", \2, {"ocr_override": "\3"}]',
        json_str,
    )
    # Collapse [id, x, {"optional": true}]
    json_str = re.sub(
        r'\[\s*\n\s*"([^"]+)",\s*\n\s*(\d+),\s*\n\s*\{\s*\n\s*"optional":\s*true\s*\n\s*\}\s*\n\s*\]',
        r'["\1", \2, {"optional": true}]',
        json_str,
    )
    # Collapse {"max_false_positives": N}
    json_str = re.sub(
        r'\{\s*\n\s*"max_false_positives":\s*(\d+)\s*\n\s*\}',
        r'{"max_false_positives": \1}',
        json_str,
    )
    return json_str


def update_baseline_trains(image_name: str, trains: list[dict]) -> int:
    """Add or update an entry in baseline_trains.json.

    Args:
        image_name: Filename (e.g. 'IMG_9791.jpg')
        trains: List of dicts with 'id' and 'x' keys

    Returns:
        Number of trains written.
    """
    data = json.loads(BASELINE_TRAINS_PATH.read_text())

    train_entries = [[t['id'], t['x']] for t in sorted(trains, key=lambda t: t['x'])]
    data['images_with_trains'][image_name] = train_entries

    # Remove from train_free_images if present
    data.get('train_free_images', {}).pop(image_name, None)

    BASELINE_TRAINS_PATH.write_text(_compact_json(json.dumps(data, indent=2)) + '\n')
    return len(train_entries)


def write_baseline_entries(image_name: str, entries: list[list]) -> int:
    """Write pre-formed baseline entries, preserving metadata (overrides, optional flags).

    Args:
        image_name: Filename (e.g. 'IMG_9791.jpg')
        entries: List of [id, x] or [id, x, {metadata}] lists, already sorted by x.

    Returns:
        Number of entries written.
    """
    data = json.loads(BASELINE_TRAINS_PATH.read_text())
    data['images_with_trains'][image_name] = entries
    data.get('train_free_images', {}).pop(image_name, None)
    BASELINE_TRAINS_PATH.write_text(_compact_json(json.dumps(data, indent=2)) + '\n')
    return len(entries)


def update_baseline_delays(image_name: str, status: str, summaries: list[str]) -> None:
    """Add or update an entry in baseline_delay_summaries.json."""
    data = json.loads(BASELINE_DELAYS_PATH.read_text())

    data['images'][image_name] = {
        'status': status,
        'delay_summaries': summaries,
    }

    BASELINE_DELAYS_PATH.write_text(json.dumps(data, indent=2) + '\n')
