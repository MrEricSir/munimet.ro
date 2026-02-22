#!/usr/bin/env python3
"""
Download archived snapshot images from GCS for local examination.

Usage:
    python scripts/download-archive-images.py                    # Today's images
    python scripts/download-archive-images.py --date 2026-02-21  # Specific date
    python scripts/download-archive-images.py --from 2026-02-01 --to 2026-02-21  # Date range
    python scripts/download-archive-images.py --reason transition # Filter by reason
    python scripts/download-archive-images.py --list             # List only, don't download
    python scripts/download-archive-images.py --output-dir ./my-images  # Custom output dir
"""

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from lib.config import DEFAULT_ARCHIVE_BUCKET
from lib.gcs_utils import gcs_list_blobs, gcs_download_to_file


def date_range(start: str, end: str) -> list[str]:
    """Generate YYYY/MM/DD prefixes for each date in the range (inclusive)."""
    start_dt = datetime.strptime(start, '%Y-%m-%d')
    end_dt = datetime.strptime(end, '%Y-%m-%d')
    prefixes = []
    current = start_dt
    while current <= end_dt:
        prefixes.append(current.strftime('%Y/%m/%d/'))
        current += timedelta(days=1)
    return prefixes


def main():
    parser = argparse.ArgumentParser(
        description='Download archived snapshot images from GCS.'
    )
    parser.add_argument('--date', help='Specific date (YYYY-MM-DD)')
    parser.add_argument('--from', dest='from_date', help='Start date for range (YYYY-MM-DD)')
    parser.add_argument('--to', dest='to_date', help='End date for range (YYYY-MM-DD)')
    parser.add_argument(
        '--reason',
        choices=['transition', 'override', 'baseline'],
        help='Filter by archive reason'
    )
    parser.add_argument('--list', action='store_true', help='List only, do not download')
    parser.add_argument(
        '--output-dir',
        default=str(PROJECT_ROOT / 'artifacts' / 'runtime' / 'archive'),
        help='Output directory (default: artifacts/runtime/archive/)'
    )
    parser.add_argument(
        '--bucket',
        default=DEFAULT_ARCHIVE_BUCKET,
        help=f'GCS bucket name (default: {DEFAULT_ARCHIVE_BUCKET})'
    )

    args = parser.parse_args()

    # Determine date prefixes to search
    if args.from_date and args.to_date:
        prefixes = date_range(args.from_date, args.to_date)
    elif args.date:
        dt = datetime.strptime(args.date, '%Y-%m-%d')
        prefixes = [dt.strftime('%Y/%m/%d/')]
    else:
        prefixes = [datetime.now().strftime('%Y/%m/%d/')]

    # Collect matching blobs
    all_blobs = []
    for prefix in prefixes:
        blobs = gcs_list_blobs(args.bucket, prefix=prefix)
        all_blobs.extend(blobs)

    # Filter by reason if specified
    if args.reason:
        all_blobs = [b for b in all_blobs if f'_{args.reason}.jpg' in b]

    if not all_blobs:
        print('No archived images found matching the criteria.')
        return

    print(f'Found {len(all_blobs)} archived image(s):')
    for blob_name in sorted(all_blobs):
        print(f'  gs://{args.bucket}/{blob_name}')

    if args.list:
        return

    # Download
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    downloaded = 0
    for blob_name in sorted(all_blobs):
        filename = Path(blob_name).name
        local_path = output_dir / filename
        print(f'  Downloading {filename}...', end=' ')
        if gcs_download_to_file(args.bucket, blob_name, str(local_path)):
            print('OK')
            downloaded += 1
        else:
            print('FAILED (not found)')

    print(f'\nDownloaded {downloaded}/{len(all_blobs)} images to {output_dir}')


if __name__ == '__main__':
    main()
