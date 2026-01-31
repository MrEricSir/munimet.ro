#!/usr/bin/env python3
"""
Download a Muni snapshot image.

Usage:
    python download_muni_image.py                    # Download to default location
    python download_muni_image.py /path/to/folder   # Download to specified folder
"""
import sys
from pathlib import Path

# Add parent directory to path for lib imports
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from lib.muni_lib import download_muni_image

# Default output folder
DEFAULT_OUTPUT = str(PROJECT_ROOT / "artifacts" / "runtime" / "downloads")


def main():
    # Use command line arg or default
    output_folder = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_OUTPUT

    print(f"Downloading to: {output_folder}")
    result = download_muni_image(output_folder=output_folder, validate_dimensions=True)

    if result['success']:
        print(f"Downloaded: {result['filepath']}")
        print(f"Dimensions: {result['width']}x{result['height']}")
    else:
        print(f"Failed: {result['error']}")
        sys.exit(1)


if __name__ == "__main__":
    main()
