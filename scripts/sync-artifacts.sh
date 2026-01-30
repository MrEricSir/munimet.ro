#!/bin/bash
# Sync artifacts (training images) with Google Cloud Storage
#
# This is an alias for sync-training-data.sh for convenience.
# ML models are no longer used (replaced with OpenCV-based detection).
#
# Usage:
#   ./scripts/sync-artifacts.sh upload    # Upload local changes to GCS
#   ./scripts/sync-artifacts.sh download  # Download changes from GCS
#   ./scripts/sync-artifacts.sh both      # Sync both directions
#   ./scripts/sync-artifacts.sh delete <path>  # Delete from GCS

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Just forward to sync-training-data.sh
exec "$SCRIPT_DIR/sync-training-data.sh" "$@"
