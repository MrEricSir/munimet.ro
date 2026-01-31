#!/bin/bash
# Sync artifacts (reference images) with Google Cloud Storage
#
# This is an alias for sync-reference-data.sh for convenience.
#
# Usage:
#   ./scripts/sync-artifacts.sh upload    # Upload local changes to GCS
#   ./scripts/sync-artifacts.sh download  # Download changes from GCS
#   ./scripts/sync-artifacts.sh both      # Sync both directions
#   ./scripts/sync-artifacts.sh delete <path>  # Delete from GCS

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Just forward to sync-reference-data.sh
exec "$SCRIPT_DIR/sync-reference-data.sh" "$@"
