#!/bin/bash
# Sync artifacts with Google Cloud Storage
#
# Usage:
#   ./scripts/sync-artifacts.sh upload    # Upload local changes to GCS
#   ./scripts/sync-artifacts.sh download  # Download changes from GCS
#   ./scripts/sync-artifacts.sh both      # Sync both directions
#
# To delete files from GCS, use the individual scripts:
#   ./scripts/sync-models.sh delete <path>
#   ./scripts/sync-training-data.sh delete <path>

set -e

COMMAND="${1:-both}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=========================================="
echo "Sync Artifacts with Google Cloud Storage"
echo "=========================================="
echo ""

# Check authentication
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q .; then
    echo "Error: Not authenticated with Google Cloud"
    echo "Run: gcloud auth login"
    exit 1
fi

case "$COMMAND" in
    upload|download|both)
        echo "[1/2] Syncing training data..."
        "$SCRIPT_DIR/sync-training-data.sh" "$COMMAND"
        echo ""

        echo "[2/2] Syncing models..."
        "$SCRIPT_DIR/sync-models.sh" "$COMMAND"
        ;;

    *)
        echo "Error: Unknown command '$COMMAND'"
        echo ""
        echo "Usage:"
        echo "  ./scripts/sync-artifacts.sh upload    # Upload local changes to GCS"
        echo "  ./scripts/sync-artifacts.sh download  # Download changes from GCS"
        echo "  ./scripts/sync-artifacts.sh both      # Sync both directions"
        echo ""
        echo "To delete files from GCS, use the individual scripts:"
        echo "  ./scripts/sync-models.sh delete <path>"
        echo "  ./scripts/sync-training-data.sh delete <path>"
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo "Sync complete!"
echo "=========================================="
