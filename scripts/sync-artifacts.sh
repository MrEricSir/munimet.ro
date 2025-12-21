#!/bin/bash
# Sync artifacts with Google Cloud Storage
#
# Usage:
#   ./scripts/sync-artifacts.sh upload    # Upload local changes to GCS
#   ./scripts/sync-artifacts.sh download  # Download changes from GCS
#   ./scripts/sync-artifacts.sh both      # Sync both directions

set -e

BUCKET="gs://munimetro-annex"
COMMAND="${1:-both}"

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
    upload)
        echo "Uploading artifacts to GCS..."
        echo ""

        echo "[1/2] Uploading training data..."
        gsutil -m rsync -r artifacts/training_data "$BUCKET/training_data"
        echo "✓ Training data uploaded"
        echo ""

        echo "[2/2] Uploading models..."
        gsutil -m rsync -r artifacts/models "$BUCKET/models"
        echo "✓ Models uploaded"
        ;;

    download)
        echo "Downloading artifacts from GCS..."
        echo ""

        echo "[1/2] Downloading training data (~270MB)..."
        mkdir -p artifacts/training_data
        gsutil -m rsync -r "$BUCKET/training_data" artifacts/training_data
        echo "✓ Training data downloaded"
        echo ""

        echo "[2/2] Downloading models (~856MB)..."
        mkdir -p artifacts/models
        gsutil -m rsync -r "$BUCKET/models" artifacts/models
        echo "✓ Models downloaded"
        ;;

    both)
        echo "Syncing artifacts bidirectionally..."
        echo ""

        echo "[1/2] Syncing training data..."
        gsutil -m rsync -r artifacts/training_data "$BUCKET/training_data"
        gsutil -m rsync -r "$BUCKET/training_data" artifacts/training_data
        echo "✓ Training data synced"
        echo ""

        echo "[2/2] Syncing models..."
        gsutil -m rsync -r artifacts/models "$BUCKET/models"
        gsutil -m rsync -r "$BUCKET/models" artifacts/models
        echo "✓ Models synced"
        ;;

    *)
        echo "Error: Unknown command '$COMMAND'"
        echo ""
        echo "Usage:"
        echo "  ./scripts/sync-artifacts.sh upload    # Upload local changes to GCS"
        echo "  ./scripts/sync-artifacts.sh download  # Download changes from GCS"
        echo "  ./scripts/sync-artifacts.sh both      # Sync both directions"
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo "Sync complete!"
echo "=========================================="
