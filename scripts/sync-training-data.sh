#!/bin/bash
# Sync training data with Google Cloud Storage
#
# Usage:
#   ./scripts/sync-training-data.sh upload    # Upload local changes to GCS
#   ./scripts/sync-training-data.sh download  # Download changes from GCS
#   ./scripts/sync-training-data.sh both      # Sync both directions (default)

set -e

BUCKET="gs://munimetro-annex"
COMMAND="${1:-both}"

# Exclusions based on .gitignore (regex patterns for gsutil -x flag)
EXCLUDE_PATTERNS=(
    ".*\.DS_Store$"        # macOS metadata
    ".*__MACOSX.*"         # macOS archive metadata
    ".*__pycache__.*"      # Python cache
    ".*\.pyc$"             # Python bytecode
    ".*\.pyo$"             # Python optimized bytecode
    ".*\.pyd$"             # Python DLL
    ".*\.log$"             # Log files
    ".*\.ipynb_checkpoints.*"  # Jupyter checkpoints
    ".*\.swp$"             # Vim swap files
    ".*\.swo$"             # Vim swap files
    ".*~$"                 # Backup files
)

echo "=========================================="
echo "Sync Training Data with GCS"
echo "=========================================="
echo ""

# Check authentication
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q .; then
    echo "Error: Not authenticated with Google Cloud"
    echo "Run: gcloud auth login"
    exit 1
fi

# Build exclusion flags for gsutil
EXCLUDE_FLAGS=()
for pattern in "${EXCLUDE_PATTERNS[@]}"; do
    EXCLUDE_FLAGS+=("-x" "$pattern")
done

case "$COMMAND" in
    upload)
        echo "Uploading training data to GCS..."
        echo ""
        gsutil -m rsync -r "${EXCLUDE_FLAGS[@]}" artifacts/training_data "$BUCKET/training_data"
        echo ""
        echo "✓ Training data uploaded to $BUCKET/training_data"
        ;;

    download)
        echo "Downloading training data from GCS (~270MB)..."
        echo ""
        mkdir -p artifacts/training_data
        gsutil -m rsync -r "${EXCLUDE_FLAGS[@]}" "$BUCKET/training_data" artifacts/training_data
        echo ""
        echo "✓ Training data downloaded to artifacts/training_data"
        ;;

    both)
        echo "Syncing training data bidirectionally..."
        echo ""
        gsutil -m rsync -r "${EXCLUDE_FLAGS[@]}" artifacts/training_data "$BUCKET/training_data"
        gsutil -m rsync -r "${EXCLUDE_FLAGS[@]}" "$BUCKET/training_data" artifacts/training_data
        echo ""
        echo "✓ Training data synced"
        ;;

    *)
        echo "Error: Unknown command '$COMMAND'"
        echo ""
        echo "Usage:"
        echo "  ./scripts/sync-training-data.sh upload    # Upload local changes to GCS"
        echo "  ./scripts/sync-training-data.sh download  # Download changes from GCS"
        echo "  ./scripts/sync-training-data.sh both      # Sync both directions"
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo "Complete!"
echo "=========================================="
