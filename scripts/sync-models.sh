#!/bin/bash
# Sync trained models with Google Cloud Storage
#
# Usage:
#   ./scripts/sync-models.sh upload    # Upload local changes to GCS
#   ./scripts/sync-models.sh download  # Download changes from GCS
#   ./scripts/sync-models.sh both      # Sync both directions (default)

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
echo "Sync Models with GCS"
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
        echo "Uploading models to GCS (~856MB)..."
        echo ""
        gsutil -m rsync -r "${EXCLUDE_FLAGS[@]}" artifacts/models "$BUCKET/models"
        echo ""
        echo "✓ Models uploaded to $BUCKET/models"
        ;;

    download)
        echo "Downloading models from GCS (~856MB)..."
        echo ""
        mkdir -p artifacts/models
        gsutil -m rsync -r "${EXCLUDE_FLAGS[@]}" "$BUCKET/models" artifacts/models
        echo ""
        echo "✓ Models downloaded to artifacts/models"
        ;;

    both)
        echo "Syncing models bidirectionally..."
        echo ""
        gsutil -m rsync -r "${EXCLUDE_FLAGS[@]}" artifacts/models "$BUCKET/models"
        gsutil -m rsync -r "${EXCLUDE_FLAGS[@]}" "$BUCKET/models" artifacts/models
        echo ""
        echo "✓ Models synced"
        ;;

    *)
        echo "Error: Unknown command '$COMMAND'"
        echo ""
        echo "Usage:"
        echo "  ./scripts/sync-models.sh upload    # Upload local changes to GCS"
        echo "  ./scripts/sync-models.sh download  # Download changes from GCS"
        echo "  ./scripts/sync-models.sh both      # Sync both directions"
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo "Complete!"
echo "=========================================="
