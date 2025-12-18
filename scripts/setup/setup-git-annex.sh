#!/bin/bash
# Configure git-annex for automatic large file handling
# Run this after: git annex init "your-computer-name"

set -e

echo "=========================================="
echo "Git-Annex Configuration"
echo "=========================================="
echo ""

# Get the project root directory (2 levels up from this script)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT" || exit 1

# Check if git-annex is initialized
if ! git annex version >/dev/null 2>&1; then
    echo "ERROR: git-annex is not installed"
    echo "Please install git-annex first (see SETUP.md)"
    exit 1
fi

# Check if repository is initialized
if ! git config --local annex.uuid >/dev/null 2>&1; then
    echo "WARNING: Git-annex not initialized for this repository"
    echo "Run: git annex init \"your-computer-name\""
    echo ""
    read -p "Would you like to initialize now? [y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        read -p "Enter a descriptive name for this computer: " computer_name
        git annex init "$computer_name"
    else
        exit 1
    fi
fi

echo "[1/3] Configuring git-annex filter..."
# Enable automatic annexing via git filter
git config filter.annex.process 'git-annex filter-process'
echo "✓ Git-annex filter enabled"
echo ""

echo "[2/3] Setting largefiles configuration..."
# Configure which files should be automatically annexed
git annex config --set annex.largefiles 'largerthan=100kb or mimetype=image/*'
echo "✓ Files >100KB or images will be automatically annexed"
echo ""

echo "[3/5] Verifying .gitattributes..."
if [ -f .gitattributes ]; then
    echo "✓ .gitattributes exists"
else
    echo "WARNING: .gitattributes not found"
    echo "This file should be committed to the repository"
fi
echo ""

echo "[4/5] Enabling gcs remote..."
# Check if remote exists
if git annex info gcs >/dev/null 2>&1; then
    echo "✓ gcs remote is already available"
else
    echo "Attempting to enable gcs remote..."
    if git annex enableremote gcs 2>/dev/null; then
        echo "✓ gcs remote enabled successfully"
    else
        echo "⚠ Could not enable remote (this is normal for first-time setup)"
        echo "  The remote will be configured automatically when you clone"
    fi
fi
echo ""

echo "[5/5] Downloading pre-trained model..."
# Check if model files are already present
if [ -e "artifacts/models/v1/model.safetensors" ] && [ ! -L "artifacts/models/v1/model.safetensors" ]; then
    echo "✓ Model files already downloaded"
elif git annex whereis artifacts/models/v1/ >/dev/null 2>&1; then
    echo "Downloading model files (856MB, this may take a few minutes)..."
    if git annex get artifacts/models/v1/ --jobs=4; then
        echo "✓ Model files downloaded successfully"
    else
        echo "⚠ Could not download model files"
        echo "  You can download them later with: git annex get artifacts/models/v1/"
    fi
else
    echo "⚠ Model files not available yet (repository may need to be pushed)"
    echo "  You can download them later with: git annex get artifacts/models/v1/"
fi
echo ""

echo "=========================================="
echo "✓ Git-annex configuration complete!"
echo "=========================================="
echo ""
echo "Summary:"
echo "• Automatic annexing enabled for files >100KB or images"
echo "• Large files will be symlinked instead of added to git"
echo "• Pre-commit hook prevents accidental large file commits"
echo "• Model files downloaded (if available)"
echo ""
echo "For collaborators with training data access:"
echo "  1. Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY"
echo "  2. Run: git annex enableremote gcs"
echo "  3. Download training data: git annex get artifacts/training_data/"
echo ""
echo "See GCS_SETUP.md for detailed instructions."
echo ""
