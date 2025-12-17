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

echo "[3/3] Verifying .gitattributes..."
if [ -f .gitattributes ]; then
    echo "✓ .gitattributes exists"
else
    echo "WARNING: .gitattributes not found"
    echo "This file should be committed to the repository"
fi
echo ""

echo "=========================================="
echo "✓ Git-annex configuration complete!"
echo "=========================================="
echo ""
echo "Automatic annexing is now enabled. When you run:"
echo "  git add artifacts/training_data/images/*.jpg"
echo ""
echo "Large files will be automatically annexed (symlinked)"
echo "instead of being added to regular git."
echo ""
echo "The pre-commit hook will prevent accidental commits"
echo "of large files that weren't properly annexed."
echo ""
