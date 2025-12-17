#!/bin/bash
# Script to automatically annex new training images
# Run this periodically (e.g., via cron) or manually after downloading images

cd "$(dirname "$0")/.." || exit 1

# Find any regular (non-symlink) JPEG files and annex them
regular_files=$(find artifacts/training_data/images -name "*.jpg" -type f 2>/dev/null)

if [ -n "$regular_files" ]; then
    count=$(echo "$regular_files" | wc -l | tr -d ' ')
    echo "Found $count new image(s) to annex..."

    find artifacts/training_data/images -name "*.jpg" -type f -exec git annex add {} +

    echo "✓ Annexed $count image(s)"
    echo "Don't forget to commit: git commit -m 'Add new training images'"
else
    echo "No new images to annex"
fi
