#!/bin/bash
# Run tests in the same container environment as CI and GCP Cloud Run.
# This ensures OCR results match production exactly.
#
# Usage:
#   ./scripts/test-in-docker.sh           # Run all tests
#   ./scripts/test-in-docker.sh --quick   # Quick tests (skip train detection)
#   ./scripts/test-in-docker.sh -v        # Verbose output
#   ./scripts/test-in-docker.sh -k bunch  # Run only bunching tests

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Check for --quick flag
QUICK_MODE=false
REMAINING_ARGS=()
for arg in "$@"; do
    if [ "$arg" = "--quick" ]; then
        QUICK_MODE=true
    else
        REMAINING_ARGS+=("$arg")
    fi
done

# Use same image as CI and Dockerfile
IMAGE="python:3.13-slim-bookworm"

# Check if Docker is available and running
if ! command -v docker &> /dev/null; then
    echo "Error: Docker is not installed."
    echo ""
    echo "Please install Docker Desktop or Colima:"
    echo "  brew install --cask docker    # Docker Desktop"
    echo "  brew install colima && colima start   # Colima (lightweight)"
    exit 1
fi

if ! docker info &> /dev/null; then
    echo "Error: Docker daemon is not running."
    echo ""
    echo "Please start Docker:"
    echo "  - Docker Desktop: Open the Docker Desktop app"
    echo "  - Colima: Run 'colima start'"
    echo ""
    echo "Then re-run this script."
    exit 1
fi

echo "========================================"
echo "Running tests in Docker ($IMAGE)"
echo "========================================"
echo "This matches the CI and GCP Cloud Run environment."
echo ""

# Set up pytest arguments
if [ ${#REMAINING_ARGS[@]} -gt 0 ]; then
    PYTEST_ARGS="${REMAINING_ARGS[*]}"
else
    PYTEST_ARGS="-v --tb=short"
fi

# Select test files based on quick mode
if [ "$QUICK_MODE" = true ]; then
    TEST_FILES="tests/test_system_status.py"
    echo "(Quick mode: skipping train detection tests)"
    echo ""
else
    TEST_FILES="tests/test_system_status.py tests/test_train_detection.py"
fi

docker run --rm \
    -v "$PROJECT_ROOT:/app" \
    -w /app \
    "$IMAGE" \
    bash -c "
        set -e
        echo 'Installing dependencies...'
        apt-get update -qq
        apt-get install -y -qq tesseract-ocr tesseract-ocr-eng >/dev/null 2>&1
        pip install -q -r api/requirements.txt pytest

        echo ''
        echo 'Environment:'
        echo \"  Python: \$(python --version)\"
        echo \"  Tesseract: \$(tesseract --version 2>&1 | head -1)\"
        echo ''
        echo '========================================'
        echo 'Running pytest...'
        echo '========================================'
        python -m pytest $TEST_FILES $PYTEST_ARGS
    "

echo ""
echo "========================================"
echo "Docker tests completed successfully!"
echo "========================================"
