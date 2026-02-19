#!/bin/bash
set -e

echo "=========================================="
echo "Muni Metro - Python Environment Setup"
echo "=========================================="
echo ""

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

# Get the project root directory (2 levels up from this script)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "Project root: $PROJECT_ROOT"
echo ""

cd "$PROJECT_ROOT"

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
else
    echo "Virtual environment already exists."
fi

# Activate virtual environment
source .venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
python -m pip install --upgrade pip

# Install dependencies
echo "Installing dependencies from api/requirements.txt..."
pip install -r api/requirements.txt

# Deactivate virtual environment
deactivate

echo -e "${GREEN}=========================================="
echo "Python environment setup complete!"
echo "==========================================${NC}"
echo ""
echo "To activate the environment:"
echo "  source .venv/bin/activate"
echo ""
echo "Next steps:"
echo "1. Download reference data: ./scripts/sync-reference-data.sh download"
echo "2. Follow SETUP.md for project-specific configuration"
echo ""
