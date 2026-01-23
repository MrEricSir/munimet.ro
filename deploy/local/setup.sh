#!/bin/bash
# Setup local development environment for MuniMetro

set -e

# Navigate to project root
cd "$(dirname "$0")/../.."

echo "=========================================="
echo "MuniMetro Local Development Setup"
echo "=========================================="
echo ""

# Check Python version
echo "[1/4] Checking Python version..."
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found"
    echo "   Install from: https://www.python.org/downloads/"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1-2)
echo "✓ Python $PYTHON_VERSION found"
echo ""

# Create virtual environment
echo "[2/4] Creating Python virtual environment..."
if [ -d "api/venv" ]; then
    echo "⚠️  Virtual environment already exists: api/venv"
    read -p "Recreate? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf api/venv
        python3 -m venv api/venv
        echo "✓ Virtual environment recreated"
    else
        echo "  Skipping..."
    fi
else
    python3 -m venv api/venv
    echo "✓ Virtual environment created: api/venv"
fi
echo ""

# Install dependencies
echo "[3/4] Installing Python dependencies..."
source api/venv/bin/activate
pip install --upgrade pip --quiet
pip install -r api/requirements.txt
echo "✓ Dependencies installed"
echo ""

# Create runtime directories
echo "[4/4] Creating runtime directories..."
mkdir -p artifacts/runtime/cache
mkdir -p artifacts/runtime/downloads
echo "✓ Runtime directories created"
echo ""

echo "=========================================="
echo "✓ Local development setup complete!"
echo "=========================================="
echo ""

# Check if credentials are configured
if [ ! -f ".env" ]; then
    echo "Optional: Configure credentials for Bluesky posting"
    read -p "Setup credentials now? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        python3 scripts/setup-credentials.py
    else
        echo "  Skipped. Run later with: python3 scripts/setup-credentials.py"
    fi
    echo ""
fi

echo "Next steps:"
echo "  1. Start services:  ./deploy/local/start.sh"
echo "  2. Verify status:   ./deploy/local/verify.sh"
echo "  3. Stop services:   ./deploy/local/stop.sh"
echo ""
echo "Or run individual components:"
echo "  Activate venv:      source api/venv/bin/activate"
echo "  Run checker:        python api/check_status.py --write-cache --continuous"
echo "  Run API:            python -m gunicorn api:app --bind 0.0.0.0:8000"
