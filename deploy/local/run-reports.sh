#!/bin/bash
# Manually run the analytics reports job locally
#
# Usage:
#   ./deploy/local/run-reports.sh

set -e

# Navigate to project root
cd "$(dirname "$0")/../.."

echo "=========================================="
echo "Running MuniMetro Analytics Reports (Local)"
echo "=========================================="
echo ""

# Check if venv exists
if [ ! -d ".venv" ]; then
    echo "Error: Virtual environment not found: .venv"
    echo "  Run: ./deploy/local/setup.sh"
    exit 1
fi

# Check if analytics database exists
if [ ! -f "artifacts/runtime/analytics.db" ]; then
    echo "Warning: No analytics database found at artifacts/runtime/analytics.db"
    echo "  Reports will be empty until status checks are logged."
    echo "  Start local services: ./deploy/local/start.sh"
    echo ""
fi

# Activate virtual environment
source .venv/bin/activate

echo "Generating analytics reports..."
echo ""

# Run the reports job
python -m api.generate_reports_job

echo ""
echo "=========================================="
echo "Reports generated successfully!"
echo "=========================================="
echo ""
echo "Report cache location:"
echo "  artifacts/runtime/cache/"
echo ""
echo "View cached reports:"
echo "  ls -la artifacts/runtime/cache/analytics_*.json"
echo ""
echo "View in browser:"
echo "  http://localhost:8000/analytics"
