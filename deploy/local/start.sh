#!/bin/bash
# Start local MuniMetro services

set -e

# Navigate to project root
cd "$(dirname "$0")/../.."

echo "=========================================="
echo "Starting MuniMetro Local Services"
echo "=========================================="
echo ""

# Check if venv exists
if [ ! -d ".venv" ]; then
    echo "❌ Virtual environment not found: .venv"
    echo "   Run: ./deploy/local/setup.sh"
    exit 1
fi

# Create PID directory
mkdir -p artifacts/runtime/pids

# Activate virtual environment
source .venv/bin/activate

# Verify required packages are installed
echo "Checking dependencies..."
MISSING=0
for pkg in falcon gunicorn cv2 numpy requests PIL pytesseract; do
    if ! python -c "import $pkg" 2>/dev/null; then
        echo "  ❌ Missing: $pkg"
        MISSING=$((MISSING + 1))
    fi
done
if [ $MISSING -gt 0 ]; then
    echo ""
    echo "❌ $MISSING required package(s) missing from .venv"
    echo "   Run: ./deploy/local/setup.sh"
    exit 1
fi
echo "✓ All dependencies available"
echo ""

echo "[1/2] Starting cache writer..."
# Start cache writer in background
nohup python api/check_status.py --continuous --write-cache --interval 60 \
    > artifacts/runtime/cache-writer.log 2>&1 &
CACHE_WRITER_PID=$!
echo $CACHE_WRITER_PID > artifacts/runtime/pids/cache-writer.pid
echo "✓ Cache writer started (PID: $CACHE_WRITER_PID)"
echo "  Log: artifacts/runtime/cache-writer.log"
echo ""

# Wait a moment for first cache write
echo "  Waiting 5 seconds for initial cache..."
sleep 5

echo "[2/2] Starting API server..."
# Start API in cache-only mode (fast responses)
ENABLE_FALLBACK=false nohup python -m gunicorn api.api:app \
    --bind 0.0.0.0:8000 \
    --timeout 120 \
    --workers 2 \
    --access-logfile artifacts/runtime/api-access.log \
    --error-logfile artifacts/runtime/api-error.log \
    > artifacts/runtime/api.log 2>&1 &
API_PID=$!
echo $API_PID > artifacts/runtime/pids/api.pid
echo "✓ API server started (PID: $API_PID)"
echo "  Access log: artifacts/runtime/api-access.log"
echo "  Error log: artifacts/runtime/api-error.log"
echo ""

# Wait for API to be ready
echo "  Waiting for API to be ready..."
for i in {1..10}; do
    if curl -sf http://localhost:8000/health > /dev/null 2>&1; then
        echo "  ✓ API is responding"
        break
    fi
    if [ $i -eq 10 ]; then
        echo "  ⚠️  API may not be ready yet (check logs)"
    fi
    sleep 1
done
echo ""

echo "=========================================="
echo "✓ Local services started successfully!"
echo "=========================================="
echo ""
echo "Services running:"
echo "  Cache Writer: PID $CACHE_WRITER_PID"
echo "  API Server:   PID $API_PID"
echo ""
echo "Access points:"
echo "  Dashboard:    http://localhost:8000"
echo "  Health:       http://localhost:8000/health"
echo "  Status API:   http://localhost:8000/status"
echo ""
echo "Management:"
echo "  View status:  ./deploy/local/verify.sh"
echo "  Stop all:     ./deploy/local/stop.sh"
echo ""
echo "Logs:"
echo "  tail -f artifacts/runtime/cache-writer.log"
echo "  tail -f artifacts/runtime/api-error.log"
