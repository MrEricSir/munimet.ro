#!/bin/bash
# Verify local MuniMetro deployment status

set -e

# Navigate to project root
cd "$(dirname "$0")/../.."

echo "=========================================="
echo "MuniMetro Local Deployment Verification"
echo "=========================================="
echo ""

PID_DIR="artifacts/runtime/pids"
RUNTIME_DIR="artifacts/runtime"
ISSUES=0

echo "[1/5] Checking Python environment..."
if [ -d "api/venv" ]; then
    echo "✓ Virtual environment exists: api/venv"

    # Check if venv has dependencies (falcon, cv2, numpy for OpenCV-based detection)
    if source api/venv/bin/activate 2>/dev/null && python -c "import falcon, cv2, numpy" 2>/dev/null; then
        echo "✓ Dependencies installed"
        deactivate
    else
        echo "⚠️  Dependencies may be missing"
        echo "   Run: ./deploy/local/setup.sh"
        ISSUES=$((ISSUES + 1))
    fi
else
    echo "❌ Virtual environment not found"
    echo "   Run: ./deploy/local/setup.sh"
    ISSUES=$((ISSUES + 1))
fi
echo ""

echo "[2/5] Checking runtime directories..."
for dir in cache downloads pids; do
    if [ -d "$RUNTIME_DIR/$dir" ]; then
        echo "✓ $RUNTIME_DIR/$dir exists"
    else
        echo "⚠️  $RUNTIME_DIR/$dir missing (will be created on start)"
        mkdir -p "$RUNTIME_DIR/$dir"
    fi
done
echo ""

echo "[3/5] Checking cache writer status..."
if [ -f "$PID_DIR/cache-writer.pid" ]; then
    CACHE_PID=$(cat "$PID_DIR/cache-writer.pid")
    if ps -p $CACHE_PID > /dev/null 2>&1; then
        echo "✓ Cache writer running (PID: $CACHE_PID)"

        # Check if cache file exists
        if [ -f "$RUNTIME_DIR/cache/latest_status.json" ]; then
            echo "✓ Cache file exists"

            # Show cache age
            CACHE_AGE=$(( $(date +%s) - $(stat -f %m "$RUNTIME_DIR/cache/latest_status.json" 2>/dev/null || stat -c %Y "$RUNTIME_DIR/cache/latest_status.json" 2>/dev/null || echo 0) ))
            echo "  Age: ${CACHE_AGE}s"

            # Check cache content
            if command -v jq &> /dev/null; then
                STATUS=$(cat "$RUNTIME_DIR/cache/latest_status.json" | jq -r '.best_status.status // .statuses[0].status // "unknown"' 2>/dev/null || echo "unknown")
                TRAINS=$(cat "$RUNTIME_DIR/cache/latest_status.json" | jq -r '.best_status.detection.trains | length // 0' 2>/dev/null || echo "0")
                echo "  Status: $STATUS ($TRAINS trains detected)"
            fi
        else
            echo "⚠️  Cache file not found (may still be initializing)"
        fi
    else
        echo "❌ Cache writer not running (stale PID)"
        ISSUES=$((ISSUES + 1))
    fi
else
    echo "❌ Cache writer not running"
    echo "   Start with: ./deploy/local/start.sh"
    ISSUES=$((ISSUES + 1))
fi
echo ""

echo "[4/5] Checking API server status..."
if [ -f "$PID_DIR/api.pid" ]; then
    API_PID=$(cat "$PID_DIR/api.pid")
    if ps -p $API_PID > /dev/null 2>&1; then
        echo "✓ API server running (PID: $API_PID)"

        # Test health endpoint
        echo "  Testing /health..."
        if curl -sf http://localhost:8000/health > /dev/null 2>&1; then
            echo "  ✓ Health check passed"
        else
            echo "  ❌ Health check failed"
            ISSUES=$((ISSUES + 1))
        fi

        # Test status endpoint
        echo "  Testing /status..."
        if curl -sf http://localhost:8000/status > /dev/null 2>&1; then
            echo "  ✓ Status endpoint responding"
        else
            echo "  ⚠️  Status endpoint error (cache may be empty)"
        fi

        # Show port binding
        PORT_INFO=$(lsof -nP -iTCP:8000 -sTCP:LISTEN 2>/dev/null | tail -1 || echo "")
        if [ -n "$PORT_INFO" ]; then
            echo "  ✓ Listening on port 8000"
        fi
    else
        echo "❌ API server not running (stale PID)"
        ISSUES=$((ISSUES + 1))
    fi
else
    echo "❌ API server not running"
    echo "   Start with: ./deploy/local/start.sh"
    ISSUES=$((ISSUES + 1))
fi
echo ""

echo "[5/5] Recent logs (last 5 entries)..."
echo ""
if [ -f "$RUNTIME_DIR/cache-writer.log" ]; then
    echo "--- Cache Writer ---"
    tail -5 "$RUNTIME_DIR/cache-writer.log" 2>/dev/null || echo "No logs available"
else
    echo "--- Cache Writer ---"
    echo "No log file found"
fi

echo ""
if [ -f "$RUNTIME_DIR/api-error.log" ]; then
    echo "--- API Server (errors) ---"
    tail -5 "$RUNTIME_DIR/api-error.log" 2>/dev/null || echo "No logs available"
else
    echo "--- API Server (errors) ---"
    echo "No log file found"
fi

echo ""
echo "=========================================="
if [ $ISSUES -eq 0 ]; then
    echo "✓ Verification Complete - No Issues"
else
    echo "⚠️  Verification Complete - $ISSUES issue(s) found"
fi
echo "=========================================="
echo ""

# Summary
if ps -p $(cat "$PID_DIR/api.pid" 2>/dev/null || echo "0") > /dev/null 2>&1; then
    echo "🌐 Local Dashboard: http://localhost:8000"
    echo ""
    echo "Quick commands:"
    echo "  Test API:       curl http://localhost:8000/status"
    echo "  View logs:      tail -f $RUNTIME_DIR/api-error.log"
    echo "  Stop services:  ./deploy/local/stop.sh"
fi
