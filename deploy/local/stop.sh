#!/bin/bash
# Stop local MuniMetro services

set -e

# Navigate to project root
cd "$(dirname "$0")/../.."

echo "=========================================="
echo "Stopping MuniMetro Local Services"
echo "=========================================="
echo ""

PID_DIR="artifacts/runtime/pids"
STOPPED=0

# Stop API server
if [ -f "$PID_DIR/api.pid" ]; then
    API_PID=$(cat "$PID_DIR/api.pid")
    if ps -p $API_PID > /dev/null 2>&1; then
        echo "[1/2] Stopping API server (PID: $API_PID)..."
        kill $API_PID
        # Wait for graceful shutdown
        for i in {1..5}; do
            if ! ps -p $API_PID > /dev/null 2>&1; then
                echo "✓ API server stopped"
                STOPPED=$((STOPPED + 1))
                break
            fi
            sleep 1
        done
        # Force kill if still running
        if ps -p $API_PID > /dev/null 2>&1; then
            echo "  Force killing API server..."
            kill -9 $API_PID
            echo "✓ API server stopped (forced)"
            STOPPED=$((STOPPED + 1))
        fi
    else
        echo "[1/2] API server not running (stale PID file)"
    fi
    rm -f "$PID_DIR/api.pid"
else
    echo "[1/2] API server PID file not found"
fi
echo ""

# Stop cache writer
if [ -f "$PID_DIR/cache-writer.pid" ]; then
    CACHE_PID=$(cat "$PID_DIR/cache-writer.pid")
    if ps -p $CACHE_PID > /dev/null 2>&1; then
        echo "[2/2] Stopping cache writer (PID: $CACHE_PID)..."
        kill $CACHE_PID
        # Wait for graceful shutdown
        for i in {1..5}; do
            if ! ps -p $CACHE_PID > /dev/null 2>&1; then
                echo "✓ Cache writer stopped"
                STOPPED=$((STOPPED + 1))
                break
            fi
            sleep 1
        done
        # Force kill if still running
        if ps -p $CACHE_PID > /dev/null 2>&1; then
            echo "  Force killing cache writer..."
            kill -9 $CACHE_PID
            echo "✓ Cache writer stopped (forced)"
            STOPPED=$((STOPPED + 1))
        fi
    else
        echo "[2/2] Cache writer not running (stale PID file)"
    fi
    rm -f "$PID_DIR/cache-writer.pid"
else
    echo "[2/2] Cache writer PID file not found"
fi
echo ""

# Also check for any orphaned gunicorn processes
echo "Checking for orphaned processes..."
ORPHANED=$(pgrep -f "gunicorn.*api:app" || echo "")
if [ -n "$ORPHANED" ]; then
    echo "⚠️  Found orphaned gunicorn processes: $ORPHANED"
    echo "   Kill manually with: kill $ORPHANED"
else
    echo "✓ No orphaned processes found"
fi
echo ""

echo "=========================================="
if [ $STOPPED -eq 0 ]; then
    echo "⚠️  No services were running"
else
    echo "✓ Stopped $STOPPED service(s)"
fi
echo "=========================================="
echo ""
echo "Cleanup:"
echo "  Logs remain in artifacts/runtime/*.log"
echo "  Clear cache: rm artifacts/runtime/cache/*.json"
