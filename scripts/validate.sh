#!/bin/bash
# Local validation script - runs the same checks as GitHub CI
#
# Usage:
#   ./scripts/validate.sh          # Run all validations
#   ./scripts/validate.sh --quick  # Skip slow tests (train detection)
#   ./scripts/validate.sh --lint   # Only run linting
#   ./scripts/validate.sh --test   # Only run tests

set -e

# Navigate to project root
cd "$(dirname "$0")/.."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Parse arguments
RUN_LINT=true
RUN_TESTS=true
QUICK_MODE=false

for arg in "$@"; do
    case $arg in
        --quick)
            QUICK_MODE=true
            ;;
        --lint)
            RUN_TESTS=false
            ;;
        --test)
            RUN_LINT=false
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --quick    Skip slow tests (train detection)"
            echo "  --lint     Only run linting"
            echo "  --test     Only run tests"
            echo "  --help     Show this help"
            exit 0
            ;;
    esac
done

echo "=========================================="
echo "MuniMetro Local Validation"
echo "=========================================="
echo ""

ERRORS=0

# Check for virtual environment
if [ ! -d ".venv" ]; then
    echo -e "${RED}Error: Virtual environment not found${NC}"
    echo "Run: ./deploy/local/setup.sh"
    exit 1
fi

# Activate virtual environment
source .venv/bin/activate

# 1. Linting
if [ "$RUN_LINT" = true ]; then
    echo "[1/3] Running FawltyDeps (dependency linting)..."
    echo "----------------------------------------------"

    # Install fawltydeps if not present
    if ! command -v fawltydeps &> /dev/null; then
        pip install fawltydeps --quiet
    fi

    cd api
    if fawltydeps --detailed; then
        echo -e "${GREEN}✓ FawltyDeps passed${NC}"
    else
        echo -e "${RED}✗ FawltyDeps failed${NC}"
        ERRORS=$((ERRORS + 1))
    fi
    cd ..
    echo ""
fi

# 2. Unit tests (run in Docker for environment parity with CI/production)
if [ "$RUN_TESTS" = true ]; then
    echo "[2/3] Running unit tests in Docker..."
    echo "----------------------------------------------"
    echo "(Using Docker ensures OCR results match CI and production)"
    echo ""

    # Use test-in-docker.sh which has tessdata_best like production
    if [ "$QUICK_MODE" = true ]; then
        if ./scripts/test-in-docker.sh --quick; then
            echo -e "${GREEN}✓ Unit tests passed${NC}"
        else
            echo -e "${RED}✗ Unit tests failed${NC}"
            ERRORS=$((ERRORS + 1))
        fi
    else
        if ./scripts/test-in-docker.sh; then
            echo -e "${GREEN}✓ All tests passed${NC}"
        else
            echo -e "${RED}✗ Tests failed${NC}"
            ERRORS=$((ERRORS + 1))
        fi
    fi
    echo ""
fi

# 3. Import validation (ensures modules load without errors)
echo "[3/3] Validating module imports..."
echo "----------------------------------------------"

IMPORT_ERRORS=0

# Test key imports
for module in "lib.detection" "lib.analytics" "lib.muni_lib" "api.api"; do
    if python -c "import $module" 2>/dev/null; then
        echo "  ✓ $module"
    else
        echo -e "  ${RED}✗ $module${NC}"
        IMPORT_ERRORS=$((IMPORT_ERRORS + 1))
    fi
done

if [ $IMPORT_ERRORS -eq 0 ]; then
    echo -e "${GREEN}✓ All imports valid${NC}"
else
    echo -e "${RED}✗ $IMPORT_ERRORS import(s) failed${NC}"
    ERRORS=$((ERRORS + 1))
fi
echo ""

# Summary
echo "=========================================="
if [ $ERRORS -eq 0 ]; then
    echo -e "${GREEN}✓ All validations passed${NC}"
    echo "=========================================="
    echo ""
    echo "Ready to commit/deploy!"
else
    echo -e "${RED}✗ $ERRORS validation(s) failed${NC}"
    echo "=========================================="
    echo ""
    echo "Please fix the issues before committing."
    exit 1
fi
