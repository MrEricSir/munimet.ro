#!/bin/bash
# Install git hooks for MuniMetro development
#
# Usage: ./scripts/install-hooks.sh

set -e

cd "$(dirname "$0")/.."

HOOKS_DIR=".git/hooks"

echo "Installing git hooks..."

# Create pre-commit hook
cat > "$HOOKS_DIR/pre-commit" << 'EOF'
#!/bin/bash
# Pre-commit hook: runs validation before commits
#
# Uses Docker-based tests when available (matches CI/production environment).
# Falls back to native tests if Docker is not running.
#
# To skip this hook (use sparingly): git commit --no-verify

echo "Running pre-commit validation..."
echo ""

# Check if Docker is available and running
if command -v docker &> /dev/null && docker info &> /dev/null 2>&1; then
    # Docker is running - use containerized tests for CI/production parity
    echo "Using Docker-based tests (matches CI/GCP environment)"
    echo ""

    if [ -f "scripts/test-in-docker.sh" ]; then
        # Run quick tests (system status only, skip slow train detection)
        if ./scripts/test-in-docker.sh --quick; then
            echo ""
            echo "Pre-commit validation passed."
            exit 0
        else
            echo ""
            echo "Pre-commit validation failed."
            echo "Fix the issues above, or use 'git commit --no-verify' to skip."
            exit 1
        fi
    fi
fi

# Fallback to native tests if Docker not available
if [ -f "scripts/validate.sh" ]; then
    echo "Docker not available - using native tests"
    echo "(For full CI parity, start Docker and re-run)"
    echo ""

    if ./scripts/validate.sh --quick; then
        echo ""
        echo "Pre-commit validation passed."
        exit 0
    else
        echo ""
        echo "Pre-commit validation failed."
        echo "Fix the issues above, or use 'git commit --no-verify' to skip."
        exit 1
    fi
fi

# No validation scripts found
echo "Warning: No validation scripts found, skipping pre-commit checks."
exit 0
EOF

chmod +x "$HOOKS_DIR/pre-commit"
echo "✓ Installed pre-commit hook"

echo ""
echo "Git hooks installed successfully!"
echo ""
echo "The pre-commit hook will:"
echo "  - Use Docker tests when Docker is running (matches CI/GCP)"
echo "  - Fall back to native tests if Docker is not available"
echo ""
echo "To skip the hook (use sparingly): git commit --no-verify"
