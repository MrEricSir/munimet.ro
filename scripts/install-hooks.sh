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
# Pre-commit hook: runs quick validation before commits
#
# To skip this hook (use sparingly): git commit --no-verify

# Only run if validate.sh exists
if [ ! -f "scripts/validate.sh" ]; then
    exit 0
fi

echo "Running pre-commit validation..."
echo ""

# Run quick validation (skip slow train detection tests)
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
EOF

chmod +x "$HOOKS_DIR/pre-commit"
echo "✓ Installed pre-commit hook"

echo ""
echo "Git hooks installed successfully!"
echo ""
echo "The pre-commit hook will run './scripts/validate.sh --quick' before each commit."
echo "To skip the hook (use sparingly): git commit --no-verify"
