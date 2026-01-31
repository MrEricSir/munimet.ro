#!/bin/bash
set -e

echo "=========================================="
echo "Muni Metro - Python Environment Setup"
echo "=========================================="
echo ""

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get the project root directory (2 levels up from this script)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "Project root: $PROJECT_ROOT"
echo ""

# Function to setup environment for a component
setup_component() {
    local component=$1
    local component_path="$PROJECT_ROOT/$component"

    echo -e "${BLUE}Setting up $component environment...${NC}"

    if [ ! -d "$component_path" ]; then
        echo -e "${RED}Error: $component_path does not exist${NC}"
        return 1
    fi

    cd "$component_path"

    # Check if requirements.txt exists
    if [ ! -f "requirements.txt" ]; then
        echo -e "${YELLOW}No requirements.txt found in $component, skipping...${NC}"
        return 0
    fi

    # Create virtual environment if it doesn't exist
    if [ ! -d "venv" ]; then
        echo "Creating virtual environment..."
        python3 -m venv venv
    else
        echo "Virtual environment already exists."
    fi

    # Activate virtual environment
    source venv/bin/activate

    # Upgrade pip
    echo "Upgrading pip..."
    python -m pip install --upgrade pip

    # Install dependencies
    echo "Installing dependencies from requirements.txt..."
    pip install -r requirements.txt

    # Deactivate virtual environment
    deactivate

    echo -e "${GREEN}$component environment setup complete!${NC}"
    echo ""
}

# Ask which components to set up
echo "Which components would you like to set up?"
echo ""
echo "1) All components (training, api, tests)"
echo "2) Training only"
echo "3) API only"
echo "4) Tests only"
echo "5) Custom selection"
echo ""
read -p "Enter your choice [1-5]: " choice

case $choice in
    1)
        setup_component "training"
        setup_component "api"
        setup_component "tests"
        ;;
    2)
        setup_component "training"
        ;;
    3)
        setup_component "api"
        ;;
    4)
        setup_component "tests"
        ;;
    5)
        read -p "Setup training? [y/N]: " setup_training
        read -p "Setup api? [y/N]: " setup_api
        read -p "Setup tests? [y/N]: " setup_tests

        [[ $setup_training =~ ^[Yy]$ ]] && setup_component "training"
        [[ $setup_api =~ ^[Yy]$ ]] && setup_component "api"
        [[ $setup_tests =~ ^[Yy]$ ]] && setup_component "tests"
        ;;
    *)
        echo -e "${RED}Invalid choice. Exiting.${NC}"
        exit 1
        ;;
esac

echo -e "${GREEN}=========================================="
echo "Python environment setup complete!"
echo "==========================================${NC}"
echo ""
echo "To activate an environment:"
echo "  cd $PROJECT_ROOT/api && source venv/bin/activate"
echo "  cd $PROJECT_ROOT/tests && source venv/bin/activate"
echo ""
echo "Next steps:"
echo "1. Download reference data: ./scripts/sync-reference-data.sh download"
echo "2. Follow SETUP.md for project-specific configuration"
echo ""
