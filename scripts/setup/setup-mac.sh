#!/bin/bash
set -e

echo "=========================================="
echo "Muni Metro - macOS Setup Script"
echo "=========================================="
echo ""

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if Homebrew is installed
if ! command -v brew &> /dev/null; then
    echo -e "${YELLOW}Homebrew not found. Installing Homebrew...${NC}"
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
else
    echo -e "${GREEN}Homebrew is already installed.${NC}"
fi

# Update Homebrew
echo ""
echo "Updating Homebrew..."
brew update

# Install Python 3.13+
echo ""
echo "Checking Python installation..."
if ! command -v python3.13 &> /dev/null; then
    echo -e "${YELLOW}Installing Python 3.13...${NC}"
    brew install python@3.13
else
    echo -e "${GREEN}Python 3.13 is already installed.${NC}"
fi

# Install Git
echo ""
echo "Checking Git installation..."
if ! command -v git &> /dev/null; then
    echo -e "${YELLOW}Installing Git...${NC}"
    brew install git
else
    echo -e "${GREEN}Git is already installed.${NC}"
fi

# Install git-annex
echo ""
echo "Checking git-annex installation..."
if ! command -v git-annex &> /dev/null; then
    echo -e "${YELLOW}Installing git-annex...${NC}"
    brew install git-annex
else
    echo -e "${GREEN}git-annex is already installed.${NC}"
fi

# Install tkinter for labeling GUI
echo ""
echo "Checking tkinter installation..."
if ! python3.13 -c "import tkinter" &> /dev/null; then
    echo -e "${YELLOW}Installing tkinter...${NC}"
    brew install python-tk@3.13
else
    echo -e "${GREEN}tkinter is already installed.${NC}"
fi

# Optional: Install Google Cloud SDK
echo ""
read -p "Do you want to install Google Cloud SDK? (required for cloud deployment) [y/N]: " install_gcloud
if [[ $install_gcloud =~ ^[Yy]$ ]]; then
    if ! command -v gcloud &> /dev/null; then
        echo -e "${YELLOW}Installing Google Cloud SDK...${NC}"
        brew install google-cloud-sdk
    else
        echo -e "${GREEN}Google Cloud SDK is already installed.${NC}"
    fi
fi

# Optional: Install Docker
echo ""
read -p "Do you want to install Docker? (required for containerized deployment) [y/N]: " install_docker
if [[ $install_docker =~ ^[Yy]$ ]]; then
    if ! command -v docker &> /dev/null; then
        echo -e "${YELLOW}Installing Docker...${NC}"
        brew install --cask docker
        echo -e "${YELLOW}Please start Docker Desktop manually from Applications.${NC}"
    else
        echo -e "${GREEN}Docker is already installed.${NC}"
    fi
fi

echo ""
echo -e "${GREEN}=========================================="
echo "Base dependencies installed successfully!"
echo "==========================================${NC}"
echo ""
echo "Next steps:"
echo "1. Run: source scripts/setup/setup-python-env.sh"
echo "2. Follow the setup guide in SETUP.md for project configuration"
echo ""
