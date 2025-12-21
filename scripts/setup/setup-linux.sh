#!/bin/bash
set -e

echo "=========================================="
echo "Muni Metro - Linux Setup Script"
echo "=========================================="
echo ""

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Detect package manager
if command -v apt-get &> /dev/null; then
    PKG_MANAGER="apt"
    INSTALL_CMD="sudo apt-get install -y"
    UPDATE_CMD="sudo apt-get update"
elif command -v yum &> /dev/null; then
    PKG_MANAGER="yum"
    INSTALL_CMD="sudo yum install -y"
    UPDATE_CMD="sudo yum update -y"
elif command -v dnf &> /dev/null; then
    PKG_MANAGER="dnf"
    INSTALL_CMD="sudo dnf install -y"
    UPDATE_CMD="sudo dnf update -y"
else
    echo -e "${RED}Error: No supported package manager found (apt, yum, or dnf)${NC}"
    exit 1
fi

echo "Detected package manager: $PKG_MANAGER"
echo ""

# Update package lists
echo "Updating package lists..."
$UPDATE_CMD

# Install Python 3.13+ (or latest available)
echo ""
echo "Checking Python installation..."
if ! command -v python3 &> /dev/null; then
    echo -e "${YELLOW}Installing Python 3...${NC}"
    if [ "$PKG_MANAGER" = "apt" ]; then
        $INSTALL_CMD python3 python3-pip python3-venv python3-dev
    else
        $INSTALL_CMD python3 python3-pip python3-devel
    fi
else
    echo -e "${GREEN}Python 3 is already installed.${NC}"
fi

# Display Python version
python3 --version

# Install Git
echo ""
echo "Checking Git installation..."
if ! command -v git &> /dev/null; then
    echo -e "${YELLOW}Installing Git...${NC}"
    $INSTALL_CMD git
else
    echo -e "${GREEN}Git is already installed.${NC}"
fi

# Install tkinter for labeling GUI
echo ""
echo "Checking tkinter installation..."
if ! python3 -c "import tkinter" &> /dev/null; then
    echo -e "${YELLOW}Installing tkinter...${NC}"
    if [ "$PKG_MANAGER" = "apt" ]; then
        $INSTALL_CMD python3-tk
    else
        $INSTALL_CMD python3-tkinter
    fi
else
    echo -e "${GREEN}tkinter is already installed.${NC}"
fi

# Install build tools (needed for some Python packages)
echo ""
echo "Checking build tools..."
if [ "$PKG_MANAGER" = "apt" ]; then
    $INSTALL_CMD build-essential
else
    $INSTALL_CMD gcc gcc-c++ make
fi

# Optional: Install Google Cloud SDK
echo ""
read -p "Do you want to install Google Cloud SDK? (required for cloud deployment) [y/N]: " install_gcloud
if [[ $install_gcloud =~ ^[Yy]$ ]]; then
    if ! command -v gcloud &> /dev/null; then
        echo -e "${YELLOW}Installing Google Cloud SDK...${NC}"
        if [ "$PKG_MANAGER" = "apt" ]; then
            echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | sudo tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
            curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key --keyring /usr/share/keyrings/cloud.google.gpg add -
            sudo apt-get update
            $INSTALL_CMD google-cloud-sdk
        else
            # Install via script for yum/dnf
            curl https://sdk.cloud.google.com | bash
            exec -l $SHELL
        fi
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
        if [ "$PKG_MANAGER" = "apt" ]; then
            # Install Docker on Ubuntu/Debian
            curl -fsSL https://get.docker.com -o get-docker.sh
            sudo sh get-docker.sh
            sudo usermod -aG docker $USER
            rm get-docker.sh
        else
            # Install Docker on RHEL/CentOS
            $INSTALL_CMD docker
            sudo systemctl start docker
            sudo systemctl enable docker
            sudo usermod -aG docker $USER
        fi
        echo -e "${YELLOW}Please log out and back in for Docker group membership to take effect.${NC}"
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
