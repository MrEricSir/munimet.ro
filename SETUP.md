# Comprehensive Setup Guide

Complete installation and configuration guide for the Muni Metro project. This guide covers first-time setup for all supported platforms.

## Table of Contents

- [First Time Setup](#first-time-setup)
- [Platform-Specific Installation](#platform-specific-installation)
  - [macOS](#macos)
  - [Linux](#linux)
  - [Windows](#windows)
- [Python Environment Setup](#python-environment-setup)
- [Accessing Reference Data](#accessing-training-data)
- [Troubleshooting](#troubleshooting)

## First Time Setup

This section guides you through a complete installation from scratch. If you're new to the project, follow these steps in order.

### Prerequisites Overview

Before starting, you'll need to install these base dependencies:

| Tool | Purpose | Required For |
|------|---------|--------------|
| **Python 3.13+** | Runtime for all Python scripts | All workflows |
| **Git** | Version control | All workflows |
| **tkinter** | GUI framework for image labeling | Training only |
| **Google Cloud SDK** | Cloud deployment and GCS access | Cloud deployment, accessing reference data |
| **Docker** | Containerization | Local/cloud deployment |

### Quick Start (Automated Installation)

The fastest way to get started is using our automated setup scripts:

#### macOS
```bash
# Clone the repository
git clone https://github.com/MrEricSir/munimet.ro.git
cd munimet.ro

# Run automated setup
bash scripts/setup/setup-mac.sh

# Setup Python environments
source scripts/setup/setup-python-env.sh
```

#### Linux
```bash
# Clone the repository
git clone https://github.com/MrEricSir/munimet.ro.git
cd munimet.ro

# Run automated setup
bash scripts/setup/setup-linux.sh

# Setup Python environments
source scripts/setup/setup-python-env.sh
```

#### Windows
```powershell
# Clone the repository
git clone https://github.com/MrEricSir/munimet.ro.git
cd munimet.ro

# Allow running scripts
Set-ExecutionPolicy RemoteSigned –Scope Process

# Run automated setup
.\scripts\setup\setup-windows.ps1

# Setup Python environments
.\scripts\setup\setup-python-env.ps1
```

After running the automated scripts, skip to [Accessing Reference Data](#accessing-training-data).

### Manual Installation

If you prefer manual installation or the automated scripts don't work for your environment, follow the platform-specific instructions below.

## Platform-Specific Installation

### macOS

#### Install Homebrew (Package Manager)

Homebrew is the recommended package manager for macOS:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

#### Install Base Dependencies

```bash
# Update Homebrew
brew update

# Install Python 3.13+
brew install python@3.13

# Install Git
brew install git

# Install tkinter (for labeling GUI)
brew install python-tk@3.13
```

#### Optional: Install Development Tools

```bash
# Google Cloud SDK (for cloud deployment)
brew install gcs-sdk

# Docker (for containerized deployment)
brew install --cask docker
```

After installation, start Docker Desktop from Applications if you installed it.

### Linux

#### Debian/Ubuntu

```bash
# Update package lists
sudo apt-get update

# Install Python 3.13+ (or latest available)
sudo apt-get install -y python3 python3-pip python3-venv python3-dev

# Install Git
sudo apt-get install -y git

# Install build tools (needed for some Python packages)
sudo apt-get install -y build-essential

# Install tkinter (for labeling GUI)
sudo apt-get install -y python3-tk
```

#### RHEL/CentOS/Fedora

```bash
# Update package lists
sudo yum update -y
# OR for newer systems:
# sudo dnf update -y

# Install Python 3
sudo yum install -y python3 python3-pip python3-devel
# OR: sudo dnf install -y python3 python3-pip python3-devel

# Install Git
sudo yum install -y git

# Install build tools
sudo yum install -y gcc gcc-c++ make

# Install tkinter
sudo yum install -y python3-tkinter
```

#### Optional: Install Development Tools

```bash
# Google Cloud SDK (Debian/Ubuntu)
echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | sudo tee -a /etc/apt/sources.list.d/gcs-sdk.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key --keyring /usr/share/keyrings/cloud.google.gpg add -
sudo apt-get update
sudo apt-get install -y gcs-sdk

# Docker (Ubuntu)
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER
rm get-docker.sh

# Note: Log out and back in for Docker group membership to take effect
```

### Windows

#### Recommended: Windows Subsystem for Linux (WSL2)

For the best development experience on Windows, we recommend using WSL2:

```powershell
# Install WSL2 (PowerShell as Administrator)
wsl --install

# Install Ubuntu distribution
wsl --install -d Ubuntu

# After installation, restart your computer
# Then open Ubuntu from the Start menu and follow the Linux setup instructions
```

Once WSL2 is installed, open Ubuntu and follow the [Linux (Debian/Ubuntu)](#debianubuntu) instructions above.

#### Native Windows Installation

If you prefer native Windows without WSL2, follow these instructions:

##### Install Package Manager

Windows doesn't come with a package manager by default. We recommend using **winget** (Windows Package Manager) or **Scoop**:

**Option 1: winget (Recommended for Windows 11)**

winget comes pre-installed on Windows 11. For Windows 10, install from the [Microsoft Store](https://apps.microsoft.com/store/detail/app-installer/9NBLGGH4NNS1).

**Option 2: Scoop**

```powershell
# Install Scoop (PowerShell)
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser -Force
Invoke-RestMethod get.scoop.sh | Invoke-Expression
```

##### Install Base Dependencies

**Using winget:**

```powershell
# Install Git
winget install --id Git.Git -e

# Install Python 3.13
winget install --id Python.Python.3.13 -e
```

**Using Scoop:**

```powershell
# Install Git
scoop install git

# Install Python
scoop install python
```

**Note about tkinter:** tkinter is included with the official Python installer on Windows. If you encounter issues, reinstall Python and ensure the "tcl/tk and IDLE" option is checked during installation.

##### Optional: Install Development Tools

```powershell
# Google Cloud SDK
winget install --id Google.CloudSDK -e

# Docker Desktop
winget install --id Docker.DockerDesktop -e
```

After installing Docker Desktop, start it manually from the Start menu.

## Python Environment Setup

After installing base dependencies, set up isolated Python virtual environments for each project component.

### Understanding Project Components

The project has three separate Python environments:

- **scripts/** - Detection scripts and utilities
- **api/** - Production web API and prediction service
- **tests/** - Automated test suite

Each has its own `requirements.txt` and virtual environment.

### Automated Setup

Use our setup script to create all environments:

**macOS/Linux:**
```bash
source scripts/setup/setup-python-env.sh
```

**Windows:**
```powershell
.\scripts\setup\setup-python-env.ps1
```

The script will prompt you to select which components to set up.

### Manual Setup

If you prefer to set up environments manually:

#### API Environment

```bash
cd api

# Create virtual environment
python3 -m venv venv

# Activate environment
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate   # Windows

# Upgrade pip
python -m pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

# Deactivate when done
deactivate
```

#### Test Environment

```bash
cd tests

# Create virtual environment
python3 -m venv venv

# Activate environment
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate   # Windows

# Upgrade pip
python -m pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

# Deactivate when done
deactivate
```

## Google Cloud Authentication

To access reference data and models stored in Google Cloud Storage, you need to authenticate with gcloud.

### Setup Google Cloud CLI

If you haven't installed the Google Cloud SDK yet, refer to the platform-specific installation sections above.

### Authenticate

**macOS/Linux:**
```bash
# Authenticate with your Google account
gcloud auth login

# Set the project (if you're a collaborator)
gcloud config set project munimetro
```

**Windows:**
```powershell
# Authenticate with your Google account
gcloud auth login

# Set the project (if you're a collaborator)
gcloud config set project munimetro
```

After authentication, you can use the sync scripts to download reference data and models.

## Accessing Reference Data

### Option 1: Use Existing Data (Collaborators with GCS Access)

If you have access to the shared Google Cloud Storage bucket:

1. Authenticate with Google Cloud (see [Google Cloud Authentication](#google-cloud-authentication) above)

2. Download reference data and models using sync scripts:
   ```bash
   # Download both reference data and models
   ./scripts/sync-artifacts.sh download    # macOS/Linux
   .\scripts\sync-artifacts.ps1 download   # Windows

   # Or download reference data only:
   ./scripts/sync-reference-data.sh download  # Reference data only (~270MB)
   ```

The sync scripts use `gsutil rsync` to efficiently download only changed files.

### Option 2: Collect Your Own Data (No GCS Access Required)

Contributors can collect their own reference images:

```bash
# Run data collector (downloads images periodically)
python scripts/download_muni_image.py

# Label collected images (requires tkinter)
python scripts/label_images.py
```

## Verification

After setup, verify your installation:

```bash
# Check Python version
python3 --version  # Should be 3.13+

# Check Git
git --version

# Check Google Cloud SDK
gcloud --version

# Check Python environment (from api/)
cd api
source venv/bin/activate  # Windows: venv\Scripts\activate
python -c "import falcon; import cv2; print('Dependencies OK')"
deactivate
```

## Troubleshooting

### Command Not Found Errors

**Symptom:** `command not found: gcloud` or similar errors

**Solution:**
- Verify installation completed successfully
- Check PATH environment variable
- Restart terminal/shell to refresh environment
- For pip-installed packages: ensure `~/.local/bin` (Linux/Mac) or Python Scripts directory (Windows) is in PATH

### Python Version Issues

**Symptom:** `python3: command not found` or wrong Python version

**Solution:**
```bash
# Check available Python versions
python --version
python3 --version
python3.13 --version

# Use specific version for venv
python3.13 -m venv venv
```

### Virtual Environment Activation Issues

**Symptom:** Environment not activating or `activate` command not found

**Solution:**
```bash
# macOS/Linux: Use dot notation if 'source' doesn't work
. venv/bin/activate

# Windows: Ensure PowerShell execution policy allows scripts
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Windows: Try alternative activation method
venv\Scripts\activate.bat  # CMD
venv\Scripts\Activate.ps1  # PowerShell
```

### pip Install Failures

**Symptom:** Compilation errors during `pip install -r requirements.txt`

**Solution:**
```bash
# Upgrade pip first
python -m pip install --upgrade pip

# macOS: Install Xcode Command Line Tools
xcode-select --install

# Linux (Debian/Ubuntu): Install build tools
sudo apt-get install python3-dev build-essential

# Linux (RHEL/CentOS): Install build tools
sudo yum install python3-devel gcc gcc-c++

# Windows: Install Visual Studio Build Tools
# Download from: https://visualstudio.microsoft.com/downloads/
```

### GCS Authentication Issues

**Symptom:** `gsutil` or sync scripts fail with authentication errors

**Solution:**
```bash
# Authenticate with Google Cloud
gcloud auth login

# Verify authentication
gcloud auth list

# Test access to bucket
gsutil ls gs://munimetro-annex/
```

### Missing Model or Reference Data Files

**Symptom:** Files in `artifacts/` are missing or empty

**Solution:**
```bash
# Download reference data (optional - for development/testing)
./scripts/sync-reference-data.sh download   # Reference data (~270MB)

# Verify files downloaded
ls -lh artifacts/reference_data/images/
```

### tkinter Import Errors

**Symptom:** `ModuleNotFoundError: No module named 'tkinter'`

**Solution:**
```bash
# macOS
brew install python-tk@3.13

# Debian/Ubuntu
sudo apt-get install python3-tk

# RHEL/CentOS
sudo yum install python3-tkinter

# Windows: Reinstall Python with tkinter option enabled
# Or download from: https://www.python.org/downloads/
```

### WSL2 Networking Issues (Windows)

**Symptom:** Unable to access internet from WSL2

**Solution:**
```powershell
# Restart WSL (PowerShell as Administrator)
wsl --shutdown
wsl

# Or reset network stack in WSL
# From within WSL:
sudo systemctl restart systemd-resolved
```

### Permission Denied Errors

**Symptom:** Permission errors when running scripts

**Solution:**
```bash
# Make scripts executable (macOS/Linux)
chmod +x scripts/setup/*.sh

# Windows: Run PowerShell as Administrator for system-level installations
# Right-click PowerShell -> "Run as Administrator"
```

## Next Steps

After completing setup:

1. **Download Data**: Use sync scripts to download reference data (see [Accessing Reference Data](#accessing-reference-data))
2. **Run Locally**: See [deploy/README.md](deploy/README.md) for deployment instructions

## Related Documentation

- **[Main README](README.md)** - Project overview and quick start
- **[Deployment Guide](deploy/README.md)** - Local and cloud deployment
- **[API Documentation](api/README.md)** - API endpoints and configuration
- **[Testing Guide](tests/README.md)** - Running automated tests

## Getting Help

If you encounter issues not covered in this guide:

1. Check the [Troubleshooting](#troubleshooting) section above
2. Review component-specific documentation linked above
3. Open an issue on GitHub with details about your environment and error messages
