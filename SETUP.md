# Environment Setup Guide

Configuration instructions for Python virtual environments and dependency management.

## Project Structure

The project uses isolated Python environments for each component:

- **training/** - Data collection, labeling, model training
- **api/** - Production web API and deployment
- **tests/** - Test suite

## Virtual Environment Setup

### Training Environment

```bash
cd training

# Create virtual environment
python3 -m venv venv

# Activate environment (macOS/Linux)
source venv/bin/activate

# Activate environment (Windows)
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install tkinter for labeling GUI (macOS)
brew install python-tk@3.13
```

### API Environment

```bash
cd api

# Create virtual environment
python3 -m venv venv

# Activate environment
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### Test Environment

```bash
cd tests

# Create virtual environment
python3 -m venv venv

# Activate environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Docker Deployment

For production deployments, Docker manages all dependencies automatically:

```bash
# Local deployment
cd deploy/local
./setup.sh
./start.sh

# Cloud deployment
cd deploy/cloud
./deploy-services.sh
```

See [deploy/README.md](deploy/README.md) for deployment documentation.

## Dependency Reference

### Training Dependencies (training/requirements.txt)

- **requests, pillow** - Image downloading and processing
- **torch, transformers** - ML framework and models
- **numpy, tqdm** - Data processing and progress tracking
- **tkinter** - Installed separately via system package manager

### API Dependencies (api/requirements.txt)

- **falcon, gunicorn** - Web framework and WSGI server
- **requests, pillow** - Image downloading and processing
- **torch, transformers** - ML inference
- **numpy** - Data processing
- **google-cloud-storage** - Cloud storage client (for Cloud Run deployment)

### Test Dependencies (tests/requirements.txt)

- **selenium** - Browser automation for frontend testing
- **webdriver-manager** - ChromeDriver management

## Troubleshooting

### Error: "No such file or directory"

Cause: Command executed from incorrect directory

Solution:
```bash
# Navigate to correct directory
cd PROJECT_ROOT/training

# Verify location
pwd

# Create environment
python3 -m venv venv
```

### Error: "command not found: source"

Cause: Shell does not support `source` command

Solution:
```bash
# Use dot notation instead
. venv/bin/activate
```

### Error: Python version mismatch

Cause: Multiple Python installations on system

Solution:
```bash
# Specify Python version explicitly
python3.13 -m venv venv

# Or use full path
/usr/bin/python3 -m venv venv

# Verify Python version
python3 --version
```

### Error: "externally-managed-environment"

Cause: Distribution-managed Python installation (Debian/Ubuntu)

Solution:
```bash
# Option 1: Create venv with system packages
python3 -m venv --system-site-packages venv

# Option 2: Use pipx for global tools
pipx install PACKAGE

# Option 3: Override (not recommended)
pip install --break-system-packages PACKAGE
```

### Error: Command concatenation with &&

Cause: Some shells handle `&&` differently in interactive mode

Solution:
```bash
# Execute commands separately instead of:
# python3 -m venv venv && source venv/bin/activate

# Run sequentially:
python3 -m venv venv
source venv/bin/activate
```

### Error: pip install failures

Cause: Missing system dependencies or outdated pip

Solution:
```bash
# Upgrade pip
python3 -m pip install --upgrade pip

# Install build tools (if needed)
# macOS
xcode-select --install

# Debian/Ubuntu
sudo apt-get install python3-dev build-essential

# RHEL/CentOS
sudo yum install python3-devel gcc
```

## Dependency Management Tasks

| Task | Directory | Requirements File | Notes |
|------|-----------|-------------------|-------|
| Download images | `training/` | `training/requirements.txt` | - |
| Label images | `training/` | `training/requirements.txt` | Requires tkinter |
| Train model | `training/` | `training/requirements.txt` | Requires CUDA for GPU |
| Run API (local) | `api/` | `api/requirements.txt` | - |
| Run API (Docker) | `api/` | N/A | Dockerfile handles deps |
| Run tests | `tests/` | `tests/requirements.txt` | Requires Chrome browser |

## Platform-Specific Notes

### macOS

- tkinter: Install via `brew install python-tk@3.13`
- ChromeDriver: Managed automatically by webdriver-manager

### Linux

- tkinter: Install via package manager (`python3-tk`)
- Chrome: Required for Selenium tests
- Build tools: May be needed for PyTorch compilation

### Windows

- Activation: Use `venv\Scripts\activate` instead of `source venv/bin/activate`
- Visual Studio: May be required for building certain packages
- Chrome: Required for Selenium tests

## Next Steps

Component-specific documentation:

- **Training**: [training/README.md](training/README.md)
- **API**: [api/README.md](api/README.md)
- **Deployment**: [deploy/README.md](deploy/README.md)
- **Testing**: [tests/README.md](tests/README.md)
