# Setup Guide

Quick setup instructions for the Muni Status Monitor.

## Virtual Environment Setup

The project is split into two parts with separate dependencies:
- **training/** - Data collection, labeling, and model training
- **api/** - Production web API and deployment

### Training Setup

```bash
cd ~/Development/munimetro/training

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On macOS/Linux (bash/zsh):
source venv/bin/activate

# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install tkinter for labeling GUI (macOS)
brew install python-tk@3.13
```

### API Setup (Local Development)

```bash
cd ~/Development/munimetro/api

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

## Troubleshooting Virtual Environment Errors

### Error: "No such file or directory"

**Cause**: Running command from wrong directory

**Solution**:
```bash
# Make sure you're in the correct directory first
cd ~/Development/munimetro/training
pwd  # Verify you're in the right place

# Then create venv
python3 -m venv venv
```

### Error: Split commands with `&&`

**Cause**: Some shells don't handle `&&` well in interactive mode

**Solution**: Run commands separately:
```bash
# Instead of: python3 -m venv venv && source venv/bin/activate
# Do this:
python3 -m venv venv
source venv/bin/activate
```

### Error: "command not found: source"

**Cause**: Using a shell that doesn't have `source` (some sh variants)

**Solution**:
```bash
# Use dot instead of source
. venv/bin/activate
```

### Error: Python version mismatch

**Cause**: System has multiple Python versions

**Solution**:
```bash
# Use specific Python version
python3.13 -m venv venv
# or
/usr/bin/python3 -m venv venv
```

### Error: "externally-managed-environment"

**Cause**: Python installed via package manager on some Linux distros

**Solution**:
```bash
# Create venv with --system-site-packages if needed
python3 -m venv --system-site-packages venv

# Or use pipx/poetry for dependency management
```

## Docker Setup (Recommended for Production)

No virtual environment needed! Docker handles all dependencies.

```bash
cd ~/Development/munimetro/api
docker-compose up -d
```

See [api/README.md](api/README.md) for full Docker documentation.

## Quick Reference

| Task | Directory | Requirements File |
|------|-----------|-------------------|
| Download images | `training/` | `requirements.txt` |
| Label images | `training/` | `requirements.txt` (+ tkinter) |
| Train model | `training/` | `requirements.txt` |
| Run API locally | `api/` | `requirements.txt` |
| Run API with Docker | `api/` | None (handled by Dockerfile) |
| Run tests | `tests/` | `requirements.txt` |

## What's in Each Requirements File?

**training/requirements.txt:**
- `requests`, `pillow` - Image downloading
- `torch`, `transformers` - ML training
- `numpy`, `tqdm` - Data processing
- Note: tkinter installed separately via system package manager

**api/requirements.txt:**
- `falcon`, `gunicorn` - Web server
- `requests`, `pillow` - Image downloading
- `torch`, `transformers` - ML prediction
- `numpy` - Data processing

**tests/requirements.txt:**
- `selenium` - Browser automation for frontend testing

## Next Steps

Once your environment is set up:
1. Training: See [training/README.md](training/README.md)
2. API: See [api/README.md](api/README.md)
3. Testing: See [tests/README.md](tests/README.md)
