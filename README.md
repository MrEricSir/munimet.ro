# Muni Metro Subway Status

[![lint](https://github.com/MrEricSir/munimet.ro/actions/workflows/lint.yml/badge.svg)](https://github.com/MrEricSir/munimet.ro/actions/workflows/lint.yml)

Real-time monitoring of San Francisco's Muni Metro subway using OpenCV-based computer vision to detect system status from internal SFMTA status images.

URL: https://munimet.ro

This project was "vibe coded" using Anthropic's Claude Code. Uses deterministic computer vision analysis; no ML models or external AI services required.

## Quick Start

### Prerequisites

**First time setup?** See the comprehensive [Setup Guide (SETUP.md)](SETUP.md) for detailed installation instructions for macOS, Linux, and Windows.

Quick checklist:
- Python 3.13+
- Git
- Google Cloud SDK with gsutil (for accessing reference data and cloud deployment)

### Installation

```bash
# Clone repository
git clone https://github.com/MrEricSir/munimet.ro.git
cd munimet.ro

# Download reference data from GCS (optional - for development/testing)
./scripts/sync-artifacts.sh download    # macOS/Linux
# or
.\scripts\sync-artifacts.ps1 download   # Windows
```

The sync script will download:
- Reference dataset (~270MB) - labeled images for testing

**Note:** Requires `gcloud` authentication. Run `gcloud auth login` first.

**Need help with installation?** See [SETUP.md](SETUP.md) for platform-specific installation instructions.

### Deployment

```bash
# Local deployment
cd deploy/local
./setup.sh   # Creates venv, installs dependencies
./start.sh   # Starts cache writer + API server

# Access services
open http://localhost:8000              # Landing page
open http://localhost:8000/dashboard    # Status dashboard

# Cloud deployment (Google Cloud Run)
cd deploy/cloud
./setup-infrastructure.sh
./deploy-services.sh
./setup-scheduler.sh
```

See [deploy/README.md](deploy/README.md) for detailed deployment instructions.

### Credentials Setup (Optional)

To enable Bluesky status posting, configure credentials:

```bash
# Local development - saves to .env file
python3 scripts/setup/setup-credentials.py

# Cloud deployment - saves to Google Cloud Secret Manager
python3 scripts/setup/setup-credentials.py --cloud
```

The setup script will prompt for:
- **Bluesky handle** - Your account (e.g., `munimetro.bsky.social`)
- **Bluesky app password** - Generate at [bsky.app/settings/app-passwords](https://bsky.app/settings/app-passwords)

Credentials are optional - the app works without them, but won't post status updates to Bluesky.

## Accessing Training Data

The reference dataset (2,666 labeled images, ~270MB) is stored in Google Cloud Storage and synced via rsync scripts.

### For Collaborators with GCS Access

1. Follow [SETUP.md](SETUP.md) to install base dependencies (gcloud CLI, etc.)

2. Authenticate with Google Cloud:
   ```bash
   gcloud auth login
   gcloud config set project munimetro
   ```

3. Download reference data using sync scripts:
   ```bash
   ./scripts/sync-artifacts.sh download    # macOS/Linux
   .\scripts\sync-artifacts.ps1 download   # Windows

   # Or download reference data only:
   ./scripts/sync-reference-data.sh download  # Reference data only (~270MB)
   ```

The sync scripts use `gsutil rsync` to efficiently download only changed files.

### For Contributors Without GCS Access

Contributors can collect their own test images:

1. Collect status images:
   ```bash
   python scripts/download_muni_image.py  # Run periodically to build dataset
   ```

2. Analyze a single image:
   ```bash
   python scripts/analyze.py path/to/image.jpg           # Pretty output
   python scripts/analyze.py path/to/image.jpg --json    # JSON output
   python scripts/analyze.py path/to/image.jpg --verbose # Full details
   ```

3. Visualize detection interactively:
   ```bash
   python scripts/station_viewer.py  # Interactive detection viewer
   ```

4. Run tests:
   ```bash
   pytest tests/test_system_status.py -v  # Verify detection accuracy
   ```

## Project Structure

```
munimet.ro/
├── lib/                    # Shared library code
│   ├── muni_lib.py        # Core download & detection functions
│   ├── detection.py       # OpenCV-based status detection
│   ├── analytics.py       # SQLite-based delay analytics
│   └── notifiers/         # Notification channels (Bluesky, RSS)
│
├── scripts/               # Detection and utility scripts
│   ├── analyze.py              # CLI tool for image analysis
│   ├── station_detector.py     # Station position detection
│   ├── train_detector.py       # Train ID detection (OCR)
│   ├── detect_stations.py      # Station configuration
│   └── station_viewer.py       # Debug visualization tool
│
├── api/                   # Production web API
│   ├── api.py             # Falcon web server
│   ├── check_status.py    # Status checker
│   ├── check_status_job.py # Cloud Run Job entry point
│   ├── html/              # Frontend files
│   │   ├── index.html     # Landing page
│   │   └── dashboard.html # Status dashboard
│   └── requirements.txt   # API dependencies
│
├── deploy/                # Deployment configuration
│   ├── local/             # Local development scripts
│   └── cloud/             # Google Cloud Run deployment
│
├── tests/                 # Test suite
│   ├── images/            # Test images (~2MB)
│   ├── test_system_status.py   # Status detection tests
│   └── test_train_detection.py # Train detection tests
│
└── artifacts/             # Generated data (synced via GCS)
    ├── reference_data/     # Reference images (~270MB)
    │   ├── images/        # Labeled snapshots
    │   └── labels.json    # Training labels
    └── runtime/           # Transient runtime data (gitignored)
        ├── cache/         # API response cache
        └── downloads/     # Recent snapshots
```

## Documentation

- **[Setup Guide](SETUP.md)** - Comprehensive installation guide for macOS, Linux, and Windows
- **[Deployment Guide](deploy/README.md)** - Local and cloud deployment instructions
- **[API Documentation](api/README.md)** - API endpoints and configuration
- **[Testing](tests/README.md)** - Automated test suite

## Architecture

### Detection Pipeline

The system uses OpenCV-based computer vision to analyze SFMTA status images:

1. **Image Download** - Fetches real-time status image from SFMTA internal system
2. **Station Detection** - Identifies platform colors (blue=normal, yellow=hold)
3. **Track Analysis** - Detects track segment status (cyan=normal, red=disabled)
4. **Train Detection** - Locates trains and reads IDs via OCR (optional)
5. **Status Calculation** - Determines system status (green/yellow/red) based on:
   - Platforms in hold mode
   - Disabled track segments
   - Train bunching (4+ trains clustered together)

### Production Deployment

#### Local Development
```
Cache Writer (background process)
  ↓ downloads image every 60s
  ↓ runs OpenCV detection
  ↓ writes JSON to local disk
Local Cache File
  ↑ reads JSON (~30ms)
API Server (gunicorn)
  ↓ serves dashboard & endpoints
Browser
```

#### Cloud Run (Production)
```
Cloud Scheduler (every 3 min)
  ↓ triggers via OAuth
Cloud Run Job (munimetro-checker)
  ↓ downloads image + detects status
  ↓ writes JSON + exits
Cloud Storage (gs://munimetro-cache/)
  ↑ reads JSON (~100-200ms)
Cloud Run Service (munimetro-api)
  ↓ serves dashboard & endpoints
Users
```

## Features

- **Computer Vision Detection** - OpenCV-based analysis for deterministic status classification
- **Rich Detection Data** - Detects trains, platform holds, disabled tracks, and bunching
- **Production API** - Falcon web framework with health checks, caching, and graceful degradation
- **Lightweight Frontend** - Vanilla JavaScript with zero runtime dependencies
- **Containerized Deployment** - Multi-stage Docker builds with security best practices
- **Smart Caching** - Best-of-three smoothing reduces false positives (~30ms local response time)
- **Cloud Native** - Serverless deployment on Google Cloud Run with automatic scaling
- **No ML Dependencies** - No PyTorch or large model files required
- **Multi-Channel Notifications** - Status updates via Bluesky and RSS feed
- **Delay Analytics** - SQLite-based tracking with visual dashboard for delay patterns

## Development Workflow

1. **Collect Data** - Run `download_muni_image.py` periodically
2. **Test Detection** - Use `scripts/station_viewer.py` to visualize detection
3. **Run Tests** - Execute `pytest tests/` to verify detection accuracy
4. **Test Locally** - Deploy with `./deploy/local/setup.sh && ./deploy/local/start.sh`
5. **Deploy Cloud** - Deploy to Cloud Run with `./deploy/cloud/deploy-services.sh`

## Technology Stack

- **Computer Vision**: OpenCV for image analysis
- **OCR**: Tesseract for train ID recognition (optional)
- **Web Framework**: Falcon (async-ready, production WSGI)
- **Frontend**: Vanilla JavaScript (no build step)
- **Deployment**: Docker, Google Cloud Run, Cloud Scheduler
- **Storage**: Google Cloud Storage (reference data, cache)

## Requirements

### API Environment
- Python 3.13+
- OpenCV (opencv-python-headless)
- Falcon, Gunicorn
- Optional: Tesseract OCR for train ID detection
- Google Cloud Storage client (for cloud deployment)

### Development Tools
- Docker & Docker Compose
- Google Cloud SDK with gsutil (for accessing reference data and cloud deployment)
- pytest for running tests

## License

MIT
