# Muni Metro Subway Status

[![lint](https://github.com/MrEricSir/munimet.ro/actions/workflows/lint.yml/badge.svg)](https://github.com/MrEricSir/munimet.ro/actions/workflows/lint.yml)

Real-time monitoring of San Francisco's Muni Metro subway using computer vision to classify status from internal SFMTA status images.

URL: https://munimet.ro

This project was "vibe coded" using Anthropic's Claude Code. Runs without requiring any LLM or external AI services.

## Quick Start

### Prerequisites

**First time setup?** See the comprehensive [Setup Guide (SETUP.md)](SETUP.md) for detailed installation instructions for macOS, Linux, and Windows.

Quick checklist:
- Python 3.13+
- Git
- Google Cloud SDK with gsutil (for accessing training data and cloud deployment)

### Installation

```bash
# Clone repository
git clone https://github.com/MrEricSir/munimet.ro.git
cd munimet.ro

# Download models and training data from GCS
./scripts/sync-artifacts.sh download    # macOS/Linux
# or
.\scripts\sync-artifacts.ps1 download   # Windows
```

The sync script will download:
- Pre-trained models (~856MB)
- Training dataset (~270MB)

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
python3 scripts/setup-credentials.py

# Cloud deployment - saves to Google Cloud Secret Manager
python3 scripts/setup-credentials.py --cloud
```

The setup script will prompt for:
- **Bluesky handle** - Your account (e.g., `munimetro.bsky.social`)
- **Bluesky app password** - Generate at [bsky.app/settings/app-passwords](https://bsky.app/settings/app-passwords)

Credentials are optional - the app works without them, but won't post status updates to Bluesky.

## Accessing Training Data

The training dataset (2,666 labeled images, ~270MB) and model files (856MB) are stored in Google Cloud Storage and synced via rsync scripts.

### For Collaborators with GCS Access

1. Follow [SETUP.md](SETUP.md) to install base dependencies (gcloud CLI, etc.)

2. Authenticate with Google Cloud:
   ```bash
   gcloud auth login
   gcloud config set project munimetro
   ```

3. Download training data and models using sync scripts:
   ```bash
   # Download both training data and models
   ./scripts/sync-artifacts.sh download    # macOS/Linux
   .\scripts\sync-artifacts.ps1 download   # Windows

   # Or download individually:
   ./scripts/sync-training-data.sh download  # Training data only (~270MB)
   ./scripts/sync-models.sh download         # Models only (~856MB)
   ```

The sync scripts use `gsutil rsync` to efficiently download only changed files.

### For Contributors Without GCS Access

Contributors can train their own models using collected data:

1. Collect status images:
   ```bash
   cd training
   source venv/bin/activate
   python download_muni_image.py  # Run periodically to build dataset
   ```

2. Label images:
   ```bash
   python label_images.py  # GUI for labeling status (red/yellow/green)
   ```

3. Train model:
   ```bash
   python train_model.py  # Fine-tune BLIP model on labeled data
   ```

See [training/README.md](training/README.md) for detailed training instructions.

## Project Structure

```
munimet.ro/
├── lib/                    # Shared library code
│   └── muni_lib.py        # Core download & prediction functions
│
├── training/              # Data collection & ML training
│   ├── download_muni_image.py  # Status image collector
│   ├── label_images.py         # Image labeling GUI
│   ├── train_model.py          # Model training script
│   └── requirements.txt        # ML dependencies
│
├── api/                   # Production web API
│   ├── api.py             # Falcon web server
│   ├── check_status.py    # Status checker
│   ├── check_status_job.py # Cloud Run Job entry point
│   ├── predict_status.py  # Prediction script
│   ├── index.html         # Landing page
│   ├── dashboard.html     # Status dashboard
│   └── requirements.txt   # API dependencies
│
├── deploy/                # Deployment configuration
│   ├── local/             # Local development scripts
│   └── cloud/             # Google Cloud Run deployment
│
├── tests/                 # Test suite
│   └── test_frontend.py   # Frontend integration tests
│
└── artifacts/             # Generated data (synced via GCS)
    ├── training_data/     # ML training dataset (~270MB)
    │   ├── images/        # 2,666 labeled snapshots
    │   └── labels.json    # Training labels (570KB)
    ├── models/            # Trained models (~856MB)
    │   └── v1/            # BLIP model + classifier
    └── runtime/           # Transient runtime data (gitignored)
        ├── cache/         # API response cache
        └── downloads/     # Recent snapshots
```

## Documentation

- **[Setup Guide](SETUP.md)** - Comprehensive installation guide for macOS, Linux, and Windows
- **[Deployment Guide](deploy/README.md)** - Local and cloud deployment instructions
- **[Training Guide](training/README.md)** - Data collection, labeling, and model training
- **[API Documentation](api/README.md)** - API endpoints and configuration
- **[Testing](tests/README.md)** - Automated test suite

## Architecture

### Training Pipeline

1. **Data Collection** - `download_muni_image.py` periodically captures status images
2. **Labeling** - `label_images.py` provides GUI for manual classification
3. **Training** - `train_model.py` fine-tunes BLIP vision-language model
4. **Evaluation** - Model achieves >95% accuracy on held-out test set

### Production Deployment

#### Local Development
```
Cache Writer (background process)
  ↓ downloads image every 60s
  ↓ runs ML prediction
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
  ↓ downloads image + predicts status
  ↓ writes JSON + exits
Cloud Storage (gs://munimetro-cache/)
  ↑ reads JSON (~100-200ms)
Cloud Run Service (munimetro-api)
  ↓ serves dashboard & endpoints
Users
```

## Features

- **ML-Powered Classification** - BLIP vision-language model for status classification and description generation
- **Production API** - Falcon web framework with health checks, caching, and graceful degradation
- **Lightweight Frontend** - Vanilla JavaScript with zero runtime dependencies
- **Containerized Deployment** - Multi-stage Docker builds with security best practices
- **Smart Caching** - Best-of-two logic reduces false positives (~30ms local response time)
- **Cloud Native** - Serverless deployment on Google Cloud Run with automatic scaling

## Development Workflow

1. **Collect Data** - Run `download_muni_image.py` periodically
2. **Label Data** - Use `label_images.py` to classify images
3. **Train Model** - Execute `train_model.py` to fine-tune BLIP
4. **Test Locally** - Deploy with `./deploy/local/setup.sh && ./deploy/local/start.sh`
5. **Deploy Cloud** - Deploy to Cloud Run with `./deploy/cloud/deploy-services.sh`

## Technology Stack

- **ML Framework**: PyTorch, Transformers (HuggingFace)
- **Vision Model**: BLIP (Salesforce) fine-tuned for status classification
- **Web Framework**: Falcon (async-ready, production WSGI)
- **Frontend**: Vanilla JavaScript (no build step)
- **Deployment**: Docker, Google Cloud Run, Cloud Scheduler
- **Storage**: Google Cloud Storage (model files, training data, cache)

## Requirements

### Training Environment
- Python 3.12+ (3.13+ recommended)
- PyTorch 2.0+ (with CUDA support for GPU training)
- Transformers (HuggingFace)
- Pillow
- tkinter (for labeling GUI)
- Optional: NVIDIA GPU with CUDA for faster training (~10x speedup)

### API Environment
- Python 3.13+
- Falcon
- Gunicorn
- Google Cloud Storage client (for cloud deployment)

### Development Tools
- Docker & Docker Compose
- Google Cloud SDK with gsutil (for accessing training data and cloud deployment)

## License

MIT
