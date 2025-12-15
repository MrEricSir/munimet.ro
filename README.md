# Muni Metro Subway Status

Real-time monitoring of San Francisco's Muni Metro subway using computer vision to classify status from internal SFMTA status images.

URL: https://munimet.ro

This project was "vibe coded" using Anthropic's Claude Code. Runs without requiring any LLM or external AI services.

## Quick Start

### Prerequisites

**First time setup?** See the comprehensive [Setup Guide (SETUP.md)](SETUP.md) for detailed installation instructions for macOS, Linux, and Windows.

Quick checklist:
- Python 3.13+
- Git with git-annex
- Google Cloud SDK (for cloud deployment only)

### Installation

```bash
# Clone repository
git clone https://github.com/MrEricSir/munimet.ro.git
cd munimet.ro

# Initialize git-annex with a descriptive name for this computer
# Replace "your-computer-name" with any name (e.g., "laptop", "work-desktop")
git annex init "your-computer-name"

# Download pre-trained model (856MB)
git annex get artifacts/models/v1/
```

**Need help with installation?** See [SETUP.md](SETUP.md) for automated setup scripts and platform-specific instructions.

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

## Accessing Training Data

The training dataset (2,666 labeled images, ~270MB) and model files (856MB) are managed via git-annex and stored in Google Cloud Storage.

### For Collaborators with GCS Access

1. Follow [SETUP.md](SETUP.md) to install base dependencies (git-annex, rclone, etc.)

2. Initialize git-annex with a descriptive name for this computer:
   ```bash
   # Replace "your-computer-name" with any name (e.g., "laptop", "alice-desktop")
   git annex init "your-computer-name"
   ```

3. Configure rclone for Google Cloud Storage:
   ```bash
   rclone config create munimetro-gcs gcs project_number=munimetro bucket_policy_only=true
   # Follow OAuth flow in browser
   ```

4. Enable the remote:
   ```bash
   git annex enableremote google-cloud
   ```

5. Download files:
   ```bash
   # Download all training data
   git annex get artifacts/training_data/images/

   # Download model files
   git annex get artifacts/models/v1/
   ```

See [GCS_SETUP.md](GCS_SETUP.md) for detailed Google Cloud Storage configuration.

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
└── artifacts/             # Generated data
    ├── training_data/     # ML training dataset (git-annex)
    │   ├── images/        # 2,666 labeled snapshots (~270MB)
    │   └── labels.json    # Training labels (570KB)
    ├── models/            # Trained models (git-annex)
    │   └── v1/            # BLIP model + classifier (856MB)
    └── runtime/           # Transient runtime data (gitignored)
        ├── cache/         # API response cache
        └── downloads/     # Recent snapshots
```

## Documentation

- **[Setup Guide](SETUP.md)** - Comprehensive installation guide for macOS, Linux, and Windows
- **[Deployment Guide](deploy/README.md)** - Local and cloud deployment instructions
- **[Training Guide](training/README.md)** - Data collection, labeling, and model training
- **[API Documentation](api/README.md)** - API endpoints and configuration
- **[Data Management](artifacts/README.md)** - Git-annex workflows and storage
- **[Testing](tests/README.md)** - Automated test suite
- **[GCS Setup](GCS_SETUP.md)** - Google Cloud Storage configuration for collaborators

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
- **Storage**: Google Cloud Storage (model files, cache)
- **Data Management**: git-annex with rclone GCS backend

## Requirements

### Training Environment
- Python 3.13+
- PyTorch 2.0+
- Transformers (HuggingFace)
- Pillow
- tkinter (for labeling GUI)

### API Environment
- Python 3.13+
- Falcon
- Gunicorn
- Google Cloud Storage client (for cloud deployment)

### Development Tools
- Docker & Docker Compose
- git-annex (for training data access)
- rclone (for GCS backend)
- Google Cloud SDK (for cloud deployment)

## License

MIT
