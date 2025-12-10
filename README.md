# SF Muni Metro Status Monitor

ML-powered real-time monitoring system for SF Muni Metro subway status. Downloads status board images, trains vision-language models to classify and describe status, and serves predictions via a production-ready web API.

## Quick Start

### For New Users (First Time Setup)

This project uses **git-annex** to manage large files (1.1GB of training data and models) stored in Google Cloud Storage.

```bash
# 1. Install git-annex (one-time setup)
brew install git-annex rclone git-annex-remote-rclone

# 2. Clone the repository
git clone <your-repo-url>
cd munimetro

# 3. Initialize git-annex and download files from cloud
git annex init "your-laptop"
git annex enableremote google-cloud  # Authenticates with Google Cloud
git annex get artifacts/models/v1/  # Download model (856MB)
# Optional: git annex get data/        # Download all training data (268MB)

# 4. Train the model (see training/README.md)
cd training
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python download_muni_image.py  # Collect new data
python label_images.py          # Label images
python train_model.py           # Train model

# 5. Run the API (see api/README.md)
cd ../api
docker-compose up -d
open http://localhost:8000
```

### Adding New Training Data

When you collect new snapshots or update labels:

```bash
# New images are automatically tracked by git-annex
git annex add artifacts/training_data/images/*.jpg

# Upload to Google Cloud Storage
git annex copy artifacts/training_data/images/ --to=google-cloud

# Commit the changes
git add data/
git commit -m "Add new training snapshots"
git push
```

See [DATA_MANAGEMENT.md](DATA_MANAGEMENT.md) for complete git-annex workflow.

## Project Structure

```
munimetro/
├── lib/                    # Shared library code
│   └── muni_lib.py        # Core functions for download & prediction
│
├── training/              # Data collection & ML training → See training/README.md
│   ├── download_muni_image.py  # Download status images
│   ├── label_images.py         # GUI for labeling images
│   ├── train_model.py          # Train BLIP vision-language model
│   └── requirements_ml.txt     # ML dependencies
│
├── api/                   # Production web API & deployment → See api/README.md
│   ├── api.py             # Falcon web API
│   ├── check_status.py    # Download + predict combined
│   ├── predict_status.py  # Standalone prediction script
│   ├── index.html         # Web dashboard (8.6KB, vanilla JS)
│   ├── Dockerfile         # Production container image
│   └── docker-compose.yml # Local deployment orchestration
│
├── tests/                 # Test suite → See tests/README.md
│   └── test_frontend.py   # Frontend integration tests
│
├── data/                  # Training data (tracked with git-annex)
│   ├── muni_snapshots/    # 2,601 status images (268MB)
│   ├── training_labels.json  # Labeled training data (570KB)
│   └── cache/             # API cache files (gitignored)
│
└── models/                # Trained models (tracked with git-annex)
    └── trained_model/     # BLIP model + classifier weights (856MB)
```

## Documentation

- **[Setup Guide](SETUP.md)** - Virtual environment setup and troubleshooting
- **[Training Guide](training/README.md)** - Download images, label data, train models
- **[API & Deployment Guide](api/README.md)** - Run API locally or deploy to Google Cloud Run
- **[Testing Guide](tests/README.md)** - Run automated tests
- **[Data Management Guide](DATA_MANAGEMENT.md)** - Complete git-annex workflow and commands

## Workflow

1. **Data Collection** - Run `download_muni_image.py` to collect status images over time
2. **Labeling** - Use `label_images.py` GUI to label 50-100+ images with status + descriptions
3. **Training** - Run `train_model.py` to fine-tune BLIP model on your labeled data
4. **Deployment** - Use Docker to deploy the API locally or to Google Cloud Run
5. **Monitoring** - Access real-time status via web dashboard or API endpoints

## Features

- **ML-Powered Classification** - BLIP vision-language model classifies status (🟢/🟡/🔴) and generates descriptions
- **Production-Ready API** - Falcon web framework with health checks, caching, and graceful degradation
- **Lightweight Frontend** - 8.6KB vanilla JavaScript dashboard with zero dependencies
- **Containerized Deployment** - Multi-stage Docker build with security best practices
- **Smart Caching** - Best-of-two status logic smooths transient failures (~30ms response time)

## Requirements

- **Training**: Python 3.13+, PyTorch, Transformers, Pillow, tkinter
- **API**: Docker & Docker Compose (or Python 3.13+ for local development)
- **Cloud Deployment**: Google Cloud SDK (optional)
- **Data Management**: git-annex (for accessing training data/models from cloud storage)

## License

MIT
