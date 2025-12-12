# Muni Metro Subway Status

Is the Muni Metro subway in San Francisco running? This web app monitors the internal status image and provides a brief summary using computer vision.

See it in action:
https://munimet.ro

This project was largely "vibe coded" using Anthropic's Claude Code. The project itself does not rely on Claude or any other LLM AI.

## Quick Start

### For New Users (First Time Setup)

This project uses **git-annex** to manage large files (1.1GB of training data and models) stored in Google Cloud Storage.

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd munimetro

# 2. Set up git-annex and download data (see artifacts/README.md)
brew install git-annex rclone git-annex-remote-rclone
git annex init "your-laptop"
git annex enableremote google-cloud
git annex get artifacts/models/v1/        # Download model (856MB)

# 3. Train the model (see training/README.md)
cd training
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python download_muni_image.py  # Collect data
python label_images.py          # Label images
python train_model.py           # Train model

# 4. Run the API (see api/README.md)
cd ../api
docker-compose up -d
open http://localhost:8000
```

See **[artifacts/README.md](artifacts/README.md)** for complete data management workflow.

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
│   └── requirements.txt        # ML dependencies
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
└── artifacts/             # Generated data → See artifacts/README.md
    ├── training_data/     # ML training dataset (git-annex tracked)
    │   ├── images/        # 2,666 labeled snapshots (~270MB)
    │   └── labels.json    # Training labels (570KB, unlocked)
    ├── models/            # Trained models (git-annex tracked)
    │   └── v1/            # BLIP model + classifier (856MB)
    └── runtime/           # Transient runtime data (gitignored)
        ├── cache/         # API response cache
        └── downloads/     # Recent snapshots for predictions
```

## Documentation

- **[Data Management](artifacts/README.md)** - Git-annex workflows for training data and models
- **[Training Guide](training/README.md)** - Download images, label data, train models
- **[API & Deployment](api/README.md)** - Run API locally or deploy to Google Cloud Run
- **[Testing](tests/README.md)** - Run automated tests
- **[Setup](SETUP.md)** - Virtual environment setup and troubleshooting
- **[GCS Setup](GCS_SETUP.md)** - Initial Google Cloud Storage configuration

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
