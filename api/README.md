# API & Deployment Guide

Production-ready web API for SF Muni status monitoring with Cloud Run deployment support.

## Deployment Options

### 🚀 Cloud Run (Production - Recommended)

**Fully managed, serverless, auto-scaling deployment**

```bash
./deploy/cloud/setup-infrastructure.sh  # One-time setup
./deploy/cloud/deploy-services.sh       # Deploy
./deploy/cloud/setup-scheduler.sh       # Configure auto-updates
./deploy/cloud/verify.sh               # Check deployment status
```

**Cost:** ~$0.12/month for typical usage
**Guide:** See [../deploy/README.md](../deploy/README.md) for complete instructions

### 💻 Local Development (Shell Scripts - Recommended)

**Simple local development without Docker**

```bash
./deploy/local/setup.sh     # One-time setup
./deploy/local/start.sh     # Start services
./deploy/local/verify.sh    # Check status
./deploy/local/stop.sh      # Stop services
```

**Guide:** See [../deploy/README.md](../deploy/README.md) for complete instructions

## Running Locally (Manual Setup)

For automated setup, use the deployment scripts in [../deploy/local/](../deploy/local/).

For manual setup:

```bash
# Setup
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Option 1: Single status check
python check_status.py

# Option 2: Continuous monitoring
python check_status.py --continuous --interval 60

# Option 3: Web API (standalone, ~11s per request)
gunicorn api.api:app --bind 0.0.0.0:8000 --timeout 120

# Option 4: Web API (lightweight, cache-only) - Recommended
# Terminal 1 - Cache writer
python check_status.py --continuous --write-cache --interval 60

# Terminal 2 - Lightweight API
ENABLE_FALLBACK=false gunicorn api.api:app --bind 0.0.0.0:8000 --timeout 120
```

## API Endpoints

### GET /

Serves the web dashboard (HTML)

### GET /health

Health check endpoint

**Response:**
```json
{
  "status": "ok",
  "service": "muni-status-api",
  "timestamp": "2025-12-08T14:36:24.876950"
}
```

### GET /status

Get current Muni status (best of last 2 checks)

**Response:**
```json
{
  "status": "green",
  "description": "normal",
  "confidence": 0.9998,
  "probabilities": {
    "green": 0.9998,
    "yellow": 0.0002,
    "red": 0.0
  },
  "image_path": "../artifacts/training_data/images/muni_snapshot_20251208_143638.jpg",
  "image_dimensions": {"width": 1860, "height": 800},
  "timestamp": "2025-12-08T14:36:28.364915",
  "cached": true,
  "cache_age": 4.5,
  "status_history": [
    {"status": "green", "timestamp": "2025-12-08T14:36:28.364915"},
    {"status": "green", "timestamp": "2025-12-08T14:35:58.123456"}
  ],
  "is_best_of_two": true
}
```

**Cache behavior:**
- Keeps last 2 status checks
- Returns **best** (most optimistic): green > yellow > red
- Smooths out brief transient failures
- Uses cache if < 5 minutes old (~30ms)
- Falls back to live download + predict if cache stale (~11s)

## Web Frontend

**Lightweight dashboard** (`index.html`) - Just **8.6KB** with zero dependencies!

**Features:**
- 🟢🟡🔴 Large status indicator with emoji
- Real-time status with confidence percentage
- Visual probability bars
- Auto-refreshes every 30 seconds
- Mobile-responsive
- Pure vanilla JavaScript (no frameworks!)

## Cloud Run Deployment

For Cloud Run deployment, use the automated scripts:

```bash
./deploy/cloud/setup-infrastructure.sh  # One-time GCP setup
./deploy/cloud/deploy-services.sh       # Build and deploy services
./deploy/cloud/setup-scheduler.sh       # Configure auto-updates
./deploy/cloud/verify.sh               # Verify deployment
```

See [../deploy/README.md](../deploy/README.md) for complete Cloud Run deployment guide.

## Monitoring

### Local Development

```bash
# View logs
tail -f artifacts/runtime/cache-writer.log
tail -f artifacts/runtime/api-error.log

# Check status
./deploy/local/verify.sh
```

### Cloud Run

```bash
# View logs
gcloud run services logs read munimetro-api --region us-west1 --limit 50
gcloud run services logs read munimetro-checker --region us-west1 --limit 50

# Verify deployment
./deploy/cloud/verify.sh
```

## Troubleshooting

**Local services won't start:**
```bash
# Check logs
tail -20 artifacts/runtime/cache-writer.log
tail -20 artifacts/runtime/api-error.log

# Verify model exists
ls -la artifacts/models/v1/status_classifier.pt

# Run verification
./deploy/local/verify.sh
```

**Port already in use:**
```bash
lsof -i :8000
kill <PID>
```

**Cloud Run deployment issues:**
```bash
# Run verification script
./deploy/cloud/verify.sh

# Check recent logs
gcloud run services logs read munimetro-checker --region us-west1 --limit 100
```

## Performance

- **Standalone API**: ~11 seconds (downloads + predicts every request)
- **Cached API (fallback enabled)**: ~40ms cached, ~11s on cache miss
- **Lightweight cached API (fallback disabled)**: ~30ms cached, 503 on cache miss ⚡

**Recommended**: Use cache writer + lightweight API for production (~30ms responses).

## Next Steps

- See [Testing Guide](../tests/README.md) for running automated tests
- See [Training Guide](../training/README.md) to retrain with more data
