# API & Deployment Guide

Production-ready web API for SF Muni status monitoring with Docker containerization and Google Cloud Run deployment support.

## Deployment Options

### 🚀 Cloud Run (Production - Recommended)

**Fully managed, serverless, auto-scaling deployment**

```bash
cd api/deploy
./setup-infrastructure.sh  # One-time setup
./deploy-services.sh       # Deploy
./setup-scheduler.sh       # Configure auto-updates
```

**Cost:** ~$0.12/month for typical usage
**Guide:** See [deploy/README.md](deploy/README.md) for complete instructions

### 🐳 Docker (Local Development)

## Quick Start (Docker)

```bash
# From the api/ directory
docker-compose up -d

# Access the web dashboard
open http://localhost:8000

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

This starts:
- **cache-writer** - Downloads images and updates predictions every 60 seconds
- **api** - Lightweight API serving cached results (~30ms response time)

## Running Locally (Without Docker)

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
gunicorn api:app --bind 0.0.0.0:8000 --timeout 120

# Option 4: Web API (lightweight, cache-only)
# Terminal 1 - Cache writer
python check_status.py --continuous --write-cache --interval 60

# Terminal 2 - Lightweight API
ENABLE_FALLBACK=false gunicorn api:app --bind 0.0.0.0:8000 --timeout 120
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

## Docker Deployment

### Prerequisites

1. Docker installed - [Install Docker Desktop](https://www.docker.com/products/docker-desktop) or use Colima
2. Trained model in `../artifacts/models/v1/`

### Security Features

- **Minimal base image** - `python:3.13-slim-bookworm`
- **Multi-stage build** - Separates build deps from runtime
- **Non-root user** - Runs as `muni` user (not root)
- **Read-only volumes** - Model directory mounted read-only
- **Health checks** - Automatic container monitoring
- **Graceful shutdown** - Proper signal handling with gunicorn

### Building Manually

```bash
# Build from project root
cd ..
docker build -f api/Dockerfile -t muni-status:latest .

# Run with docker-compose
cd api
docker-compose up -d
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ENABLE_FALLBACK` | `false` | Enable live predictions when cache unavailable |
| `PYTHONUNBUFFERED` | `1` | Disable Python output buffering for real-time logs |
| `PORT` | `8000` | HTTP server port |

### Volume Mounts

| Volume | Purpose | Mount Mode |
|--------|---------|------------|
| `../artifacts/models/v1/` | ML model files | Read-only (`:ro`) |
| `cache-data` (named volume) | Cached predictions | Read-write (API: read-only) |
| `snapshot-data` (named volume) | Downloaded images | Read-write |

## Google Cloud Run Deployment

Serverless, auto-scaling deployment ideal for this application.

### Step 1: Prepare

```bash
# Set project
export PROJECT_ID=your-gcp-project-id
export REGION=us-central1

# Enable APIs
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com
```

### Step 2: Create Artifact Registry

```bash
gcloud artifacts repositories create muni-status \
  --repository-format=docker \
  --location=$REGION \
  --description="Muni status API containers"
```

### Step 3: Build and Push

```bash
# Configure Docker auth
gcloud auth configure-docker ${REGION}-docker.pkg.dev

# Build and push from project root
cd ..
docker build -f api/Dockerfile -t ${REGION}-docker.pkg.dev/${PROJECT_ID}/muni-status/api:latest .
docker push ${REGION}-docker.pkg.dev/${PROJECT_ID}/muni-status/api:latest
```

### Step 4: Upload Model to Cloud Storage

```bash
# Create bucket
gsutil mb -l $REGION gs://${PROJECT_ID}-muni-models

# Upload model
gsutil -m cp -r artifacts/models/v1/* gs://${PROJECT_ID}-muni-artifacts/models/v1/
```

### Step 5: Deploy to Cloud Run

```bash
# Deploy cache writer (always-on, updates cache)
gcloud run deploy muni-cache-writer \
  --image ${REGION}-docker.pkg.dev/${PROJECT_ID}/muni-status/api:latest \
  --region $REGION \
  --memory 2Gi \
  --cpu 1 \
  --timeout 3600 \
  --no-allow-unauthenticated \
  --set-env-vars PYTHONUNBUFFERED=1 \
  --command python \
  --args check_status.py,--continuous,--write-cache,--interval,60

# Deploy API (auto-scaling, cache-only mode)
gcloud run deploy muni-api \
  --image ${REGION}-docker.pkg.dev/${PROJECT_ID}/muni-status/api:latest \
  --region $REGION \
  --memory 512Mi \
  --cpu 1 \
  --timeout 120 \
  --allow-unauthenticated \
  --set-env-vars ENABLE_FALLBACK=false,PYTHONUNBUFFERED=1 \
  --port 8000

# Get URL
gcloud run services describe muni-api --region $REGION --format 'value(status.url)'
```

**Note**: Cloud Run is stateless. Share cache between services using:
1. Cloud Storage (slower, ~100-200ms)
2. Redis/Memorystore (faster, ~5-10ms)
3. Always-on cache writer + auto-scaling API

## Monitoring

```bash
# View logs (Docker)
docker-compose logs -f api
docker-compose logs -f cache-writer

# Check container health
docker inspect muni-api | grep -A 10 Health

# View logs (Cloud Run)
gcloud run services logs read muni-api --region $REGION --limit 50
```

## Troubleshooting

**Container won't start:**
```bash
docker logs muni-api
ls -la ../artifacts/models/v1/  # Verify model exists
```

**Health check failing:**
```bash
docker exec muni-api curl -f http://localhost:8000/health
docker exec muni-api ps aux | grep gunicorn
```

**Cache not updating:**
```bash
docker logs muni-cache-writer
docker exec muni-cache-writer ls -la ../artifacts/runtime/cache/
```

**Port already in use:**
```bash
lsof -i :8000
# Kill the process or change port in docker-compose.yml
```

## Performance

- **Standalone API**: ~11 seconds (downloads + predicts every request)
- **Cached API (fallback enabled)**: ~40ms cached, ~11s on cache miss
- **Lightweight cached API (fallback disabled)**: ~30ms cached, 503 on cache miss ⚡

**Recommended**: Use cache writer + lightweight API for production (~30ms responses).

## Next Steps

- See [Testing Guide](../tests/README.md) for running automated tests
- See [Training Guide](../training/README.md) to retrain with more data
