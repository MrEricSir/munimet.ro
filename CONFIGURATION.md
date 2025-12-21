# Configuration Reference

This document records the actual configuration values used for this deployment.

## Google Cloud Platform

### Project Configuration
- **Project ID**: `munimetro`
- **Project Number**: `438243686292`
- **Region**: `us-west1` (Oregon - closest to San Francisco)
- **Billing Account**: Configured

### Google Cloud Storage

#### Cache Bucket (Production)
- **Bucket Name**: `munimetro-cache`
- **Region**: `us-west1`
- **Storage Class**: Standard
- **Purpose**: API response cache for Cloud Run deployment
- **Access**: Private (service account only)

#### Training Data and Models Bucket
- **Bucket Name**: `munimetro-annex`
- **Region**: `us-west1`
- **Storage Class**: Standard
- **Purpose**: Training data and model storage (synced via GCS rsync scripts)
- **Size**: ~1.1GB
- **Access**: Private (collaborators with GCS access)

### Cloud Run Services

#### API Service
- **Service Name**: `munimetro-api`
- **Type**: Cloud Run Service (long-running HTTP server)
- **Region**: `us-west1`
- **URL**: `https://munimetro-api-438243686292.us-west1.run.app`
- **Custom Domain**: `https://munimet.ro`
- **Memory**: 2Gi
- **CPU**: 1
- **Concurrency**: 80
- **Min Instances**: 0
- **Max Instances**: 10
- **Access**: Public (unauthenticated)
- **Port**: 8000

#### Checker Job
- **Job Name**: `munimetro-checker`
- **Type**: Cloud Run Job (batch execution)
- **Region**: `us-west1`
- **Memory**: 2Gi
- **CPU**: 1
- **Task Timeout**: 120s
- **Max Retries**: 3
- **Access**: Restricted (Cloud Scheduler only)

### Cloud Scheduler

- **Job Name**: `munimetro-status-check`
- **Location**: `us-west1`
- **Schedule**: `*/3 * * * *` (every 3 minutes)
- **Target**: `munimetro-checker` Cloud Run Job
- **Authentication**: OAuth (service account)
- **Retry Policy**: 3 attempts, exponential backoff (5s-3600s)

### Service Account

- **Name**: `munimetro-api`
- **Email**: `munimetro-api@munimetro.iam.gserviceaccount.com`
- **Roles**:
  - Storage Object Admin on `gs://munimetro-cache`
  - Cloud Run Invoker on `munimetro-checker` job

## GCS Sync Configuration

### Sync Scripts

Training data and models are synced with Google Cloud Storage using `gsutil rsync`:

- **Bucket**: `gs://munimetro-annex`
- **Scripts**:
  - `scripts/sync-models.sh` / `scripts/sync-models.ps1` - Sync model files (~856MB)
  - `scripts/sync-training-data.sh` / `scripts/sync-training-data.ps1` - Sync training data (~270MB)
  - `scripts/sync-artifacts.sh` / `scripts/sync-artifacts.ps1` - Sync all artifacts (~1.1GB)
- **Authentication**: Google Cloud SDK (`gcloud auth login`)
- **Protocol**: HTTPS via gsutil

### Repository

- **GitHub URL**: `https://github.com/MrEricSir/munimet.ro.git`
- **SSH URL**: `git@github.com:MrEricSir/munimet.ro.git`
- **Branch**: `main`

## Domain Configuration

- **Primary Domain**: `munimet.ro`
- **DNS Provider**: (Configure via domain registrar)
- **SSL/TLS**: Managed by Google Cloud Load Balancer
- **HTTP→HTTPS**: Automatic redirect

## Local Development

### Ports

- **API Server**: 8000
- **Cache Writer**: Background process (no port)

### Directories

- **Runtime Cache**: `artifacts/runtime/cache/`
- **Downloads**: `artifacts/runtime/downloads/`
- **Logs**: `artifacts/runtime/*.log`

### Virtual Environments

- **Training**: `training/venv/`
- **API**: `api/venv/`
- **Tests**: `tests/venv/`

## File Locations

### Models (GCS synced)

- **Directory**: `artifacts/models/v1/`
- **Size**: ~856MB
- **Files**:
  - `model.safetensors` (854MB) - BLIP vision transformer
  - `status_classifier.pt` (775KB) - Status classification head
  - `config.json`, `generation_config.json`, `preprocessor_config.json`
  - `tokenizer.json`, `tokenizer_config.json`, `vocab.txt`, `special_tokens_map.json`
- **Sync**: Use `scripts/sync-models.sh download` to download from GCS

### Training Data (GCS synced)

- **Images Directory**: `artifacts/training_data/images/`
- **Count**: 2,666 labeled images
- **Size**: ~270MB
- **Labels File**: `artifacts/training_data/labels.json` (570KB)
- **Sync**: Use `scripts/sync-training-data.sh download` to download from GCS

## Environment Variables

### Cloud Run (Production)

```bash
CLOUD_RUN=true
GCS_BUCKET=munimetro-cache
ENABLE_FALLBACK=false  # Don't download+predict on API, read cache only
PORT=8000
```

### Local Development

```bash
# No environment variables needed for default configuration
# Optional overrides:
# CLOUD_RUN=false (default)
# ENABLE_FALLBACK=true (default)
```

## Cost Estimates

### Current Monthly Costs

- **Cloud Run Service (API)**: ~$0 (within free tier at current traffic)
- **Cloud Run Jobs (Checker)**: ~$0.70 (14,400 executions/month @ ~10s each)
- **Cloud Storage (Cache)**: ~$0.02 (1KB file, 100K reads/month)
- **Cloud Storage (Training Data)**: ~$0 (1.1GB, within 5GB free tier)
- **Cloud Scheduler**: $0.10 (1 job)
- **Cloud Monitoring**: $0 (within free tier limits)
- **Total**: ~$0.82/month

### Scaling Projections

At 100K API requests/day:
- **Cloud Run Service**: ~$2-3/month
- **Other services**: Unchanged
- **Total**: ~$2-3/month

## Image Source

- **URL**: `http://sfmunicentral.com/sfmunicentral_Snapshot_Objects/Mimic1_A7SE582P.jpg`
- **Update Frequency**: Real-time (SFMTA internal status board)
- **Dimensions**: 1860 × 800 pixels
- **Format**: JPEG
- **Size**: ~200KB per image

## Model Configuration

- **Base Model**: Salesforce/blip-image-captioning-base
- **Architecture**: Vision transformer + language model
- **Fine-tuning**: Status classification head
- **Classes**: 3 (red, yellow, green)
- **Training Accuracy**: >95% on held-out test set
- **Inference Time**: ~2-3 seconds (CPU)

## Notes

- All timestamps use UTC
- Cloud Run uses Pacific time for scheduler display
- Training data and models synced with GCS using `gsutil rsync`
- Docker images are multi-stage builds (smaller production images)
