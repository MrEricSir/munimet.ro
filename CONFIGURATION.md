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

#### Reference Data and Models Bucket
- **Bucket Name**: `munimetro-annex`
- **Region**: `us-west1`
- **Storage Class**: Standard
- **Purpose**: Reference data and model storage (synced via GCS rsync scripts)
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

Reference data is synced with Google Cloud Storage using `gsutil rsync`:

- **Bucket**: `gs://munimetro-annex`
- **Scripts**:
  - `scripts/sync-reference-data.sh` / `scripts/sync-reference-data.ps1` - Sync training data (~270MB)
  - `scripts/sync-artifacts.sh` / `scripts/sync-artifacts.ps1` - Sync all artifacts (~270MB)
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

- **API**: `api/venv/`
- **Tests**: `tests/venv/`

## File Locations

### Reference Data (GCS synced)

- **Images Directory**: `artifacts/reference_data/images/`
- **Count**: 2,666 labeled images
- **Size**: ~270MB
- **Labels File**: `artifacts/reference_data/labels.json` (570KB)
- **Sync**: Use `scripts/sync-reference-data.sh download` to download from GCS

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
- **Cloud Storage (Reference Data)**: ~$0 (1.1GB, within 5GB free tier)
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

## Detection Configuration

The system uses OpenCV-based computer vision for deterministic status detection.

### Detection Logic

**Location**: `lib/detection.py`

The system status is determined by:

1. **Red (Not Operating)**:
   - Fewer than 2 trains with valid route suffixes detected
   - Indicates overnight/maintenance mode

2. **Yellow (Delays)**:
   - 2+ platforms in hold mode (yellow color), OR
   - Any track sections disabled (red color), OR
   - Train bunching detected (4+ trains clustered together)

3. **Green (Normal)**:
   - 2+ trains with valid routes operating
   - No significant delays detected

### Detection Parameters

**HSV Color Ranges** (`scripts/station_detector.py`):
- Platform blue (normal): H=95-115, S=100-255, V=100-255
- Platform yellow (hold): H=20-35, S=150-255, V=150-255
- Track cyan (normal): H=80-100, S=100-255, V=100-255
- Track red (disabled): H=0-10 or 170-180, S=100-255, V=100-255

**Train Bunching** (`lib/detection.py`):
- Threshold: 4+ trains in cluster
- Cluster distance: 70 pixels
- Excluded stations: CT, EM (turnaround points)

### Testing Detection

```bash
# Run detection on a single image
python lib/detection.py tests/images/muni_snapshot_20251207_092107.jpg

# Run all detection tests
pytest tests/test_system_status.py -v

# Interactive detection viewer
python scripts/station_viewer.py
```

## Notes

- All timestamps use UTC
- Cloud Run uses Pacific time for scheduler display
- Reference data and models synced with GCS using `gsutil rsync`
- Docker images are multi-stage builds (smaller production images)
