# API Documentation

Production web API for Muni Metro status monitoring with local and cloud deployment support.

## Quick Start

### Local Deployment (Recommended for Development)

```bash
cd deploy/local
./setup.sh      # One-time environment setup
./start.sh      # Start cache writer + API server
./verify.sh     # Verify services running
./stop.sh       # Stop all services
```

See [deploy/README.md](../deploy/README.md) for detailed deployment instructions.

### Cloud Deployment (Production)

```bash
cd deploy/cloud
./setup-infrastructure.sh  # One-time GCP setup
./deploy-services.sh       # Deploy services
./setup-scheduler.sh       # Configure scheduler
./verify.sh               # Verify deployment
```

Cost: ~$0.13/month for typical usage.

See [deploy/cloud/README.md](../deploy/cloud/README.md) for complete cloud deployment guide.

## Manual Setup (Development)

For manual local development without deployment scripts:

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run status checker (single execution)
python check_status.py

# Run continuous monitoring
python check_status.py --continuous --interval 60

# Run API server (standalone mode, ~11s per request)
gunicorn api:app --bind 0.0.0.0:8000 --timeout 120

# Recommended: Cache writer + lightweight API
# Terminal 1: Start cache writer
python check_status.py --continuous --write-cache --interval 60

# Terminal 2: Start lightweight API (cache-only mode)
ENABLE_FALLBACK=false gunicorn api:app --bind 0.0.0.0:8000 --timeout 120
```

## API Endpoints

### GET /

Web dashboard interface.

**Response**: HTML page with real-time status display.

### GET /health

Health check endpoint with component-level status.

**Response** (200 OK):
```json
{
  "status": "healthy",
  "service": "muni-status-api",
  "timestamp": "2025-12-11T23:00:00.000000",
  "components": {
    "cache": {
      "status": "healthy",
      "cache_age_seconds": 45.2,
      "is_stale": false,
      "last_status": "green"
    },
    "analytics": {
      "status": "healthy",
      "total_checks": 20160
    }
  }
}
```

**Overall Status Values**:
- `healthy`: All components working normally
- `degraded`: Some components have issues but service is functional
- `unhealthy`: Critical failure (not currently returned as 200)

**Component Status**:
- `cache`: Status cache freshness (stale if > 5 minutes old)
- `analytics`: SQLite database connectivity

### GET /status

Current Muni Metro status with best-of-three smoothing logic.

**Response** (200 OK):
```json
{
  "status": "green",
  "description": "normal",
  "confidence": 0.9998,
  "probabilities": {
    "green": 0.9998,
    "yellow": 0.0002,
    "red": 0.0000
  },
  "image_path": "artifacts/runtime/downloads/muni_snapshot_20251211_230000.jpg",
  "image_dimensions": {
    "width": 1860,
    "height": 800
  },
  "timestamp": "2025-12-11T23:00:15.920941",
  "cached": true,
  "cache_age": 12.7,
  "status_history": [
    {
      "status": "green",
      "timestamp": "2025-12-11T23:00:15.920941"
    },
    {
      "status": "green",
      "timestamp": "2025-12-11T22:59:38.506073"
    }
  ],
  "is_best_of_two": true
}
```

**Fields**:
- `status`: Classification result (green/yellow/red)
- `description`: Natural language description
- `confidence`: Classification confidence (0-1)
- `probabilities`: Per-class probabilities
- `cached`: Whether response served from cache
- `cache_age`: Seconds since cache update
- `status_history`: Last 3 status checks
- `is_best_of_two`: Whether best-of-three smoothing applied

**Caching Behavior**:
- Maintains last 2 status checks
- Returns most optimistic status (green > yellow > red)
- Reduces false positives from transient issues
- Cache valid for 5 minutes
- Fallback to live prediction if cache stale (when `ENABLE_FALLBACK=true`)

**Performance**:
- Cached response: ~30ms
- Live detection: ~11s (download + OpenCV analysis)
- Cache-only mode: ~30ms cached, 503 on miss

### GET /feed.xml

RSS feed of status updates. Updates when status changes.

**Response** (200 OK): RSS 2.0 XML feed with recent status changes.

**Caching**: 1 minute (`Cache-Control: public, max-age=60`)

### GET /analytics

Analytics dashboard showing delay statistics.

**Response**: HTML page with charts for:
- Status distribution (pie chart)
- Delays by station (bar chart)
- Delays by hour of day (bar chart)
- Delays by day of week (bar chart)

### GET /analytics-data

JSON API for delay analytics.

**Query Parameters**:
- `days`: Number of days to analyze (default: 7, max: 365)

**Response** (200 OK):
```json
{
  "period_days": 7,
  "frequency": {
    "total_checks": 20160,
    "delayed_checks": 245,
    "delay_rate": 0.0122,
    "by_status": {"green": 19800, "yellow": 245, "red": 115}
  },
  "by_station": [
    {"station": "PO", "name": "Powell", "count": 45, "types": {"platform_hold": 40, "bunching": 5}}
  ],
  "by_time": {
    "by_hour": {"0": 0, "1": 0, ..., "17": 45, "18": 38, ...},
    "by_day": {"0": 50, "1": 48, ..., "6": 12}
  },
  "generated_at": "2026-01-31T14:30:00",
  "from_cache": true,
  "cache_age": 120.5
}
```

**Notes**:
- Reports are cached for 24 hours to reduce database load
- Only yellow status counts as "delayed" (red = maintenance)
- `by_day`: 0=Monday, 6=Sunday

**Caching**: 24 hours (`Cache-Control: public, max-age=86400`)

## Web Dashboard

Lightweight single-page application (`dashboard.html`, 8.6KB).

**Access**: Navigate to `/dashboard` to view the status dashboard.

**Features**:
- Real-time status indicator (green/yellow/red)
- Confidence percentage
- Visual probability bars
- Auto-refresh (30-second interval)
- Mobile-responsive design
- Zero JavaScript dependencies

**Technology**:
- Pure vanilla JavaScript
- No build process required
- No external dependencies

## Architecture

### Local Development

```
Cache Writer Process (background)
  ↓ downloads image every 60s
  ↓ runs OpenCV detection
  ↓ writes JSON to disk
Local Cache File
  ↑ reads JSON (~30ms)
API Server (gunicorn)
  ↓ serves /status, /health, /
Client Browser
```

### Cloud Run Production

```
Cloud Scheduler (every 3 min)
  ↓ triggers via OAuth
Cloud Run Job (munimetro-checker)
  ↓ downloads image + detects status
  ↓ writes JSON to GCS
Cloud Storage (gs://munimetro-cache/)
  ↑ reads JSON (~100-200ms)
Cloud Run Service (munimetro-api)
  ↓ serves endpoints
Client Browser
```

See [CONFIGURATION.md](../CONFIGURATION.md) for deployment configuration details.

## Monitoring

### Local Development

```bash
# View logs
tail -f artifacts/runtime/cache-writer.log
tail -f artifacts/runtime/api-error.log

# Check service status
./deploy/local/verify.sh

# Test endpoints
curl http://localhost:8000/health
curl http://localhost:8000/status
```

### Cloud Run

```bash
# View API logs
gcloud run services logs read munimetro-api --region us-west1 --limit 50

# View job logs
gcloud logging read 'resource.type="cloud_run_job" AND resource.labels.job_name="munimetro-checker"' --limit 50

# Verify deployment
./deploy/cloud/verify.sh

# Test endpoints
curl https://munimetro-api-438243686292.us-west1.run.app/health
curl https://munimetro-api-438243686292.us-west1.run.app/status
```

## Environment Variables

### Local Development

- `CLOUD_RUN`: `false` (default) - Use local file cache
- `ENABLE_FALLBACK`: `true` (default) - Download+predict on cache miss

### Cloud Run

- `CLOUD_RUN`: `true` - Use Google Cloud Storage cache
- `GCS_BUCKET`: `munimetro-cache` - Cache bucket name
- `ENABLE_FALLBACK`: `false` - Cache-only mode (no fallback)
- `PORT`: `8000` - HTTP server port

See [CONFIGURATION.md](../CONFIGURATION.md) for complete configuration reference.

## Troubleshooting

### Local Services Not Starting

Check logs and verify model files:

```bash
# View recent log entries
tail -20 artifacts/runtime/cache-writer.log
tail -20 artifacts/runtime/api-error.log

# Run verification script
./deploy/local/verify.sh
```

### Port Already in Use

```bash
# Find process using port 8000
lsof -i :8000

# Terminate process
kill PID

# Or use stop script
./deploy/local/stop.sh
```

### Cloud Run Deployment Issues

```bash
# Run comprehensive verification
./deploy/cloud/verify.sh

# Check service logs
gcloud run services logs read munimetro-api --region us-west1 --limit 100

# Check job logs
gcloud logging read 'resource.type="cloud_run_job" AND resource.labels.job_name="munimetro-checker"' --limit 100

# Verify service status
gcloud run services describe munimetro-api --region us-west1

# Verify job status
gcloud run jobs describe munimetro-checker --region us-west1
```

### Cache Not Updating

Local development:

```bash
# Check cache writer process
ps aux | grep check_status.py

# View cache writer logs
tail -f artifacts/runtime/cache-writer.log

# Verify cache file
ls -lh artifacts/runtime/cache/latest_status.json
cat artifacts/runtime/cache/latest_status.json | python3 -m json.tool
```

Cloud Run:

```bash
# Check recent job executions
gcloud run jobs executions list --job=munimetro-checker --region=us-west1 --limit=10

# View job execution logs
gcloud logging read 'resource.type="cloud_run_job"' --limit 50

# Verify cache in GCS
gsutil stat gs://munimetro-cache/latest_status.json
```

## Performance Optimization

### Recommended Configuration

**Local Development**:
- Cache writer with 60-second interval
- Lightweight API with `ENABLE_FALLBACK=false`
- Response time: ~30ms (cached)

**Cloud Run**:
- Cloud Run Job triggered every 3 minutes
- API Service in cache-only mode (`ENABLE_FALLBACK=false`)
- Response time: ~100-200ms (GCS latency + processing)

### Performance Metrics

| Configuration | Cached Response | Cache Miss |
|--------------|-----------------|------------|
| Standalone API | N/A | ~11s |
| Cache + Fallback | ~30ms | ~11s |
| Cache Only | ~30ms | 503 error |
| Cloud Run | ~100-200ms | 503 error |

## Related Documentation

- **Deployment Guide**: [deploy/README.md](../deploy/README.md) - Local and cloud deployment
- **Cloud Deployment**: [deploy/cloud/README.md](../deploy/cloud/README.md) - Google Cloud Run
- **Training Guide**: [training/README.md](../training/README.md) - Model training workflow
- **Testing Guide**: [tests/README.md](../tests/README.md) - Automated test suite
- **Configuration**: [CONFIGURATION.md](../CONFIGURATION.md) - System configuration values
