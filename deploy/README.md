# MuniMetro Deployment Guide

Production and local development deployment for SF Muni Metro status monitoring.

See [CONFIGURATION.md](../CONFIGURATION.md) for actual deployment configuration values.

## Quick Reference

### Local Development

```bash
# First time setup
./deploy/local/setup.sh      # Install dependencies, create venv

# Start services
./deploy/local/start.sh      # Start cache writer + API

# Verify status
./deploy/local/verify.sh     # Check everything is running

# Stop services
./deploy/local/stop.sh       # Stop all services
```

Access: http://localhost:8000

### Cloud Run (Production)

```bash
# First time setup (one-time)
./deploy/cloud/setup-infrastructure.sh    # Create GCS bucket, service account, enable APIs
./deploy/cloud/deploy-services.sh         # Deploy API + checker services
./deploy/cloud/setup-scheduler.sh         # Configure auto-updates every 5 min

# Verify deployment
./deploy/cloud/verify.sh                  # Check all services, logs, permissions

# Updates (re-run anytime)
./deploy/cloud/deploy-services.sh         # Deploy new version (zero downtime)
```

Access: See service URL in [CONFIGURATION.md](../CONFIGURATION.md)

---

## Local Development

### Architecture

```
Terminal 1: Cache Writer (background)
  ↓ downloads image every 60s
  ↓ runs ML prediction
  ↓ writes JSON
artifacts/runtime/cache/latest_status.json
  ↑ reads JSON (~30ms)
Terminal 2: API Server (gunicorn)
  ↓ serves to users
Browser: http://localhost:8000
```

### Prerequisites

- Python 3.11+
- Trained ML model in `artifacts/models/v1/` (status_classifier.pt + model.safetensors)
- ~2GB RAM

### Setup (First Time)

```bash
./deploy/local/setup.sh
```

This script:
1. Checks Python version
2. Creates virtual environment: `api/venv`
3. Installs dependencies from `api/requirements.txt`
4. Creates runtime directories

### Starting Services

```bash
./deploy/local/start.sh
```

This starts:
- **Cache writer** - Downloads images + predicts status every 60s (background)
- **API server** - Serves cached results with ~30ms response time (gunicorn)

Both services log to `artifacts/runtime/*.log`.

### Verifying Status

```bash
./deploy/local/verify.sh
```

Checks:
1. Python environment and dependencies
2. ML model exists
3. Runtime directories
4. Cache writer process
5. API server process
6. Endpoint health (/health, /status)

### Stopping Services

```bash
./deploy/local/stop.sh
```

Gracefully stops both services, falling back to force kill if needed.

### Troubleshooting

**Services won't start:**
```bash
# Check logs
tail -f artifacts/runtime/cache-writer.log
tail -f artifacts/runtime/api-error.log

# Verify model exists
ls -lh artifacts/models/v1/status_classifier.pt

# Reinstall dependencies
./deploy/local/setup.sh
```

**Port 8000 already in use:**
```bash
lsof -i :8000
kill <PID>
```

**Cache not updating:**
```bash
# Check cache writer is running
./deploy/local/verify.sh

# View cache writer logs
tail -f artifacts/runtime/cache-writer.log
```

---

## Cloud Run Deployment

### Architecture

```
Cloud Scheduler (every 5 min)
  ↓ triggers (OAuth)
munimetro-checker (Cloud Run Job)
  ↓ downloads image + predicts status
  ↓ writes JSON + exits
Cloud Storage (gs://munimetro-cache/latest_status.json)
  ↑ reads JSON (~100-200ms)
munimetro-api (Cloud Run Service)
  ↓ serves to users
Users
```

### Prerequisites

1. **Google Cloud SDK**
   ```bash
   brew install google-cloud-sdk
   gcloud auth login
   ```

2. **GCP Project with billing enabled**
   ```bash
   gcloud projects create PROJECT_ID --name="PROJECT_NAME"
   gcloud billing projects link PROJECT_ID --billing-account=BILLING_ACCOUNT_ID
   ```

3. **Environment variables** (optional - override defaults)
   ```bash
   export GCP_PROJECT_ID="PROJECT_ID"
   export GCP_REGION="REGION"
   export GCS_BUCKET="BUCKET_NAME"
   ```

See [CONFIGURATION.md](../CONFIGURATION.md) for the actual values used in this deployment.

### Setup (First Time)

```bash
./deploy/cloud/setup-infrastructure.sh
```

This script:
1. Enables required APIs (Cloud Run, Cloud Build, Cloud Scheduler, Storage)
2. Creates private GCS bucket for cache
3. Creates service account with minimal permissions
4. Sets up App Engine (required for Cloud Scheduler)

**Takes ~5 minutes**

### Deploying Services

```bash
./deploy/cloud/deploy-services.sh
```

This script:
1. Builds Docker image from project root
2. Deploys `munimetro-api` (public Cloud Run Service, serves frontend + API)
3. Deploys `munimetro-checker` (Cloud Run Job, updates status when triggered)
4. Grants service account permissions

**First deployment: ~10-15 minutes** (downloading ML model layers)
**Subsequent deployments: ~3-5 minutes**

### Setting Up Scheduler

```bash
./deploy/cloud/setup-scheduler.sh
```

This script:
1. Creates Cloud Scheduler job
2. Configures 5-minute interval (`*/5 * * * *`)
3. Sets up OAuth authentication to trigger Cloud Run Job
4. Runs test execution

### Verifying Deployment

```bash
./deploy/cloud/verify.sh
```

Comprehensive verification checks:
1. GCS bucket exists and has cache file
2. API service deployed and healthy
3. Checker job deployed and execution history
4. Cloud Scheduler configured and running
5. Service account permissions
6. Recent logs from API and job executions

### Cost Estimate

Typical usage costs approximately $1.17/month:

| Service | Usage | Cost |
|---------|-------|------|
| Cloud Run Service (API) | ~1000 requests/day | $0 (free tier) |
| Cloud Run Jobs (Checker) | 21,600 executions/month @ ~10s each | $1.04 |
| Cloud Storage | 1KB file, 75K reads/month | $0.016 |
| Cloud Scheduler | 1 job | $0.10 |

Cloud Run Jobs are more cost-effective than Services for scheduled tasks due to the absence of idle costs.

At 100K API requests/day, estimated cost increases to approximately $2-3/month.

See [CONFIGURATION.md](../CONFIGURATION.md) for detailed cost breakdown.

### Configuration

See [CONFIGURATION.md](../CONFIGURATION.md) for actual service names and regions.

**Change update frequency:**
```bash
# Edit deploy/cloud/setup-scheduler.sh, change SCHEDULE
export SCHEDULE="*/10 * * * *"  # Every 10 minutes
./deploy/cloud/setup-scheduler.sh
```

**Change region:**
```bash
export GCP_REGION="REGION"
./deploy/cloud/setup-infrastructure.sh
./deploy/cloud/deploy-services.sh
./deploy/cloud/setup-scheduler.sh
```

**Manual trigger:**
```bash
# Option 1: Via scheduler
gcloud scheduler jobs run SCHEDULER_JOB_NAME --location=REGION

# Option 2: Run job directly (faster for testing)
gcloud run jobs execute CHECKER_JOB_NAME --region=REGION
```

### Monitoring

**View logs:**
```bash
# API logs
gcloud run services logs read API_SERVICE_NAME --region REGION --limit 50

# Checker job logs
gcloud logging read 'resource.type="cloud_run_job" AND resource.labels.job_name="CHECKER_JOB_NAME"' --limit 50

# Job executions
gcloud run jobs executions list --job=CHECKER_JOB_NAME --region=REGION

# Scheduler logs
gcloud logging read 'resource.type="cloud_scheduler_job"' --limit 20
```

**View metrics (Cloud Console):**
```bash
open "https://console.cloud.google.com/run?project=PROJECT_ID"
```

### Updates

**Deploy new version:**
```bash
./deploy/cloud/deploy-services.sh  # Zero-downtime deployment
```

**Rollback:**
```bash
# List revisions
gcloud run revisions list --service API_SERVICE_NAME --region REGION

# Rollback
gcloud run services update-traffic API_SERVICE_NAME \
    --region REGION \
    --to-revisions REVISION_NAME=100
```

### Troubleshooting

**Permission denied errors:**
```bash
# Check bucket permissions
gsutil iam get gs://BUCKET_NAME

# Re-grant if needed
gsutil iam ch \
  serviceAccount:SERVICE_ACCOUNT_EMAIL:objectAdmin \
  gs://BUCKET_NAME
```

**Checker job fails:**
```bash
# View recent executions
gcloud run jobs executions list --job=CHECKER_JOB_NAME --region=REGION --limit=10

# View logs
gcloud logging read 'resource.type="cloud_run_job" AND resource.labels.job_name="CHECKER_JOB_NAME"' --limit 100

# Common issues:
# - Model download timeout → increase --task-timeout in deploy script
# - GCS permission denied → check service account IAM
# - Out of memory → increase --memory to 4Gi in deploy script
```

**Scheduler not triggering:**
```bash
# Check status
gcloud scheduler jobs describe SCHEDULER_JOB_NAME --location REGION

# View execution history
gcloud logging read 'resource.type="cloud_scheduler_job"' --limit 20

# Manual test
gcloud scheduler jobs run SCHEDULER_JOB_NAME --location REGION
```

### Security

**IAM Permissions:**
- Service account has Storage Object Admin on bucket only
- API service is public
- Checker job only executable by authorized service accounts
- Cloud Scheduler uses OAuth to trigger job executions

**Network:**
- API Service and Checker Job run in Google's managed VPC
- Outbound: HTTPS to sfmunicentral.com only
- Bucket is private (no public access)
- Checker Job has no inbound network (batch execution only)

### Cleanup

```bash
# Delete service
gcloud run services delete API_SERVICE_NAME --region REGION --quiet

# Delete job
gcloud run jobs delete CHECKER_JOB_NAME --region REGION --quiet

# Delete scheduler
gcloud scheduler jobs delete SCHEDULER_JOB_NAME --location REGION --quiet

# Delete bucket
gsutil rm -r gs://BUCKET_NAME

# Delete service account
gcloud iam service-accounts delete SERVICE_ACCOUNT_EMAIL --quiet
```

---

## Comparison

| Feature | Local | Cloud Run |
|---------|-------|-----------|
| Setup time | 2 minutes | 15 minutes (first time) |
| Cost | Free | ~$0.54/month |
| Auto-scaling | No | Yes |
| Auto-updates | No | Yes (every 5 min) |
| Public URL | No | Yes |
| Persistence | Local files | Cloud Storage |
| Ideal for | Development, testing | Production, demos |

---

## Notes

- **Dockerfile**: The `Dockerfile` at project root is only used for Cloud Run deployments. Local development uses shell scripts and does not require Docker.
- **Cloud Build**: Cloud Run deployments automatically build container images using Google Cloud Build.

