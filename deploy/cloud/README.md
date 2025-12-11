# Cloud Run Deployment Guide

This directory contains scripts to deploy MuniMetro to Google Cloud Run.

## Architecture

```
Cloud Scheduler (every 5 min)
  ↓
munimetro-checker (Cloud Run)
  ↓ downloads image + predicts status
  ↓ writes JSON
Cloud Storage (gs://munimetro-cache/latest_status.json)
  ↑ reads JSON
munimetro-api (Cloud Run)
  ↓ serves to users
Users
```

## Prerequisites

1. **Google Cloud SDK** installed
   ```bash
   # Install gcloud CLI
   brew install google-cloud-sdk

   # Login
   gcloud auth login
   ```

2. **GCP Project** created
   ```bash
   # Create project (if needed)
   gcloud projects create munimetro --name="MuniMetro"

   # Set billing account (required)
   gcloud billing accounts list
   gcloud billing projects link munimetro --billing-account=BILLING_ACCOUNT_ID
   ```

3. **Environment variables** (optional)
   ```bash
   export GCP_PROJECT_ID="munimetro"       # Your GCP project ID
   export GCP_REGION="us-west1"             # Region (default: us-west1)
   export GCS_BUCKET="munimetro-cache"      # Bucket name (default: munimetro-cache)
   ```

## Deployment Steps

### 1. Setup Infrastructure (One-time)

```bash
./deploy/cloud/setup-infrastructure.sh
```

This script:
- ✅ Enables required GCP APIs
- ✅ Creates Cloud Storage bucket (private)
- ✅ Creates service account with minimal permissions
- ✅ Sets up App Engine (for Cloud Scheduler)

### 2. Deploy Services

```bash
./deploy/cloud/deploy-services.sh
```

This script:
- ✅ Builds container image using Cloud Build
- ✅ Deploys `munimetro-api` (public, serves frontend + API)
- ✅ Deploys `munimetro-checker` (private, updates status)

**First deployment takes ~10-15 minutes** (downloading ML model layers).

### 3. Setup Scheduler

```bash
./deploy/cloud/setup-scheduler.sh
```

This script:
- ✅ Creates Cloud Scheduler job
- ✅ Configures 5-minute interval
- ✅ Runs test execution

## Verification

### Check API is running
```bash
# Get API URL
API_URL=$(gcloud run services describe munimetro-api \
    --region us-west1 \
    --format 'value(status.url)')

# Test health endpoint
curl $API_URL/health

# Test status endpoint (may be empty on first deploy)
curl $API_URL/status
```

### Trigger manual status update
```bash
gcloud scheduler jobs run munimetro-status-check --location=us-west1

# Wait 30 seconds for job to complete, then check API
sleep 30
curl $API_URL/status
```

### View logs
```bash
# API logs
gcloud run services logs read munimetro-api --region us-west1 --limit 50

# Checker logs
gcloud run services logs read munimetro-checker --region us-west1 --limit 50

# Scheduler logs
gcloud logging read 'resource.type="cloud_scheduler_job"' --limit 20
```

## Cost Estimate

**Expected monthly cost: ~$0-5**

| Service | Usage | Cost |
|---------|-------|------|
| Cloud Run (API) | ~1000 requests/day | $0 (free tier) |
| Cloud Run (Checker) | 8,640 invocations/month | $0 (free tier) |
| Cloud Storage | 1KB file, 30K reads/month | $0.016 |
| Cloud Scheduler | 1 job | $0.10 |
| **Total** | | **~$0.12/month** |

*Scaling: At 100K API requests/day, cost increases to ~$2-3/month.*

## Configuration

### Change update frequency
```bash
# Edit schedule (cron format)
export SCHEDULE="*/10 * * * *"  # Every 10 minutes
./deploy/cloud/setup-scheduler.sh
```

### Change bucket name
```bash
export GCS_BUCKET="my-custom-bucket"
./deploy/cloud/setup-infrastructure.sh
./deploy/cloud/deploy-services.sh  # Redeploy with new bucket
```

### Enable fallback mode (download+predict on API)
```bash
# Edit deploy/cloud/deploy-services.sh, change:
# ENABLE_FALLBACK=false → ENABLE_FALLBACK=true

# Redeploy
./deploy/cloud/deploy-services.sh
```

## Security

### IAM Permissions

**Service Account** (`munimetro-api@PROJECT.iam.gserviceaccount.com`):
- ✅ Storage Object Admin on `gs://munimetro-cache` (read/write cache)
- ❌ No other permissions

**API Service** (`munimetro-api`):
- ✅ Public (unauthenticated) access
- ✅ Serves frontend and `/status` endpoint

**Checker Service** (`munimetro-checker`):
- ✅ Private (authenticated) access only
- ✅ Only callable by Cloud Scheduler via service account

### Network Security
- Both services run in Google's managed VPC
- Outbound: HTTPS to sfmunicentral.com only
- Inbound: API is public, Checker is private

### Data Security
- GCS bucket is private (no public access)
- Cache contains only public transit status (no PII)
- Service account has minimal permissions

## Troubleshooting

### "Permission denied" errors
```bash
# Verify service account has bucket access
gsutil iam get gs://munimetro-cache

# Re-grant permissions
gsutil iam ch \
  serviceAccount:munimetro-api@PROJECT.iam.gserviceaccount.com:objectAdmin \
  gs://munimetro-cache
```

### Checker service fails
```bash
# View recent logs
gcloud run services logs read munimetro-checker \
    --region us-west1 \
    --limit 100

# Common issues:
# - ML model download timeout (increase timeout in deploy script)
# - GCS permission denied (check service account IAM)
# - Out of memory (increase --memory to 4Gi)
```

### Scheduler not triggering
```bash
# Check job status
gcloud scheduler jobs describe munimetro-status-check \
    --location us-west1

# View execution history
gcloud logging read 'resource.type="cloud_scheduler_job"' \
    --limit 20 \
    --format=json

# Manual trigger for testing
gcloud scheduler jobs run munimetro-status-check \
    --location us-west1
```

## Cleanup

### Delete everything
```bash
# Delete Cloud Run services
gcloud run services delete munimetro-api --region us-west1 --quiet
gcloud run services delete munimetro-checker --region us-west1 --quiet

# Delete scheduler job
gcloud scheduler jobs delete munimetro-status-check --location us-west1 --quiet

# Delete GCS bucket
gsutil rm -r gs://munimetro-cache

# Delete service account
gcloud iam service-accounts delete \
    munimetro-api@PROJECT.iam.gserviceaccount.com \
    --quiet
```

## Local Testing

Test the code locally before deploying:

```bash
# Set environment to simulate Cloud Run
export CLOUD_RUN=true
export GCS_BUCKET=munimetro-cache

# Test checker (requires gcloud auth)
python3 api/check_status.py --write-cache

# Test API (requires cache to exist)
python3 -m gunicorn api:app --bind 0.0.0.0:8000
```

## Updates

### Deploy new version
```bash
# Just re-run deploy script
./deploy/cloud/deploy-services.sh

# Cloud Run will:
# - Build new image
# - Deploy with zero downtime
# - Route traffic to new version
```

### Rollback
```bash
# List revisions
gcloud run revisions list --service munimetro-api --region us-west1

# Rollback to previous
gcloud run services update-traffic munimetro-api \
    --region us-west1 \
    --to-revisions REVISION-NAME=100
```

## Monitoring

### View metrics
```bash
# Cloud Console
open "https://console.cloud.google.com/run?project=munimetro"

# Key metrics:
# - Request count
# - Request latency
# - Error rate
# - Container instance count
```

### Set up alerts
```bash
# Create alert policy for errors
gcloud alpha monitoring policies create \
    --notification-channels=CHANNEL_ID \
    --display-name="MuniMetro API Errors" \
    --condition-display-name="Error rate > 5%" \
    ...
```

## Support

- Cloud Run docs: https://cloud.google.com/run/docs
- Cloud Scheduler docs: https://cloud.google.com/scheduler/docs
- Troubleshooting: Check logs with `gcloud run services logs read`
