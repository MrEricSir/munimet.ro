# Cloud Run Deployment Guide

This directory contains scripts to deploy MuniMetro to Google Cloud Run.

## Architecture

```
Cloud Scheduler (every 5 min)
  ↓ triggers
munimetro-checker (Cloud Run Job)
  ↓ downloads image + predicts status
  ↓ writes JSON + exits
Cloud Storage (gs://munimetro-cache/latest_status.json)
  ↑ reads JSON
munimetro-api (Cloud Run Service)
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
- ✅ Deploys `munimetro-api` (public Cloud Run Service, serves frontend + API)
- ✅ Deploys `munimetro-checker` (Cloud Run Job, updates status when triggered)

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
# Option 1: Trigger via scheduler
gcloud scheduler jobs run munimetro-status-check --location=us-west1

# Option 2: Run job directly (faster for testing)
gcloud run jobs execute munimetro-checker --region=us-west1

# Wait 30 seconds for job to complete, then check API
sleep 30
curl $API_URL/status
```

### View logs
```bash
# API logs
gcloud run services logs read munimetro-api --region us-west1 --limit 50

# Checker job logs
gcloud logging read 'resource.type="cloud_run_job" AND resource.labels.job_name="munimetro-checker"' --limit 50

# Job executions
gcloud run jobs executions list --job=munimetro-checker --region=us-west1

# Scheduler logs
gcloud logging read 'resource.type="cloud_scheduler_job"' --limit 20
```

## Cost Estimate

**Expected monthly cost: ~$0.10-0.15**

| Service | Usage | Cost |
|---------|-------|------|
| Cloud Run Service (API) | ~1000 requests/day | $0 (free tier) |
| Cloud Run Jobs (Checker) | 8,640 executions/month @ ~10s each | $0.01 (very minimal) |
| Cloud Storage | 1KB file, 30K reads/month | $0.016 |
| Cloud Scheduler | 1 job | $0.10 |
| **Total** | | **~$0.13/month** |

*Cloud Run Jobs are cheaper than Services for scheduled tasks since you only pay for execution time (no idle costs).*

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

**Checker Job** (`munimetro-checker`):
- ✅ Only executable by authorized service accounts
- ✅ Triggered by Cloud Scheduler via OAuth authentication
- ✅ No HTTP endpoint (runs as batch job)

### Network Security
- API Service and Checker Job run in Google's managed VPC
- Outbound: HTTPS to sfmunicentral.com only
- Inbound: API is public, Checker Job has no inbound (batch execution only)

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

### Checker job fails
```bash
# View recent job executions
gcloud run jobs executions list \
    --job=munimetro-checker \
    --region=us-west1 \
    --limit=10

# View logs for failed execution
gcloud logging read \
    'resource.type="cloud_run_job" AND resource.labels.job_name="munimetro-checker"' \
    --limit=100 \
    --format=json

# Common issues:
# - ML model download timeout (increase --task-timeout in deploy script)
# - GCS permission denied (check service account IAM)
# - Out of memory (increase --memory to 4Gi in deploy script)
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
# Delete Cloud Run service
gcloud run services delete munimetro-api --region us-west1 --quiet

# Delete Cloud Run job
gcloud run jobs delete munimetro-checker --region us-west1 --quiet

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

# Test checker job script (requires gcloud auth)
python3 -m api.check_status_job

# Or use the local deployment scripts
./deploy/local/setup.sh   # First time only
./deploy/local/start.sh   # Start cache writer + API
./deploy/local/verify.sh  # Check status
./deploy/local/stop.sh    # Stop services
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
