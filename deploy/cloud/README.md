# Cloud Run Deployment Guide

Scripts for deploying MuniMetro to Google Cloud Run.

See [../../CONFIGURATION.md](../../CONFIGURATION.md) for actual deployment configuration values.

## Architecture

```
Cloud Scheduler (every 2 min)
  ↓ triggers
Checker (Cloud Run Job)
  ↓ downloads image + predicts status
  ↓ writes JSON + exits
Cloud Storage (cache file)
  ↑ reads JSON
API (Cloud Run Service)
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

2. **GCP Project** created with billing enabled
   ```bash
   # Create project
   gcloud projects create PROJECT_ID --name="PROJECT_NAME"

   # Set billing account
   gcloud billing accounts list
   gcloud billing projects link PROJECT_ID --billing-account=BILLING_ACCOUNT_ID
   ```

3. **Environment variables** (optional - override defaults)
   ```bash
   export GCP_PROJECT_ID="PROJECT_ID"
   export GCP_REGION="REGION"
   export GCS_BUCKET="BUCKET_NAME"
   ```

See [../../CONFIGURATION.md](../../CONFIGURATION.md) for the actual values used in this deployment.

## Deployment Steps

### 1. Setup Infrastructure (One-time)

```bash
./deploy/cloud/setup-infrastructure.sh
```

This script:
- Enables required GCP APIs
- Creates Cloud Storage bucket (private)
- Creates service account with minimal permissions
- Sets up App Engine (required for Cloud Scheduler)

### 2. Deploy Services

```bash
./deploy/cloud/deploy-services.sh
```

This script:
- Builds container image using Cloud Build
- Deploys API service (public Cloud Run Service, serves frontend + API)
- Deploys checker job (Cloud Run Job, updates status when triggered)

First deployment takes approximately 10-15 minutes (downloading ML model layers).

### 3. Setup Scheduler

```bash
./deploy/cloud/setup-scheduler.sh
```

This script:
- Creates Cloud Scheduler job
- Configures 2-minute interval
- Runs test execution

## Verification

### Check API is running
```bash
# Get API URL
API_URL=$(gcloud run services describe API_SERVICE_NAME \
    --region REGION \
    --format 'value(status.url)')

# Test health endpoint
curl $API_URL/health

# Test status endpoint
curl $API_URL/status
```

### Trigger manual status update
```bash
# Option 1: Trigger via scheduler
gcloud scheduler jobs run SCHEDULER_JOB_NAME --location=REGION

# Option 2: Run job directly (faster for testing)
gcloud run jobs execute CHECKER_JOB_NAME --region=REGION

# Wait for job to complete, then check API
sleep 30
curl $API_URL/status
```

### View logs
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

## Cost Estimate

Typical usage costs approximately $1.17/month:

| Service | Usage | Cost |
|---------|-------|------|
| Cloud Run Service (API) | ~1000 requests/day | $0 (free tier) |
| Cloud Run Jobs (Checker) | 21,600 executions/month @ ~10s each | $1.04 |
| Cloud Storage | 1KB file, 75K reads/month | $0.016 |
| Cloud Scheduler | 1 job | $0.10 |
| **Total** | | **~$1.17/month** |

Cloud Run Jobs are more cost-effective than Services for scheduled tasks since execution time is billed without idle costs.

At 100K API requests/day, estimated cost increases to approximately $2-3/month.

See [../../CONFIGURATION.md](../../CONFIGURATION.md) for detailed cost breakdown.

## Configuration

### Change update frequency
```bash
# Edit schedule (cron format)
export SCHEDULE="*/10 * * * *"  # Every 10 minutes
./deploy/cloud/setup-scheduler.sh
```

### Change bucket name
```bash
export GCS_BUCKET="BUCKET_NAME"
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

**Service Account**:
- Storage Object Admin on cache bucket (read/write cache)
- Run Invoker on checker job
- No additional permissions

**API Service**:
- Public (unauthenticated) access
- Serves frontend and `/status` endpoint

**Checker Job**:
- Only executable by authorized service accounts
- Triggered by Cloud Scheduler via OAuth authentication
- No HTTP endpoint (batch execution only)

### Network Security
- API Service and Checker Job run in Google's managed VPC
- Outbound: HTTPS to sfmunicentral.com only
- Inbound: API is public, Checker Job has no inbound (batch execution only)

### Data Security
- GCS bucket is private (no public access)
- Cache contains only public transit status (no PII)
- Service account has minimal permissions

See [../../CONFIGURATION.md](../../CONFIGURATION.md) for actual service account and bucket details.

## Troubleshooting

### Permission denied errors
```bash
# Verify service account has bucket access
gsutil iam get gs://BUCKET_NAME

# Re-grant permissions
gsutil iam ch \
  serviceAccount:SERVICE_ACCOUNT_EMAIL:objectAdmin \
  gs://BUCKET_NAME
```

### Checker job fails
```bash
# View recent job executions
gcloud run jobs executions list \
    --job=CHECKER_JOB_NAME \
    --region=REGION \
    --limit=10

# View logs for failed execution
gcloud logging read \
    'resource.type="cloud_run_job" AND resource.labels.job_name="CHECKER_JOB_NAME"' \
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
gcloud scheduler jobs describe SCHEDULER_JOB_NAME \
    --location REGION

# View execution history
gcloud logging read 'resource.type="cloud_scheduler_job"' \
    --limit 20 \
    --format=json

# Manual trigger for testing
gcloud scheduler jobs run SCHEDULER_JOB_NAME \
    --location REGION
```

## Cleanup

### Delete everything
```bash
# Delete Cloud Run service
gcloud run services delete API_SERVICE_NAME --region REGION --quiet

# Delete Cloud Run job
gcloud run jobs delete CHECKER_JOB_NAME --region REGION --quiet

# Delete scheduler job
gcloud scheduler jobs delete SCHEDULER_JOB_NAME --location REGION --quiet

# Delete GCS bucket
gsutil rm -r gs://BUCKET_NAME

# Delete service account
gcloud iam service-accounts delete SERVICE_ACCOUNT_EMAIL --quiet
```

## Local Testing

Test the code locally before deploying:

```bash
# Set environment to simulate Cloud Run
export CLOUD_RUN=true
export GCS_BUCKET=BUCKET_NAME

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
gcloud run revisions list --service API_SERVICE_NAME --region REGION

# Rollback to previous
gcloud run services update-traffic API_SERVICE_NAME \
    --region REGION \
    --to-revisions REVISION_NAME=100
```

## Monitoring

### View metrics
```bash
# Cloud Console
open "https://console.cloud.google.com/run?project=PROJECT_ID"

# Key metrics:
# - Request count
# - Request latency
# - Error rate
# - Container instance count
```

