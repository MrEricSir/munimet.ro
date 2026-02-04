# Cloud Run Deployment Guide

Scripts for deploying MuniMetro to Google Cloud Run.

See [../../CONFIGURATION.md](../../CONFIGURATION.md) for actual deployment configuration values.

## Architecture

```
Cloud Scheduler (every 3 min)          Cloud Scheduler (daily midnight)
  ↓ triggers                             ↓ triggers
Checker Job                            Reports Job
  ↓ downloads image                      ↓ generates analytics
  ↓ detects status                       ↓ caches reports
  ↓ writes JSON                          ↓ exits
Cloud Storage (cache)                  Cloud Storage (reports cache)
  ↑ reads                                ↑ reads
API Service  ←───────────────────────────┘
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
- Auto-detects the currently deployed model version (or prompts on first deploy)
- Builds container image using Cloud Build (lightweight, no model baked in)
- Deploys API service and checker job
- Services download the model from GCS at startup

**First deploy only**: Set `MODEL_VERSION` since there's no existing deployment:
```bash
export MODEL_VERSION=20251223_224331
./deploy/cloud/deploy-services.sh
```

**Subsequent deploys**: The script automatically uses the currently deployed model version.

### 3. Setup Schedulers

```bash
./deploy/cloud/setup-scheduler.sh
```

This script creates two Cloud Scheduler jobs:
- **Status checker** - Runs every 3 minutes, triggers status detection
- **Analytics reports** - Runs daily at midnight UTC, generates cached reports

The analytics reports job is isolated from user requests to prevent report generation issues from affecting the main API.

### 4. Setup Monitoring (Optional but Recommended)

```bash
export ALERT_EMAIL="your-email@example.com"  # Set your email first
./deploy/cloud/setup-monitoring.sh
```

This script:
- Creates uptime check for API `/health` endpoint (checks every 5 minutes)
- Creates alert policy for API downtime (alerts if down for 5+ minutes)
- Creates alert policy for job failures (alerts if checker job fails repeatedly)
- Creates alert policy for high error rates (alerts if API error rate >10%)
- Sends email notifications to `ALERT_EMAIL`

**Cost**: FREE (within Google Cloud Monitoring free tier limits)

**Note**: You must run `setup-infrastructure.sh` with `ALERT_EMAIL` set first to create the notification channel.

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

Typical usage costs approximately $1.20/month:

| Service | Usage | Cost |
|---------|-------|------|
| Cloud Run Service (API) | ~1000 requests/day | $0 (free tier) |
| Cloud Run Jobs (Checker) | 14,400 executions/month @ ~10s each | $0.70 |
| Cloud Run Jobs (Reports) | 30 executions/month @ ~5s each | $0.01 |
| Cloud Storage | Cache files, ~100K reads/month | $0.02 |
| Cloud Scheduler | 2 jobs | $0.20 |
| **Total** | | **~$1.00/month** |

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

## Social Media Notifications (Optional)

The checker job can post status updates to Bluesky and Mastodon when the system status changes. To enable this:

### 1. Setup credentials locally (generates app passwords/tokens)

```bash
python scripts/setup/setup-credentials.py --cloud
```

This will prompt you for:
- **Bluesky**: Handle (e.g., `munimetro.bsky.social`) and app password
- **Mastodon**: Instance URL (e.g., `https://mastodon.social`) and access token

Credentials are stored in Google Cloud Secret Manager.

### 2. Create Bluesky app password

1. Go to Bluesky Settings → App Passwords
2. Create a new app password
3. Copy the generated password

### 3. Create Mastodon access token

1. Go to your Mastodon instance → Preferences → Development
2. Create a new application with `write:statuses` scope
3. Copy the access token

### 4. Redeploy to pick up secrets

```bash
./deploy/cloud/deploy-services.sh
```

The deploy script automatically detects secrets in Secret Manager and mounts them.

### Secrets used

| Secret Name | Description |
|-------------|-------------|
| `BLUESKY_HANDLE` | Bluesky account handle |
| `BLUESKY_APP_PASSWORD` | Bluesky app password |
| `MASTODON_INSTANCE` | Mastodon instance URL |
| `MASTODON_ACCESS_TOKEN` | Mastodon access token |

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

## Data Persistence

### Analytics Database

The analytics system uses SQLite stored in GCS for persistence across container restarts.

**Storage location**: `gs://munimetro-cache/analytics/analytics.db`

**How it works**:
1. Job starts → downloads `analytics.db` from GCS to local filesystem
2. Status check runs → data logged to local SQLite
3. Job ends → uploads local `analytics.db` back to GCS

**Safety features**:
- **Size check**: Won't overwrite a larger GCS backup with a smaller local file (prevents data loss from failed restores)
- **GCS versioning**: Enabled on the bucket - old versions are retained when files are overwritten

### GCS Versioning

Versioning is enabled on `gs://munimetro-cache` to protect against accidental data loss.

**List all versions of analytics DB**:
```bash
gsutil ls -la gs://munimetro-cache/analytics/analytics.db
```

**Restore a previous version** (if data loss occurs):
```bash
# Find the generation number from the list command above
gsutil cp "gs://munimetro-cache/analytics/analytics.db#<GENERATION_NUMBER>" \
    gs://munimetro-cache/analytics/analytics.db
```

**View version metadata**:
```bash
gsutil stat "gs://munimetro-cache/analytics/analytics.db#<GENERATION_NUMBER>"
```

### Cache Files

| File | Purpose | Location |
|------|---------|----------|
| `analytics.db` | Status check history, delay incidents | `gs://munimetro-cache/analytics/` |
| `analytics_report_*d.json` | Pre-generated analytics reports (1d, 7d, 30d, 365d) | `gs://munimetro-cache/analytics/` |
| `latest_status.json` | Current status cache (for API responses) | `gs://munimetro-cache/` |
| `latest_image.jpg` | Most recent MIMIC screenshot | `gs://munimetro-cache/` |

### Lifecycle Management (Optional)

To automatically delete old versions after 30 days (reduces storage costs):

```bash
cat > /tmp/lifecycle.json << 'EOF'
{
  "rule": [
    {
      "action": {"type": "Delete"},
      "condition": {
        "numNewerVersions": 10,
        "isLive": false
      }
    }
  ]
}
EOF

gsutil lifecycle set /tmp/lifecycle.json gs://munimetro-cache
```

This keeps the 10 most recent versions of each file and deletes older non-current versions.

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

### Analytics data missing or incomplete

**Symptoms**: Analytics page shows 0 red events, or very few total checks

**Diagnostic steps**:

1. **Check job execution logs** for GCS restore/backup issues:
   ```bash
   gcloud logging read "resource.type=cloud_run_job AND resource.labels.job_name=munimetro-checker" \
       --limit=50 --format="table(timestamp,textPayload)"
   ```

   Look for:
   - `Restored analytics DB from GCS (X bytes)` - successful restore
   - `WARNING: Local DB smaller than GCS backup` - backup skipped to prevent data loss
   - `ERROR` messages during restore/backup

2. **Check database size in GCS**:
   ```bash
   gsutil ls -l gs://munimetro-cache/analytics/analytics.db
   ```

3. **Check scheduler is running consistently**:
   ```bash
   gcloud scheduler jobs describe munimetro-status-check --location=us-west1
   ```

   Verify `lastAttemptTime` is recent and `state` is `ENABLED`.

4. **View database version history** (if data was lost):
   ```bash
   gsutil ls -la gs://munimetro-cache/analytics/analytics.db
   ```

5. **Restore from a previous version** if needed:
   ```bash
   gsutil cp "gs://munimetro-cache/analytics/analytics.db#<GENERATION>" \
       gs://munimetro-cache/analytics/analytics.db
   ```

**Common causes**:
- Scheduler paused or misconfigured
- GCS restore fails silently, creating empty DB that overwrites backup
- Container restarts during overnight hours when no traffic keeps it warm

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

See [ROLLBACk.md](ROLLBACK.md) for how to run a rollback via the script or manually.

Quick guide:
```bash
# List revisions
gcloud run revisions list --service API_SERVICE_NAME --region REGION

# Rollback to previous
gcloud run services update-traffic API_SERVICE_NAME \
    --region REGION \
    --to-revisions REVISION_NAME=100
```

## Model Management

Models are stored in GCS at `gs://munimetro-annex/models/snapshots/<version>/`. Services download the model at startup based on the `MODEL_VERSION` environment variable.

### Commands

```bash
# List all available model versions with metrics
python3 scripts/manage-models.py list

# Show currently deployed model
python3 scripts/manage-models.py current

# View detailed metrics for a specific model
python3 scripts/manage-models.py info 20251223_224331

# Switch to a different model (updates env var, no rebuild needed)
python3 scripts/manage-models.py switch 20251223_224331
```

### Example output

```
$ python3 scripts/manage-models.py list
Available model snapshots:
======================================================================
  20251230_142847  acc:95.9%  red-R:96.6%  yel-P:81.7%
  20251227_151141  acc:95.1%  red-R:97.7%  yel-P:88.9%
  20251225_021726  acc:95.2%  red-R:94.3%  yel-P:87.2%
  20251223_224331  acc:94.2%  red-R:98.8%  yel-P:81.1% <-- DEPLOYED

Use 'switch <version>' to deploy a different model.
```

### Key metrics

- **acc**: Overall test accuracy
- **red-R**: Red recall - ability to detect red (system down) states
- **yel-P**: Yellow precision - avoids false yellow alarms

For overnight issues (system shutdown), prioritize high red recall.

### Switching vs Redeploying

- **`manage-models.py switch`**: Updates the `MODEL_VERSION` env var on running services. Fast (~30s), no rebuild. Model downloads on next cold start.
- **`deploy-services.sh`**: Full redeploy with new code. Use when code changes, not just model changes.

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

