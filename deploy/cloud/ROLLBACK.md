# Cloud Run Deployment Rollback Guide

This guide explains how to quickly rollback a bad deployment to minimize downtime.

## Quick Start

If you need to rollback immediately after a bad deployment:

```bash
# Rollback both API and checker job
./deploy/cloud/rollback.sh --yes

# Or run interactively with confirmation prompts
./deploy/cloud/rollback.sh
```

## Usage Options

### Full Rollback (Default)
Rollback both the API service and checker job to their previous versions:
```bash
./deploy/cloud/rollback.sh
```

### Rollback API Service Only
If only the API service is broken:
```bash
./deploy/cloud/rollback.sh --api-only
```

### Rollback Checker Job Only
If only the checker job is broken:
```bash
./deploy/cloud/rollback.sh --job-only
```

### Skip Confirmation
For automated rollbacks or when you're certain:
```bash
./deploy/cloud/rollback.sh --yes
```

## How It Works

### API Service Rollback
- Cloud Run maintains all previous revisions
- The script routes 100% of traffic back to the most recent working revision
- Rollback happens **instantly** with zero downtime
- No new build or deployment required

### Checker Job Rollback
- The script finds the previous Docker image from Container Registry
- Updates the job to use the previous image
- Next scheduled run will use the rolled-back version

## What Gets Rolled Back

| Component | What Happens | Downtime |
|-----------|-------------|----------|
| API Service | Traffic routes to previous revision | **None** (instant) |
| Checker Job | Updated to use previous image | None (takes effect on next run) |

## When to Use Rollback

**Use rollback when:**
- New deployment returns errors (500, 503, etc.)
- API is returning incorrect data
- Checker job is failing to update cache
- Performance degradation after deployment

**Don't use rollback when:**
- Issue is with external dependencies (SFMTA website, GCS, etc.)
- Issue existed before the deployment
- You need to fix data/configuration (not code)

## Verification After Rollback

### 1. Check API Service
```bash
# Test the API endpoint
curl https://munimetro-api-438243686292.us-west1.run.app/status

# Verify the revision
gcloud run services describe munimetro-api \
  --region=us-west1 \
  --format='value(status.traffic[0].revisionName)'
```

### 2. Check Checker Job
```bash
# Check job status
gcloud run jobs describe munimetro-checker \
  --region=us-west1 \
  --format='value(metadata.name,spec.template.spec.containers[0].image)'

# Optionally run the job manually to test
gcloud run jobs execute munimetro-checker --region=us-west1
```

### 3. Monitor Logs
```bash
# API service logs
gcloud run services logs read munimetro-api \
  --region=us-west1 \
  --limit=50

# Checker job logs (replace with actual execution name)
gcloud logging read "resource.type=cloud_run_job AND \
  resource.labels.job_name=munimetro-checker" \
  --limit=50 \
  --format='value(textPayload)'
```

## Rollback to Specific Revision

If you need to rollback to a specific revision (not just the previous one):

### For API Service
```bash
# List available revisions
gcloud run revisions list \
  --service=munimetro-api \
  --region=us-west1 \
  --format='table(name,traffic,creation_timestamp,status)'

# Rollback to specific revision
gcloud run services update-traffic munimetro-api \
  --region=us-west1 \
  --to-revisions=munimetro-api-00020-xyz=100
```

### For Checker Job
```bash
# List available images
gcloud container images list-tags gcr.io/munimetro/munimetro \
  --format='table(digest.slice(0:12),timestamp)' \
  --limit=10

# Update job to specific image
gcloud run jobs update munimetro-checker \
  --region=us-west1 \
  --image=gcr.io/munimetro/munimetro@sha256:abc123...
```

## Troubleshooting

### Error: "No previous revision found"
**Cause:** This is the first deployment or all old revisions were deleted.

**Solution:**
- You cannot rollback automatically
- Deploy a known-good version manually using `./deploy/cloud/deploy-services.sh`

### Error: "No previous image found"
**Cause:** Container Registry doesn't have previous images.

**Solution:**
- Check Container Registry: `gcloud container images list-tags gcr.io/munimetro/munimetro`
- If no images exist, you must deploy from source

### Rollback Succeeded but API Still Broken
**Possible causes:**
1. **Cache is stale**: Wait 3 minutes for next scheduled cache update, or run job manually
2. **Issue is external**: Check SFMTA website or GCS bucket access
3. **Issue exists in multiple revisions**: Need to rollback further or fix the code

**Investigation steps:**
```bash
# Check cache age
curl https://munimetro-api-438243686292.us-west1.run.app/status | \
  jq '.cache_age'

# Check if fallback mode is enabled
gcloud run services describe munimetro-api \
  --region=us-west1 \
  --format='value(spec.template.spec.containers[0].env[?(@.name=="ENABLE_FALLBACK")].value)'

# Check GCS bucket
gsutil ls -l gs://munimetro-cache/latest_status.json
```

## Prevention: Testing Before Deployment

To avoid needing rollbacks, test deployments before they go live:

### 1. Local Testing
```bash
# Run API locally with production model
cd api
gunicorn api:app --bind 0.0.0.0:8000

# Test in another terminal
curl http://localhost:8000/status
```

### 2. Canary Deployment (Advanced)
Deploy new revision with limited traffic:
```bash
# Deploy new revision without serving traffic
gcloud run deploy munimetro-api \
  --no-traffic \
  --tag=canary \
  # ... other flags ...

# Send 10% of traffic to canary
gcloud run services update-traffic munimetro-api \
  --to-revisions=munimetro-api-00023-xyz=10,munimetro-api-00022-abc=90

# If canary looks good, send 100% traffic
gcloud run services update-traffic munimetro-api \
  --to-revisions=munimetro-api-00023-xyz=100
```

## Emergency Contact

If rollback doesn't resolve the issue:

1. **Check Cloud Run status**: https://status.cloud.google.com/
2. **Review deployment validation**: Ensure `deploy-services.sh` checks passed
3. **Manual intervention**: May need to update environment variables or fix GCS permissions

## Related Documentation

- [deploy-services.sh](./deploy-services.sh) - Main deployment script
- [verify.sh](./verify.sh) - Deployment verification script
- [Cloud Run Rollback Docs](https://cloud.google.com/run/docs/rollouts-rollbacks-traffic-migration)
