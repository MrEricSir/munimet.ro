#!/bin/bash
# Setup Cloud Scheduler to trigger status checker periodically

set -e  # Exit on error

# Configuration
PROJECT_ID="${GCP_PROJECT_ID:-munimetro}"
REGION="${GCP_REGION:-us-west1}"
CHECKER_SERVICE="munimetro-checker"
JOB_NAME="munimetro-status-check"
SCHEDULE="${SCHEDULE:-*/5 * * * *}"  # Every 5 minutes by default
SERVICE_ACCOUNT_NAME="munimetro-api"
SERVICE_ACCOUNT_EMAIL="${SERVICE_ACCOUNT_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"

echo "=========================================="
echo "MuniMetro Cloud Scheduler Setup"
echo "=========================================="
echo "Project: $PROJECT_ID"
echo "Region: $REGION"
echo "Schedule: $SCHEDULE (cron format)"
echo "=========================================="
echo ""

# Get the checker service URL
echo "[1/3] Getting checker service URL..."
CHECKER_URL=$(gcloud run services describe "$CHECKER_SERVICE" \
    --platform managed \
    --region "$REGION" \
    --project "$PROJECT_ID" \
    --format 'value(status.url)')

if [ -z "$CHECKER_URL" ]; then
    echo "❌ Error: Checker service not found. Deploy services first:"
    echo "   ./deploy/cloud/deploy-services.sh"
    exit 1
fi

echo "✓ Checker service: $CHECKER_URL"
echo ""

# Delete existing job if it exists
echo "[2/3] Checking for existing scheduler job..."
if gcloud scheduler jobs describe "$JOB_NAME" \
    --location="$REGION" \
    --project="$PROJECT_ID" &> /dev/null; then
    echo "  - Deleting existing job..."
    gcloud scheduler jobs delete "$JOB_NAME" \
        --location="$REGION" \
        --project="$PROJECT_ID" \
        --quiet
    echo "✓ Existing job deleted"
else
    echo "  - No existing job found"
fi
echo ""

# Create new scheduler job
echo "[3/3] Creating scheduler job..."
gcloud scheduler jobs create http "$JOB_NAME" \
    --location="$REGION" \
    --schedule="$SCHEDULE" \
    --uri="$CHECKER_URL" \
    --http-method=GET \
    --oidc-service-account-email="$SERVICE_ACCOUNT_EMAIL" \
    --oidc-token-audience="$CHECKER_URL" \
    --max-retry-attempts=3 \
    --max-backoff=3600s \
    --min-backoff=5s \
    --description="Triggers MuniMetro status check every 5 minutes" \
    --project="$PROJECT_ID"

echo "✓ Scheduler job created: $JOB_NAME"
echo ""

# Trigger a test run
echo "Testing scheduler job..."
gcloud scheduler jobs run "$JOB_NAME" \
    --location="$REGION" \
    --project="$PROJECT_ID" || {
    echo "⚠️  Test run failed. This is OK on first setup."
    echo "   The job will run on schedule: $SCHEDULE"
}
echo ""

echo "=========================================="
echo "✓ Cloud Scheduler configured!"
echo "=========================================="
echo ""
echo "Job details:"
echo "  Name: $JOB_NAME"
echo "  Schedule: $SCHEDULE"
echo "  Target: $CHECKER_URL"
echo ""
echo "Next status check: $(date -u -v+5M '+%Y-%m-%d %H:%M:%S UTC' 2>/dev/null || date -u -d '+5 minutes' '+%Y-%m-%d %H:%M:%S UTC' 2>/dev/null || echo 'in 5 minutes')"
echo ""
echo "View job status:"
echo "  gcloud scheduler jobs describe $JOB_NAME --location=$REGION"
echo ""
echo "Manual trigger:"
echo "  gcloud scheduler jobs run $JOB_NAME --location=$REGION"
