#!/bin/bash
# Setup Cloud Scheduler to trigger status checker periodically

set -e  # Exit on error

# Configuration
PROJECT_ID="${GCP_PROJECT_ID:-munimetro}"
REGION="${GCP_REGION:-us-west1}"
CHECKER_JOB="munimetro-checker"
SCHEDULER_JOB_NAME="munimetro-status-check"
SCHEDULE="${SCHEDULE:-*/2 * * * *}"  # Every 2 minutes by default
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

# Verify the checker job exists
echo "[1/3] Verifying checker job exists..."
if ! gcloud run jobs describe "$CHECKER_JOB" \
    --region "$REGION" \
    --project "$PROJECT_ID" &> /dev/null; then
    echo "❌ Error: Checker job not found. Deploy services first:"
    echo "   ./deploy/cloud/deploy-services.sh"
    exit 1
fi

echo "✓ Checker job found: $CHECKER_JOB"
echo ""

# Delete existing scheduler job if it exists
echo "[2/3] Checking for existing scheduler job..."
if gcloud scheduler jobs describe "$SCHEDULER_JOB_NAME" \
    --location="$REGION" \
    --project="$PROJECT_ID" &> /dev/null; then
    echo "  - Deleting existing scheduler job..."
    gcloud scheduler jobs delete "$SCHEDULER_JOB_NAME" \
        --location="$REGION" \
        --project="$PROJECT_ID" \
        --quiet
    echo "✓ Existing scheduler job deleted"
else
    echo "  - No existing scheduler job found"
fi
echo ""

# Create new scheduler job to trigger Cloud Run Job
echo "[3/3] Creating scheduler job..."
# Cloud Run Jobs API endpoint
RUN_JOB_URL="https://${REGION}-run.googleapis.com/apis/run.googleapis.com/v1/namespaces/${PROJECT_ID}/jobs/${CHECKER_JOB}:run"

gcloud scheduler jobs create http "$SCHEDULER_JOB_NAME" \
    --location="$REGION" \
    --schedule="$SCHEDULE" \
    --uri="$RUN_JOB_URL" \
    --http-method=POST \
    --oauth-service-account-email="$SERVICE_ACCOUNT_EMAIL" \
    --max-retry-attempts=3 \
    --max-backoff=3600s \
    --min-backoff=5s \
    --description="Triggers MuniMetro status check every 2 minutes" \
    --project="$PROJECT_ID"

echo "✓ Scheduler job created: $SCHEDULER_JOB_NAME"
echo ""

# Trigger a test run
echo "Testing scheduler job..."
gcloud scheduler jobs run "$SCHEDULER_JOB_NAME" \
    --location="$REGION" \
    --project="$PROJECT_ID" || {
    echo "⚠️  Test run failed. This is normal on first setup."
    echo "   The job will run on schedule: $SCHEDULE"
}
echo ""

echo "=========================================="
echo "✓ Cloud Scheduler configured!"
echo "=========================================="
echo ""
echo "Scheduler job details:"
echo "  Name: $SCHEDULER_JOB_NAME"
echo "  Schedule: $SCHEDULE"
echo "  Target: Cloud Run Job '$CHECKER_JOB'"
echo ""
echo "Next status check: $(date -u -v+2M '+%Y-%m-%d %H:%M:%S UTC' 2>/dev/null || date -u -d '+2 minutes' '+%Y-%m-%d %H:%M:%S UTC' 2>/dev/null || echo 'in 2 minutes')"
echo ""
echo "View scheduler job status:"
echo "  gcloud scheduler jobs describe $SCHEDULER_JOB_NAME --location=$REGION"
echo ""
echo "Manual trigger options:"
echo "  Via scheduler: gcloud scheduler jobs run $SCHEDULER_JOB_NAME --location=$REGION"
echo "  Direct job:    gcloud run jobs execute $CHECKER_JOB --region=$REGION"
