#!/bin/bash
# Setup Cloud Scheduler to trigger status checker and analytics reports

set -e  # Exit on error

# Configuration
PROJECT_ID="${GCP_PROJECT_ID:-munimetro}"
REGION="${GCP_REGION:-us-west1}"
SERVICE_ACCOUNT_NAME="munimetro-api"
SERVICE_ACCOUNT_EMAIL="${SERVICE_ACCOUNT_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"

# Status checker job
CHECKER_JOB="munimetro-checker"
CHECKER_SCHEDULER="munimetro-status-check"
CHECKER_SCHEDULE="${CHECKER_SCHEDULE:-*/3 * * * *}"  # Every 3 minutes

# Analytics reports job
REPORTS_JOB="munimetro-reports"
REPORTS_SCHEDULER="munimetro-reports-periodic"
REPORTS_SCHEDULE="${REPORTS_SCHEDULE:-*/30 * * * *}"  # Every 30 minutes

echo "=========================================="
echo "MuniMetro Cloud Scheduler Setup"
echo "=========================================="
echo "Project: $PROJECT_ID"
echo "Region: $REGION"
echo ""
echo "Schedules:"
echo "  Status checker: $CHECKER_SCHEDULE (every 3 min)"
echo "  Analytics reports: $REPORTS_SCHEDULE (every 30 min)"
echo "=========================================="
echo ""

# Verify jobs exist
echo "[1/4] Verifying Cloud Run jobs exist..."
if ! gcloud run jobs describe "$CHECKER_JOB" \
    --region "$REGION" \
    --project "$PROJECT_ID" &> /dev/null; then
    echo "❌ Error: Checker job not found. Deploy services first:"
    echo "   ./deploy/cloud/deploy-services.sh"
    exit 1
fi
echo "  ✓ Checker job found: $CHECKER_JOB"

if ! gcloud run jobs describe "$REPORTS_JOB" \
    --region "$REGION" \
    --project "$PROJECT_ID" &> /dev/null; then
    echo "❌ Error: Reports job not found. Deploy services first:"
    echo "   ./deploy/cloud/deploy-services.sh"
    exit 1
fi
echo "  ✓ Reports job found: $REPORTS_JOB"
echo ""

# Delete existing scheduler jobs if they exist
echo "[2/4] Cleaning up existing scheduler jobs..."
for SCHEDULER_NAME in "$CHECKER_SCHEDULER" "$REPORTS_SCHEDULER"; do
    if gcloud scheduler jobs describe "$SCHEDULER_NAME" \
        --location="$REGION" \
        --project="$PROJECT_ID" &> /dev/null; then
        echo "  - Deleting $SCHEDULER_NAME..."
        gcloud scheduler jobs delete "$SCHEDULER_NAME" \
            --location="$REGION" \
            --project="$PROJECT_ID" \
            --quiet
    fi
done
echo "✓ Cleanup complete"
echo ""

# Create status checker scheduler
echo "[3/4] Creating status checker scheduler..."
CHECKER_URL="https://${REGION}-run.googleapis.com/apis/run.googleapis.com/v1/namespaces/${PROJECT_ID}/jobs/${CHECKER_JOB}:run"

gcloud scheduler jobs create http "$CHECKER_SCHEDULER" \
    --location="$REGION" \
    --schedule="$CHECKER_SCHEDULE" \
    --uri="$CHECKER_URL" \
    --http-method=POST \
    --oauth-service-account-email="$SERVICE_ACCOUNT_EMAIL" \
    --max-retry-attempts=3 \
    --max-backoff=3600s \
    --min-backoff=5s \
    --description="Triggers MuniMetro status check every 3 minutes" \
    --project="$PROJECT_ID"

echo "✓ Status checker scheduler created: $CHECKER_SCHEDULER"
echo ""

# Create analytics reports scheduler
echo "[4/4] Creating analytics reports scheduler..."
REPORTS_URL="https://${REGION}-run.googleapis.com/apis/run.googleapis.com/v1/namespaces/${PROJECT_ID}/jobs/${REPORTS_JOB}:run"

gcloud scheduler jobs create http "$REPORTS_SCHEDULER" \
    --location="$REGION" \
    --schedule="$REPORTS_SCHEDULE" \
    --uri="$REPORTS_URL" \
    --http-method=POST \
    --oauth-service-account-email="$SERVICE_ACCOUNT_EMAIL" \
    --max-retry-attempts=2 \
    --max-backoff=3600s \
    --min-backoff=60s \
    --description="Generates MuniMetro analytics reports every 30 minutes" \
    --project="$PROJECT_ID"

echo "✓ Analytics reports scheduler created: $REPORTS_SCHEDULER"
echo ""

# Trigger test runs
echo "Testing schedulers..."
echo "  Testing status checker..."
gcloud scheduler jobs run "$CHECKER_SCHEDULER" \
    --location="$REGION" \
    --project="$PROJECT_ID" 2>/dev/null || echo "  ⚠️  Test run pending (normal on first setup)"

echo "  Testing reports generator..."
gcloud scheduler jobs run "$REPORTS_SCHEDULER" \
    --location="$REGION" \
    --project="$PROJECT_ID" 2>/dev/null || echo "  ⚠️  Test run pending (normal on first setup)"
echo ""

echo "=========================================="
echo "✓ Cloud Schedulers configured!"
echo "=========================================="
echo ""
echo "Configured schedulers:"
echo ""
echo "  Status Checker:"
echo "    Name: $CHECKER_SCHEDULER"
echo "    Schedule: $CHECKER_SCHEDULE (every 3 min)"
echo "    Target: $CHECKER_JOB"
echo ""
echo "  Analytics Reports:"
echo "    Name: $REPORTS_SCHEDULER"
echo "    Schedule: $REPORTS_SCHEDULE (every 30 min)"
echo "    Target: $REPORTS_JOB"
echo ""
echo "View scheduler status:"
echo "  gcloud scheduler jobs list --location=$REGION"
echo ""
echo "Manual trigger options:"
echo "  gcloud scheduler jobs run $CHECKER_SCHEDULER --location=$REGION"
echo "  gcloud scheduler jobs run $REPORTS_SCHEDULER --location=$REGION"
