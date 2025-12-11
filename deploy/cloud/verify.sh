#!/bin/bash
# Verify Cloud Run deployment status and health

set -e

# Configuration
PROJECT_ID="${GCP_PROJECT_ID:-munimetro}"
REGION="${GCP_REGION:-us-west1}"
API_SERVICE="munimetro-api"
CHECKER_JOB="munimetro-checker"
SCHEDULER_JOB_NAME="munimetro-status-check"
BUCKET_NAME="${GCS_BUCKET:-munimetro-cache}"

echo "=========================================="
echo "MuniMetro Cloud Deployment Verification"
echo "=========================================="
echo "Project: $PROJECT_ID"
echo "Region: $REGION"
echo "=========================================="
echo ""

# Check if gcloud is authenticated
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" &> /dev/null; then
    echo "❌ Not authenticated with gcloud"
    echo "   Run: gcloud auth login"
    exit 1
fi

# Set project
gcloud config set project "$PROJECT_ID" --quiet

echo "[1/6] Checking GCS Bucket..."
if gsutil ls -b "gs://$BUCKET_NAME" &> /dev/null; then
    echo "✓ Bucket exists: gs://$BUCKET_NAME"

    # Check if cache file exists
    if gsutil ls "gs://$BUCKET_NAME/latest_status.json" &> /dev/null; then
        echo "✓ Cache file exists"

        # Show cache age
        UPDATED=$(gsutil stat "gs://$BUCKET_NAME/latest_status.json" | grep 'Update time:' | cut -d: -f2-)
        echo "  Last updated: $UPDATED"
    else
        echo "⚠️  Cache file not found (normal on first deployment)"
    fi
else
    echo "❌ Bucket not found: gs://$BUCKET_NAME"
    echo "   Run: ./setup-infrastructure.sh"
fi
echo ""

echo "[2/6] Checking API Service..."
if gcloud run services describe "$API_SERVICE" \
    --region "$REGION" --project "$PROJECT_ID" &> /dev/null; then

    API_URL=$(gcloud run services describe "$API_SERVICE" \
        --region "$REGION" --project "$PROJECT_ID" \
        --format 'value(status.url)')

    STATUS=$(gcloud run services describe "$API_SERVICE" \
        --region "$REGION" --project "$PROJECT_ID" \
        --format 'value(status.conditions[0].status)')

    echo "✓ Service deployed: $API_SERVICE"
    echo "  URL: $API_URL"
    echo "  Status: $STATUS"

    # Test health endpoint
    echo "  Testing /health..."
    if curl -sf "$API_URL/health" > /dev/null; then
        echo "  ✓ Health check passed"
    else
        echo "  ❌ Health check failed"
    fi

    # Test status endpoint
    echo "  Testing /status..."
    if curl -sf "$API_URL/status" > /dev/null; then
        echo "  ✓ Status endpoint responding"
    else
        echo "  ⚠️  Status endpoint returned error (may be empty cache)"
    fi
else
    echo "❌ API service not deployed"
    echo "   Run: ./deploy-services.sh"
fi
echo ""

echo "[3/6] Checking Checker Job..."
if gcloud run jobs describe "$CHECKER_JOB" \
    --region "$REGION" --project "$PROJECT_ID" &> /dev/null; then

    IMAGE=$(gcloud run jobs describe "$CHECKER_JOB" \
        --region "$REGION" --project "$PROJECT_ID" \
        --format 'value(spec.template.spec.containers[0].image)')

    echo "✓ Job deployed: $CHECKER_JOB"
    echo "  Image: $IMAGE"

    # Check last execution
    echo "  Checking recent executions..."
    LAST_EXECUTION=$(gcloud run jobs executions list \
        --job="$CHECKER_JOB" \
        --region="$REGION" \
        --project="$PROJECT_ID" \
        --limit=1 \
        --format='value(metadata.name)' 2>/dev/null)

    if [ -n "$LAST_EXECUTION" ]; then
        EXEC_STATUS=$(gcloud run jobs executions describe "$LAST_EXECUTION" \
            --region="$REGION" \
            --project="$PROJECT_ID" \
            --format='value(status.conditions[0].type)' 2>/dev/null)
        echo "  Last execution: $LAST_EXECUTION"
        echo "  Status: $EXEC_STATUS"
    else
        echo "  ⚠️  No executions yet (run manually or wait for scheduler)"
    fi
else
    echo "❌ Checker job not deployed"
    echo "   Run: ./deploy/cloud/deploy-services.sh"
fi
echo ""

echo "[4/6] Checking Cloud Scheduler..."
if gcloud scheduler jobs describe "$SCHEDULER_JOB_NAME" \
    --location="$REGION" --project="$PROJECT_ID" &> /dev/null; then

    SCHEDULE=$(gcloud scheduler jobs describe "$SCHEDULER_JOB_NAME" \
        --location="$REGION" --project="$PROJECT_ID" \
        --format 'value(schedule)')

    STATE=$(gcloud scheduler jobs describe "$SCHEDULER_JOB_NAME" \
        --location="$REGION" --project="$PROJECT_ID" \
        --format 'value(state)')

    echo "✓ Scheduler job configured: $SCHEDULER_JOB_NAME"
    echo "  Schedule: $SCHEDULE"
    echo "  State: $STATE"

    # Get last run time
    echo "  Checking recent executions..."
    LAST_RUN=$(gcloud logging read \
        "resource.type=cloud_scheduler_job AND resource.labels.job_id=$SCHEDULER_JOB_NAME" \
        --limit 1 --format 'value(timestamp)' --project "$PROJECT_ID" 2>/dev/null)

    if [ -n "$LAST_RUN" ]; then
        echo "  Last execution: $LAST_RUN"
    else
        echo "  ⚠️  No recent executions found"
    fi
else
    echo "❌ Scheduler job not configured"
    echo "   Run: ./deploy/cloud/setup-scheduler.sh"
fi
echo ""

echo "[5/6] Checking Service Account..."
SERVICE_ACCOUNT_EMAIL="munimetro-api@${PROJECT_ID}.iam.gserviceaccount.com"
if gcloud iam service-accounts describe "$SERVICE_ACCOUNT_EMAIL" \
    --project "$PROJECT_ID" &> /dev/null; then
    echo "✓ Service account exists: $SERVICE_ACCOUNT_EMAIL"

    # Check bucket permissions
    echo "  Checking bucket permissions..."
    PERMS=$(gsutil iam get "gs://$BUCKET_NAME" | grep "$SERVICE_ACCOUNT_EMAIL" || echo "")
    if [ -n "$PERMS" ]; then
        echo "  ✓ Has bucket access"
    else
        echo "  ⚠️  Missing bucket permissions"
    fi
else
    echo "❌ Service account not found"
    echo "   Run: ./setup-infrastructure.sh"
fi
echo ""

echo "[6/6] Recent Logs (last 10 entries)..."
echo ""
echo "--- API Service ---"
gcloud run services logs read "$API_SERVICE" \
    --region "$REGION" --project "$PROJECT_ID" \
    --limit 5 --format 'value(textPayload)' 2>/dev/null | head -5 || echo "No logs available"

echo ""
echo "--- Checker Job ---"
gcloud logging read \
    "resource.type=cloud_run_job AND resource.labels.job_name=$CHECKER_JOB" \
    --limit 10 --project "$PROJECT_ID" \
    --format 'value(textPayload)' 2>/dev/null | head -5 || echo "No logs available"

echo ""
echo "=========================================="
echo "Verification Complete"
echo "=========================================="
echo ""

# Summary
if [ -n "$API_URL" ]; then
    echo "🌐 Live URL: $API_URL"
    echo ""
    echo "Quick commands:"
    echo "  Test API:         curl $API_URL/status"
    echo "  Trigger via scheduler: gcloud scheduler jobs run $SCHEDULER_JOB_NAME --location=$REGION"
    echo "  Trigger job directly:  gcloud run jobs execute $CHECKER_JOB --region=$REGION"
    echo "  View API logs:    gcloud run services logs read $API_SERVICE --region=$REGION"
fi
