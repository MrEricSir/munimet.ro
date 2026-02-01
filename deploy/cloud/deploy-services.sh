#!/bin/bash
# Deploy Cloud Run services for MuniMetro
# Deploys both the API service and the status checker job

set -e  # Exit on error

# Configuration
PROJECT_ID="${GCP_PROJECT_ID:-munimetro}"
REGION="${GCP_REGION:-us-west1}"
BUCKET_NAME="${GCS_BUCKET:-munimetro-cache}"
SERVICE_ACCOUNT_NAME="munimetro-api"
SERVICE_ACCOUNT_EMAIL="${SERVICE_ACCOUNT_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"

# Service and job names
API_SERVICE="munimetro-api"
CHECKER_JOB="munimetro-checker"
REPORTS_JOB="munimetro-reports"

# Image names
IMAGE_NAME="munimetro"
API_IMAGE="gcr.io/${PROJECT_ID}/${IMAGE_NAME}:latest"

echo "=========================================="
echo "MuniMetro Cloud Run Deployment"
echo "=========================================="
echo "Project: $PROJECT_ID"
echo "Region: $REGION"
echo "Bucket: gs://$BUCKET_NAME"
echo "Service Account: $SERVICE_ACCOUNT_EMAIL"
echo "=========================================="
echo ""

# Navigate to project root
cd "$(dirname "$0")/../.."

# Build and push Docker image
echo "[1/3] Building Docker image..."
gcloud builds submit \
    --tag "$API_IMAGE" \
    --project "$PROJECT_ID" \
    --timeout=30m \
    .

BUILD_STATUS=$?
echo ""

# Check build status
if [ $BUILD_STATUS -ne 0 ]; then
    echo "❌ Docker build failed"
    exit $BUILD_STATUS
fi

echo "✓ Image built and pushed: $API_IMAGE"
echo ""

# Deploy services
echo "[2/4] Deploying services..."
echo ""
echo "  Deploying API service..."
gcloud run deploy "$API_SERVICE" \
    --image "$API_IMAGE" \
    --platform managed \
    --region "$REGION" \
    --project "$PROJECT_ID" \
    --service-account "$SERVICE_ACCOUNT_EMAIL" \
    --allow-unauthenticated \
    --memory 1Gi \
    --cpu 1 \
    --timeout 60s \
    --max-instances 10 \
    --min-instances 0 \
    --port 8000 \
    --set-env-vars="CLOUD_RUN=true,GCS_BUCKET=${BUCKET_NAME},ENABLE_FALLBACK=false" \
    --command="gunicorn" \
    --args="api.api:app,--bind,0.0.0.0:8000,--workers,2,--timeout,60,--graceful-timeout,30,--log-level,info"

API_URL=$(gcloud run services describe "$API_SERVICE" \
    --platform managed \
    --region "$REGION" \
    --project "$PROJECT_ID" \
    --format 'value(status.url)')

echo "✓ API service deployed: $API_URL"
echo ""

# Deploy status checker job
echo "[3/4] Deploying status checker job..."

# Build secrets flag if secrets exist in Secret Manager
SECRETS_FLAG=""
if gcloud secrets describe BLUESKY_HANDLE --project="$PROJECT_ID" &>/dev/null; then
    SECRETS_FLAG="--set-secrets=BLUESKY_HANDLE=BLUESKY_HANDLE:latest,BLUESKY_APP_PASSWORD=BLUESKY_APP_PASSWORD:latest"
    echo "  (with Bluesky credentials from Secret Manager)"
fi

gcloud run jobs deploy "$CHECKER_JOB" \
    --image "$API_IMAGE" \
    --region "$REGION" \
    --project "$PROJECT_ID" \
    --service-account "$SERVICE_ACCOUNT_EMAIL" \
    --memory 1Gi \
    --cpu 1 \
    --task-timeout 120s \
    --max-retries 3 \
    --set-env-vars="CLOUD_RUN=true,GCS_BUCKET=${BUCKET_NAME}" \
    $SECRETS_FLAG \
    --command="python" \
    --args="-m,api.check_status_job"

echo "✓ Checker job deployed: $CHECKER_JOB"
echo ""

# Deploy analytics reports job
echo "[4/4] Deploying analytics reports job..."

gcloud run jobs deploy "$REPORTS_JOB" \
    --image "$API_IMAGE" \
    --region "$REGION" \
    --project "$PROJECT_ID" \
    --service-account "$SERVICE_ACCOUNT_EMAIL" \
    --memory 512Mi \
    --cpu 1 \
    --task-timeout 300s \
    --max-retries 2 \
    --set-env-vars="CLOUD_RUN=true,GCS_BUCKET=${BUCKET_NAME}" \
    --command="python" \
    --args="-m,api.generate_reports_job"

echo "✓ Reports job deployed: $REPORTS_JOB"
echo ""

# Grant service account permission to invoke jobs (for Cloud Scheduler)
echo "  Granting job execution permissions to service account..."
gcloud run jobs add-iam-policy-binding "$CHECKER_JOB" \
    --region="$REGION" \
    --member="serviceAccount:$SERVICE_ACCOUNT_EMAIL" \
    --role="roles/run.invoker" \
    --project="$PROJECT_ID" > /dev/null

gcloud run jobs add-iam-policy-binding "$REPORTS_JOB" \
    --region="$REGION" \
    --member="serviceAccount:$SERVICE_ACCOUNT_EMAIL" \
    --role="roles/run.invoker" \
    --project="$PROJECT_ID" > /dev/null

echo "✓ Job invoker permissions granted"
echo ""

echo "=========================================="
echo "✓ Deployment complete!"
echo "=========================================="
echo ""
echo "Deployed components:"
echo "  API Service:   $API_URL"
echo "  Checker Job:   $CHECKER_JOB (every 3 min)"
echo "  Reports Job:   $REPORTS_JOB (daily at midnight)"
echo ""
echo "Next steps:"
echo "1. Test API: curl $API_URL/status"
echo "2. Setup schedulers: ./deploy/cloud/setup-scheduler.sh"
echo "3. Test jobs manually:"
echo "   gcloud run jobs execute $CHECKER_JOB --region=$REGION"
echo "   gcloud run jobs execute $REPORTS_JOB --region=$REGION"
echo ""
echo "Environment info:"
echo "  Cache: gs://$BUCKET_NAME/latest_status.json"
echo "  Service Account: $SERVICE_ACCOUNT_EMAIL"
echo ""
echo "If issues occur, rollback with:"
echo "  ./deploy/cloud/rollback.sh --yes"
