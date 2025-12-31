#!/bin/bash
# Deploy Cloud Run services for MuniMetro
# Deploys both the API service and the status checker service

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

# Verify model files are present
echo "[1/4] Verifying model files..."
MODEL_DIR="artifacts/models/v1"
MODEL_FILE="$MODEL_DIR/model.safetensors"

if [ ! -f "$MODEL_FILE" ]; then
    echo "  ERROR: Model file not found: $MODEL_FILE"
    echo "  Run: ./scripts/sync-models.sh download"
    exit 1
fi

# Verify the file is the actual model (should be ~856MB)
ACTUAL_SIZE=$(wc -c < "$MODEL_FILE" 2>/dev/null || echo "0")
if [ "$ACTUAL_SIZE" -lt 100000000 ]; then
    echo "  ERROR: Model file is too small ($ACTUAL_SIZE bytes)"
    echo "  Expected ~856MB. Run: ./scripts/sync-models.sh download"
    exit 1
fi
echo "  ✓ Model file size: $(numfmt --to=iec --suffix=B $ACTUAL_SIZE 2>/dev/null || echo "$ACTUAL_SIZE bytes")"
echo ""

# Build and push Docker image
echo "[2/4] Building Docker image..."
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
echo "[3/4] Deploying services..."
echo ""
echo "  Deploying API service..."
gcloud run deploy "$API_SERVICE" \
    --image "$API_IMAGE" \
    --platform managed \
    --region "$REGION" \
    --project "$PROJECT_ID" \
    --service-account "$SERVICE_ACCOUNT_EMAIL" \
    --allow-unauthenticated \
    --memory 2Gi \
    --cpu 1 \
    --timeout 60s \
    --max-instances 10 \
    --min-instances 0 \
    --port 8000 \
    --set-env-vars="CLOUD_RUN=true,GCS_BUCKET=${BUCKET_NAME},ENABLE_FALLBACK=false" \
    --command="gunicorn" \
    --args="api.api:app,--bind,0.0.0.0:8000,--workers,1,--timeout,60,--graceful-timeout,30,--log-level,info"

API_URL=$(gcloud run services describe "$API_SERVICE" \
    --platform managed \
    --region "$REGION" \
    --project "$PROJECT_ID" \
    --format 'value(status.url)')

echo "✓ API service deployed: $API_URL"
echo ""

# Deploy status checker job
echo ""
echo "  Deploying status checker job..."
gcloud run jobs deploy "$CHECKER_JOB" \
    --image "$API_IMAGE" \
    --region "$REGION" \
    --project "$PROJECT_ID" \
    --service-account "$SERVICE_ACCOUNT_EMAIL" \
    --memory 2Gi \
    --cpu 1 \
    --task-timeout 120s \
    --max-retries 3 \
    --set-env-vars="CLOUD_RUN=true,GCS_BUCKET=${BUCKET_NAME}" \
    --command="python" \
    --args="-m,api.check_status_job"

echo "✓ Checker job deployed: $CHECKER_JOB"
echo ""

# Grant service account permission to invoke the job (for Cloud Scheduler)
echo "  Granting job execution permission to service account..."
gcloud run jobs add-iam-policy-binding "$CHECKER_JOB" \
    --region="$REGION" \
    --member="serviceAccount:$SERVICE_ACCOUNT_EMAIL" \
    --role="roles/run.invoker" \
    --project="$PROJECT_ID" > /dev/null

echo "✓ Job invoker permission granted"
echo ""

echo "=========================================="
echo "✓ Deployment complete!"
echo "=========================================="
echo ""
echo "Deployed components:"
echo "  API Service:  $API_URL"
echo "  Checker Job:  $CHECKER_JOB (in region $REGION)"
echo ""
echo "Next steps:"
echo "1. Test API: curl $API_URL/status"
echo "2. Setup scheduler: ./deploy/cloud/setup-scheduler.sh"
echo "3. Test job manually: gcloud run jobs execute $CHECKER_JOB --region=$REGION"
echo ""
echo "Environment info:"
echo "  Cache: gs://$BUCKET_NAME/latest_status.json"
echo "  Service Account: $SERVICE_ACCOUNT_EMAIL"
echo ""
echo "If issues occur, rollback with:"
echo "  ./deploy/cloud/rollback.sh --yes"
