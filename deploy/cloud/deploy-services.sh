#!/bin/bash
# Deploy Cloud Run services for MuniMetro
# Deploys both the API service and the status checker service

set -e  # Exit on error

# Configuration
PROJECT_ID="${GCP_PROJECT_ID:-munimetro}"
REGION="${GCP_REGION:-us-west1}"
BUCKET_NAME="${GCS_BUCKET:-munimetro-cache}"
MODEL_VERSION="${MODEL_VERSION:-}"  # Required: model version to deploy (e.g., 20251223_224331)
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

# Determine MODEL_VERSION
echo "[1/4] Checking model version..."

if [ -z "$MODEL_VERSION" ]; then
    # Try to get currently deployed version
    CURRENT=$(gcloud run jobs describe "$CHECKER_JOB" --region="$REGION" --project="$PROJECT_ID" --format="value(spec.template.spec.containers[0].env)" 2>/dev/null | tr ';' '\n' | grep MODEL_VERSION | sed 's/.*value=//')

    if [ -n "$CURRENT" ]; then
        MODEL_VERSION="$CURRENT"
        echo "  Using currently deployed model: $MODEL_VERSION"
    else
        # First deploy - show available models and exit
        echo "  No model currently deployed. Available versions:"
        echo ""
        gsutil ls gs://munimetro-annex/models/snapshots/ 2>/dev/null | sed 's|.*/||' | grep -v '^$' | sort -r | head -5 | while read v; do echo "    $v"; done
        echo ""
        echo "  Set MODEL_VERSION for first deploy:"
        echo "    export MODEL_VERSION=20251223_224331"
        echo "    ./deploy/cloud/deploy-services.sh"
        echo ""
        echo "  Or use manage-models.py to see metrics:"
        echo "    python3 scripts/manage-models.py list"
        exit 1
    fi
else
    echo "  Using specified model: $MODEL_VERSION"
fi

# Verify model exists in GCS
MODEL_PATH="gs://munimetro-annex/models/snapshots/${MODEL_VERSION}/model/status_classifier.pt"
if ! gsutil ls "$MODEL_PATH" &>/dev/null; then
    echo "  ERROR: Model version '$MODEL_VERSION' not found in GCS"
    echo "  Run: python3 scripts/manage-models.py list"
    exit 1
fi
echo "  ✓ Model verified in GCS"
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
    --set-env-vars="CLOUD_RUN=true,GCS_BUCKET=${BUCKET_NAME},ENABLE_FALLBACK=false,MODEL_VERSION=${MODEL_VERSION}" \
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
    --memory 2Gi \
    --cpu 1 \
    --task-timeout 120s \
    --max-retries 3 \
    --set-env-vars="CLOUD_RUN=true,GCS_BUCKET=${BUCKET_NAME},MODEL_VERSION=${MODEL_VERSION}" \
    $SECRETS_FLAG \
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
echo "  Model: $MODEL_VERSION (from GCS)"
echo "  Cache: gs://$BUCKET_NAME/latest_status.json"
echo "  Service Account: $SERVICE_ACCOUNT_EMAIL"
echo ""
echo "If issues occur, rollback with:"
echo "  ./deploy/cloud/rollback.sh --yes"
