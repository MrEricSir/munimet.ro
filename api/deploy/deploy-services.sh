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

# Service names
API_SERVICE="munimetro-api"
CHECKER_SERVICE="munimetro-checker"

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
    --timeout=20m \
    .
echo "✓ Image built and pushed: $API_IMAGE"
echo ""

# Deploy API service (reads cache, serves frontend)
echo "[2/3] Deploying API service..."
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
    --set-env-vars="CLOUD_RUN=true,GCS_BUCKET=${BUCKET_NAME},ENABLE_FALLBACK=false" \
    --command="gunicorn" \
    --args="api:app,--bind,0.0.0.0:8000,--workers,1,--timeout,60,--graceful-timeout,30,--log-level,info"

API_URL=$(gcloud run services describe "$API_SERVICE" \
    --platform managed \
    --region "$REGION" \
    --project "$PROJECT_ID" \
    --format 'value(status.url)')

echo "✓ API service deployed: $API_URL"
echo ""

# Deploy status checker service (downloads image, writes cache)
echo "[3/3] Deploying status checker service..."
gcloud run deploy "$CHECKER_SERVICE" \
    --image "$API_IMAGE" \
    --platform managed \
    --region "$REGION" \
    --project "$PROJECT_ID" \
    --service-account "$SERVICE_ACCOUNT_EMAIL" \
    --no-allow-unauthenticated \
    --memory 2Gi \
    --cpu 1 \
    --timeout 120s \
    --max-instances 1 \
    --min-instances 0 \
    --concurrency 1 \
    --set-env-vars="CLOUD_RUN=true,GCS_BUCKET=${BUCKET_NAME}" \
    --command="python3" \
    --args="check_status.py,--write-cache"

CHECKER_URL=$(gcloud run services describe "$CHECKER_SERVICE" \
    --platform managed \
    --region "$REGION" \
    --project "$PROJECT_ID" \
    --format 'value(status.url)')

echo "✓ Checker service deployed: $CHECKER_URL"
echo ""

echo "=========================================="
echo "✓ Deployment complete!"
echo "=========================================="
echo ""
echo "Service URLs:"
echo "  API (public):     $API_URL"
echo "  Checker (private): $CHECKER_URL"
echo ""
echo "Next steps:"
echo "1. Test API: curl $API_URL/status"
echo "2. Setup scheduler: ./setup-scheduler.sh"
echo ""
echo "Environment info:"
echo "  Cache: gs://$BUCKET_NAME/latest_status.json"
echo "  Service Account: $SERVICE_ACCOUNT_EMAIL"
