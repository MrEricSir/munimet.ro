#!/bin/bash
# Setup GCP infrastructure for MuniMetro Cloud Run deployment
# Run this once before deploying services

set -e  # Exit on error

# Configuration
PROJECT_ID="${GCP_PROJECT_ID:-munimetro}"
REGION="${GCP_REGION:-us-west1}"
BUCKET_NAME="${GCS_BUCKET:-munimetro-cache}"
SERVICE_ACCOUNT_NAME="munimetro-api"

echo "=========================================="
echo "MuniMetro Infrastructure Setup"
echo "=========================================="
echo "Project: $PROJECT_ID"
echo "Region: $REGION"
echo "Bucket: gs://$BUCKET_NAME"
echo "=========================================="
echo ""

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo "❌ Error: gcloud CLI not found"
    echo "Install from: https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# Set project
echo "[1/6] Setting GCP project..."
gcloud config set project "$PROJECT_ID"
echo "✓ Project set to $PROJECT_ID"
echo ""

# Enable required APIs
echo "[2/6] Enabling required APIs..."
gcloud services enable \
    run.googleapis.com \
    cloudbuild.googleapis.com \
    cloudscheduler.googleapis.com \
    storage.googleapis.com \
    artifactregistry.googleapis.com
echo "✓ APIs enabled"
echo ""

# Create GCS bucket (private by default)
echo "[3/6] Creating Cloud Storage bucket..."
if gsutil ls -b "gs://$BUCKET_NAME" &> /dev/null; then
    echo "⚠️  Bucket already exists: gs://$BUCKET_NAME"
else
    gsutil mb -p "$PROJECT_ID" -l "$REGION" "gs://$BUCKET_NAME"
    echo "✓ Bucket created: gs://$BUCKET_NAME"
fi

# Ensure bucket is private
echo "  - Ensuring bucket is private..."
gsutil iam ch -d allUsers:objectViewer "gs://$BUCKET_NAME" 2>/dev/null || true
echo "✓ Bucket access restricted"
echo ""

# Create service account
echo "[4/6] Creating service account..."
SERVICE_ACCOUNT_EMAIL="${SERVICE_ACCOUNT_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"

if gcloud iam service-accounts describe "$SERVICE_ACCOUNT_EMAIL" &> /dev/null; then
    echo "⚠️  Service account already exists: $SERVICE_ACCOUNT_EMAIL"
else
    gcloud iam service-accounts create "$SERVICE_ACCOUNT_NAME" \
        --display-name="MuniMetro API Service Account" \
        --description="Used by Cloud Run services to access GCS cache"
    echo "✓ Service account created: $SERVICE_ACCOUNT_EMAIL"
fi
echo ""

# Grant service account access to bucket
echo "[5/6] Granting service account access to bucket..."
gsutil iam ch \
    "serviceAccount:${SERVICE_ACCOUNT_EMAIL}:roles/storage.objectAdmin" \
    "gs://$BUCKET_NAME"
echo "✓ Service account has objectAdmin access to gs://$BUCKET_NAME"
echo ""

# Create App Engine app (required for Cloud Scheduler)
echo "[6/6] Setting up Cloud Scheduler (requires App Engine)..."
if gcloud app describe &> /dev/null; then
    echo "⚠️  App Engine app already exists"
else
    echo "Creating App Engine app in region $REGION..."
    gcloud app create --region="$REGION" || {
        echo "⚠️  App Engine creation failed. This is OK if it already exists."
    }
fi
echo "✓ Cloud Scheduler ready"
echo ""

echo "=========================================="
echo "✓ Infrastructure setup complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Deploy services: ./deploy-services.sh"
echo "2. Configure scheduler: ./setup-scheduler.sh"
echo ""
echo "Environment variables for services:"
echo "  GCS_BUCKET=$BUCKET_NAME"
echo "  SERVICE_ACCOUNT=$SERVICE_ACCOUNT_EMAIL"
