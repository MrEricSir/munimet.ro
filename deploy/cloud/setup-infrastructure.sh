#!/bin/bash
# Setup GCP infrastructure for MuniMetro Cloud Run deployment
# Run this once before deploying services

set -e  # Exit on error

# Configuration
PROJECT_ID="${GCP_PROJECT_ID:-munimetro}"
REGION="${GCP_REGION:-us-west1}"
BUCKET_NAME="${GCS_BUCKET:-munimetro-cache}"
SERVICE_ACCOUNT_NAME="munimetro-api"
ALERT_EMAIL="${ALERT_EMAIL:-}"  # Set via environment variable

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
echo "[1/9] Setting GCP project..."
gcloud config set project "$PROJECT_ID"
echo "✓ Project set to $PROJECT_ID"
echo ""

# Enable required APIs
echo "[2/9] Enabling required APIs..."
gcloud services enable \
    run.googleapis.com \
    cloudbuild.googleapis.com \
    cloudscheduler.googleapis.com \
    storage.googleapis.com \
    artifactregistry.googleapis.com \
    monitoring.googleapis.com \
    secretmanager.googleapis.com
echo "✓ APIs enabled"
echo ""

# Note: Monitoring setup now uses REST API instead of alpha commands (step removed)
# Renumber: 3/10 becomes 3/9, etc.

# Create GCS bucket (private by default)
echo "[3/9] Creating Cloud Storage bucket..."
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
echo "[4/9] Creating service account..."
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
echo "[5/9] Granting service account access to bucket..."
gsutil iam ch \
    "serviceAccount:${SERVICE_ACCOUNT_EMAIL}:roles/storage.objectAdmin" \
    "gs://$BUCKET_NAME"
echo "✓ Service account has objectAdmin access to gs://$BUCKET_NAME"
echo ""

# Grant service account access to Secret Manager (for social media credentials)
echo "[5b/9] Granting service account access to Secret Manager..."
gcloud projects add-iam-policy-binding "$PROJECT_ID" \
    --member="serviceAccount:${SERVICE_ACCOUNT_EMAIL}" \
    --role="roles/secretmanager.secretAccessor" \
    --condition=None \
    --quiet
echo "✓ Service account can access secrets"
echo ""

# Create App Engine app (required for Cloud Scheduler)
echo "[6/9] Setting up Cloud Scheduler (requires App Engine)..."
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

# Setup monitoring and alerts
if [ -n "$ALERT_EMAIL" ]; then
    echo "[7/9] Setting up email notification channel..."

    # Get access token for REST API
    ACCESS_TOKEN=$(gcloud auth print-access-token)

    # Check if notification channel already exists using REST API
    EXISTING_CHANNELS=$(curl -s -H "Authorization: Bearer $ACCESS_TOKEN" \
        "https://monitoring.googleapis.com/v3/projects/$PROJECT_ID/notificationChannels")

    # Use Python to parse JSON reliably
    CHANNEL_ID=$(echo "$EXISTING_CHANNELS" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    for channel in data.get('notificationChannels', []):
        if channel.get('labels', {}).get('email_address') == '$ALERT_EMAIL':
            print(channel['name'])
            sys.exit(0)
except:
    pass
" 2>/dev/null)

    if [ -n "$CHANNEL_ID" ]; then
        echo "⚠️  Email notification channel already exists for $ALERT_EMAIL"
    else
        # Create notification channel using REST API
        CHANNEL_JSON=$(cat <<EOF
{
  "type": "email",
  "displayName": "MuniMetro Alerts",
  "labels": {
    "email_address": "$ALERT_EMAIL"
  },
  "enabled": true
}
EOF
)
        RESPONSE=$(curl -s -X POST \
            -H "Authorization: Bearer $ACCESS_TOKEN" \
            -H "Content-Type: application/json" \
            -d "$CHANNEL_JSON" \
            "https://monitoring.googleapis.com/v3/projects/$PROJECT_ID/notificationChannels")

        CHANNEL_ID=$(echo "$RESPONSE" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    print(data.get('name', ''))
except:
    pass
" 2>/dev/null)

        if [ -n "$CHANNEL_ID" ]; then
            echo "✓ Email notification channel created: $ALERT_EMAIL"
        else
            echo "⚠️  Warning: Could not create notification channel"
            echo "   Response: $RESPONSE"
            echo "   You can create it manually or run setup-monitoring.sh later"
        fi
    fi
    echo ""

    echo "[8/9] Creating uptime check..."
    # Note: Uptime check will be created after services are deployed
    # This will be done in setup-monitoring.sh (separate script to run after deployment)
    echo "⚠️  Uptime check will be configured after service deployment"
    echo "   Run: ./deploy/cloud/setup-monitoring.sh (after deploying services)"
    echo ""

    echo "[9/9] Creating alert policies..."
    # Note: Alert policies need the Cloud Run services to exist first
    echo "⚠️  Alert policies will be configured after service deployment"
    echo "   Run: ./deploy/cloud/setup-monitoring.sh (after deploying services)"
    echo ""
else
    echo "[7/9] Skipping email notification setup..."
    echo "⚠️  To enable email alerts, set ALERT_EMAIL environment variable"
    echo "   Example: export ALERT_EMAIL=your-email@example.com"
    echo ""

    echo "[8/9] Skipping uptime check..."
    echo "⚠️  Uptime check requires ALERT_EMAIL to be set"
    echo ""

    echo "[9/9] Skipping alert policies..."
    echo "⚠️  Alert policies require ALERT_EMAIL to be set"
    echo ""
fi

echo "=========================================="
echo "✓ Infrastructure setup complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Deploy services: ./deploy/cloud/deploy-services.sh"
echo "2. Configure scheduler: ./deploy/cloud/setup-scheduler.sh"
if [ -n "$ALERT_EMAIL" ]; then
    echo "3. Setup monitoring: ./deploy/cloud/setup-monitoring.sh"
fi
echo ""
echo "Environment variables for services:"
echo "  GCS_BUCKET=$BUCKET_NAME"
echo "  SERVICE_ACCOUNT=$SERVICE_ACCOUNT_EMAIL"
if [ -n "$ALERT_EMAIL" ]; then
    echo "  ALERT_EMAIL=$ALERT_EMAIL"
    echo "  NOTIFICATION_CHANNEL=$CHANNEL_ID"
fi
