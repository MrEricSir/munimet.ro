#!/bin/bash
# Setup Cloud Monitoring uptime checks and alert policies
# Run this after deploying services (deploy-services.sh)

set -e  # Exit on error

# Configuration
PROJECT_ID="${GCP_PROJECT_ID:-munimetro}"
REGION="${GCP_REGION:-us-west1}"
API_SERVICE="munimetro-api"
CHECKER_JOB="munimetro-checker"
REPORTS_JOB="munimetro-reports"
ALERT_EMAIL="${ALERT_EMAIL:-}"

echo "=========================================="
echo "MuniMetro Monitoring Setup"
echo "=========================================="
echo "Project: $PROJECT_ID"
echo "Region: $REGION"
echo "Alert Email: ${ALERT_EMAIL:-Not set}"
echo "=========================================="
echo ""

# Check if alert email is set
if [ -z "$ALERT_EMAIL" ]; then
    echo "❌ Error: ALERT_EMAIL environment variable not set"
    echo "   Example: export ALERT_EMAIL=your-email@example.com"
    echo "   Then run: ./deploy/cloud/setup-infrastructure.sh first"
    exit 1
fi

# Get notification channel ID
echo "[1/3] Finding notification channel..."

# Get access token for REST API
ACCESS_TOKEN=$(gcloud auth print-access-token)

# Get notification channels using REST API
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

if [ -z "$CHANNEL_ID" ]; then
    echo "❌ Error: No notification channel found for $ALERT_EMAIL"
    echo "   Run: ./deploy/cloud/setup-infrastructure.sh first"
    exit 1
fi
echo "✓ Notification channel found: $CHANNEL_ID"
echo ""

# Get API service URL
echo "[2/3] Getting API service URL..."
API_URL=$(gcloud run services describe "$API_SERVICE" \
    --region="$REGION" \
    --project="$PROJECT_ID" \
    --format="value(status.url)" 2>/dev/null)

if [ -z "$API_URL" ]; then
    echo "❌ Error: API service not found. Deploy services first:"
    echo "   ./deploy/cloud/deploy-services.sh"
    exit 1
fi
echo "✓ API URL: $API_URL"
echo ""

# Create uptime check
echo "[3/3] Creating uptime check and alert policies..."

# Check if uptime check already exists using REST API
EXISTING_UPTIMES=$(curl -s -H "Authorization: Bearer $ACCESS_TOKEN" \
    "https://monitoring.googleapis.com/v3/projects/$PROJECT_ID/uptimeCheckConfigs")

UPTIME_CHECK_ID=$(echo "$EXISTING_UPTIMES" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    for check in data.get('uptimeCheckConfigs', []):
        if check.get('displayName') == 'MuniMetro API Health Check':
            print(check['name'])
            sys.exit(0)
except:
    pass
" 2>/dev/null)

if [ -n "$UPTIME_CHECK_ID" ]; then
    echo "⚠️  Uptime check already exists, deleting old one..."
    curl -s -X DELETE -H "Authorization: Bearer $ACCESS_TOKEN" \
        "https://monitoring.googleapis.com/v3/$UPTIME_CHECK_ID" > /dev/null
fi

# Create uptime check using REST API
API_HOST=$(echo $API_URL | sed 's|https://||' | sed 's|http://||')
UPTIME_JSON=$(cat <<EOF
{
  "displayName": "MuniMetro API Health Check",
  "monitoredResource": {
    "type": "uptime_url",
    "labels": {
      "project_id": "$PROJECT_ID",
      "host": "$API_HOST"
    }
  },
  "httpCheck": {
    "path": "/health",
    "port": 443,
    "useSsl": true,
    "validateSsl": true
  },
  "period": "300s",
  "timeout": "10s"
}
EOF
)

UPTIME_RESPONSE=$(curl -s -X POST \
    -H "Authorization: Bearer $ACCESS_TOKEN" \
    -H "Content-Type: application/json" \
    -d "$UPTIME_JSON" \
    "https://monitoring.googleapis.com/v3/projects/$PROJECT_ID/uptimeCheckConfigs")

UPTIME_CHECK_ID=$(echo "$UPTIME_RESPONSE" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    print(data.get('name', ''))
except:
    pass
" 2>/dev/null)
echo "✓ Uptime check created"

# Create alert policy for uptime check failures
cat > /tmp/alert-uptime.json <<EOF
{
  "displayName": "MuniMetro API Down",
  "conditions": [{
    "displayName": "API health check failing",
    "conditionThreshold": {
      "filter": "metric.type=\"monitoring.googleapis.com/uptime_check/check_passed\" AND resource.type=\"uptime_url\" AND metric.label.check_id=\"${UPTIME_CHECK_ID##*/}\"",
      "comparison": "COMPARISON_LT",
      "thresholdValue": 1,
      "duration": "300s",
      "aggregations": [{
        "alignmentPeriod": "60s",
        "perSeriesAligner": "ALIGN_FRACTION_TRUE"
      }]
    }
  }],
  "alertStrategy": {
    "autoClose": "1800s"
  },
  "combiner": "OR",
  "enabled": true,
  "notificationChannels": ["$CHANNEL_ID"],
  "documentation": {
    "content": "The MuniMetro API health check is failing. The service may be down or experiencing issues.",
    "mimeType": "text/markdown"
  }
}
EOF

# Check if alert policy exists using REST API
EXISTING_POLICIES=$(curl -s -H "Authorization: Bearer $ACCESS_TOKEN" \
    "https://monitoring.googleapis.com/v3/projects/$PROJECT_ID/alertPolicies")

EXISTING_ALERT=$(echo "$EXISTING_POLICIES" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    for policy in data.get('alertPolicies', []):
        if policy.get('displayName') == 'MuniMetro API Down':
            print(policy['name'])
            sys.exit(0)
except:
    pass
" 2>/dev/null)

if [ -n "$EXISTING_ALERT" ]; then
    echo "⚠️  Uptime alert policy already exists, updating..."
    curl -s -X DELETE -H "Authorization: Bearer $ACCESS_TOKEN" \
        "https://monitoring.googleapis.com/v3/$EXISTING_ALERT" > /dev/null
fi

# Create alert policy using REST API
UPTIME_ALERT_JSON=$(cat < /tmp/alert-uptime.json)
curl -s -X POST \
    -H "Authorization: Bearer $ACCESS_TOKEN" \
    -H "Content-Type: application/json" \
    -d "$UPTIME_ALERT_JSON" \
    "https://monitoring.googleapis.com/v3/projects/$PROJECT_ID/alertPolicies" > /dev/null
rm /tmp/alert-uptime.json
echo "✓ Uptime alert policy created"

# Create alert policy for Cloud Run job failures
cat > /tmp/alert-job-failures.json <<EOF
{
  "displayName": "MuniMetro Checker Job Failures",
  "conditions": [{
    "displayName": "Checker job failing",
    "conditionThreshold": {
      "filter": "resource.type=\"cloud_run_job\" AND resource.labels.job_name=\"$CHECKER_JOB\" AND metric.type=\"run.googleapis.com/job/completed_execution_count\" AND metric.labels.result=\"failed\"",
      "comparison": "COMPARISON_GT",
      "thresholdValue": 2,
      "duration": "0s",
      "aggregations": [{
        "alignmentPeriod": "600s",
        "perSeriesAligner": "ALIGN_DELTA"
      }]
    }
  }],
  "alertStrategy": {
    "autoClose": "3600s"
  },
  "combiner": "OR",
  "enabled": true,
  "notificationChannels": ["$CHANNEL_ID"],
  "documentation": {
    "content": "The MuniMetro status checker job has failed multiple times. Check Cloud Run logs for details.",
    "mimeType": "text/markdown"
  }
}
EOF

EXISTING_JOB_ALERT=$(echo "$EXISTING_POLICIES" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    for policy in data.get('alertPolicies', []):
        if policy.get('displayName') == 'MuniMetro Checker Job Failures':
            print(policy['name'])
            sys.exit(0)
except:
    pass
" 2>/dev/null)

if [ -n "$EXISTING_JOB_ALERT" ]; then
    echo "⚠️  Job failure alert policy already exists, updating..."
    curl -s -X DELETE -H "Authorization: Bearer $ACCESS_TOKEN" \
        "https://monitoring.googleapis.com/v3/$EXISTING_JOB_ALERT" > /dev/null
fi

JOB_ALERT_JSON=$(cat < /tmp/alert-job-failures.json)
curl -s -X POST \
    -H "Authorization: Bearer $ACCESS_TOKEN" \
    -H "Content-Type: application/json" \
    -d "$JOB_ALERT_JSON" \
    "https://monitoring.googleapis.com/v3/projects/$PROJECT_ID/alertPolicies" > /dev/null
rm /tmp/alert-job-failures.json
echo "✓ Checker job failure alert policy created"

# Create alert policy for Reports job failures (includes missing database errors)
cat > /tmp/alert-reports-failures.json <<EOF
{
  "displayName": "MuniMetro Reports Job Failures",
  "conditions": [{
    "displayName": "Reports job failing",
    "conditionThreshold": {
      "filter": "resource.type=\"cloud_run_job\" AND resource.labels.job_name=\"$REPORTS_JOB\" AND metric.type=\"run.googleapis.com/job/completed_execution_count\" AND metric.labels.result=\"failed\"",
      "comparison": "COMPARISON_GT",
      "thresholdValue": 0,
      "duration": "0s",
      "aggregations": [{
        "alignmentPeriod": "86400s",
        "perSeriesAligner": "ALIGN_DELTA"
      }]
    }
  }],
  "alertStrategy": {
    "autoClose": "86400s"
  },
  "combiner": "OR",
  "enabled": true,
  "notificationChannels": ["$CHANNEL_ID"],
  "documentation": {
    "content": "The MuniMetro analytics reports job has failed. This may indicate a missing or corrupted analytics database. Check Cloud Run logs for ANALYTICS_NO_DATABASE or ANALYTICS_ERROR messages.",
    "mimeType": "text/markdown"
  }
}
EOF

EXISTING_REPORTS_ALERT=$(echo "$EXISTING_POLICIES" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    for policy in data.get('alertPolicies', []):
        if policy.get('displayName') == 'MuniMetro Reports Job Failures':
            print(policy['name'])
            sys.exit(0)
except:
    pass
" 2>/dev/null)

if [ -n "$EXISTING_REPORTS_ALERT" ]; then
    echo "⚠️  Reports job failure alert policy already exists, updating..."
    curl -s -X DELETE -H "Authorization: Bearer $ACCESS_TOKEN" \
        "https://monitoring.googleapis.com/v3/$EXISTING_REPORTS_ALERT" > /dev/null
fi

REPORTS_ALERT_JSON=$(cat < /tmp/alert-reports-failures.json)
curl -s -X POST \
    -H "Authorization: Bearer $ACCESS_TOKEN" \
    -H "Content-Type: application/json" \
    -d "$REPORTS_ALERT_JSON" \
    "https://monitoring.googleapis.com/v3/projects/$PROJECT_ID/alertPolicies" > /dev/null
rm /tmp/alert-reports-failures.json
echo "✓ Reports job failure alert policy created"

# Create alert policy for API error rate
cat > /tmp/alert-error-rate.json <<EOF
{
  "displayName": "MuniMetro API Error Rate High",
  "conditions": [{
    "displayName": "API error rate above 10%",
    "conditionThreshold": {
      "filter": "resource.type=\"cloud_run_revision\" AND resource.labels.service_name=\"$API_SERVICE\" AND metric.type=\"run.googleapis.com/request_count\" AND metric.labels.response_code_class=\"5xx\"",
      "comparison": "COMPARISON_GT",
      "thresholdValue": 0.1,
      "duration": "300s",
      "aggregations": [{
        "alignmentPeriod": "60s",
        "perSeriesAligner": "ALIGN_RATE",
        "crossSeriesReducer": "REDUCE_SUM"
      }]
    }
  }],
  "alertStrategy": {
    "autoClose": "1800s"
  },
  "combiner": "OR",
  "enabled": true,
  "notificationChannels": ["$CHANNEL_ID"],
  "documentation": {
    "content": "The MuniMetro API is experiencing a high error rate (>10% 5xx responses). Check Cloud Run logs for details.",
    "mimeType": "text/markdown"
  }
}
EOF

EXISTING_ERROR_ALERT=$(echo "$EXISTING_POLICIES" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    for policy in data.get('alertPolicies', []):
        if policy.get('displayName') == 'MuniMetro API Error Rate High':
            print(policy['name'])
            sys.exit(0)
except:
    pass
" 2>/dev/null)

if [ -n "$EXISTING_ERROR_ALERT" ]; then
    echo "⚠️  Error rate alert policy already exists, updating..."
    curl -s -X DELETE -H "Authorization: Bearer $ACCESS_TOKEN" \
        "https://monitoring.googleapis.com/v3/$EXISTING_ERROR_ALERT" > /dev/null
fi

ERROR_ALERT_JSON=$(cat < /tmp/alert-error-rate.json)
curl -s -X POST \
    -H "Authorization: Bearer $ACCESS_TOKEN" \
    -H "Content-Type: application/json" \
    -d "$ERROR_ALERT_JSON" \
    "https://monitoring.googleapis.com/v3/projects/$PROJECT_ID/alertPolicies" > /dev/null
rm /tmp/alert-error-rate.json
echo "✓ API error rate alert policy created"

echo ""
echo "=========================================="
echo "✓ Monitoring setup complete!"
echo "=========================================="
echo ""
echo "Configured alerts:"
echo "  1. API Down - Health check failing for 5+ minutes"
echo "  2. Checker Job Failures - Status checker job failing repeatedly"
echo "  3. Reports Job Failures - Analytics reports job failing (includes missing database)"
echo "  4. High Error Rate - API returning >10% errors"
echo ""
echo "Notifications will be sent to: $ALERT_EMAIL"
echo ""
echo "View monitoring dashboard:"
echo "  https://console.cloud.google.com/monitoring/dashboards?project=$PROJECT_ID"
echo ""
echo "Manage alert policies:"
echo "  https://console.cloud.google.com/monitoring/alerting/policies?project=$PROJECT_ID"
echo ""
