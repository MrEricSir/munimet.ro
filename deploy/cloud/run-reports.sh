#!/bin/bash
# Manually run the analytics reports job on GCP Cloud Run
#
# Usage:
#   ./deploy/cloud/run-reports.sh         # Run and wait for completion
#   ./deploy/cloud/run-reports.sh --async # Run in background (don't wait)

set -e

# Configuration
PROJECT_ID="${GCP_PROJECT_ID:-munimetro}"
REGION="${GCP_REGION:-us-west1}"
REPORTS_JOB="munimetro-reports"

# Parse arguments
ASYNC_MODE=false
for arg in "$@"; do
    case $arg in
        --async)
            ASYNC_MODE=true
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Manually run the analytics reports job on GCP Cloud Run."
            echo ""
            echo "Options:"
            echo "  --async    Run in background (don't wait for completion)"
            echo "  --help     Show this help"
            exit 0
            ;;
    esac
done

echo "=========================================="
echo "Running MuniMetro Analytics Reports Job"
echo "=========================================="
echo "Project: $PROJECT_ID"
echo "Region:  $REGION"
echo "Job:     $REPORTS_JOB"
echo ""

# Verify job exists
if ! gcloud run jobs describe "$REPORTS_JOB" \
    --region "$REGION" \
    --project "$PROJECT_ID" &> /dev/null; then
    echo "Error: Reports job not found. Deploy services first:"
    echo "  ./deploy/cloud/deploy-services.sh"
    exit 1
fi

# Execute the job
if [ "$ASYNC_MODE" = true ]; then
    echo "Starting job (async mode)..."
    gcloud run jobs execute "$REPORTS_JOB" \
        --region "$REGION" \
        --project "$PROJECT_ID" \
        --async
    echo ""
    echo "Job started in background."
    echo "View status: gcloud run jobs executions list --job=$REPORTS_JOB --region=$REGION"
else
    echo "Running job (waiting for completion)..."
    echo ""
    gcloud run jobs execute "$REPORTS_JOB" \
        --region "$REGION" \
        --project "$PROJECT_ID" \
        --wait
    echo ""
    echo "Job completed successfully."
fi

echo ""
echo "View recent executions:"
echo "  gcloud run jobs executions list --job=$REPORTS_JOB --region=$REGION --limit=5"
echo ""
echo "View logs:"
echo "  gcloud logging read 'resource.type=\"cloud_run_job\" resource.labels.job_name=\"$REPORTS_JOB\"' --limit=50"
