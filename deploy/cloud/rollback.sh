#!/bin/bash
# Rollback MuniMetro Cloud Run deployment to previous version
#
# Usage:
#   ./deploy/cloud/rollback.sh              # Interactive rollback
#   ./deploy/cloud/rollback.sh --api-only   # Rollback API service only
#   ./deploy/cloud/rollback.sh --job-only   # Rollback checker job only
#   ./deploy/cloud/rollback.sh --yes        # Skip confirmation prompt

set -e  # Exit on error

# Configuration
PROJECT_ID="${GCP_PROJECT_ID:-munimetro}"
REGION="${GCP_REGION:-us-west1}"
API_SERVICE="munimetro-api"
CHECKER_JOB="munimetro-checker"

# Parse arguments
ROLLBACK_API=true
ROLLBACK_JOB=true
SKIP_CONFIRM=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --api-only)
            ROLLBACK_JOB=false
            shift
            ;;
        --job-only)
            ROLLBACK_API=false
            shift
            ;;
        --yes|-y)
            SKIP_CONFIRM=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Rollback MuniMetro Cloud Run deployment to previous version"
            echo ""
            echo "Options:"
            echo "  --api-only    Rollback API service only"
            echo "  --job-only    Rollback checker job only"
            echo "  --yes, -y     Skip confirmation prompt"
            echo "  --help, -h    Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "=========================================="
echo "MuniMetro Cloud Run Rollback"
echo "=========================================="
echo "Project: $PROJECT_ID"
echo "Region: $REGION"
echo "=========================================="
echo ""

# Function to show current and previous revisions
show_api_revisions() {
    echo "Current API Service Revisions:"
    echo "------------------------------"
    gcloud run revisions list \
        --service="$API_SERVICE" \
        --region="$REGION" \
        --project="$PROJECT_ID" \
        --limit=5 \
        --format='table(name,traffic,creation_timestamp,status)' 2>/dev/null || {
        echo "Error: Could not fetch API service revisions"
        return 1
    }
    echo ""
}

# Function to show job information
show_job_info() {
    echo "Current Checker Job Information:"
    echo "--------------------------------"
    gcloud run jobs describe "$CHECKER_JOB" \
        --region="$REGION" \
        --project="$PROJECT_ID" \
        --format='value(metadata.name,spec.template.spec.containers[0].image,metadata.generation)' 2>/dev/null | \
    awk '{
        print "Job: " $1
        print "Current Image: " $2
        print "Generation: " $3
    }' || {
        echo "Error: Could not fetch job information"
        return 1
    }
    echo ""
}

# Rollback API Service
rollback_api() {
    echo "[1/2] Rolling back API service..."
    echo ""

    # Get current revision (the one receiving traffic)
    CURRENT_REVISION=$(gcloud run services describe "$API_SERVICE" \
        --region="$REGION" \
        --project="$PROJECT_ID" \
        --format='value(status.traffic[0].revisionName)' 2>/dev/null)

    if [ -z "$CURRENT_REVISION" ]; then
        echo "❌ Error: Could not determine current revision"
        return 1
    fi

    echo "Current revision: $CURRENT_REVISION"

    # Get list of all revisions, sorted by creation time (newest first)
    REVISIONS=$(gcloud run revisions list \
        --service="$API_SERVICE" \
        --region="$REGION" \
        --project="$PROJECT_ID" \
        --sort-by="~metadata.creationTimestamp" \
        --format='value(name)' 2>/dev/null)

    # Find the previous revision (first one that's not the current one)
    PREVIOUS_REVISION=""
    for rev in $REVISIONS; do
        if [ "$rev" != "$CURRENT_REVISION" ]; then
            PREVIOUS_REVISION="$rev"
            break
        fi
    done

    if [ -z "$PREVIOUS_REVISION" ]; then
        echo "❌ Error: No previous revision found to rollback to"
        return 1
    fi

    echo "Previous revision: $PREVIOUS_REVISION"
    echo ""

    if [ "$SKIP_CONFIRM" = false ]; then
        read -p "Rollback API service from $CURRENT_REVISION to $PREVIOUS_REVISION? (y/N): " confirm
        if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
            echo "Rollback cancelled"
            return 1
        fi
    fi

    echo "Rolling back API service..."
    gcloud run services update-traffic "$API_SERVICE" \
        --region="$REGION" \
        --project="$PROJECT_ID" \
        --to-revisions="$PREVIOUS_REVISION=100" \
        --quiet

    echo "✓ API service rolled back to: $PREVIOUS_REVISION"
    echo ""

    # Get the service URL
    API_URL=$(gcloud run services describe "$API_SERVICE" \
        --region="$REGION" \
        --project="$PROJECT_ID" \
        --format='value(status.url)')

    echo "Service URL: $API_URL"
    echo ""
}

# Rollback Checker Job
rollback_job() {
    echo "[2/2] Rolling back checker job..."
    echo ""

    # For jobs, we need to get the previous image from the build history
    # Get current image
    CURRENT_IMAGE=$(gcloud run jobs describe "$CHECKER_JOB" \
        --region="$REGION" \
        --project="$PROJECT_ID" \
        --format='value(spec.template.spec.containers[0].image)' 2>/dev/null)

    if [ -z "$CURRENT_IMAGE" ]; then
        echo "❌ Error: Could not determine current job image"
        return 1
    fi

    echo "Current image: $CURRENT_IMAGE"

    # Extract the image name (without tag/digest)
    IMAGE_BASE=$(echo "$CURRENT_IMAGE" | cut -d'@' -f1 | cut -d':' -f1)

    # Get list of images from Container Registry
    echo "Fetching image history from Container Registry..."
    IMAGES=$(gcloud container images list-tags "$IMAGE_BASE" \
        --project="$PROJECT_ID" \
        --format='value(digest)' \
        --limit=10 \
        --sort-by="~timestamp" 2>/dev/null)

    if [ -z "$IMAGES" ]; then
        echo "❌ Error: Could not fetch image history"
        return 1
    fi

    # Get current digest
    CURRENT_DIGEST=$(echo "$CURRENT_IMAGE" | grep -o 'sha256:[a-f0-9]*')

    # Find previous image (first one that's not current)
    PREVIOUS_DIGEST=""
    for digest in $IMAGES; do
        if [ "$digest" != "$CURRENT_DIGEST" ]; then
            PREVIOUS_DIGEST="$digest"
            break
        fi
    done

    if [ -z "$PREVIOUS_DIGEST" ]; then
        echo "❌ Error: No previous image found to rollback to"
        return 1
    fi

    PREVIOUS_IMAGE="$IMAGE_BASE@$PREVIOUS_DIGEST"
    echo "Previous image: $PREVIOUS_IMAGE"
    echo ""

    if [ "$SKIP_CONFIRM" = false ]; then
        read -p "Rollback checker job to previous image? (y/N): " confirm
        if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
            echo "Rollback cancelled"
            return 1
        fi
    fi

    echo "Rolling back checker job..."
    gcloud run jobs update "$CHECKER_JOB" \
        --region="$REGION" \
        --project="$PROJECT_ID" \
        --image="$PREVIOUS_IMAGE" \
        --quiet

    echo "✓ Checker job rolled back to: $PREVIOUS_IMAGE"
    echo ""
}

# Show current state
if [ "$ROLLBACK_API" = true ]; then
    show_api_revisions || exit 1
fi

if [ "$ROLLBACK_JOB" = true ]; then
    show_job_info || exit 1
fi

# Perform rollback
ROLLBACK_FAILED=false

if [ "$ROLLBACK_API" = true ]; then
    if ! rollback_api; then
        ROLLBACK_FAILED=true
    fi
fi

if [ "$ROLLBACK_JOB" = true ]; then
    if ! rollback_job; then
        ROLLBACK_FAILED=true
    fi
fi

# Summary
echo "=========================================="
if [ "$ROLLBACK_FAILED" = true ]; then
    echo "❌ Rollback completed with errors"
    echo "=========================================="
    exit 1
else
    echo "✓ Rollback completed successfully!"
    echo "=========================================="
    echo ""
    echo "Verification steps:"
    if [ "$ROLLBACK_API" = true ]; then
        API_URL=$(gcloud run services describe "$API_SERVICE" \
            --region="$REGION" \
            --project="$PROJECT_ID" \
            --format='value(status.url)')
        echo "1. Test API: curl $API_URL/status"
    fi
    if [ "$ROLLBACK_JOB" = true ]; then
        echo "2. Test job: gcloud run jobs execute $CHECKER_JOB --region=$REGION"
    fi
    echo ""
fi
