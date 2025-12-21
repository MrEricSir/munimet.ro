#!/bin/bash
# Migrate from git-annex to direct Google Cloud Storage
#
# This script helps transition from git-annex to simple GCS bucket access
# Run this ONCE on the machine with all the data (Mac with files present)

set -e

BUCKET="gs://munimetro-annex"
PROJECT_ID="${GCP_PROJECT_ID:-munimetro}"

echo "=========================================="
echo "Migrate to Direct Google Cloud Storage"
echo "=========================================="
echo ""
echo "This will:"
echo "1. Upload all training data to GCS"
echo "2. Upload all model data to GCS"
echo "3. Create download script for other machines"
echo ""

# Check authentication
echo "[1/4] Checking Google Cloud authentication..."
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q .; then
    echo "Not authenticated. Running gcloud auth login..."
    gcloud auth login
fi

echo "Authenticated as: $(gcloud auth list --filter=status:ACTIVE --format='value(account)')"
echo ""

# Set project
echo "[2/4] Setting project to $PROJECT_ID..."
gcloud config set project "$PROJECT_ID"
echo ""

# Upload training data
echo "[3/4] Uploading training data to $BUCKET/training_data/..."
echo "This may take a few minutes (~270MB)..."

# Get files from git-annex if needed
if [ -d "artifacts/training_data/images" ]; then
    # Check if files are symlinks (git-annex) and get them
    if [ -L "artifacts/training_data/images/$(ls artifacts/training_data/images | head -1)" ]; then
        echo "Getting files from git-annex first..."
        git annex get artifacts/training_data/
    fi
fi

gsutil -m rsync -r artifacts/training_data "$BUCKET/training_data"
echo "✓ Training data uploaded"
echo ""

# Upload models
echo "[4/4] Uploading models to $BUCKET/models/..."
if [ -d "artifacts/models/v1" ]; then
    # Get from git-annex if needed
    if [ -L "artifacts/models/v1/model.safetensors" ]; then
        echo "Getting model from git-annex first..."
        git annex get artifacts/models/v1/
    fi

    gsutil -m rsync -r artifacts/models "$BUCKET/models"
    echo "✓ Models uploaded"
else
    echo "No models found, skipping..."
fi
echo ""

# Create download script
cat > scripts/setup/download-from-gcs.sh << 'EOF'
#!/bin/bash
# Download training data and models from Google Cloud Storage
#
# Run this on any machine to get the latest data

set -e

BUCKET="gs://munimetro-annex"
PROJECT_ID="${GCP_PROJECT_ID:-munimetro}"

echo "=========================================="
echo "Download from Google Cloud Storage"
echo "=========================================="
echo ""

# Check authentication
echo "[1/3] Checking Google Cloud authentication..."
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q .; then
    echo "Not authenticated. Running gcloud auth login..."
    gcloud auth login
fi

echo "Authenticated as: $(gcloud auth list --filter=status:ACTIVE --format='value(account)')"
gcloud config set project "$PROJECT_ID"
echo ""

# Download training data
echo "[2/3] Downloading training data from GCS (~270MB)..."
mkdir -p artifacts/training_data
gsutil -m rsync -r "$BUCKET/training_data" artifacts/training_data
echo "✓ Training data downloaded"
echo ""

# Download models
echo "[3/3] Downloading models from GCS (~856MB)..."
mkdir -p artifacts/models
gsutil -m rsync -r "$BUCKET/models" artifacts/models
echo "✓ Models downloaded"
echo ""

echo "=========================================="
echo "Download complete!"
echo "=========================================="
echo ""
echo "Training data: artifacts/training_data/"
echo "Models: artifacts/models/"
EOF

chmod +x scripts/setup/download-from-gcs.sh

echo "=========================================="
echo "Migration complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo ""
echo "1. On other machines, run:"
echo "   ./scripts/setup/download-from-gcs.sh"
echo ""
echo "2. To upload new data/models, use:"
echo "   gsutil -m rsync -r artifacts/training_data $BUCKET/training_data"
echo "   gsutil -m rsync -r artifacts/models $BUCKET/models"
echo ""
echo "3. Optional: Remove git-annex (keep or remove as you prefer):"
echo "   git annex uninit"
echo ""
