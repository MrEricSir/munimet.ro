# Generated Artifacts

This directory contains training data, models, and runtime files for the Muni Metro status classifier.

## Directory Structure

```
artifacts/
├── training_data/      # ML training dataset (~270MB)
│   ├── images/         # 2,666 labeled Muni subway status snapshots
│   └── labels.json     # Image labels with status and descriptions
├── models/             # Trained ML models (~856MB)
│   └── v1/             # Model version 1 (BLIP-based classifier)
│       ├── model.safetensors      # Vision transformer weights (854MB)
│       ├── status_classifier.pt   # Classification head (775KB)
│       ├── outlier_report.json    # Model evaluation outliers
│       └── [config files]         # Model configuration
└── runtime/            # Transient runtime data (gitignored)
    ├── cache/          # API response cache
    └── downloads/      # Recent snapshot downloads
```

## Storage Management

Training data and models are stored in Google Cloud Storage and synced using rsync scripts. The actual files are **not** stored in git to keep the repository lightweight.

### Storage Allocation

```
~270MB  artifacts/training_data/     # Training images and labels
~856MB  artifacts/models/v1/         # BLIP model + classifier
```

**Total**: ~1.1GB stored in Google Cloud Storage bucket `munimetro-annex` (within 5GB free tier).

For configuration details, see [CONFIGURATION.md](../CONFIGURATION.md).

## Downloading Artifacts

### Prerequisites

1. Install Google Cloud SDK (see [SETUP.md](../SETUP.md))
2. Authenticate with Google Cloud:
   ```bash
   gcloud auth login
   gcloud config set project munimetro
   ```

### Download All Artifacts

```bash
# Download both training data and models
./scripts/sync-artifacts.sh download    # macOS/Linux
.\scripts\sync-artifacts.ps1 download   # Windows
```

### Download Individually

```bash
# Training data only (~270MB)
./scripts/sync-training-data.sh download    # macOS/Linux
.\scripts\sync-training-data.ps1 download   # Windows

# Models only (~856MB)
./scripts/sync-models.sh download    # macOS/Linux
.\scripts\sync-models.ps1 download   # Windows
```

The sync scripts use `gsutil rsync` to efficiently download only changed files.

## Uploading Artifacts

After collecting new training data or training a new model, upload to Google Cloud Storage:

### Upload All Artifacts

```bash
# Upload both training data and models
./scripts/sync-artifacts.sh upload    # macOS/Linux
.\scripts\sync-artifacts.ps1 upload   # Windows
```

### Upload Individually

```bash
# Training data only
./scripts/sync-training-data.sh upload    # macOS/Linux
.\scripts\sync-training-data.ps1 upload   # Windows

# Models only
./scripts/sync-models.sh upload    # macOS/Linux
.\scripts\sync-models.ps1 upload   # Windows
```

## Workflows

### Collecting and Uploading Training Data

```bash
# 1. Collect new images
cd training
python download_muni_image.py

# 2. Label images
python label_images.py

# 3. Upload to cloud storage
cd ..
./scripts/sync-training-data.sh upload    # macOS/Linux
.\scripts\sync-training-data.ps1 upload   # Windows
```

### Training and Uploading Models

```bash
# 1. Train model
cd training
python train_model.py  # Saves to artifacts/models/v1/

# 2. Upload to cloud storage
cd ..
./scripts/sync-models.sh upload    # macOS/Linux
.\scripts\sync-models.ps1 upload   # Windows
```

### Updating Training Labels

The `labels.json` file is a regular file that can be edited directly:

```bash
# 1. Run labeling tool
cd training
python label_images.py  # Modifies artifacts/training_data/labels.json

# 2. Upload changes to cloud
cd ..
./scripts/sync-training-data.sh upload    # macOS/Linux
.\scripts\sync-training-data.ps1 upload   # Windows
```

## Bidirectional Sync

To sync changes in both directions (upload local changes and download remote changes):

```bash
# Sync everything
./scripts/sync-artifacts.sh both    # macOS/Linux
.\scripts\sync-artifacts.ps1 both   # Windows

# Or sync individually
./scripts/sync-training-data.sh both
./scripts/sync-models.sh both
```

## Troubleshooting

### Authentication Errors

**Symptom:** Access denied when trying to download/upload files

**Solution:**
```bash
# Verify authentication
gcloud auth list

# Re-authenticate if needed
gcloud auth login

# Set project
gcloud config set project munimetro

# Verify access to bucket
gsutil ls gs://munimetro-annex
```

### Slow Downloads/Uploads

**Symptom:** Transfers are very slow

**Solution:**
- The sync scripts use `gsutil -m` for parallel transfers
- Check network connectivity
- Try different network (WiFi vs Ethernet)
- Verify bucket region is optimal (currently `us-west1`)

### Files Not Syncing

**Symptom:** Files don't appear after running sync script

**Solution:**
```bash
# Verify files exist in cloud
gsutil ls gs://munimetro-annex/training_data/
gsutil ls gs://munimetro-annex/models/

# Run sync with verbose output
gsutil -m rsync -r gs://munimetro-annex/training_data artifacts/training_data

# Check local directory exists
ls -la artifacts/
```

### Disk Space Issues

**Symptom:** Not enough space to download all artifacts

**Solution:**
- Download only what you need (use individual sync scripts)
- Training data: ~270MB
- Models: ~856MB
- Free up space before downloading
- Consider downloading to external drive

## Related Documentation

- **Setup Guide**: [SETUP.md](../SETUP.md) - Environment setup and Google Cloud authentication
- **Training Guide**: [training/README.md](../training/README.md) - Model training workflow
- **Deployment Guide**: [deploy/README.md](../deploy/README.md) - API deployment
- **Configuration**: [CONFIGURATION.md](../CONFIGURATION.md) - System configuration values
