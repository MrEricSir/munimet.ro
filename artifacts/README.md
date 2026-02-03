# Generated Artifacts

This directory contains reference data and runtime files for the Muni Metro status detection system.

## Directory Structure

```
artifacts/
├── reference_data/     # Labeled reference images (~270MB) - synced from GCS
│   ├── images/         # Labeled Muni subway status snapshots
│   └── labels.json     # Image labels with status and descriptions
└── runtime/            # Transient runtime data (gitignored)
    ├── cache/          # API response cache
    └── downloads/      # Recent snapshot downloads
```

## Storage Management

Reference data is stored in Google Cloud Storage. The actual files are **not** stored in git to keep the repository lightweight.

### Storage Allocation

```
~270MB  reference_data/  # Reference images and labels
```

**Location**: Google Cloud Storage bucket `munimetro-annex` (within 5GB free tier).

For configuration details, see [CONFIGURATION.md](../CONFIGURATION.md).

## Downloading Artifacts

### Prerequisites

1. Install Google Cloud SDK (see [SETUP.md](../SETUP.md))
2. Authenticate with Google Cloud:
   ```bash
   gcloud auth login
   gcloud config set project munimetro
   ```

### Download Reference Data

```bash
# Download reference data
./scripts/sync-artifacts.sh download    # macOS/Linux
.\scripts\sync-artifacts.ps1 download   # Windows

# Or use the specific script
./scripts/sync-reference-data.sh download    # macOS/Linux
.\scripts\sync-reference-data.ps1 download   # Windows
```

The sync scripts use `gsutil rsync` to efficiently download only changed files.

## Uploading Artifacts

After collecting new reference data, upload to Google Cloud Storage:

```bash
# Upload reference data
./scripts/sync-reference-data.sh upload    # macOS/Linux
.\scripts\sync-reference-data.ps1 upload   # Windows
```

## Workflows

### Collecting Reference Images

```bash
# 1. Collect new images
python scripts/download_muni_image.py

# 2. View detection results (auto-opens browser)
python scripts/detection_viewer.py path/to/image.jpg

# 3. Upload to cloud storage
./scripts/sync-reference-data.sh upload    # macOS/Linux
.\scripts\sync-reference-data.ps1 upload   # Windows
```

### Viewing Detection Results

Use the detection viewer to visualize station/train detection on images:

```bash
# View a single image (auto-opens browser)
python scripts/detection_viewer.py path/to/image.jpg

# Browse all reference images
python scripts/detection_viewer.py

# Browse images from a custom folder
python scripts/detection_viewer.py --folder path/to/images/
```

## Bidirectional Sync

To sync changes in both directions (upload local changes and download remote changes):

```bash
./scripts/sync-reference-data.sh both    # macOS/Linux
.\scripts\sync-reference-data.ps1 both   # Windows
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
gsutil ls gs://munimetro-annex/reference_data/

# Run sync with verbose output
gsutil -m rsync -r gs://munimetro-annex/reference_data artifacts/reference_data

# Check local directory exists
ls -la artifacts/
```

## Related Documentation

- **Setup Guide**: [SETUP.md](../SETUP.md) - Environment setup and Google Cloud authentication
- **Deployment Guide**: [deploy/README.md](../deploy/README.md) - API deployment
- **Configuration**: [CONFIGURATION.md](../CONFIGURATION.md) - System configuration values
