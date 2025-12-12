# Generated Artifacts

Storage and management of training data, models, and runtime files using git-annex for large file handling.

## Directory Structure

```
artifacts/
├── training_data/      # ML training dataset (git-annex tracked)
│   ├── images/         # 2,666 labeled Muni subway status snapshots
│   └── labels.json     # Image labels with status and descriptions
├── models/             # Trained ML models (git-annex tracked)
│   └── v1/             # Model version 1 (BLIP-based classifier)
└── runtime/            # Transient runtime data (gitignored)
    ├── cache/          # API response cache
    └── downloads/      # Recent snapshot downloads
```

## Data Management with git-annex

Training data and models use git-annex for efficient large file storage:

- **Symlinks in Git**: Only lightweight symlinks are version controlled
- **Actual Files**: Stored in `.git/annex/objects/` and synced to Google Cloud Storage
- **Selective Download**: Retrieve only required files

### Storage Allocation

```
~270MB  artifacts/training_data/images/    # 2,666 training images
856MB   artifacts/models/v1/               # BLIP model + classifier
570KB   artifacts/training_data/labels.json  # Training labels (unlocked)
```

**Total**: ~1.1GB tracked via git-annex in Google Cloud Storage bucket `munimetro-annex` (within 5GB free tier).

For configuration details, see [CONFIGURATION.md](../CONFIGURATION.md).

## Initial Setup

### Collaborator Onboarding

After cloning the repository:

```bash
# Initialize git-annex
git annex init "machine-name"

# Enable Google Cloud Storage remote
# Note: Requires rclone configuration for GCS access
git annex enableremote google-cloud

# Download required files
git annex get artifacts/models/v1/        # Model files (856MB)
git annex get artifacts/training_data/    # Training data (270MB)

# Unlock labels for editing (required for labeling workflow)
git annex unlock artifacts/training_data/labels.json
```

For rclone configuration, see [GCS_SETUP.md](../GCS_SETUP.md).

### Verification

```bash
# Check file locations (local/cloud/both)
git annex whereis artifacts/training_data/
git annex whereis artifacts/models/v1/

# Display repository statistics
git annex info
```

## Workflows

### Collecting Training Data

```bash
# Download new snapshots
cd training
python download_muni_image.py

# Add to git-annex
cd ..
git annex add artifacts/training_data/images/

# Upload to cloud storage
git annex copy artifacts/training_data/images/ --to=google-cloud --jobs=4

# Commit and push
git commit -m "Add training snapshots"
git push
```

### Updating Training Labels

The `labels.json` file must be unlocked to enable editing:

```bash
# Unlock file (first time only per repository clone)
git annex unlock artifacts/training_data/labels.json

# Run labeling tool
cd training
python label_images.py  # Modifies artifacts/training_data/labels.json

# Commit changes
cd ..
git add artifacts/training_data/labels.json
git commit -m "Update training labels"

# Upload to cloud
git annex copy artifacts/training_data/labels.json --to=google-cloud
git push
```

**Rationale**: Labels file requires write access for the labeling workflow. The `unlock` command replaces the symlink with the actual file while maintaining git-annex tracking.

**Persistence**: Unlocked files remain unlocked across commits, requiring unlock only once per clone.

### Training and Uploading Models

```bash
# Train model
cd training
python train_model.py  # Saves to artifacts/models/v1/

# Add new model files
cd ..
git annex add artifacts/models/v1/

# Upload to cloud storage
git annex copy artifacts/models/v1/ --to=google-cloud

# Commit and push
git commit -m "Update trained model"
git push
```

## Command Reference

| Command | Function |
|---------|----------|
| `git annex get FILE` | Download file from cloud storage |
| `git annex drop FILE` | Remove local copy (preserves cloud copy) |
| `git annex copy --to=google-cloud` | Upload files to cloud storage |
| `git annex whereis FILE` | Show file locations |
| `git annex unlock FILE` | Make file writable |
| `git annex sync --content` | Synchronize all changes |
| `git annex info` | Display repository statistics |

### Disk Space Management

```bash
# Remove local copies to free disk space
# Symlinks remain, files stay in cloud
git annex drop artifacts/training_data/images/
git annex drop artifacts/models/

# Restore files from cloud
git annex get artifacts/training_data/
git annex get artifacts/models/
```

## Storage Configuration

- **Backend**: Google Cloud Storage (`gs://munimetro-annex`)
- **Region**: `us-west1` (Oregon)
- **Cost**: $0/month (within 5GB free tier)
- **Current Usage**: ~1.1GB (training data + models)
- **Chunk Size**: 50MiB for transfer optimization

See [CONFIGURATION.md](../CONFIGURATION.md) for complete configuration details.

## Troubleshooting

### Remote Not Available

```bash
# Re-enable cloud remote
git annex enableremote google-cloud

# Verify remote configuration
git annex info google-cloud
```

### Authentication Errors

```bash
# Re-authenticate with Google Cloud
gcloud auth application-default login

# Verify active account
gcloud auth list
```

### Broken Symlinks

Cause: Files not downloaded from cloud storage

Solution:
```bash
# Download all annexed files
git annex get .

# Or download specific directories
git annex get artifacts/training_data/
git annex get artifacts/models/
```

### Verify Cloud Storage

```bash
# Show all files and their locations
git annex whereis

# Count files in cloud storage
git annex find --in=google-cloud | wc -l

# List annexed files
git annex list
```

### Transfer Failures

```bash
# Enable debug logging
git annex --debug copy --to=google-cloud

# Test rclone connectivity
rclone lsd munimetro-gcs:

# Verify bucket access
gsutil ls gs://munimetro-annex
```

## Related Documentation

- **Git-annex Setup**: [GCS_SETUP.md](../GCS_SETUP.md) - Google Cloud Storage configuration
- **Configuration**: [CONFIGURATION.md](../CONFIGURATION.md) - Deployment configuration values
- **Training**: [training/README.md](../training/README.md) - Model training workflow
- **Deployment**: [deploy/README.md](../deploy/README.md) - API deployment guide
