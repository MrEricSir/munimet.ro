# Generated Artifacts

This directory contains all generated data for the MuniMet.ro project, organized by type and persistence.

## Directory Structure

```
artifacts/
├── training_data/      # ML training dataset (git-annex tracked)
│   ├── images/         # ~2800 labeled Muni subway status snapshots
│   └── labels.json     # Image labels with status and descriptions
├── models/             # Trained ML models (git-annex tracked)
│   └── v1/             # Model version 1 (BLIP-based classifier)
└── runtime/            # Transient runtime data (gitignored)
    ├── cache/          # API response cache
    └── downloads/      # Recent snapshot downloads
```

## Data Management with git-annex

The `training_data/` and `models/` directories use **git-annex** for efficient large file storage:

- **Symlinks in Git**: Only lightweight symlinks are tracked in the Git repository
- **Actual files**: Stored in `.git/annex/objects/` and synced to Google Cloud Storage
- **Selective download**: Download only the files you need

### Current Data Size

```bash
~270MB  artifacts/training_data/images/    # 2,666 training images
856MB   artifacts/models/v1/               # BLIP model + classifier
570KB   artifacts/training_data/labels.json  # Training labels (unlocked for editing)
```

**Total: ~1.1GB** tracked with git-annex, stored in Google Cloud Storage (FREE - under 5GB tier).

---

## Quick Start

### For New Collaborators

After cloning the repository, initialize git-annex and download the files you need:

```bash
# 1. Initialize git-annex
git annex init "your-laptop-name"

# 2. Enable Google Cloud Storage remote (authenticates via browser)
git annex enableremote google-cloud

# 3. Download what you need
git annex get artifacts/models/v1/                    # For running API (856MB)
git annex get artifacts/training_data/                # For training (270MB)
```

### Check what's available

```bash
# See which files are present locally vs in cloud
git annex whereis artifacts/training_data/
git annex whereis artifacts/models/v1/

# Show repository stats
git annex info
```

---

## Daily Workflows

### Collecting New Training Data

```bash
# 1. Download new snapshots
cd training
python download_muni_image.py

# 2. Add to git-annex and upload to cloud
cd ..
git annex add artifacts/training_data/images/
git annex copy artifacts/training_data/images/ --to=google-cloud --jobs=4

# 3. Commit and push
git commit -m "Add new training snapshots"
git push
```

### Updating Training Labels

The `labels.json` file is **unlocked** in git-annex, so you can edit it directly:

```bash
# Labels file is already unlocked - just edit it
cd training
python label_images.py  # Modifies artifacts/training_data/labels.json

# After labeling, commit and upload
cd ..
git add artifacts/training_data/labels.json
git commit -m "Update training labels"
git annex copy artifacts/training_data/labels.json --to=google-cloud
git push
```

**Why unlocked?** The labels file needs to be writable for the labeling workflow. Git-annex `unlock` makes annexed files editable while still tracking them in the annex for cloud storage.

If you ever need to unlock other files for editing:
```bash
git annex unlock <file>
```

### Training and Uploading New Model

```bash
# 1. Train the model
cd training
python train_model.py  # Saves to artifacts/models/v1/

# 2. Add and upload new model files
cd ..
git annex add artifacts/models/v1/
git annex copy artifacts/models/v1/ --to=google-cloud

# 3. Commit and push
git commit -m "Update trained model"
git push
```

---

## Useful Commands

| Command | Purpose |
|---------|---------|
| `git annex get <file>` | Download file from cloud |
| `git annex drop <file>` | Remove local copy (keeps in cloud) |
| `git annex copy --to=google-cloud` | Upload files to cloud |
| `git annex whereis <file>` | Show where file is stored |
| `git annex unlock <file>` | Make file editable |
| `git annex sync --content` | Sync everything |
| `git annex info` | Show repository stats |

### Free up local disk space

```bash
# Remove local copies (symlinks remain, files stay in cloud)
git annex drop artifacts/training_data/images/
git annex drop artifacts/models/

# Download them again anytime with:
git annex get artifacts/training_data/
```

---

## Storage Details

- **Backend**: Google Cloud Storage (gs://munimetro-annex)
- **Region**: us-west1 (close to San Francisco)
- **Cost**: $0/month (under Google Cloud's 5GB free tier)
- **Current Usage**: ~1.1GB (training data + models)
- **Chunking**: 50MiB chunks for efficient transfer

---

## Troubleshooting

### "Remote not available"
```bash
git annex enableremote google-cloud
```

### "Permission denied"
```bash
# Re-authenticate with Google Cloud
gcloud auth application-default login
```

### Files showing as broken symlinks
```bash
# Download the files from cloud
git annex get .
```

### Check what's uploaded to cloud
```bash
# Show files and their locations
git annex whereis

# Count files in cloud
git annex find --in=google-cloud | wc -l
```

---

## More Information

- Full git-annex setup guide: [`../GCS_SETUP.md`](../GCS_SETUP.md)
- Training workflow: [`../training/README.md`](../training/README.md)
- API deployment: [`../api/README.md`](../api/README.md)
