# Data Management Guide

This project uses **git-annex with Google Cloud Storage** to manage large files (training data and models).

## Quick Reference

| What you want to do | Command |
|---------------------|---------|
| Download model from cloud | `git annex get artifacts/models/v1/` |
| Download training data | `git annex get artifacts/training_data/images/` |
| Upload new snapshots to cloud | `git annex copy artifacts/training_data/images/ --to=google-cloud` |
| Check what's in cloud | `git annex whereis` |
| Free up local disk space | `git annex drop artifacts/training_data/images/` |

## Current Data Size

```bash
~270MB  artifacts/training_data/images/       # 2,600+ training images
856MB   artifacts/models/v1/      # BLIP model + classifier
570KB   artifacts/training_data/labels.json  # Training labels (unlocked for editing)
```

**Total: ~1.1GB** tracked with git-annex, stored in Google Cloud Storage (FREE - under 5GB tier).

## How It Works

Large files are tracked as symlinks in Git, with actual content stored in Google Cloud Storage:

- **In Git**: Small symlink files (~200 bytes each) - fast to clone and pull
- **In GCS**: Actual large files (bucket: gs://munimetro-annex/) - permanent storage
- **Local**: Files you've downloaded via `git annex get` - on-demand

This keeps the Git repository small (~10MB) while making large files (1.1GB) available on-demand.

---

## Daily Workflow (Project Owner)

### Collecting New Training Data

```bash
# 1. Run download script to collect new snapshots
cd training
python download_muni_image.py

# 2. New images are automatically tracked by git-annex
cd ..
git annex add artifacts/training_data/images/

# 3. Upload to Google Cloud Storage (runs in background)
git annex copy artifacts/training_data/images/ --to=google-cloud --jobs=4

# 4. Commit and push the symlinks
git add artifacts/training_data/images/
git commit -m "Add new training snapshots"
git push
```

### Updating Training Labels

```bash
# training_labels.json is "unlocked" - you can edit it directly
cd training
python label_images.py  # This modifies artifacts/training_data/labels.json

# After labeling, commit and upload
cd ..
git add artifacts/training_data/labels.json
git commit -m "Update training labels"
git annex copy artifacts/training_data/labels.json --to=google-cloud
git push
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
git add models/
git commit -m "Update trained model"
git push
```

---

## For Collaborators

When you clone this repository, you'll see symlinks for large files. Follow these steps to get the actual files:

### 1. Install Prerequisites

```bash
brew install git-annex rclone git-annex-remote-rclone
```

### 2. Initialize git-annex

```bash
cd munimetro
git annex init "your-laptop-name"
```

### 3. Enable Google Cloud Storage Remote

```bash
git annex enableremote google-cloud
```

This will prompt you to authenticate with Google Cloud via your browser.

### 4. Download Files You Need

```bash
# Option A: Get everything (1.1GB)
git annex get .

# Option B: Get only the model (for running the API)
git annex get artifacts/models/v1/

# Option C: Get only training data (for retraining)
git annex get data/

# Option D: Get specific file
git annex get artifacts/training_data/images/muni_snapshot_20251206_134756.jpg
```

### 5. (Optional) Unlock Labels for Editing

If you want to edit training labels:

```bash
git annex unlock artifacts/training_data/labels.json
```

Now you can modify the file with `label_images.py`.

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
