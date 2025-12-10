# Data Management Guide

This project uses **git-annex with Google Cloud Storage** to manage large files (training data and models).

## Current Data Size

```bash
268MB  data/muni_snapshots/       # 2,601 training images
856MB  models/trained_model/      # BLIP model + classifier
570KB  data/training_labels.json  # Training labels
```

**Total: ~1.1GB** tracked with git-annex, stored in Google Cloud Storage (FREE - under 5GB tier).

## How It Works

Large files are tracked as symlinks in Git, with actual content stored in Google Cloud Storage:

- **In Git**: Small symlink files (~200 bytes each)
- **In GCS**: Actual large files (bucket: gs://munimetro-annex/)
- **Local**: Files you've downloaded via `git annex get`

This keeps the Git repository small while making large files available on demand.

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
git annex get models/trained_model/

# Option C: Get only training data (for retraining)
git annex get data/

# Option D: Get specific file
git annex get data/muni_snapshots/muni_snapshot_20251206_134756.jpg
```

### 5. (Optional) Unlock Labels for Editing

If you want to edit training labels:

```bash
git annex unlock data/training_labels.json
```

Now you can modify the file with `label_images.py`.

---

## For Project Owner

### Adding New Files

When you add new images or update the model, git-annex handles it automatically:

```bash
# Add files (git-annex tracks them automatically)
git annex add data/muni_snapshots/new_image.jpg

# Upload to Google Cloud Storage
git annex copy data/muni_snapshots/new_image.jpg --to=google-cloud

# Commit the symlink
git add data/muni_snapshots/new_image.jpg
git commit -m "Add new training image"
```

### Uploading All Changes

```bash
# Upload everything not yet in cloud storage
git annex copy --to=google-cloud --jobs=4
```

### Checking File Locations

```bash
# See where files are stored
git annex whereis data/muni_snapshots/

# See what's available locally
git annex find --in=here

# See what's in Google Cloud
git annex find --in=google-cloud
```

### Freeing Up Local Disk Space

```bash
# Remove local copies (keeps them in cloud)
git annex drop data/muni_snapshots/

# Get them back later
git annex get data/muni_snapshots/
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

---

## Storage Details

- **Bucket**: gs://munimetro-annex
- **Region**: us-west1 (close to San Francisco)
- **Cost**: $0/month (under Google Cloud's 5GB free tier)
- **Chunking**: 50MiB chunks for efficient transfer
- **Encryption**: None (data is not sensitive)

For more details on the setup, see [GCS_SETUP.md](GCS_SETUP.md).

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
# Download the files
git annex get .
```

### Want to switch to a different storage backend?
git-annex supports multiple remotes. You can add S3, another GCS bucket, or even a local USB drive:
```bash
git annex initremote backup-usb type=directory directory=/Volumes/backup/munimetro encryption=none
git annex copy --to=backup-usb
```
