# Google Cloud Storage Setup for git-annex

This guide explains how to configure Google Cloud Storage (GCS) as the remote backend for git-annex to store your 1.1GB of training data and models.

**Status**: ✅ **Already configured** for the munimetro project! This doc explains the setup for reference and for collaborators.

## Prerequisites

- Google Cloud account
- `gcloud` CLI installed (`brew install google-cloud-sdk`)
- `rclone` installed (`brew install rclone`)
- `git-annex-remote-rclone` installed (`brew install git-annex-remote-rclone`)
- Git-annex already initialized (done)

## Why Google Cloud Storage?

- **FREE**: Your 1.1GB fits within the 5GB free tier
- **Reliable**: Google's infrastructure
- **Fast**: Good upload/download speeds
- **No lock-in**: Can switch to S3 or other backends later

---

## Setup Steps (Already Completed)

The munimetro project is already set up with GCS. Here's what was done:

### 1. Install Required Tools

```bash
brew install google-cloud-sdk rclone git-annex-remote-rclone
```

### 2. Authenticate with Google Cloud

```bash
gcloud auth login
gcloud config set project munimetro
```

### 3. Enable API and Create Bucket

```bash
gcloud services enable storage-api.googleapis.com
gsutil mb -p munimetro -l us-west1 gs://munimetro-annex
```

### 4. Configure rclone for GCS

```bash
rclone config create munimetro-gcs gcs project_number=munimetro bucket_policy_only=true
# This opens a browser for OAuth authentication
```

### 5. Initialize git-annex Remote

```bash
git annex initremote google-cloud \
  type=external \
  externaltype=rclone \
  target=munimetro-gcs \
  prefix=munimetro-annex \
  chunk=50MiB \
  encryption=none \
  rclone_layout=lower
```

### 6. Upload Files

```bash
# Upload all annexed files (1.1GB - takes ~10-20 minutes)
git annex copy --to=google-cloud --jobs=4
```

---

## Uploading Files to GCS

Once the remote is configured, upload your annexed files:

```bash
# Copy all annexed files to Google Cloud
git annex copy --to=google-cloud

# Or copy specific directories
git annex copy artifacts/training_data/images/ --to=google-cloud
git annex copy artifacts/models/v1/ --to=google-cloud
git annex copy artifacts/training_data/labels.json --to=google-cloud

# Check what's been uploaded
git annex whereis
```

This will take some time (~1.1GB upload).

---

## Collaborator Workflow

When a collaborator clones your repo, they'll see symlinks. Here's how they get the actual files:

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/munimetro.git
cd munimetro

# 2. Initialize git-annex
git annex init "collaborator-laptop"

# 3. Enable the Google Cloud remote
git annex enableremote google-cloud

# 4. Download only the files they need
# Option A: Get everything
git annex get .

# Option B: Get only the model (for API usage)
git annex get artifacts/models/v1/

# Option C: Get only the training data (for retraining)
git annex get data/

# 5. If editing training labels, unlock the file
git annex unlock artifacts/training_data/labels.json

# 6. Verify files are available
ls -lh artifacts/training_data/images/ | head
ls -lh artifacts/models/v1/
```

**Important**: `training_labels.json` must be unlocked to edit it with `label_images.py`. The file stays tracked in git-annex but becomes writable.

---

## Useful Commands

### Check file locations
```bash
# See where files are stored (local, cloud, both)
git annex whereis artifacts/training_data/images/

# See what's available locally
git annex find --in=here

# See what's in Google Cloud
git annex find --in=google-cloud
```

### Free up local disk space
```bash
# Remove local copies (keeps them in cloud)
git annex drop artifacts/training_data/images/

# Get them back later
git annex get artifacts/training_data/images/
```

### Sync with cloud
```bash
# Upload new files
git annex copy --to=google-cloud

# Download updates from cloud
git annex sync --content
```

---

## Cost Monitoring

Monitor your Google Cloud Storage usage:

```bash
# Check bucket size
gsutil du -sh gs://$BUCKET_NAME

# List all objects
gsutil ls -lh gs://$BUCKET_NAME/munimetro/
```

**Expected cost**: $0/month (under 5GB free tier)

---

## Troubleshooting

### "Remote not available"
```bash
# Re-enable the remote
git annex enableremote google-cloud

# Check remote status
git annex info google-cloud
```

### "Permission denied"
```bash
# Re-authenticate
gcloud auth application-default login
```

### "Bucket not found"
```bash
# Verify bucket exists
gsutil ls

# Check bucket permissions
gsutil iam get gs://$BUCKET_NAME
```

---

## Next Steps

1. **Test the setup**: Upload a few files first
2. **Push to GitHub**: Push your git repo (only symlinks, not actual files)
3. **Document for collaborators**: Add instructions to README

See [STORAGE_COMPARISON.md](STORAGE_COMPARISON.md) for alternatives if you want to switch later.
