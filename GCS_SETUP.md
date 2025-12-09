# Google Cloud Storage Setup for git-annex

This guide explains how to configure Google Cloud Storage (GCS) as the remote backend for git-annex to store your 1.1GB of training data and models.

## Prerequisites

- Google Cloud account
- `gcloud` CLI installed (`brew install google-cloud-sdk`)
- Git-annex already initialized (done)

## Why Google Cloud Storage?

- **FREE**: Your 1.1GB fits within the 5GB free tier
- **Reliable**: Google's infrastructure
- **Fast**: Good upload/download speeds
- **No lock-in**: Can switch to S3 or other backends later

---

## Setup Steps

### 1. Install Google Cloud SDK (if not already installed)

```bash
brew install google-cloud-sdk
```

### 2. Authenticate with Google Cloud

```bash
gcloud auth login
gcloud auth application-default login
```

### 3. Create a Google Cloud Project

```bash
# Set your project ID (choose a unique name)
export PROJECT_ID="munimetro-data"

# Create project
gcloud projects create $PROJECT_ID --name="Muni Metro Data"

# Set as active project
gcloud config set project $PROJECT_ID
```

### 4. Enable Cloud Storage API

```bash
gcloud services enable storage-api.googleapis.com
```

### 5. Create a GCS Bucket

```bash
# Choose a globally unique bucket name
export BUCKET_NAME="munimetro-annex-$(whoami)"

# Create bucket in us-west region (close to SF)
gsutil mb -l us-west1 gs://$BUCKET_NAME

# Verify bucket was created
gsutil ls
```

### 6. Configure git-annex to use GCS

```bash
# Initialize GCS remote in git-annex
git annex initremote google-cloud \
  type=external \
  externaltype=rclone \
  target=google-cloud \
  prefix=munimetro/ \
  chunk=100MiB \
  encryption=none

# Alternative: Using gcrypt for encryption
# git annex initremote google-cloud \
#   type=gcrypt \
#   gitrepo=gcrypt::rclone://google-cloud:munimetro-data \
#   encryption=shared
```

**Note**: The above uses `rclone` as a bridge. Install it first:

```bash
brew install rclone
rclone config  # Follow prompts to configure Google Drive/GCS
```

### 7. Alternative: Direct GCS Setup (Simpler)

If the rclone method seems complex, use the `git-annex-remote-googledrive` special remote:

```bash
# Install the special remote
brew install git-annex-remote-googledrive

# Configure it
git annex initremote google \
  type=external \
  externaltype=googledrive \
  prefix=munimetro/ \
  root_folder_id=YOUR_FOLDER_ID \
  encryption=none
```

---

## Uploading Files to GCS

Once the remote is configured, upload your annexed files:

```bash
# Copy all annexed files to Google Cloud
git annex copy --to=google-cloud

# Or copy specific directories
git annex copy data/muni_snapshots/ --to=google-cloud
git annex copy models/trained_model/ --to=google-cloud
git annex copy data/training_labels.json --to=google-cloud

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
git annex get models/trained_model/

# Option C: Get only the training data (for retraining)
git annex get data/

# 5. If editing training labels, unlock the file
git annex unlock data/training_labels.json

# 6. Verify files are available
ls -lh data/muni_snapshots/ | head
ls -lh models/trained_model/
```

**Important**: `training_labels.json` must be unlocked to edit it with `label_images.py`. The file stays tracked in git-annex but becomes writable.

---

## Useful Commands

### Check file locations
```bash
# See where files are stored (local, cloud, both)
git annex whereis data/muni_snapshots/

# See what's available locally
git annex find --in=here

# See what's in Google Cloud
git annex find --in=google-cloud
```

### Free up local disk space
```bash
# Remove local copies (keeps them in cloud)
git annex drop data/muni_snapshots/

# Get them back later
git annex get data/muni_snapshots/
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
