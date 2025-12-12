# Google Cloud Storage Configuration for git-annex

Configuration guide for using Google Cloud Storage as the git-annex backend for training data and model storage (1.1GB total).

## Prerequisites

- Google Cloud account with billing enabled
- `gcloud` CLI: `brew install google-cloud-sdk`
- `rclone`: `brew install rclone`
- `git-annex-remote-rclone`: `brew install git-annex-remote-rclone`
- Git-annex initialized repository

## Advantages of Google Cloud Storage

- **Cost Effective**: Free tier covers 5GB storage
- **Reliability**: Google's infrastructure
- **Performance**: High-speed transfers
- **Flexibility**: Compatible with S3-compatible backends

## Initial Setup

### 1. Install Dependencies

```bash
brew install google-cloud-sdk rclone git-annex-remote-rclone
```

### 2. Authenticate with Google Cloud

```bash
gcloud auth login
gcloud config set project PROJECT_ID
```

Replace `PROJECT_ID` with the GCP project identifier.

### 3. Enable Storage API and Create Bucket

```bash
gcloud services enable storage-api.googleapis.com
gsutil mb -p PROJECT_ID -l us-west1 gs://BUCKET_NAME
```

Replace `BUCKET_NAME` with a globally unique bucket name.

### 4. Configure rclone

```bash
rclone config create REMOTE_NAME gcs project_number=PROJECT_ID bucket_policy_only=true
```

This opens a browser for OAuth authentication. Replace `REMOTE_NAME` with a descriptive name (e.g., `project-gcs`).

### 5. Initialize git-annex Remote

```bash
git annex initremote REMOTE_NAME \
  type=external \
  externaltype=rclone \
  target=REMOTE_NAME \
  prefix=BUCKET_PREFIX \
  chunk=50MiB \
  encryption=none \
  rclone_layout=lower
```

Parameters:
- `REMOTE_NAME`: git-annex remote identifier
- `target`: rclone remote name from step 4
- `BUCKET_PREFIX`: subdirectory within bucket
- `chunk`: file chunk size for large files
- `encryption`: enable for sensitive data (requires GPG key)

### 6. Upload Annexed Files

```bash
# Upload all annexed files
git annex copy --to=REMOTE_NAME --jobs=4

# Upload specific directories
git annex copy artifacts/training_data/ --to=REMOTE_NAME
git annex copy artifacts/models/ --to=REMOTE_NAME
```

Initial upload time: approximately 10-20 minutes for 1.1GB.

## Collaborator Workflow

Collaborators cloning the repository must enable the git-annex remote to access files.

### Setup Steps

```bash
# Clone repository
git clone https://github.com/MrEricSir/munimet.ro.git
cd munimet.ro

# Initialize git-annex
git annex init "machine-name"

# Enable the cloud remote
git annex enableremote google-cloud
```

Note: `enableremote` requires the same rclone configuration from step 4.

### Downloading Files

```bash
# Download all annexed files
git annex get .

# Download specific directories
git annex get artifacts/models/v1/        # Model files only
git annex get artifacts/training_data/    # Training data only

# Verify downloaded files
git annex whereis
```

### Editing Annexed Files

Some files (e.g., `training_labels.json`) must be unlocked for editing:

```bash
# Unlock file for editing
git annex unlock artifacts/training_data/labels.json

# Edit file
# File remains git-annex tracked but becomes writable

# Re-lock after editing (optional)
git annex lock artifacts/training_data/labels.json
```

## File Management Commands

### Check File Locations

```bash
# Show where files are stored (local/cloud/both)
git annex whereis PATH

# List files available locally
git annex find --in=here

# List files in cloud storage
git annex find --in=REMOTE_NAME
```

### Disk Space Management

```bash
# Remove local copies (keeps cloud copies)
git annex drop PATH

# Retrieve files from cloud
git annex get PATH

# Sync all changes with cloud
git annex sync --content
```

### Upload New Files

```bash
# Upload new or modified files
git annex copy --to=REMOTE_NAME

# Upload with parallel transfers
git annex copy --to=REMOTE_NAME --jobs=4
```

## Cost Monitoring

### Check Storage Usage

```bash
# Bucket size
gsutil du -sh gs://BUCKET_NAME

# List objects with sizes
gsutil ls -lh gs://BUCKET_NAME/PREFIX/

# View billing
gcloud billing accounts list
```

### Expected Costs

- Storage: ~1.1GB
- Monthly cost: $0 (within 5GB free tier)
- Network egress: Free within same region

## Troubleshooting

### Remote Not Available

```bash
# Re-enable remote
git annex enableremote REMOTE_NAME

# Check remote configuration
git annex info REMOTE_NAME
```

### Authentication Errors

```bash
# Re-authenticate with gcloud
gcloud auth application-default login

# Verify credentials
gcloud auth list
```

### Bucket Permission Issues

```bash
# Verify bucket exists
gsutil ls

# Check bucket IAM permissions
gsutil iam get gs://BUCKET_NAME

# Grant access if needed
gsutil iam ch user:EMAIL:objectAdmin gs://BUCKET_NAME
```

### Transfer Failures

```bash
# Check git-annex logs
git annex info --debug

# Test rclone connectivity
rclone lsd REMOTE_NAME:

# Verify bucket access
gsutil ls gs://BUCKET_NAME
```

## Security Considerations

### Encryption

For sensitive data, enable git-annex encryption:

```bash
git annex initremote REMOTE_NAME \
  type=external \
  externaltype=rclone \
  encryption=hybrid \
  keyid=GPG_KEY_ID \
  ...
```

This encrypts files before upload, preventing cloud provider access.

### Access Control

- Use service accounts for automated access
- Enable IAM policies for bucket access
- Implement bucket versioning for recovery
- Enable audit logging for compliance

## Alternative Backends

git-annex supports multiple storage backends:

- **Amazon S3**: AWS S3-compatible storage
- **Backblaze B2**: Cost-effective alternative
- **Local Server**: rsync/ssh to dedicated server
- **WebDAV**: Self-hosted storage

Configuration details: See git-annex special remote documentation.

## Reference

- [git-annex documentation](https://git-annex.branchable.com/)
- [rclone documentation](https://rclone.org/docs/)
- [GCS pricing](https://cloud.google.com/storage/pricing)
