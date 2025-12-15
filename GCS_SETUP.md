# Google Cloud Storage Configuration for git-annex

Configuration guide for using Google Cloud Storage as the git-annex backend for training data and model storage (1.1GB total).

This guide is for **collaborators with write access** to the private training dataset. If you only need the pre-trained model, see [SETUP.md](SETUP.md) instead.

## Prerequisites

Before starting, ensure you have:

- Google Cloud account with billing enabled
- Base dependencies installed (see [SETUP.md](SETUP.md) for platform-specific instructions):
  - `gcloud` CLI (Google Cloud SDK)
  - `rclone`
  - `git-annex-remote-rclone`
  - `git-annex`
- Git-annex initialized repository (`git annex init "your-computer-name"`)

## Advantages of Google Cloud Storage

- **Cost Effective**: Free tier covers 5GB storage
- **Reliability**: Google's infrastructure
- **Performance**: High-speed transfers
- **Flexibility**: Compatible with S3-compatible backends

## Initial Setup

### 1. Install Dependencies

Follow [SETUP.md](SETUP.md) for platform-specific installation instructions.

**Quick reference:**

<details>
<summary><b>macOS</b></summary>

```bash
brew install google-cloud-sdk rclone git-annex-remote-rclone
```
</details>

<details>
<summary><b>Linux (Debian/Ubuntu)</b></summary>

```bash
# Google Cloud SDK
echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | sudo tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key --keyring /usr/share/keyrings/cloud.google.gpg add -
sudo apt-get update
sudo apt-get install -y google-cloud-sdk

# rclone
curl https://rclone.org/install.sh | sudo bash

# git-annex-remote-rclone
python3 -m pip install --user git-annex-remote-rclone
```
</details>

<details>
<summary><b>Windows (Native)</b></summary>

**Using winget:**
```powershell
winget install --id Google.CloudSDK -e
winget install --id Rclone.Rclone -e
python -m pip install --user git-annex-remote-rclone
```

**Using Scoop:**
```powershell
scoop install gcloud
scoop install rclone
python -m pip install --user git-annex-remote-rclone
```
</details>

<details>
<summary><b>Windows (WSL2 - Recommended)</b></summary>

Follow the Linux (Debian/Ubuntu) instructions within your WSL2 environment.
</details>

### 2. Authenticate with Google Cloud

**macOS/Linux:**
```bash
gcloud auth login
gcloud config set project PROJECT_ID
```

**Windows (PowerShell):**
```powershell
gcloud auth login
gcloud config set project PROJECT_ID
```

Replace `PROJECT_ID` with the GCP project identifier (e.g., `munimetro`).

### 3. Enable Storage API and Create Bucket

```bash
gcloud services enable storage-api.googleapis.com
gsutil mb -p PROJECT_ID -l us-west1 gs://BUCKET_NAME
```

Replace `BUCKET_NAME` with a globally unique bucket name.

### 4. Configure rclone

**macOS/Linux:**
```bash
rclone config create REMOTE_NAME gcs project_number=PROJECT_ID bucket_policy_only=true
```

**Windows (PowerShell):**
```powershell
rclone config create REMOTE_NAME gcs project_number=PROJECT_ID bucket_policy_only=true
```

This opens a browser for OAuth authentication.

**Parameter Explanation:**
- `REMOTE_NAME`: A name you choose for this rclone connection (e.g., `munimetro-gcs`, `project-gcs`). This is user-defined and can be anything memorable.
- `gcs`: Tells rclone to use Google Cloud Storage
- `PROJECT_ID`: Your GCP project ID (e.g., `munimetro`)
- `bucket_policy_only=true`: Uses bucket-level permissions instead of ACLs

**Example:**
```bash
rclone config create munimetro-gcs gcs project_number=munimetro bucket_policy_only=true
```

### 5. Initialize git-annex Remote

**macOS/Linux:**
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

**Windows (PowerShell):**
```powershell
git annex initremote REMOTE_NAME `
  type=external `
  externaltype=rclone `
  target=REMOTE_NAME `
  prefix=BUCKET_PREFIX `
  chunk=50MiB `
  encryption=none `
  rclone_layout=lower
```

**Parameter Explanation:**
- `REMOTE_NAME`: A name for this git-annex remote (e.g., `google-cloud`). This is user-defined and identifies this remote in git-annex commands.
- `type=external`: Tells git-annex to use an external special remote
- `externaltype=rclone`: Specifies we're using the rclone backend
- `target=REMOTE_NAME`: Must match the rclone remote name from step 4
- `prefix=BUCKET_PREFIX`: Subdirectory within bucket (e.g., `annex/`, or empty for root)
- `chunk=50MiB`: Split large files into 50MB chunks for better reliability
- `encryption=none`: No encryption (use `encryption=hybrid` for encrypted storage)
- `rclone_layout=lower`: Use lowercase directory structure

**Example:**
```bash
git annex initremote google-cloud \
  type=external \
  externaltype=rclone \
  target=munimetro-gcs \
  prefix=annex/ \
  chunk=50MiB \
  encryption=none \
  rclone_layout=lower
```

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

**macOS/Linux:**
```bash
# Clone repository
git clone https://github.com/MrEricSir/munimet.ro.git
cd munimet.ro

# Initialize git-annex with a descriptive name for this computer
# Replace "your-computer-name" with any name (e.g., "laptop", "alice-desktop", "work-machine")
# This helps identify where files are stored when working across multiple computers
git annex init "your-computer-name"

# Enable the cloud remote (requires rclone configuration from step 4)
git annex enableremote google-cloud
```

**Windows (PowerShell):**
```powershell
# Clone repository
git clone https://github.com/MrEricSir/munimet.ro.git
cd munimet.ro

# Initialize git-annex with a descriptive name for this computer
git annex init "your-computer-name"

# Enable the cloud remote
git annex enableremote google-cloud
```

**Important Notes:**
- The `"your-computer-name"` parameter is user-defined. Choose something descriptive like:
  - `"macbook-pro"`
  - `"alice-workstation"`
  - `"home-desktop"`
  - `"lab-server"`
- The `enableremote` command requires you to have completed the rclone configuration (step 4) first
- You must have access permissions to the Google Cloud Storage bucket

### Downloading Files

**macOS/Linux:**
```bash
# Download all annexed files
git annex get .

# Download specific directories
git annex get artifacts/models/v1/        # Model files only
git annex get artifacts/training_data/    # Training data only

# Verify downloaded files
git annex whereis
```

**Windows (PowerShell):**
```powershell
# Download all annexed files
git annex get .

# Download specific directories
git annex get artifacts/models/v1/        # Model files only
git annex get artifacts/training_data/    # Training data only

# Verify downloaded files
git annex whereis
```

**Note:** Commands are the same on all platforms. The difference is in the shell environment (bash vs PowerShell).

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

**Windows Users:** If you encounter issues with native Windows git-annex, consider using WSL2 (Windows Subsystem for Linux) for better compatibility. See [SETUP.md](SETUP.md) for WSL2 installation instructions.

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
