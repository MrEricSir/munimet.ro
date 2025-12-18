# Google Cloud Storage Configuration for git-annex

Configuration guide for using Google Cloud Storage as the git-annex backend for training data and model storage (1.1GB total).

This guide is for **collaborators with write access** to the private training dataset. If you only need the pre-trained model, see [SETUP.md](SETUP.md) instead.

## Prerequisites

Before starting, ensure you have:

- Google Cloud account with billing enabled
- Base dependencies installed (see [SETUP.md](SETUP.md) for platform-specific instructions):
  - `gcloud` CLI (Google Cloud SDK)
  - `git-annex`
- Git-annex initialized repository (`git annex init "your-computer-name"`)

## Advantages of S3 Special Remote

- **Built into git-annex**: No external dependencies required
- **Cross-platform**: Excellent Windows, macOS, and Linux support
- **Reliable**: Direct integration with git-annex
- **Simple**: Fewer moving parts than rclone setup
- **Cost Effective**: Free tier covers 5GB storage

## Quick Start (Automated Setup)

**Recommended for most collaborators.** This automated script handles HMAC key generation and credential storage.

### Prerequisites

1. Install base dependencies (see [SETUP.md](SETUP.md))
2. Authenticate with Google Cloud and get project access from maintainer

### Run Automated Setup

**macOS/Linux:**
```bash
# Run the automated credential setup script
./scripts/setup/setup-gcs-credentials.sh
```

**Windows:**
```powershell
# Run the automated credential setup script
.\scripts\setup\setup-gcs-credentials.ps1
```

The script will:
1. Verify `gcloud` authentication
2. Check project and bucket access
3. Generate HMAC keys automatically
4. Save credentials to `~/.aws/credentials` under `[munimetro]` profile
5. Enable the git-annex remote
6. Test the connection

### Using the Credentials

After running the setup script, set the AWS profile:

**macOS/Linux:**
```bash
# Set for current session
export AWS_PROFILE=munimetro

# Or set permanently (add to ~/.bashrc or ~/.zshrc)
echo 'export AWS_PROFILE=munimetro' >> ~/.bashrc
```

**Windows:**
```powershell
# Set for current session
$env:AWS_PROFILE = "munimetro"

# Or set permanently (add to PowerShell profile)
Add-Content $PROFILE '$env:AWS_PROFILE = "munimetro"'
```

### Download Training Data

```bash
# Download all training data (270MB)
git annex get artifacts/training_data/

# Unlock labels file for editing
git annex unlock artifacts/training_data/labels.json
```

## Manual Setup (Advanced)

### 1. Install Dependencies

Follow [SETUP.md](SETUP.md) for platform-specific installation instructions.

**Quick reference:**

<details>
<summary><b>macOS</b></summary>

```bash
brew install google-cloud-sdk git-annex
```
</details>

<details>
<summary><b>Linux (Debian/Ubuntu)</b></summary>

```bash
# Google Cloud SDK
echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | sudo tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key --keyring /usr/share/keyrings/cloud.google.gpg add -
sudo apt-get update
sudo apt-get install -y google-cloud-sdk git-annex
```
</details>

<details>
<summary><b>Windows (Native)</b></summary>

**Using winget:**
```powershell
winget install --id Google.CloudSDK -e
winget install --id Git-Annex.Git-Annex -e
```

**Using Scoop:**
```powershell
# Add extras bucket for gcloud
scoop bucket add extras
scoop install gcloud git-annex
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

### 4. Create HMAC Keys for S3 API Access

Google Cloud Storage supports S3-compatible API access via HMAC keys:

```bash
# Create HMAC key for your service account
gsutil hmac create PROJECT_ID@appspot.gserviceaccount.com
```

This outputs:
```
Access ID:   GOOG1E...
Secret:      Oi4+781...
```

**Save these credentials securely!** You'll need them for git-annex configuration.

### 5. Initialize git-annex S3 Remote

**macOS/Linux:**
```bash
export AWS_ACCESS_KEY_ID="<Access ID from step 4>"
export AWS_SECRET_ACCESS_KEY="<Secret from step 4>"

git annex initremote gcs \
  type=S3 \
  encryption=none \
  host=storage.googleapis.com \
  bucket=BUCKET_NAME \
  port=443 \
  protocol=https \
  requeststyle=path \
  chunk=50MiB \
  datacenter=us-west1
```

**Windows (PowerShell):**
```powershell
$env:AWS_ACCESS_KEY_ID="<Access ID from step 4>"
$env:AWS_SECRET_ACCESS_KEY="<Secret from step 4>"

git annex initremote gcs `
  type=S3 `
  encryption=none `
  host=storage.googleapis.com `
  bucket=BUCKET_NAME `
  port=443 `
  protocol=https `
  requeststyle=path `
  chunk=50MiB `
  datacenter=us-west1
```

**Parameter Explanation:**
- `type=S3`: Use built-in S3 special remote
- `encryption=none`: No encryption (use `encryption=hybrid` for encrypted storage)
- `host=storage.googleapis.com`: Google Cloud Storage S3 endpoint
- `bucket=BUCKET_NAME`: Your GCS bucket name
- `port=443`: HTTPS port
- `protocol=https`: Secure connection
- `requeststyle=path`: GCS-compatible path style
- `chunk=50MiB`: Split large files into 50MB chunks for better reliability

### 6. Upload Annexed Files

**macOS/Linux:**
```bash
export AWS_ACCESS_KEY_ID="<Access ID>"
export AWS_SECRET_ACCESS_KEY="<Secret>"

# Upload all annexed files
git annex copy --to=gcs --jobs=4

# Upload specific directories
git annex copy artifacts/training_data/ --to=gcs
git annex copy artifacts/models/ --to=gcs
```

**Windows (PowerShell):**
```powershell
$env:AWS_ACCESS_KEY_ID="<Access ID>"
$env:AWS_SECRET_ACCESS_KEY="<Secret>"

# Upload all annexed files
git annex copy --to=gcs --jobs=4
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
git annex init "your-computer-name"

# Set up credentials (ask project maintainer for HMAC keys)
export AWS_ACCESS_KEY_ID="<Access ID>"
export AWS_SECRET_ACCESS_KEY="<Secret>"

# Enable the cloud remote
git annex enableremote gcs
```

**Windows (PowerShell):**
```powershell
# Clone repository
git clone https://github.com/MrEricSir/munimet.ro.git
cd munimet.ro

# Initialize git-annex with a descriptive name for this computer
git annex init "your-computer-name"

# Set up credentials (ask project maintainer for HMAC keys)
$env:AWS_ACCESS_KEY_ID="<Access ID>"
$env:AWS_SECRET_ACCESS_KEY="<Secret>"

# Enable the cloud remote
git annex enableremote gcs
```

**Important Notes:**
- The `"your-computer-name"` parameter is user-defined. Choose something descriptive like:
  - `"macbook-pro"`
  - `"alice-workstation"`
  - `"home-desktop"`
  - `"lab-server"`
- You must have access permissions to the Google Cloud Storage bucket
- HMAC credentials should be kept secure (never commit to git)

### Saving Credentials

To avoid setting environment variables every time:

**macOS/Linux** - Add to `~/.bashrc` or `~/.zshrc`:
```bash
export AWS_ACCESS_KEY_ID="<Access ID>"
export AWS_SECRET_ACCESS_KEY="<Secret>"
```

**Windows** - Add to PowerShell profile (`$PROFILE`):
```powershell
$env:AWS_ACCESS_KEY_ID="<Access ID>"
$env:AWS_SECRET_ACCESS_KEY="<Secret>"
```

Alternatively, use AWS credential files (`~/.aws/credentials`) which git-annex reads automatically.

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
git annex find --in=gcs
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
git annex copy --to=gcs

# Upload with parallel transfers
git annex copy --to=gcs --jobs=4
```

## Cost Monitoring

### Check Storage Usage

```bash
# Bucket size
gsutil du -sh gs://BUCKET_NAME

# List objects with sizes
gsutil ls -lh gs://BUCKET_NAME/

# View billing
gcloud billing accounts list
```

### Expected Costs

- Storage: ~1.1GB
- Monthly cost: $0 (within 5GB free tier)
- Network egress: Free within same region
- S3 API requests: Minimal cost (typically <$0.01/month)

## Troubleshooting

**Windows Users:** If you encounter issues with native Windows git-annex, consider using WSL2 (Windows Subsystem for Linux) for better compatibility. See [SETUP.md](SETUP.md) for WSL2 installation instructions.

### Remote Not Available

```bash
# Re-enable remote
git annex enableremote gcs

# Check remote configuration
git annex info gcs
```

### Authentication Errors

```bash
# Re-authenticate with gcloud
gcloud auth application-default login

# Verify credentials
gcloud auth list

# Check HMAC keys
gsutil hmac list
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

# Test connectivity
gsutil ls gs://BUCKET_NAME

# Verify HMAC credentials work
export AWS_ACCESS_KEY_ID="<Access ID>"
export AWS_SECRET_ACCESS_KEY="<Secret>"
gsutil -o "Credentials:gs_access_key_id=$AWS_ACCESS_KEY_ID" \
       -o "Credentials:gs_secret_access_key=$AWS_SECRET_ACCESS_KEY" \
       ls gs://BUCKET_NAME
```

## Security Considerations

### Encryption

For sensitive data, enable git-annex encryption:

```bash
git annex initremote gcs \
  type=S3 \
  encryption=hybrid \
  keyid=GPG_KEY_ID \
  host=storage.googleapis.com \
  bucket=BUCKET_NAME \
  ...
```

This encrypts files before upload, preventing cloud provider access.

### Access Control

- Use service accounts for automated access
- Enable IAM policies for bucket access
- Implement bucket versioning for recovery
- Enable audit logging for compliance
- Store HMAC credentials securely (never in git)
- Rotate HMAC keys periodically

## Alternative Backends

git-annex supports multiple storage backends:

- **Amazon S3**: AWS S3 storage
- **Backblaze B2**: Cost-effective alternative (S3-compatible)
- **Wasabi**: S3-compatible hot storage
- **Local Server**: rsync/ssh to dedicated server
- **WebDAV**: Self-hosted storage

All S3-compatible providers work with the same `type=S3` configuration.

## Migrating from rclone

If you previously used the rclone remote:

1. The S3 remote uses a different key structure
2. Files must be copied to the new remote
3. Both remotes can coexist during migration
4. After migration, remove the old remote:
   ```bash
   git annex enableremote google-cloud-old dead=true
   ```

## Reference

- [git-annex documentation](https://git-annex.branchable.com/)
- [git-annex S3 special remote](https://git-annex.branchable.com/special_remotes/S3/)
- [GCS S3-compatible API](https://cloud.google.com/storage/docs/interoperability)
- [GCS pricing](https://cloud.google.com/storage/pricing)
