# Generated Artifacts & Git-annex Guide

This is the authoritative guide for managing training data, models, and runtime files using git-annex. All git-annex workflows and file management operations are documented here.

## Directory Structure

```
artifacts/
├── training_data/      # ML training dataset (git-annex tracked)
│   ├── images/         # 2,666 labeled Muni subway status snapshots
│   └── labels.json     # Image labels with status and descriptions
├── models/             # Trained ML models (git-annex tracked)
│   └── v1/             # Model version 1 (BLIP-based classifier)
│       ├── model.safetensors      # Vision transformer weights
│       ├── status_classifier.pt   # Classification head
│       ├── outlier_report.json    # Model evaluation outliers
│       └── [config files]         # Model configuration
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
~270MB  artifacts/training_data/images/       # 2,666 training images
856MB   artifacts/models/v1/                  # BLIP model + classifier
570KB   artifacts/training_data/labels.json   # Training labels (unlocked)
12KB    artifacts/models/v1/outlier_report.json  # Model evaluation outliers
```

**Total**: ~1.1GB tracked via git-annex in Google Cloud Storage bucket `munimetro-annex` (within 5GB free tier).

For configuration details, see [CONFIGURATION.md](../CONFIGURATION.md).

## What is git-annex?

Git-annex is a tool that allows Git to manage large files without storing them directly in the repository. Instead of committing large files, git-annex creates lightweight symbolic links (symlinks) in Git and stores the actual files separately.

### Key Benefits

- **Efficient Storage**: Only download the files you need
- **Version Control**: Track large files without bloating the Git repository
- **Cloud Sync**: Store files in Google Cloud Storage while keeping Git lightweight
- **Selective Checkout**: Clone the repository quickly without downloading gigabytes of data

### How It Works

```
Git Repository (GitHub)
├── Symlinks tracked in Git (~1KB each)
└── .git/annex/objects/ (actual files, not in Git)
    └── Synced to Google Cloud Storage
```

When you run `git annex get`, files download from Google Cloud Storage to `.git/annex/objects/` and symlinks become functional.

### Windows: Adjusted Branch Mode

Windows doesn't handle symlinks well by default (requires admin privileges or Developer Mode). Git-annex provides an **adjusted branch** feature that stores files as regular files instead of symlinks.

**How it works:**
- Run `git annex adjust --unlock` to switch to an adjusted branch (e.g., `adjusted/main(unlocked)`)
- Files are stored as regular files in your working directory
- Git-annex automatically manages syncing between the adjusted branch and the main branch
- The Windows setup script (`setup-git-annex.ps1`) enables this automatically

**Key differences:**

| Feature | Regular Branch (macOS/Linux) | Adjusted Branch (Windows) |
|---------|------------------------------|---------------------------|
| File storage | Symlinks → `.git/annex/objects/` | Regular files |
| `git annex unlock` | Needed to edit files | Not needed (files already unlocked) |
| Branch name | `main` | `adjusted/main(unlocked)` |
| Symlink support | Required | Not required |

**Note:** The adjusted branch is a local view - commits are automatically synced to the underlying main branch when you push.

## Initial Setup

### Step 1: Install Dependencies

Before using git-annex, install required tools:

**See [SETUP.md](../SETUP.md) for detailed installation instructions** for your platform (macOS, Linux, Windows).

Quick checklist:
- git-annex
- Google Cloud SDK (gcloud) - for collaborators only

### Step 2: Run Automated Setup Script

Use the automated setup script to initialize git-annex and download the model:

```bash
cd /path/to/munimet.ro

# Run automated setup
./scripts/setup/setup-git-annex.sh      # macOS/Linux
# or
.\scripts\setup\setup-git-annex.ps1     # Windows
```

The script will:
1. Prompt for a descriptive computer name (e.g., "laptop", "alice-desktop", "work-machine")
2. Initialize git-annex with that name
3. Configure automatic annexing for large files
4. Enable the gcs remote (if available)
5. Download the pre-trained model (856MB) - no authentication required

### Step 3: For Collaborators with Private Data Access

If you're a collaborator with access to the private training dataset:

1. **Configure Google Cloud Storage access** - See [GCS_SETUP.md](../GCS_SETUP.md) for HMAC credentials setup

2. **Set up credentials and enable the remote**:
   ```bash
   # Set up HMAC credentials (ask project maintainer for keys)
   export AWS_ACCESS_KEY_ID="<your-access-id>"
   export AWS_SECRET_ACCESS_KEY="<your-secret-key>"

   # Enable the remote
   git annex enableremote gcs
   ```

**Download training data**:
```bash
# Download all training data (270MB)
git annex get artifacts/training_data/

# Unlock labels file for editing (needed for labeling workflow)
git annex unlock artifacts/training_data/labels.json
```

### Verification

After setup, verify git-annex is working:

```bash
# Check git-annex status
git annex info

# Check file locations (shows where files are: here, cloud, or both)
git annex whereis artifacts/models/v1/
git annex whereis artifacts/training_data/

# List all annexed files
git annex list
```

Expected output for `git annex whereis`:
```
whereis artifacts/models/v1/model.safetensors (2 copies)
  	00000000-0000-0000-0000-000000000001 -- web
  	a1b2c3d4-... -- your-computer-name [here]
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
git annex copy artifacts/training_data/images/ --to=gcs --jobs=4

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
git annex copy artifacts/training_data/labels.json --to=gcs
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
git annex copy artifacts/models/v1/ --to=gcs

# Commit and push
git commit -m "Update trained model"
git push
```

## Essential git-annex Commands

This section provides a comprehensive reference for all git-annex operations you'll need.

### Basic File Operations

| Command | Function | Example |
|---------|----------|---------|
| `git annex init "name"` | Initialize git-annex in repository | `git annex init "laptop"` |
| `git annex get FILE` | Download file from cloud storage | `git annex get artifacts/models/v1/` |
| `git annex drop FILE` | Remove local copy (cloud copy remains) | `git annex drop artifacts/training_data/images/` |
| `git annex add FILE` | Add new file to git-annex tracking | `git annex add artifacts/models/v2/` |
| `git annex unlock FILE` | Make file writable (keeps git-annex tracking) | `git annex unlock artifacts/training_data/labels.json` |
| `git annex lock FILE` | Make file read-only (re-create symlink) | `git annex lock artifacts/training_data/labels.json` |

### File Location & Status

| Command | Function | Example |
|---------|----------|---------|
| `git annex whereis FILE` | Show where file copies exist | `git annex whereis artifacts/models/v1/model.safetensors` |
| `git annex info` | Display repository statistics | `git annex info` |
| `git annex list` | List all annexed files with locations | `git annex list` |
| `git annex find --in=here` | List files present locally | `git annex find --in=here` |
| `git annex find --not --in=here` | List files not present locally | `git annex find --not --in=here` |

### Cloud Sync Operations

| Command | Function | Example |
|---------|----------|---------|
| `git annex copy --to=REMOTE` | Upload files to remote | `git annex copy artifacts/models/v1/ --to=gcs` |
| `git annex copy --from=REMOTE` | Download files from remote | `git annex copy --from=gcs` |
| `git annex sync --content` | Sync all changes bidirectionally | `git annex sync --content` |
| `git annex enableremote REMOTE` | Enable a configured remote | `git annex enableremote gcs` |

### Performance Options

| Option | Function | Example |
|--------|----------|---------|
| `--jobs=N` | Use N parallel transfers | `git annex get --jobs=4 artifacts/training_data/` |
| `--fast` | Skip safety checks (faster) | `git annex drop --fast FILE` |
| `--force` | Override warnings | `git annex drop --force FILE` |

### Disk Space Management

Free up local disk space while keeping files in the cloud:

```bash
# Remove local copies (symlinks remain, files stay in cloud)
git annex drop artifacts/training_data/images/
git annex drop artifacts/models/

# Check what will be dropped (dry run)
git annex drop --dry-run artifacts/training_data/

# Restore files from cloud
git annex get artifacts/training_data/
git annex get artifacts/models/

# Get only specific files
git annex get artifacts/models/v1/model.safetensors
```

### Working with Unlocked Files

Some files (like `labels.json`) need to be unlocked for editing:

```bash
# Unlock file - replaces symlink with actual file
git annex unlock artifacts/training_data/labels.json

# Now you can edit the file directly
# Edit with text editor, Python script, etc.

# Commit changes (git-annex tracking is preserved)
git add artifacts/training_data/labels.json
git commit -m "Update training labels"

# Upload to cloud
git annex copy artifacts/training_data/labels.json --to=gcs
git push

# Optional: Re-lock file (converts back to symlink)
git annex lock artifacts/training_data/labels.json
```

**Note:** Unlocked files remain unlocked across commits. You only need to unlock once per repository clone.

## Quick Start Guide

New to git-annex? Follow these steps:

```bash
# 1. Clone repository
git clone https://github.com/MrEricSir/munimet.ro.git
cd munimet.ro

# 2. Run automated setup script
./scripts/setup/setup-git-annex.sh   # Initializes, downloads model

# 3. Verify setup
git annex whereis artifacts/models/v1/

# 4. (Optional) For collaborators: configure GCS and download training data
# See GCS_SETUP.md for configuration instructions
git annex enableremote gcs
git annex get artifacts/training_data/
```

## Storage Configuration

- **Backend**: Google Cloud Storage (`gs://munimetro-annex`)
- **Region**: `us-west1` (Oregon)
- **Cost**: $0/month (within 5GB free tier)
- **Current Usage**: ~1.1GB (training data + models)
- **Chunk Size**: 50MiB for transfer optimization

See [CONFIGURATION.md](../CONFIGURATION.md) for complete configuration details.

## Troubleshooting

This section covers common git-annex issues and their solutions.

### Broken Symlinks (Files Appear Missing)

**Symptom:** Files in `artifacts/` show as broken symlinks or appear as plain text symlink paths

**Cause:** Files haven't been downloaded from cloud storage yet

**Solution:**
```bash
# Download all missing files
git annex get .

# Or download specific directories
git annex get artifacts/training_data/
git annex get artifacts/models/v1/

# Check which files are missing
git annex find --not --in=here

# Verify downloads
git annex whereis artifacts/models/v1/
```

### Remote Not Available

**Symptom:** `git annex get` fails with "remote not available" or "unknown remote"

**Solution:**
```bash
# Check if git-annex is initialized
git annex info

# If not initialized:
git annex init "your-computer-name"

# Re-enable cloud remote
git annex enableremote gcs

# Verify remote configuration
git annex info gcs

# List all remotes
git annex info | grep -A5 "remotes"
```

### Authentication Errors (Collaborators Only)

**Symptom:** Access denied when trying to download/upload files to Google Cloud Storage

**Solution:**
```bash
# Verify HMAC credentials are set
echo $AWS_ACCESS_KEY_ID
echo $AWS_SECRET_ACCESS_KEY

# Set credentials if missing (ask project maintainer for keys)
export AWS_ACCESS_KEY_ID="<your-access-id>"
export AWS_SECRET_ACCESS_KEY="<your-secret-key>"

# Re-enable remote
git annex enableremote gcs

# Test connectivity
git annex testremote gcs

# Verify with gsutil
gsutil ls gs://munimetro-annex
```

### Transfer Failures

**Symptom:** `git annex get` or `git annex copy` fails partway through

**Solution:**
```bash
# Enable debug logging to see detailed errors
git annex --debug get artifacts/models/v1/

# Try with fewer parallel jobs
git annex get --jobs=1 artifacts/models/v1/

# Verify credentials are set
echo $AWS_ACCESS_KEY_ID

# For collaborators: Verify bucket access
gsutil ls gs://munimetro-annex

# Test S3 connectivity
git annex testremote gcs

# Check network connectivity
ping google.com
```

### "git-annex: command not found"

**Symptom:** Shell cannot find git-annex command

**Solution:**
```bash
# Verify git-annex is installed
which git-annex

# If not found, install it (see SETUP.md for platform instructions)
# macOS:
brew install git-annex

# Linux (Debian/Ubuntu):
sudo apt-get install git-annex

# Linux (RHEL/CentOS):
sudo yum install git-annex

# Windows:
scoop install git-annex

# Refresh shell after installation
exec $SHELL
```

### Disk Space Issues

**Symptom:** Not enough space to download large files

**Solution:**
```bash
# Check current disk usage
git annex info

# Download only specific files instead of all
git annex get artifacts/models/v1/model.safetensors

# Drop files you don't need (keeps cloud copy)
git annex drop artifacts/training_data/images/

# Use streaming for very large files (downloads on-demand)
git annex get --auto artifacts/models/v1/
```

### File Won't Unlock

**Symptom:** `git annex unlock` fails or file remains read-only

**Solution:**
```bash
# Ensure file is downloaded first
git annex get artifacts/training_data/labels.json

# Force unlock
git annex unlock --force artifacts/training_data/labels.json

# Check file permissions
ls -la artifacts/training_data/labels.json

# If still locked, try:
chmod u+w artifacts/training_data/labels.json
```

### Verify Cloud Storage Status

Check the status of files in cloud storage:

```bash
# Show all files and their locations (local/cloud/both)
git annex whereis

# Count files in each location
git annex find --in=here | wc -l           # Files present locally
git annex find --in=gcs | wc -l   # Files in cloud
git annex find --not --in=here | wc -l     # Files missing locally

# List all annexed files with locations
git annex list

# Check specific file status
git annex whereis artifacts/models/v1/model.safetensors
```

### Performance Issues (Slow Downloads)

**Symptom:** Downloads are very slow

**Solution:**
```bash
# Use parallel transfers (4 jobs)
git annex get --jobs=4 artifacts/training_data/

# Test network speed
speedtest-cli  # Install if needed

# Verify bucket region matches your location
gsutil ls -L gs://munimetro-annex | grep Location

# Try different network
# Sometimes switching networks (e.g., WiFi to Ethernet) helps

# For very large files, downloads may be slower on first access
# Subsequent downloads use chunked resumable transfers
```

### "No space left on device" in .git/annex

**Symptom:** git-annex fails with disk space errors even though main disk has space

**Cause:** .git directory is on a separate partition or has quota limits

**Solution:**
```bash
# Check .git directory location and space
df -h .git

# Move .git/annex to larger disk
# WARNING: Advanced operation, backup first
mv .git/annex /larger/disk/location/
ln -s /larger/disk/location/annex .git/annex

# Or drop unnecessary files
git annex drop artifacts/training_data/images/
```

## Related Documentation

- **Git-annex Setup**: [GCS_SETUP.md](../GCS_SETUP.md) - Google Cloud Storage configuration
- **Configuration**: [CONFIGURATION.md](../CONFIGURATION.md) - Deployment configuration values
- **Training**: [training/README.md](../training/README.md) - Model training workflow
- **Deployment**: [deploy/README.md](../deploy/README.md) - API deployment guide
