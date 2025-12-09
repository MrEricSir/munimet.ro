# Data Management Guide

This document explains how to handle large files (training data, images, models) in the Muni Status Monitor project.

## Current Data Size

```bash
267MB  data/muni_snapshots/       # 2,601 images
856MB  models/trained_model/      # BLIP model + classifier
570KB  data/training_labels.json  # Training labels (should be committed)
```

**Total: ~1.1GB** of data that shouldn't be committed to Git.

## What's Currently Excluded from Git

The `.gitignore` file excludes:
```
data/muni_snapshots/    # Downloaded images
data/cache/             # API cache files
models/trained_model/   # Trained ML model
```

**Note**: `data/training_labels.json` is **not** ignored by default. You can commit this if you want to share your training labels with collaborators.

## Storage Alternatives

### Option 1: Git LFS (Git Large File Storage) ⭐ Recommended

**Best for**: Sharing models and data with collaborators

**Setup**:
```bash
cd ~/Development/munimetro

# Install Git LFS
brew install git-lfs
git lfs install

# Track large files
git lfs track "models/trained_model/**"
git lfs track "data/training_labels.json"

# Add .gitattributes
git add .gitattributes
git commit -m "Add Git LFS tracking for models"

# Now you can commit the model
git add models/trained_model/
git commit -m "Add trained BLIP model"
```

**Pros**:
- Seamless Git workflow
- Version history for models
- Easy collaboration
- GitHub/GitLab support

**Cons**:
- Costs money on GitHub after 1GB free storage
- Slower cloning for new contributors
- Requires Git LFS installed on all machines

**Cost**: GitHub LFS: $5/month for 50GB, GitLab: 10GB free per repo

---

### Option 2: Cloud Storage (S3, GCS, Dropbox)

**Best for**: Personal projects, avoiding Git LFS costs

**Setup with Google Cloud Storage**:
```bash
# Upload model
gsutil -m cp -r models/trained_model/* gs://your-bucket/munimetro/models/trained_model/

# Download on new machine
gsutil -m cp -r gs://your-bucket/munimetro/models/trained_model/* models/trained_model/
```

**Setup with AWS S3**:
```bash
# Upload
aws s3 sync models/trained_model/ s3://your-bucket/munimetro/models/trained_model/

# Download
aws s3 sync s3://your-bucket/munimetro/models/trained_model/ models/trained_model/
```

**Setup with Dropbox/Drive**:
```bash
# Create symlinks to shared folder
ln -s ~/Dropbox/munimetro/models/trained_model models/trained_model
ln -s ~/Dropbox/munimetro/data/training_labels.json data/training_labels.json
```

**Pros**:
- No extra tooling needed
- Can use existing cloud storage
- More storage for less money
- Easy to share via URL

**Cons**:
- Manual download step for new clones
- Not versioned with code
- Need separate credentials/setup

---

### Option 3: DVC (Data Version Control)

**Best for**: ML teams, data scientists, production ML

**Setup**:
```bash
# Install DVC
pip install dvc

# Initialize DVC
dvc init

# Add remote (S3, GCS, Azure, etc.)
dvc remote add -d storage s3://your-bucket/munimetro-dvc

# Track large files
dvc add models/trained_model/
dvc add data/training_labels.json

# Commit DVC metadata
git add models/trained_model.dvc data/training_labels.json.dvc .dvc
git commit -m "Track data with DVC"

# Push data to remote
dvc push

# On new machine, pull data
git clone your-repo
dvc pull
```

**Pros**:
- Built specifically for ML workflows
- Version models alongside code
- Supports experiments and pipelines
- Works with any cloud storage
- More cost-effective than Git LFS

**Cons**:
- Adds complexity (new tool to learn)
- Requires DVC installed on all machines
- Not as seamless as Git LFS

---

### Option 4: Regenerate on Each Machine (Current Approach)

**Best for**: Small teams, learning projects

**Current setup** - `.gitignore` excludes large files:
```bash
# Clone repo
git clone your-repo

# Download images yourself
cd training
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python download_muni_image.py  # Run for hours/days

# Label images
python label_images.py  # Label 50-100+ images

# Train model
python train_model.py  # Takes 5-20 minutes
```

**Pros**:
- No extra tools or costs
- Clean Git repo
- Learn the full workflow
- Always fresh data

**Cons**:
- Time-consuming for new contributors
- Need to collect data yourself
- Can't reproduce exact model versions
- Not practical for production

---

### Option 5: Hugging Face Hub 🤗

**Best for**: Sharing ML models publicly

**Setup**:
```bash
# Install Hugging Face CLI
pip install huggingface_hub

# Login
huggingface-cli login

# Upload model
huggingface-cli upload your-username/muni-status-model ./models/trained_model/

# Download model
from huggingface_hub import snapshot_download
snapshot_download(repo_id="your-username/muni-status-model", local_dir="./models/trained_model/")
```

**Pros**:
- Free unlimited storage for public models
- Designed for ML models
- Great discovery/sharing features
- Automatic model cards

**Cons**:
- Public by default (private repos cost money)
- Overkill for personal projects
- Need HuggingFace account

---

## Recommended Approach

**For this project, I recommend**:

1. **Commit training labels** (`data/training_labels.json` - 570KB)
   - Small enough for Git
   - Valuable for reproducibility
   - Easy to share

2. **Use Git LFS for the model** if sharing with team:
   ```bash
   git lfs track "models/trained_model/**"
   git add models/trained_model/
   ```

3. **Keep images in `.gitignore`**:
   - 267MB is manageable but grows over time
   - Easy to regenerate with `download_muni_image.py`
   - If sharing, provide a small sample dataset (10-20 images) instead

4. **Add download script** for collaborators:
   ```bash
   # scripts/download_pretrained_model.sh
   #!/bin/bash
   # Download pre-trained model from cloud storage
   gsutil cp -r gs://your-bucket/muni-model/* models/trained_model/
   ```

---

## Setup Instructions for Collaborators

Add to your `README.md`:

```markdown
## Getting the Trained Model

**Option 1: Download pre-trained model** (fastest):
\`\`\`bash
# Download from cloud storage
./scripts/download_pretrained_model.sh
\`\`\`

**Option 2: Train it yourself** (2-4 hours):
\`\`\`bash
cd training
python download_muni_image.py  # Collect images (run overnight)
python label_images.py          # Label 50-100+ images
python train_model.py           # Train model (5-20 minutes)
\`\`\`
```

---

## Decision Matrix

| Approach | Cost | Setup Complexity | Best For |
|----------|------|-----------------|----------|
| **Git LFS** | $5-10/mo | Low | Teams, version control |
| **Cloud Storage** | $1-5/mo | Medium | Personal, cost-sensitive |
| **DVC** | Storage cost only | High | ML teams, experiments |
| **Regenerate** | Free | Low | Learning, small teams |
| **Hugging Face** | Free (public) | Low | Public sharing |

---

## Current Status

✅ `.gitignore` configured to exclude:
- `data/muni_snapshots/` (267MB)
- `models/trained_model/` (856MB)
- `data/cache/` (runtime cache)

📋 Ready to commit:
- `data/training_labels.json` (570KB) - Optionally commit this

🎯 **Next step**: Choose one of the approaches above based on your needs.
