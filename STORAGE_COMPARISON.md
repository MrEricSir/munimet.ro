# Storage Solution Comparison for Muni Status Monitor

## The Requirement

**Share both training dataset AND trained model:**
- Training dataset: 268MB (2,601 images + labels) - **Required together**
- Trained model: 856MB (BLIP model + classifier)
- **Total: 1.1GB**

**Goal**: Enable collaborators to both use the model AND retrain/improve it

---

## Option Comparison

### Git-annex 🔧

**How it works:**
- Files stay in git history, but content stored separately
- Can use any backend: S3, Google Drive, rsync, USB drive, etc.
- Each clone can choose which files to actually download

**Setup:**
```bash
# Install
brew install git-annex

# Initialize
git annex init "my-laptop"

# Add files to annex
git annex add data/muni_snapshots/
git annex add data/training_labels.json
git annex add models/trained_model/

# Commit symlinks
git commit -m "Add dataset and model to git-annex"

# Set up S3 backend
git annex initremote s3-storage type=S3 encryption=none \
  bucket=munimetro-data chunk=100MiB

# Upload to S3
git annex copy --to=s3-storage

# Push to GitHub (just metadata, not files)
git push

# Collaborator workflow:
git clone repo
git annex init "their-laptop"
git annex enableremote s3-storage
git annex get data/  # Downloads only what they need
```

**Pros:**
- ✅ **FREE** (only pay for cloud storage, not git-annex itself)
- ✅ Partial checkouts (download only model, or only dataset)
- ✅ Multiple backends (S3, GCS, local server, etc.)
- ✅ Works offline once synced
- ✅ Truly distributed
- ✅ Can track file moves/renames

**Cons:**
- ❌ Steeper learning curve
- ❌ Collaborators MUST install git-annex
- ❌ Not integrated with GitHub UI
- ❌ More complex troubleshooting
- ❌ Symlinks can be confusing

**Cost:**
- git-annex: **Free**
- S3 storage: ~$0.025/GB/month = **$0.03/month** for 1.1GB
- S3 data transfer: First 100GB/month free

**Best for:** Technical teams comfortable with command-line tools

---

### Git LFS 📦

**How it works:**
- GitHub hosts the large files
- Git tracks pointers to files
- Automatic download on clone

**Setup:**
```bash
brew install git-lfs
git lfs install

git lfs track "data/muni_snapshots/**"
git lfs track "data/training_labels.json"
git lfs track "models/trained_model/**"

git add .gitattributes
git add data/ models/
git commit -m "Add data with Git LFS"
git push
```

**Pros:**
- ✅ Seamless Git workflow
- ✅ GitHub/GitLab UI integration
- ✅ Easy for collaborators (just `git clone`)
- ✅ Version control for models
- ✅ Well-documented, widely used

**Cons:**
- ❌ GitHub limit: **1GB storage + 1GB bandwidth/month FREE**
- ❌ After limit: $5/month for 50GB storage + 50GB bandwidth
- ❌ Can't do partial checkouts (downloads everything)
- ❌ Need LFS on every machine

**Cost:**
- First 1GB: **Free**
- Your 1.1GB: **$5/month** (need paid plan)
- Each clone uses bandwidth (10 clones = $5/month more)

**Best for:** Teams wanting seamless Git experience, willing to pay

---

### DVC (Data Version Control) 📊

**How it works:**
- Similar to git-annex but ML-focused
- Tracks data with `.dvc` files in git
- Stores data in cloud (S3, GCS, Azure, etc.)

**Setup:**
```bash
pip install dvc[s3]
dvc init

# Add remote storage
dvc remote add -d storage s3://munimetro-dvc/data

# Track large files
dvc add data/muni_snapshots
dvc add data/training_labels.json
dvc add models/trained_model

# Commit DVC metadata
git add data/.gitignore data/muni_snapshots.dvc \
        data/training_labels.json.dvc models/trained_model.dvc .dvc
git commit -m "Track data with DVC"

# Upload to remote
dvc push

# Collaborator workflow:
git clone repo
pip install dvc[s3]
dvc pull  # Downloads all tracked data
```

**Pros:**
- ✅ Built for ML workflows
- ✅ Experiments tracking
- ✅ Pipeline management
- ✅ Works with any cloud storage
- ✅ Partial downloads possible
- ✅ Good documentation for ML users

**Cons:**
- ❌ Requires Python + DVC installed
- ❌ Extra abstraction layer
- ❌ Overkill for simple projects
- ❌ Learning curve for non-ML users

**Cost:**
- DVC: **Free**
- S3 storage: ~$0.025/GB/month = **$0.03/month**
- S3 bandwidth: First 100GB free

**Best for:** ML teams doing experiments and versioning datasets

---

### Hugging Face Hub 🤗

**How it works:**
- Upload dataset to Hugging Face Datasets
- Upload model to Hugging Face Hub
- Download via Python API or web UI

**Setup:**
```bash
pip install huggingface_hub datasets

# Login
huggingface-cli login

# Upload model
huggingface-cli upload username/muni-status-model ./models/trained_model/

# Upload dataset (requires conversion to HF format)
python scripts/upload_to_hf.py  # Custom script to convert images+labels
```

**Pros:**
- ✅ **FREE** unlimited storage for public repos
- ✅ Beautiful web UI
- ✅ Automatic model cards
- ✅ Easy discovery/sharing
- ✅ Version control built-in
- ✅ Python API for easy downloads

**Cons:**
- ❌ Public by default (private costs money)
- ❌ Requires HuggingFace account
- ❌ Need to convert data to HF format
- ❌ Not traditional git workflow
- ❌ Tied to HuggingFace platform

**Cost:**
- Public repos: **FREE** unlimited
- Private repos: $9/month

**Best for:** Public ML projects, wanting visibility

---

### Cloud Storage + Download Script 💾

**How it works:**
- Upload archives to S3/GCS
- Provide download script in repo
- Git only tracks code

**Setup:**
```bash
# Create archives
tar -czf muni-training-dataset.tar.gz data/
tar -czf muni-trained-model.tar.gz models/trained_model/

# Upload to S3
aws s3 cp muni-training-dataset.tar.gz s3://munimetro-public/
aws s3 cp muni-trained-model.tar.gz s3://munimetro-public/

# Make public
aws s3api put-object-acl --bucket munimetro-public \
  --key muni-training-dataset.tar.gz --acl public-read

# Add download script to repo
cat > scripts/download_data.sh <<'EOF'
#!/bin/bash
curl -L https://munimetro-public.s3.amazonaws.com/muni-training-dataset.tar.gz | \
  tar xz
curl -L https://munimetro-public.s3.amazonaws.com/muni-trained-model.tar.gz | \
  tar xz
EOF
```

**Pros:**
- ✅ Simple, no special tools
- ✅ Very cheap
- ✅ Easy to understand
- ✅ Can use free tiers (GCS: 5GB free)
- ✅ Direct download URLs

**Cons:**
- ❌ No version control for data
- ❌ Manual process to update
- ❌ Extra step for collaborators
- ❌ No partial downloads

**Cost:**
- S3: **$0.03/month**
- GCS: **FREE** (under 5GB free tier)

**Best for:** Simple projects, minimal complexity

---

## Decision Matrix

| Solution | Monthly Cost | Setup Complexity | Collaborator Ease | Version Control | Best For |
|----------|--------------|------------------|-------------------|-----------------|----------|
| **git-annex** | $0.03 | High | Medium | Yes | Power users |
| **Git LFS** | $5+ | Low | High | Yes | Teams with budget |
| **DVC** | $0.03 | Medium | Medium | Yes | ML teams |
| **HuggingFace** | Free (public) | Medium | High | Yes | Public projects |
| **Cloud + Script** | $0.03 | Low | Low | No | Simple projects |

---

## Recommendation for Your Project

### 🏆 Option 1: Git-annex (If technical comfort is high)

**Why:**
- Fits perfectly in your 1.1GB size
- Essentially free (~$0.03/month)
- True version control
- Partial downloads (can get just model OR just dataset)

**When to choose:**
- You and collaborators are comfortable with CLI tools
- Want full control and flexibility
- Don't mind initial setup complexity

---

### 🏆 Option 2: Hugging Face Hub (If willing to make it public)

**Why:**
- Completely free
- Great discoverability
- Professional ML project presentation
- Easy for collaborators

**When to choose:**
- Project is open source anyway
- Want visibility in ML community
- Like polished web UIs
- Don't mind platform lock-in

---

### 🏆 Option 3: DVC + Google Cloud Storage (Good middle ground)

**Why:**
- Under 5GB = FREE (Google Cloud free tier)
- Purpose-built for ML
- Good version control
- Better docs than git-annex

**When to choose:**
- Want ML-specific features
- Team familiar with Python
- Might do experiments later
- Want better UX than git-annex

---

## My Recommendation

Given your project characteristics:
- 1.1GB total (fits everywhere)
- Both dataset AND model needed
- Likely small team or solo
- Technical project (ML/Docker already in use)

**I recommend: Git-annex with S3**

**Reasoning:**
1. **Cost**: ~$0.03/month (essentially free)
2. **Flexibility**: Can download just model or just dataset
3. **Version control**: Full git history for data
4. **No lock-in**: Can switch backends anytime
5. **Technical fit**: You're already using Docker, git-annex is similar complexity

**Alternative if you want simplicity**:
DVC + Google Cloud Storage (free under 5GB, easier than git-annex)

Would you like me to help set up either of these?
