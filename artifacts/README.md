# Generated Artifacts

This directory contains all generated data for the MuniMet.ro project, organized by type and persistence.

## Directory Structure

```
artifacts/
├── training_data/      # ML training dataset (git-annex tracked)
│   ├── images/         # ~2800 labeled Muni subway status snapshots
│   └── labels.json     # Image labels with status and descriptions
├── models/             # Trained ML models (git-annex tracked)
│   └── v1/             # Model version 1 (BLIP-based classifier)
└── runtime/            # Transient runtime data (gitignored)
    ├── cache/          # API response cache
    └── downloads/      # Recent snapshot downloads
```

## Data Management with git-annex

The `training_data/` and `models/` directories use **git-annex** for efficient large file storage:

- **Symlinks in Git**: Only lightweight symlinks are tracked in the Git repository
- **Actual files**: Stored in `.git/annex/objects/` and synced to Google Cloud Storage
- **Selective download**: Download only the files you need

### Quick Start

#### Download all training data and models

```bash
# Download training images and labels
git annex get artifacts/training_data/

# Download trained model
git annex get artifacts/models/
```

#### Check what's available

```bash
# See which files are present locally vs in cloud
git annex whereis artifacts/training_data/images/
git annex whereis artifacts/models/v1/
```

#### Upload new data to cloud

```bash
# After adding new training images or models
git annex add artifacts/training_data/images/*.jpg
git commit -m "Add new training images"
git annex copy artifacts/training_data/ --to=google-cloud
git push
```

#### Free up local disk space

```bash
# Remove local copies (symlinks remain)
git annex drop artifacts/training_data/images/
git annex drop artifacts/models/

# Files are still safely stored in cloud!
```

## Runtime Data

The `runtime/` directory contains transient data:

- **cache/**: API response cache for improved performance
- **downloads/**: Recent snapshot downloads for real-time predictions

This data is **gitignored** and not tracked by git-annex. It's regenerated as needed.

## More Information

- Full git-annex setup guide: [`../GCS_SETUP.md`](../GCS_SETUP.md)
- Data management commands: [`../DATA_MANAGEMENT.md`](../DATA_MANAGEMENT.md)
- Training workflow: [`../training/README.md`](../training/README.md)
