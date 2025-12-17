# Training Documentation

Data collection, labeling, and model training workflow for the Muni Metro status classifier.

## Prerequisites

- Python 3.13+
- Virtual environment (see [SETUP.md](../SETUP.md))
- tkinter (for labeling GUI)
- ~2GB RAM for training
- Optional: CUDA-capable GPU for faster training

## Environment Setup

```bash
# Create virtual environment
python3 -m venv venv

# Activate environment
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Install tkinter (macOS)
brew install python-tk@3.13
```

For detailed setup instructions, see [SETUP.md](../SETUP.md).

## Training Data Access

### Using Existing Data (Collaborators)

If you have access to the shared training dataset via Google Cloud Storage:

1. Follow [SETUP.md](../SETUP.md) to install git-annex and dependencies
2. See [GCS_SETUP.md](../GCS_SETUP.md) for Google Cloud Storage configuration
3. See [artifacts/README.md](../artifacts/README.md) for detailed git-annex workflows

Quick reference for downloading training data:

```bash
# Download training data (270MB)
git annex get artifacts/training_data/

# Unlock labels file for editing
git annex unlock artifacts/training_data/labels.json
```

### Collecting New Data

Build a custom training dataset:

```bash
python download_muni_image.py
```

**Behavior**:
- Downloads snapshots every 5 minutes from http://sfmunicentral.com/sfmunicentral_Snapshot_Objects/Mimic1_A7SE582P.jpg
- Saves to `artifacts/training_data/images/`
- Validates dimensions (1860×800 pixels)
- Continues until interrupted (Ctrl+C)

**Recommendation**: Collect over multiple days to capture diverse status conditions (normal operation, delays, service interruptions).

### Adding Downloaded Images to Git-Annex

Downloaded images are regular files until committed. Git-annex is configured to **automatically annex large files** when you use `git add`:

```bash
# Add new images - they'll be automatically annexed
git add artifacts/training_data/images/

# Commit the symlinks (not the actual files!)
git commit -m "Add new training images"

# Verify files are symlinks
ls -l artifacts/training_data/images/*.jpg | head -5
```

**How it works:**
1. The `.gitattributes` file tells git to use the annex filter
2. When you `git add` a large file, git-annex intercepts and creates a symlink
3. The actual file content goes to `.git/annex/objects/`
4. Only the symlink is committed to git

**Safety:** A pre-commit hook prevents accidentally committing large files that weren't properly annexed.

## Image Labeling

Launch the labeling GUI:

```bash
python label_images.py
```

### Interface Controls

| Shortcut         | Action                             |
|------------------|------------------------------------|
| `1`              | Green status (auto-fills "Normal") |
| `2`              | Yellow status                      |
| `3`              | Red status (auto-fills "Offline")  |
| `Ctrl+Enter`     | Save and advance to next image     |
| `Ctrl+←/→`       | Navigate between images            |
| `Ctrl+Shift+←/→` | Jump to next unlabeled image       |
| `Ctrl+G`         | Jump to specific index             |
| `Delete`         | Delete current image               |

### Labeling Workflow

1. GUI opens to first unlabeled image
2. Select status classification (1/2/3)
3. Enter natural language description
4. Save with `Ctrl+Enter`
5. Repeat for remaining images

**Minimum Dataset**: 50-100 labeled images recommended for baseline accuracy.

Labels are saved to `artifacts/training_data/labels.json`.

For git-annex version control workflows (committing and uploading changes), see [artifacts/README.md](../artifacts/README.md).

## Model Training

Fine-tune BLIP vision-language model:

```bash
python train_model.py
```

### Training Process

1. Loads labeled data from `artifacts/training_data/labels.json`
2. Fine-tunes BLIP vision transformer model
3. Trains classification head (3 classes: green/yellow/red)
4. Learns natural language description generation
5. Saves model to `artifacts/models/v1/`

**Duration**: ~20 minutes on GPU, 1-2 hours on CPU (for ~2000 images, 10 epochs).

### Training Configuration

The training script auto-detects hardware and configures optimal settings:

| Setting | CPU | GPU (<8GB) | GPU (8-16GB) | GPU (16GB+) |
|---------|-----|------------|--------------|-------------|
| Batch Size | 4 | 4 | 8 | 16 |
| Mixed Precision | No | Yes | Yes | Yes |
| Data Workers | 0 | 4 | 4 | 4 |

To manually override, edit `train_model.py`:

```python
EPOCHS = 10          # Training iterations (more = better fit, risk of overfitting)
LEARNING_RATE = 5e-5 # Optimizer step size (lower = stable, higher = faster)
TRAIN_SPLIT = 0.7    # Train/validation/test split ratio
```

### Model Artifacts

Trained model components in `artifacts/models/v1/`:

- `model.safetensors` (854MB) - Vision transformer weights
- `status_classifier.pt` (775KB) - Classification head
- Configuration files (JSON)
- Tokenizer files

## Shared Library Reference

Core functionality in `lib/muni_lib.py`:

```python
from lib.muni_lib import download_muni_image, predict_muni_status

# Download status image
result = download_muni_image(output_folder="artifacts/training_data/images")

# Run inference
prediction = predict_muni_status(result['filepath'])
print(f"Status: {prediction['status']}")
print(f"Description: {prediction['description']}")
print(f"Confidence: {prediction['confidence']:.2%}")
```

## Troubleshooting

### Low Classification Accuracy

Causes and solutions:

- **Insufficient training data**: Label 100-200+ images minimum
- **Inconsistent labeling**: Review and correct label inconsistencies
- **Inadequate training**: Increase `EPOCHS` in configuration
- **Class imbalance**: Ensure balanced representation (green/yellow/red)

### tkinter Module Not Found

Platform-specific installation:

```bash
# macOS
brew install python-tk@3.13

# Debian/Ubuntu
sudo apt-get install python3-tk

# RHEL/CentOS
sudo yum install python3-tkinter

# Windows
# Included with official Python installer
```

### Out of Memory Errors

Solutions:

- The script auto-adjusts batch size based on GPU VRAM
- For manual override, modify `get_training_config()` in `train_model.py`
- Close resource-intensive applications
- Use system with more RAM (8GB+ recommended)
- Enable GPU training if available (CUDA) - uses more efficient memory management

### Image Download Failures

Common issues:

```bash
# Verify network connectivity
curl -I http://sfmunicentral.com/sfmunicentral_Snapshot_Objects/Mimic1_A7SE582P.jpg

# Check output directory exists
ls -la artifacts/training_data/images/

# Verify write permissions
touch artifacts/training_data/images/test.txt
rm artifacts/training_data/images/test.txt
```

### Training Script Errors

Debug steps:

```bash
# Verify labeled data exists
cat artifacts/training_data/labels.json | python3 -m json.tool | head

# Check PyTorch installation
python3 -c "import torch; print(torch.__version__)"

# Verify CUDA availability (if using GPU)
python3 -c "import torch; print(torch.cuda.is_available())"
```

## Model Evaluation

The training script outputs validation metrics:

- **Accuracy**: Overall classification accuracy
- **Loss**: Training and validation loss curves
- **Per-class metrics**: Precision/recall for each status level

Review these metrics to assess model quality before deployment.

## Next Steps

After training:

1. **Test locally**: Run API server (see [deploy/README.md](../deploy/README.md))
2. **Validate predictions**: Use `api/predict_status.py` for manual testing
3. **Deploy**: Push model to production (see [deploy/cloud/README.md](../deploy/cloud/README.md))

## Related Documentation

- **Data Management**: [artifacts/README.md](../artifacts/README.md) - Git-annex workflows
- **API Deployment**: [deploy/README.md](../deploy/README.md) - Local and cloud deployment
- **Environment Setup**: [SETUP.md](../SETUP.md) - Virtual environment configuration
- **Configuration**: [CONFIGURATION.md](../CONFIGURATION.md) - System configuration values
