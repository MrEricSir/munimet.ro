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

1. Authenticate with Google Cloud (see [SETUP.md](../SETUP.md))
2. Download training data using sync scripts:

```bash
# Download training data (~270MB)
../scripts/sync-training-data.sh download    # macOS/Linux
..\scripts\sync-training-data.ps1 download   # Windows
```

The sync scripts use `gsutil rsync` to efficiently download only changed files.

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

### Uploading New Data

After collecting and labeling new images, you can upload them to Google Cloud Storage:

```bash
# Upload training data to GCS
../scripts/sync-training-data.sh upload    # macOS/Linux
..\scripts\sync-training-data.ps1 upload   # Windows
```

The sync scripts use `gsutil rsync` to efficiently upload only changed files.

## Image Labeling

Launch the labeling GUI:

```bash
python label_images.py
```

### Interface Modes

The labeling tool provides two modes accessible via tabs:

- **All Images**: Label all images in the dataset
- **Outliers**: Focus on model prediction errors for targeted relabeling

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

Labels are saved to `artifacts/training_data/labels.json`. To share your labeled data with collaborators, use the upload sync script (see [Uploading New Data](#uploading-new-data) above).

### Outliers Mode

After training a model, the **Outliers** tab provides focused relabeling of problematic predictions:

**What are outliers?**

During model evaluation (see `evaluate_test_set.py`), the system identifies three categories of problematic predictions:

1. **Misclassified**: Model predicted wrong status (e.g., predicted yellow but should be green)
2. **Low Confidence**: Model predicted correctly but with low confidence (<95%)
3. **High Confidence Errors**: Model was very confident but wrong (worst case)

These outliers are saved to `artifacts/models/v1/outlier_report.json`.

**Using Outliers Mode:**

1. Click the **Outliers** tab in the labeling tool
2. Images are filtered to show only outliers from the report
3. Each image displays an explanation of why it's an outlier
4. Review and correct labels as needed
5. Retrain the model with improved labels

**Benefits:**

- Targeted relabeling of problematic images
- Understand where the model struggles
- Improve model accuracy more efficiently than random relabeling
- Clear explanations for each outlier

**Example outlier explanation:**
```
High confidence error: predicted green instead of yellow (99.8%)
```

This indicates the model needs better training examples to distinguish between these states.

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

Each training run uploads a versioned snapshot to GCS:

```
gs://munimetro-annex/models/snapshots/<timestamp>/
├── model/
│   ├── model.safetensors (854MB) - Vision transformer weights
│   ├── status_classifier.pt (775KB) - Classification head
│   └── [config files]
└── run_metadata.json - Training metrics and configuration
```

Use `python3 scripts/manage-models.py list` to see available versions.

### Training History & Metrics Tracking

Every training run is automatically tracked with comprehensive metrics for performance monitoring and reproducibility.

**Automatic Tracking:**
- Training run ID (timestamp-based)
- Git commit hash and branch
- Dataset statistics and class distribution
- Hyperparameters and training configuration
- Test accuracy, loss, and per-class metrics
- Confusion matrix analysis
- Outlier counts

**Saved Locations:**

1. **Training History** (`artifacts/models/training_history.json`)
   - Append-only log of all training runs
   - Compare performance across runs
   - Track accuracy trends over time

2. **Snapshots** (`artifacts/models/snapshots/{run_id}/`)
   - Complete reproducible snapshot per run
   - Includes model, training data, metrics, environment
   - Can recreate exact training conditions

**View Training History:**

```bash
python view_training_history.py
```

**Sample Output:**

```
TRAINING HISTORY SUMMARY
================================================================
#   Run ID            Date                 Test Acc  Yellow F1  Outliers
----------------------------------------------------------------
⭐1   20251223_224331   2025-12-23 22:43:31   94.24%    0.659      31
  2   20251224_153012   2025-12-24 15:30:12   95.87%    0.782      18
----------------------------------------------------------------

PERFORMANCE TRENDS
Yellow Detection F1 Trend (minority class):
  Run 1: 0.659
  Run 2: 0.782 (↑ +0.123)

Yellow → Green Confusion Trend (critical metric):
  Run 1: 24 (44.4% of yellows)
  Run 2: 12 (22.2% of yellows) (↓ -12 better)
```

### Confusion Matrix Analysis

Training automatically generates a confusion matrix showing which classes are confused with each other.

**Location:**
- Console output during training
- `training_history.json` (in `confusion_matrix` field)
- `snapshots/{run_id}/run_metadata.json`
- `view_training_history.py` (detailed run view)

**Sample Output:**

```
Confusion Matrix:
  (Rows = True labels, Columns = Predictions)

              Pred GREEN   Pred YELLOW   Pred RED
  True GREEN         392             4          2
  True YELLOW         24            30          0
  True RED             1             0         85

Key Confusion Analysis:
  Yellow → Green misclassifications: 24 (44.4% of yellows)
  Yellow → Red misclassifications: 0 (0.0% of yellows)
  Green → Yellow false positives: 4 (1.0% of greens)
```

**Interpreting Results:**

- **Diagonal values** (392, 30, 85): Correct predictions
- **Off-diagonal values**: Confusions between classes
- **Yellow → Green**: Most critical metric - indicates missed warnings
- **Green → Yellow**: False alarms (less critical but affects user trust)

Use this to identify which classes need:
- More training examples
- Better labeling consistency
- Adjusted class weights

### Class Balancing Through Undersampling

**NEW:** The training script now automatically balances your dataset by undersampling the majority class (green).

**How it works:**

Instead of requiring you to collect equal amounts of each color, you can:
1. Collect images naturally as they occur (greens will be most common)
2. Label everything you collect
3. Training automatically uses fewer greens to balance with yellows

**Configuration:**

Edit `train_model.py` line 469:

```python
TARGET_RATIO = 4.0  # Green:Yellow ratio (4.0 = 1:4 ideal)
```

**Examples:**

| Setting | Result | Use When |
|---------|--------|----------|
| `TARGET_RATIO = 4.0` | 1:4 ratio (recommended) | You have 7x+ more greens than yellows |
| `TARGET_RATIO = 6.0` | 1:6 ratio (relaxed) | You have moderate imbalance |
| `TARGET_RATIO = 100.0` | No undersampling | Dataset already balanced |

**Sample output:**

```
📊 Class Balancing (Undersampling):
  Original counts: Green=1,878, Yellow=248, Red=408
  ✅ Undersampled greens: 1,878 → 992
  ✅ New ratio: 1:7.6 → 1:4.0
  📉 Total samples: 2,534 → 1,648
```

**Benefits:**
- ✅ No need to wait for more yellow images
- ✅ Collect naturally without bias
- ✅ Immediate training improvements
- ✅ Full dataset saved for future use

**Trade-offs:**
- ⚠️ Smaller effective training set
- ⚠️ May need +2-3 more epochs to compensate

### Hard Negative Mining

The training script applies **hard negative mining** to address class imbalance, with special focus on the yellow class (often confused with green).

**How it works:**

```python
# Automatic class weights based on frequency
green: 1.0x   (most common - baseline)
yellow: 11.3x (rare + additional 1.5x boost)
red: 4.6x     (uncommon)
```

**Why yellow gets extra boost:**
- Yellow warnings are often subtle (minor delays)
- Most frequently confused with green
- Missing a yellow is worse than a false positive
- Only 7-10% of dataset (class imbalance)

**Tuning the boost:**

Edit `train_model.py` line 442:

```python
YELLOW_BOOST_FACTOR = 1.5  # Increase to 2.0 for more yellow focus
```

Higher values = model focuses more on yellow detection (may increase false positives).

### Decision Threshold Adjustment

**Production inference** uses smart thresholds instead of simple argmax to bias toward yellow detection.

**Location:** `lib/muni_lib.py` (predict_muni_status function)

**Logic:**

```python
YELLOW_THRESHOLD = 0.08           # Consider yellow if prob > 8%
GREEN_CONFIDENCE_THRESHOLD = 0.88 # Green must be 88%+ confident

if yellow_prob > 0.08 and green_prob < 0.88:
    predict yellow  # Catch marginal yellows
```

**Example:**

| Probabilities | Without Threshold | With Threshold |
|--------------|-------------------|----------------|
| G=87%, Y=10%, R=3% | Green (wrong) | Yellow (correct) ✅ |
| G=92%, Y=5%, R=3% | Green | Green (correct) ✅ |
| G=45%, Y=9%, R=46% | Red | Red (correct) ✅ |

**Tuning thresholds:**

Edit `muni_lib.py` lines 359-360:

```python
YELLOW_THRESHOLD = 0.08  # Lower = fewer yellow predictions
GREEN_CONFIDENCE_THRESHOLD = 0.88  # Higher = more yellow predictions
```

Adjust based on:
- Too many missed yellows → Increase GREEN_CONFIDENCE_THRESHOLD to 0.90
- Too many false yellow alarms → Lower YELLOW_THRESHOLD to 0.06

### Reproducible Snapshots

Each training run uploads a complete snapshot to GCS for reproducibility.

**Snapshot Location:**

```
gs://munimetro-annex/models/snapshots/20251223_224331/
├── model/                      # Complete trained model
│   ├── model.safetensors
│   ├── config.json
│   └── status_classifier.pt
├── training_labels.json        # Exact training data used
├── run_metadata.json          # All metrics and settings
└── environment.txt            # Python/PyTorch versions, dependencies
```

**Use Cases:**

1. **Reproduce training:** Know exact code version (git commit), data, and hyperparameters
2. **Rollback:** Deploy previous best-performing model if new one regresses
3. **A/B testing:** Compare models trained with different settings
4. **Debugging:** Investigate why a specific run performed differently

**Switch to a different model:**

```bash
# List available models with metrics
python3 scripts/manage-models.py list

# Switch production to a different version (no rebuild needed)
python3 scripts/manage-models.py switch 20251223_224331
```

### Reviewed Outliers Tracking

Outliers in the labeling tool are automatically marked as "reviewed" to avoid duplicate work.

**Behavior:**

1. When viewing outliers in "Outliers" mode
2. Saving an image sets `"reviewed": true` in labels.json
3. On next labeling session, reviewed outliers are hidden
4. Console shows: `"Found 8 unreviewed outlier images (4 already reviewed)"`

**Status bar shows review status:**
```
✓ Labeled (last updated: 2025-12-23T14:30:00) | ✓ Reviewed: 2025-12-23T15:45:00
```

**Benefits:**
- No duplicate outlier reviews
- Track which problematic images you've already examined
- Focus on new outliers from latest training run

### Production Validation (False Positive Tracking)

When images are incorrectly predicted in production (e.g., a green image predicted as yellow), you can flag them for targeted validation.

**Workflow:**

1. **Flag problematic images in labeler:**
   - Open `label_images.py`
   - Navigate to the image that was a false positive
   - Check the **"Was False Positive"** checkbox in the Special section
   - Save with `Ctrl+Enter`

2. **Training automatically validates against flagged images:**
   - Images with `false_positive: true` are always placed in the test set (never used for training)
   - Training output shows dedicated "PRODUCTION VALIDATION" section
   - Metrics track accuracy specifically on previously problematic images

**Sample training output:**
```
📋 Production Validation Set:
  12 images marked as 'false_positive' will be in test set
  Distribution: {'green': 10, 'yellow': 2}

...

================================================================================
PRODUCTION VALIDATION (images marked as 'false_positive')
================================================================================
  Total: 12 images
  Accuracy: 91.7% (11/12)

  By true label:
    GREEN : 90.0% (9/10)
    YELLOW: 100.0% (2/2)

  ❌ Still failing on 1 previously problematic images:
    muni_snapshot_20260124_032145.jpg: predicted yellow, should be green
```

3. **Compare models using FP-val metric:**
   ```bash
   python scripts/manage-models.py list
   ```

   Output now includes `FP-val` column:
   ```
     20260124_103045  acc:95.2%  red-R:97.1%  yel-P:82.3%  FP-val:92% <-- DEPLOYED
     20251230_142847  acc:95.9%  red-R:96.6%  yel-P:81.7%
   ```

4. **Deploy only if FP-val improves:**
   - Higher FP-val = better handling of previously problematic images
   - If FP-val drops, the new model may reintroduce old problems

**Benefits:**
- Targeted validation on real production failures
- Automatic tracking through the normal labeling workflow
- Clear metrics to compare model improvements on problem cases
- No separate validation scripts needed

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

### Outlier Analysis

After training, run the evaluation script to generate an outlier report:

```bash
python evaluate_test_set.py
```

This creates `artifacts/models/v1/outlier_report.json` containing:

- **Misclassified images**: Wrong predictions to investigate
- **Low confidence predictions**: Correct but uncertain classifications
- **High confidence errors**: Model confidently predicted wrong status

**Usage**: Review outliers in the labeling tool's **Outliers** tab to identify and fix labeling issues or collect more diverse training examples for problematic cases.

## Next Steps

After training:

1. **Review metrics**: Check the training output for accuracy and per-class metrics
2. **List models**: `python3 scripts/manage-models.py list` to see your new snapshot
3. **Deploy to production**:
   ```bash
   # Switch to new model (no rebuild needed)
   python3 scripts/manage-models.py switch <new_version>
   ```
4. **Test locally**: Run API server (see [deploy/README.md](../deploy/README.md))

## Related Documentation

- **API Deployment**: [deploy/README.md](../deploy/README.md) - Local and cloud deployment
- **Environment Setup**: [SETUP.md](../SETUP.md) - Virtual environment configuration
- **Configuration**: [CONFIGURATION.md](../CONFIGURATION.md) - System configuration values
