# Training Guide

Collect Muni status images, label them, and train a vision-language model to classify status and generate descriptions.

## Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements_ml.txt

# Install tkinter for labeling GUI (macOS)
brew install python-tk@3.13
```

## Step 1: Download Images

Collect status board images over time to build a training dataset.

```bash
python download_muni_image.py
```

- Downloads snapshots every 5 minutes from http://sfmunicentral.com/Enterprise/MetroLO.htm
- Saves to `../data/muni_snapshots/` folder
- Validates image dimensions (1860×800px)
- Press `Ctrl+C` to stop

**Tip**: Run for several hours/days to capture different status conditions (normal, delays, offline).

## Step 2: Label Images

Use the GUI to label collected images with status and descriptions.

```bash
python label_images.py
```

### Keyboard Shortcuts

- `1` - Green status (auto-fills "Normal")
- `2` - Yellow status
- `3` - Red status (auto-fills "Offline")
- `Ctrl+Enter` - Save & next
- `Ctrl+←/→` - Navigate images
- `Ctrl+Shift+←/→` - Jump to unlabeled images
- `Delete` - Delete current image

### Labeling Tips

1. Opens to first unlabeled image automatically
2. Select status (1/2/3) and enter description
3. Save with `Ctrl+Enter` to move to next image
4. Aim for 50-100+ labeled images for good accuracy

Labels saved to: `../data/training_labels.json`

## Step 3: Train Model

Fine-tune BLIP vision-language model on your labeled data.

```bash
python train_model.py
```

**What it does:**
- Loads labeled data from `../data/training_labels.json`
- Fine-tunes BLIP-2 vision-language model
- Trains status classifier (green/yellow/red)
- Learns to generate natural language descriptions
- Takes 5-20 minutes depending on data size and hardware

**Model saved to:** `../models/trained_model/`

### Training Configuration

Edit `train_model.py` to adjust:

```python
EPOCHS = 10          # More epochs = longer training, may improve accuracy
BATCH_SIZE = 4       # Increase if you have GPU memory
LEARNING_RATE = 5e-5 # Lower = more stable, higher = faster learning
TRAIN_SPLIT = 0.8    # 80% train, 20% validation
```

## Shared Library

All scripts use `lib/muni_lib.py` for core functionality:

```python
from lib.muni_lib import download_muni_image, predict_muni_status

# Download image
result = download_muni_image(output_folder="../data/muni_snapshots")

# Predict status
prediction = predict_muni_status(result['filepath'])
print(f"Status: {prediction['status']}")
print(f"Description: {prediction['description']}")
```

## Troubleshooting

**Low ML accuracy:**
- Label more images (aim for 100-200+)
- Ensure consistent labeling across images
- Train for more epochs
- Check for class imbalance (too many green, not enough red/yellow)

**tkinter not found:**
```bash
# macOS
brew install python-tk@3.13

# Ubuntu/Debian
sudo apt-get install python3-tk

# Windows
# Included with official Python installer
```

**Out of memory during training:**
- Reduce `BATCH_SIZE` in `train_model.py`
- Close other applications
- Use a machine with more RAM or GPU memory

## Next Steps

Once you've trained a model, proceed to the [API & Deployment Guide](../api/README.md) to run the web API and deploy to production.
