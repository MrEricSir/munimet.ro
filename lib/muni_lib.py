#!/usr/bin/env python3
"""
Shared library for SF Muni Central image processing.

Contains reusable functions for:
- Downloading and validating Muni subway status images
- Predicting status from images using trained ML model
"""

import os
import requests
import time
import json
from datetime import datetime
from urllib.parse import urljoin
from pathlib import Path
from PIL import Image

# Lazy imports for ML dependencies (only needed for prediction)
# Lazy imports for Selenium (only needed for download)
# These are imported inside functions to avoid cross-environment dependencies


# Path resolution - get absolute paths relative to project root
# This works regardless of where the script is run from
LIB_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = LIB_DIR.parent

# Configuration constants
WEBPAGE_URL = "http://sfmunicentral.com/Enterprise/MetroLO.htm"
IMAGE_ID = "snapshotImage"
WAIT_TIME = 10
EXPECTED_WIDTH = 1860
EXPECTED_HEIGHT = 800
MODEL_DIR = str(PROJECT_ROOT / "artifacts" / "models" / "v1")


def _get_classifier_class():
    """Lazy import and return MuniClassifier class."""
    import torch.nn as nn

    class MuniClassifier(nn.Module):
        """BLIP model with additional classification head for status."""

        def __init__(self, base_model, num_classes=3):
            super().__init__()
            self.base_model = base_model
            self.status_classifier = nn.Sequential(
                nn.Linear(768, 256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, num_classes)
            )

        def forward(self, pixel_values):
            vision_outputs = self.base_model.vision_model(pixel_values=pixel_values)
            pooled_output = vision_outputs.pooler_output
            status_logits = self.status_classifier(pooled_output)
            return status_logits

    return MuniClassifier


def download_muni_image(output_folder="muni_snapshots", validate_dimensions=True):
    """
    Download a single Muni subway status image directly (no Selenium needed).

    The SF Muni Central page uses JavaScript to update the image every 5 seconds,
    but we can access the actual image URL directly.

    Args:
        output_folder: Directory to save the image
        validate_dimensions: If True, verify image is 1860x800 and delete if not

    Returns:
        dict: {
            'success': bool,
            'filepath': str or None,
            'width': int or None,
            'height': int or None,
            'error': str or None
        }
    """
    import random

    # Direct image URL (extracted from obfuscated JavaScript)
    IMAGE_URL = "http://sfmunicentral.com/sfmunicentral_Snapshot_Objects/Mimic1_A7SE582P.jpg"

    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    try:
        # Download with cache-busting parameter (mimics JavaScript behavior)
        params = {'nocache': random.randint(0, 999)}
        response = requests.get(IMAGE_URL, params=params, timeout=10)
        response.raise_for_status()

        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"muni_snapshot_{timestamp}.jpg"
        filepath = os.path.join(output_folder, filename)

        # Save the image
        with open(filepath, 'wb') as f:
            f.write(response.content)

        # Verify image dimensions if requested
        img = Image.open(filepath)
        width, height = img.size

        if validate_dimensions and (width != EXPECTED_WIDTH or height != EXPECTED_HEIGHT):
            os.remove(filepath)
            return {
                'success': False,
                'filepath': None,
                'width': width,
                'height': height,
                'error': f"Invalid dimensions: {width}x{height}, expected {EXPECTED_WIDTH}x{EXPECTED_HEIGHT}"
            }

        return {
            'success': True,
            'filepath': filepath,
            'width': width,
            'height': height,
            'error': None
        }

    except Exception as e:
        return {
            'success': False,
            'filepath': None,
            'width': None,
            'height': None,
            'error': str(e)
        }


def load_muni_model(model_dir=MODEL_DIR):
    """
    Load the trained Muni status prediction model.

    Args:
        model_dir: Directory containing the trained model

    Returns:
        tuple: (model, processor, label_to_status, device)

    Raises:
        FileNotFoundError: If model directory doesn't exist
    """
    # Lazy import of ML dependencies
    import torch
    from transformers import BlipProcessor, BlipForConditionalGeneration

    if not os.path.exists(model_dir):
        raise FileNotFoundError(
            f"Model directory not found: {model_dir}\n"
            f"Please train the model first using: python train_model.py"
        )

    # Get classifier class
    MuniClassifier = _get_classifier_class()

    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load processor and base model
    processor = BlipProcessor.from_pretrained(model_dir, use_fast=True)
    base_model = BlipForConditionalGeneration.from_pretrained(model_dir)

    # Load classification head
    model = MuniClassifier(base_model)
    classifier_path = os.path.join(model_dir, 'status_classifier.pt')
    if os.path.exists(classifier_path):
        model.status_classifier.load_state_dict(
            torch.load(classifier_path, map_location=device)
        )

    # Move model to device
    model = model.to(device)
    model.eval()

    # Load label mappings
    config_path = os.path.join(model_dir, 'config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
        label_to_status = {int(k): v for k, v in config['label_to_status'].items()}

    return model, processor, label_to_status, device


def predict_muni_status(image_path, model=None, processor=None, label_to_status=None, device=None, model_dir=MODEL_DIR):
    """
    Predict subway status from an image.

    Args:
        image_path: Path to the image file
        model: Pre-loaded model (optional, will load if not provided)
        processor: Pre-loaded processor (optional, will load if not provided)
        label_to_status: Label mapping dict (optional, will load if not provided)
        device: Torch device (optional, will determine if not provided)
        model_dir: Model directory (used if model not provided)

    Returns:
        dict: {
            'status': str ('green', 'yellow', 'red'),
            'status_confidence': float (0-1),
            'description': str,
            'probabilities': {
                'green': float,
                'yellow': float,
                'red': float
            }
        }

    Raises:
        ValueError: If image cannot be loaded
        FileNotFoundError: If model not found and not provided
    """
    # Load model if not provided
    if model is None or processor is None or label_to_status is None:
        model, processor, label_to_status, device = load_muni_model(model_dir)

    # Lazy import of torch for inference
    import torch

    # Load image
    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        raise ValueError(f"Error loading image: {e}")

    # Process image
    inputs = processor(images=image, return_tensors="pt")
    pixel_values = inputs.pixel_values.to(device)

    with torch.inference_mode():
        # Generate caption
        generated_ids = model.base_model.generate(
            pixel_values=pixel_values,
            max_length=64,
            num_beams=3,
            early_stopping=True
        )
        description = processor.decode(generated_ids[0], skip_special_tokens=True)

        # Classify status
        status_logits = model(pixel_values)
        status_probs = torch.softmax(status_logits, dim=1)
        predicted_label = torch.argmax(status_probs, dim=1).item()
        confidence = status_probs[0, predicted_label].item()

    predicted_status = label_to_status.get(predicted_label, 'unknown')

    return {
        'status': predicted_status,
        'status_confidence': confidence,
        'description': description,
        'probabilities': {
            'green': status_probs[0, 0].item(),
            'yellow': status_probs[0, 1].item(),
            'red': status_probs[0, 2].item()
        }
    }
