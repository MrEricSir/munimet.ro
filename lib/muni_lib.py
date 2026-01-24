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

# Load .env file for local development (not on Cloud Run)
if not os.getenv('CLOUD_RUN'):
    try:
        from dotenv import load_dotenv
        _env_path = Path(__file__).resolve().parent.parent / '.env'
        if _env_path.exists():
            load_dotenv(_env_path)
    except ImportError:
        pass  # python-dotenv not installed, skip

# Lazy imports for ML dependencies (only needed for prediction)
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

# GCS model storage
GCS_MODELS_BUCKET = os.getenv('GCS_MODELS_BUCKET', 'munimetro-annex')
GCS_MODELS_PATH = 'models/snapshots'


def get_cache_path():
    """
    Get cache file path based on environment.

    Returns local path for development, Google Cloud Storage path for production.

    Returns:
        str: Cache file path (local file path or gs:// URL)
    """
    if os.getenv('CLOUD_RUN'):
        # Cloud Run environment - use Cloud Storage
        bucket = os.getenv('GCS_BUCKET', 'munimetro-cache')
        return f'gs://{bucket}/latest_status.json'
    else:
        # Local development - use local file
        return str(PROJECT_ROOT / "artifacts" / "runtime" / "cache" / "latest_status.json")


def read_cache():
    """
    Read cached status from local file or Cloud Storage.

    Returns:
        dict: Cached status data, or None if not found
    """
    cache_path = get_cache_path()

    try:
        if cache_path.startswith('gs://'):
            # Read from Cloud Storage
            from google.cloud import storage

            # Parse gs://bucket/path
            parts = cache_path[5:].split('/', 1)
            bucket_name = parts[0]
            blob_name = parts[1] if len(parts) > 1 else 'latest_status.json'

            client = storage.Client()
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(blob_name)

            if not blob.exists():
                return None

            content = blob.download_as_string()
            return json.loads(content)
        else:
            # Read from local file
            if not os.path.exists(cache_path):
                return None

            with open(cache_path, 'r') as f:
                return json.load(f)
    except Exception as e:
        print(f"Error reading cache: {e}")
        return None


def write_cache(data):
    """
    Write status data to local file or Cloud Storage.

    Args:
        data: Dict containing status data to cache

    Returns:
        bool: True if successful, False otherwise
    """
    cache_path = get_cache_path()

    try:
        if cache_path.startswith('gs://'):
            # Write to Cloud Storage
            from google.cloud import storage

            # Parse gs://bucket/path
            parts = cache_path[5:].split('/', 1)
            bucket_name = parts[0]
            blob_name = parts[1] if len(parts) > 1 else 'latest_status.json'

            client = storage.Client()
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(blob_name)

            # Upload with content type
            blob.upload_from_string(
                json.dumps(data, indent=2),
                content_type='application/json'
            )
            return True
        else:
            # Write to local file
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, 'w') as f:
                json.dump(data, f, indent=2)
            return True
    except Exception as e:
        print(f"Error writing cache: {e}")
        return False


# Status messages matching the frontend (api/html/index.html)
STATUS_MESSAGES = {
    'green': '🟢 All aboard: Muni is on track',
    'yellow': "🟡 Uh oh: Muni's not feeling well",
    'red': '🔴 Muni is taking a nap',
}


def post_to_bluesky(status, previous_status=None):
    """
    Post a status update to Bluesky.

    Requires environment variables:
    - BLUESKY_HANDLE: The account handle (e.g., 'munimetro.bsky.social')
    - BLUESKY_APP_PASSWORD: An app password for the account

    Args:
        status: Current status ('green', 'yellow', 'red')
        previous_status: Previous status for context (optional)

    Returns:
        dict: {'success': bool, 'uri': str or None, 'error': str or None}
    """
    handle = os.getenv('BLUESKY_HANDLE')
    app_password = os.getenv('BLUESKY_APP_PASSWORD')

    if not handle or not app_password:
        return {
            'success': False,
            'uri': None,
            'error': 'BLUESKY_HANDLE and BLUESKY_APP_PASSWORD environment variables required'
        }

    message = STATUS_MESSAGES.get(status, f'Status: {status}')

    try:
        from atproto import Client

        client = Client()
        client.login(handle, app_password)

        post = client.send_post(text=message)

        return {
            'success': True,
            'uri': post.uri,
            'error': None
        }
    except Exception as e:
        return {
            'success': False,
            'uri': None,
            'error': str(e)
        }


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
    Download a single Muni subway status image.

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


def _download_model_from_gcs(version, target_dir):
    """
    Download a model version from GCS to a local directory.

    Args:
        version: Model version (e.g., '20251223_224331')
        target_dir: Local directory to download to

    Returns:
        bool: True if successful, False otherwise
    """
    from google.cloud import storage

    bucket_name = GCS_MODELS_BUCKET
    source_prefix = f"{GCS_MODELS_PATH}/{version}/model/"

    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)

        # List all blobs in the model directory
        blobs = list(bucket.list_blobs(prefix=source_prefix))

        if not blobs:
            print(f"No model files found at gs://{bucket_name}/{source_prefix}")
            return False

        # Create target directory
        os.makedirs(target_dir, exist_ok=True)

        # Download each file
        for blob in blobs:
            # Get relative path (filename only)
            filename = blob.name.replace(source_prefix, '')
            if not filename:  # Skip directory markers
                continue

            local_path = os.path.join(target_dir, filename)
            print(f"  Downloading {filename}...")
            blob.download_to_filename(local_path)

        return True

    except Exception as e:
        print(f"Error downloading model from GCS: {e}")
        return False


def _get_model_dir():
    """
    Get the model directory, downloading from GCS if MODEL_VERSION is set.

    Returns:
        str: Path to model directory
    """
    model_version = os.getenv('MODEL_VERSION')

    if not model_version:
        # No version specified, use local model
        return MODEL_DIR

    # On Cloud Run, download to /tmp for writable storage
    if os.getenv('CLOUD_RUN'):
        target_dir = f"/tmp/models/{model_version}"
    else:
        target_dir = str(PROJECT_ROOT / "artifacts" / "models" / f"v1_gcs_{model_version}")

    # Check if already downloaded
    classifier_path = os.path.join(target_dir, 'status_classifier.pt')
    if os.path.exists(classifier_path):
        return target_dir

    # Download from GCS
    print(f"Downloading model version {model_version} from GCS...")
    if _download_model_from_gcs(model_version, target_dir):
        print(f"Model downloaded to {target_dir}")
        return target_dir
    else:
        print(f"Failed to download model, falling back to local")
        return MODEL_DIR


def load_muni_model(model_dir=None):
    """
    Load the trained Muni status prediction model.

    If MODEL_VERSION environment variable is set, downloads that version from GCS.
    Otherwise, uses the local model directory.

    Args:
        model_dir: Directory containing the trained model (optional, auto-detected if not provided)

    Returns:
        tuple: (model, processor, label_to_status, device)

    Raises:
        FileNotFoundError: If model directory doesn't exist
    """
    # Lazy import of ML dependencies
    import torch
    from transformers import BlipProcessor, BlipForConditionalGeneration

    # Determine model directory
    if model_dir is None:
        model_dir = _get_model_dir()

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
    processor = BlipProcessor.from_pretrained(model_dir, use_fast=False)
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

        # Classify status with threshold adjustment
        status_logits = model(pixel_values)
        status_probs = torch.softmax(status_logits, dim=1)

        # Extract probabilities
        green_prob = status_probs[0, 0].item()
        yellow_prob = status_probs[0, 1].item()
        red_prob = status_probs[0, 2].item()

        # Apply decision thresholds to reduce false positives
        # Require high confidence for yellow predictions to avoid false alarms
        #
        # TUNING GUIDE (see CONFIGURATION.md for details):
        # - Too many false yellows? Increase YELLOW_THRESHOLD to 0.75-0.80
        # - Missing real yellows? Decrease YELLOW_THRESHOLD to 0.60-0.65
        # - Production is ~95% green, so even small false positive rates = many false alarms
        #
        RED_THRESHOLD = 0.50     # Red needs 50% confidence (serious issues are usually clear)
        YELLOW_THRESHOLD = 0.75  # Yellow needs 75% confidence (safe default for Dec 25 model)
        # Green is default if neither red nor yellow meet their thresholds

        # Decision logic prioritizing precision over recall
        if red_prob > RED_THRESHOLD:
            # Red is confident - critical issue detected
            predicted_label = 2
        elif yellow_prob > YELLOW_THRESHOLD:
            # Yellow is confident - warning condition detected
            predicted_label = 1
        else:
            # Default to green (normal operation)
            # This includes cases where yellow is highest but below threshold
            predicted_label = 0

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
