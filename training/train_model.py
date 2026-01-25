#!/usr/bin/env python3
"""
Train a vision-language model to classify Muni subway status and generate descriptions.

This script:
1. Loads labeled training data from training_labels.json
2. Fine-tunes a pre-trained vision-language model (BLIP-2)
3. Trains both status classification (green/yellow/red) and description generation
4. Saves the trained model for inference

Supports both CPU and GPU training with automatic hardware detection.
GPU training uses mixed precision (AMP) for faster training.
"""

import json
import os
import sys
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image, ImageEnhance
from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    get_linear_schedule_with_warmup
)
from tqdm import tqdm
import numpy as np
import random
from sklearn.metrics import precision_recall_fscore_support, classification_report, confusion_matrix
import subprocess
import shutil
from datetime import datetime

# Configuration
LABELS_FILE = "../artifacts/training_data/labels.json"
MODEL_OUTPUT_DIR = "../artifacts/models/v1"
SNAPSHOTS_DIR = "../artifacts/models/snapshots"
TRAINING_HISTORY_FILE = "../artifacts/models/training_history.json"
EPOCHS = 10
LEARNING_RATE = 5e-5
TRAIN_SPLIT = 0.7   # 70% train
VAL_SPLIT = 0.15    # 15% validation
TEST_SPLIT = 0.15   # 15% test (holdout set, never used during training)
MAX_LENGTH = 128

def get_training_config():
    """Get optimal training configuration based on available hardware."""
    config = {
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        'use_amp': False,  # Automatic Mixed Precision
        'batch_size': 4,
        'num_workers': 0,
        'pin_memory': False,
    }

    if torch.cuda.is_available():
        # GPU detected - enable optimizations
        config['use_amp'] = True
        config['batch_size'] = 8  # Larger batch size for GPU
        config['pin_memory'] = True  # Faster CPU->GPU transfer

        # Windows has slow multiprocessing (spawn vs fork), so keep num_workers=0
        if sys.platform == 'win32':
            config['num_workers'] = 0
        else:
            config['num_workers'] = 4  # Parallel data loading on Linux/Mac

        # Get GPU info
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3

        print(f"GPU detected: {gpu_name} ({gpu_mem:.1f} GB)")
        if sys.platform == 'win32':
            print("Enabling GPU optimizations: AMP, larger batch size")
        else:
            print("Enabling GPU optimizations: AMP, larger batch, parallel loading")

        # Adjust batch size based on GPU memory
        if gpu_mem < 8:
            config['batch_size'] = 4
            print(f"  Reduced batch size to {config['batch_size']} for lower VRAM")
        elif gpu_mem >= 16:
            config['batch_size'] = 16
            print(f"  Increased batch size to {config['batch_size']} for high VRAM")
    else:
        print("No GPU detected - using CPU (training will be slower)")

    return config


class MuniDataset(Dataset):
    """Dataset for Muni subway status images and labels."""

    def __init__(self, data, processor, augment_minority_classes=False):
        self.data = data
        self.processor = processor
        self.augment_minority_classes = augment_minority_classes

        # Map status to numeric labels
        self.status_to_label = {
            'green': 0,
            'yellow': 1,
            'red': 2,
            '': -1  # No status
        }
        self.label_to_status = {v: k for k, v in self.status_to_label.items() if v >= 0}

    def apply_augmentation(self, image):
        """Apply random augmentations to help model learn minority class features."""
        # Randomly apply brightness adjustment (±20%)
        if random.random() > 0.5:
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(random.uniform(0.8, 1.2))

        # Randomly apply contrast adjustment (±20%)
        if random.random() > 0.5:
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(random.uniform(0.8, 1.2))

        # Randomly apply color saturation adjustment (±15%)
        if random.random() > 0.5:
            enhancer = ImageEnhance.Color(image)
            image = enhancer.enhance(random.uniform(0.85, 1.15))

        return image

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Load image
        try:
            image = Image.open(item['image_path']).convert('RGB')
        except Exception as e:
            print(f"Error loading image {item['image_path']}: {e}")
            # Return a blank image as fallback
            image = Image.new('RGB', (800, 600), color='white')

        # Get description and status
        description = item.get('description', '')
        status = item.get('status', '')
        status_label = self.status_to_label.get(status, -1)

        # Apply augmentation to minority classes (yellow/red) during training
        if self.augment_minority_classes and status in ['yellow', 'red']:
            image = self.apply_augmentation(image)

        # Process image and text with BLIP processor
        encoding = self.processor(
            images=image,
            text=description,
            return_tensors="pt",
            padding="max_length",
            max_length=MAX_LENGTH,
            truncation=True
        )

        # Remove batch dimension
        encoding = {k: v.squeeze(0) for k, v in encoding.items()}

        # Add status label
        encoding['status_label'] = torch.tensor(status_label, dtype=torch.long)

        return encoding


def load_training_data():
    """Load training data from JSON file."""
    if not os.path.exists(LABELS_FILE):
        raise FileNotFoundError(f"Labels file not found: {LABELS_FILE}")

    with open(LABELS_FILE, 'r') as f:
        data = json.load(f)

    training_data = data.get('training_data', [])

    # Filter out items without both description and status
    filtered_data = [
        item for item in training_data
        if item.get('description', '').strip() and item.get('status', '')
    ]

    print(f"Total labeled images: {len(training_data)}")
    print(f"Fully labeled (with status and description): {len(filtered_data)}")

    if len(filtered_data) < 10:
        print("\nWARNING: Very few training samples! You need at least 50-100 labeled images for good results.")
        print("Continue labeling images before training for best results.")

    return filtered_data


def get_git_info():
    """Get current git commit hash and branch for reproducibility."""
    try:
        commit_hash = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            stderr=subprocess.DEVNULL,
            cwd=os.path.dirname(__file__)
        ).decode('utf-8').strip()

        branch = subprocess.check_output(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
            stderr=subprocess.DEVNULL,
            cwd=os.path.dirname(__file__)
        ).decode('utf-8').strip()

        # Check if there are uncommitted changes
        status = subprocess.check_output(
            ['git', 'status', '--porcelain'],
            stderr=subprocess.DEVNULL,
            cwd=os.path.dirname(__file__)
        ).decode('utf-8').strip()

        has_changes = bool(status)

        return {
            'commit_hash': commit_hash,
            'branch': branch,
            'has_uncommitted_changes': has_changes
        }
    except Exception as e:
        print(f"Could not get git info: {e}")
        return {
            'commit_hash': 'unknown',
            'branch': 'unknown',
            'has_uncommitted_changes': False
        }


def save_training_history(run_data):
    """Append training run data to history file."""
    history = {'training_runs': []}

    # Load existing history if it exists
    if os.path.exists(TRAINING_HISTORY_FILE):
        try:
            with open(TRAINING_HISTORY_FILE, 'r') as f:
                history = json.load(f)
        except Exception as e:
            print(f"Warning: Could not load existing history: {e}")

    # Append new run
    history['training_runs'].append(run_data)

    # Save updated history
    os.makedirs(os.path.dirname(TRAINING_HISTORY_FILE), exist_ok=True)
    with open(TRAINING_HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=2)

    print(f"Training history saved to {TRAINING_HISTORY_FILE}")


def create_snapshot(run_id, model, processor, train_dataset, run_data):
    """Create a versioned snapshot of the model and training data."""
    snapshot_dir = os.path.join(SNAPSHOTS_DIR, run_id)
    os.makedirs(snapshot_dir, exist_ok=True)

    print(f"\nCreating snapshot: {snapshot_dir}")

    # Save model files
    model_dir = os.path.join(snapshot_dir, 'model')
    os.makedirs(model_dir, exist_ok=True)
    model.base_model.save_pretrained(model_dir)
    processor.save_pretrained(model_dir)
    torch.save(
        model.status_classifier.state_dict(),
        os.path.join(model_dir, 'status_classifier.pt')
    )

    # Save config
    with open(os.path.join(model_dir, 'config.json'), 'w') as f:
        json.dump({
            'status_to_label': train_dataset.status_to_label,
            'label_to_status': train_dataset.label_to_status
        }, f, indent=2)

    # Save training data snapshot (copy of labels.json)
    shutil.copy(LABELS_FILE, os.path.join(snapshot_dir, 'training_labels.json'))

    # Save run metrics and metadata
    with open(os.path.join(snapshot_dir, 'run_metadata.json'), 'w') as f:
        json.dump(run_data, f, indent=2)

    # Save environment info
    with open(os.path.join(snapshot_dir, 'environment.txt'), 'w') as f:
        f.write(f"Python version: {subprocess.check_output(['python', '--version']).decode().strip()}\n")
        f.write(f"PyTorch version: {torch.__version__}\n")
        f.write(f"CUDA available: {torch.cuda.is_available()}\n")
        if torch.cuda.is_available():
            f.write(f"CUDA version: {torch.version.cuda}\n")
        f.write("\nPip freeze:\n")
        try:
            freeze = subprocess.check_output(['pip', 'freeze']).decode()
            f.write(freeze)
        except:
            f.write("Could not get pip freeze output\n")

    print(f"✓ Snapshot saved to {snapshot_dir}")
    return snapshot_dir


def undersample_majority_class(data, target_ratio=4.0):
    """Undersample green (majority class) to achieve better class balance.

    Args:
        data: List of training samples
        target_ratio: Desired green:yellow ratio (default 4.0 = 1:4 ratio)

    Returns:
        Balanced dataset with reduced greens
    """
    # Separate by status
    status_groups = {}
    for item in data:
        status = item.get('status', '')
        if status not in status_groups:
            status_groups[status] = []
        status_groups[status].append(item)

    green_count = len(status_groups.get('green', []))
    yellow_count = len(status_groups.get('yellow', []))
    red_count = len(status_groups.get('red', []))

    print(f"\n📊 Class Balancing (Undersampling):")
    print(f"  Original counts: Green={green_count}, Yellow={yellow_count}, Red={red_count}")

    # Skip if no yellows or already balanced
    if yellow_count == 0:
        print("  ⚠️ No yellow samples - skipping undersampling")
        return data

    current_ratio = green_count / yellow_count if yellow_count > 0 else 0
    if current_ratio <= target_ratio:
        print(f"  ✅ Already balanced (ratio 1:{current_ratio:.1f} ≤ target 1:{target_ratio:.1f})")
        return data

    # Calculate how many greens to keep
    target_green_count = int(yellow_count * target_ratio)

    # Randomly sample greens
    np.random.shuffle(status_groups['green'])
    sampled_greens = status_groups['green'][:target_green_count]

    # Combine with all yellows and reds
    balanced_data = sampled_greens + status_groups.get('yellow', []) + status_groups.get('red', [])
    np.random.shuffle(balanced_data)

    new_ratio = target_green_count / yellow_count
    print(f"  ✅ Undersampled greens: {green_count} → {target_green_count}")
    print(f"  ✅ New ratio: 1:{current_ratio:.1f} → 1:{new_ratio:.1f}")
    print(f"  📉 Total samples: {len(data)} → {len(balanced_data)}")

    return balanced_data


def split_data(data, train_ratio=0.7, val_ratio=0.15):
    """Split data into train, validation, and test sets with stratification.

    Images marked as 'false_positive' are always placed in the test set
    to ensure they are used for validation, not training.

    Args:
        data: List of labeled samples
        train_ratio: Proportion for training (default 0.7)
        val_ratio: Proportion for validation (default 0.15)

    Returns:
        train_data, val_data, test_data (remaining samples go to test)
    """
    # Separate false_positive images - they always go to test set
    false_positive_data = [item for item in data if item.get('false_positive', False)]
    regular_data = [item for item in data if not item.get('false_positive', False)]

    if false_positive_data:
        print(f"\n📋 Production Validation Set:")
        print(f"  {len(false_positive_data)} images marked as 'false_positive' will be in test set")
        fp_statuses = {}
        for item in false_positive_data:
            s = item.get('status', 'unknown')
            fp_statuses[s] = fp_statuses.get(s, 0) + 1
        print(f"  Distribution: {fp_statuses}")

    # Stratify remaining data by status
    status_groups = {}
    for item in regular_data:
        status = item.get('status', '')
        if status not in status_groups:
            status_groups[status] = []
        status_groups[status].append(item)

    train_data = []
    val_data = []
    test_data = []

    # Split each status group proportionally
    for status, items in status_groups.items():
        np.random.shuffle(items)
        n = len(items)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)

        train_data.extend(items[:train_end])
        val_data.extend(items[train_end:val_end])
        test_data.extend(items[val_end:])

    # Add false_positive images to test set
    test_data.extend(false_positive_data)

    # Shuffle the combined splits
    np.random.shuffle(train_data)
    np.random.shuffle(val_data)
    np.random.shuffle(test_data)

    return train_data, val_data, test_data


class MuniClassifier(nn.Module):
    """BLIP model with additional classification head for status."""

    def __init__(self, base_model, num_classes=3):
        super().__init__()
        self.base_model = base_model

        # Add classification head for status
        # BLIP vision encoder has hidden size 768
        self.status_classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )

    def forward(self, pixel_values, input_ids=None, attention_mask=None, labels=None):
        # Get BLIP outputs for caption generation
        outputs = self.base_model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True
        )

        # Get vision features for classification
        # Use the vision encoder's pooled output
        vision_outputs = self.base_model.vision_model(pixel_values=pixel_values)
        pooled_output = vision_outputs.pooler_output

        # Classify status
        status_logits = self.status_classifier(pooled_output)

        return outputs, status_logits


def train_model():
    """Train the Muni status classifier."""

    print("=" * 60)
    print("Muni Subway Status Model Training")
    print("=" * 60)
    print()

    # Get hardware-optimized configuration
    config = get_training_config()
    device = config['device']
    use_amp = config['use_amp']
    batch_size = config['batch_size']

    print(f"Using device: {device}")
    print(f"Batch size: {batch_size}")
    print(f"Mixed precision (AMP): {'enabled' if use_amp else 'disabled'}")
    print()

    # Load data
    print("[1/6] Loading training data...")
    all_data = load_training_data()

    if len(all_data) == 0:
        print("ERROR: No training data found. Please label some images first.")
        return

    # Apply class balancing through undersampling (optional - set ratio to high value to disable)
    # TARGET_RATIO: Desired green:yellow ratio (4.0 = 1:4, 6.0 = 1:6, 100.0 = disable)
    # NOTE: 4.0 caused high false positive rates. 8.0 is closer to production reality (~19:1)
    TARGET_RATIO = 8.0
    all_data = undersample_majority_class(all_data, target_ratio=TARGET_RATIO)

    train_data, val_data, test_data = split_data(all_data, TRAIN_SPLIT, VAL_SPLIT)
    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")
    print(f"Test samples: {len(test_data)} (holdout set)")

    # Show class distribution
    def count_statuses(data):
        counts = {}
        for item in data:
            status = item.get('status', 'unknown')
            counts[status] = counts.get(status, 0) + 1
        return counts

    print("\nClass distribution:")
    train_counts = count_statuses(train_data)
    print(f"  Train: {train_counts}")
    print(f"  Val:   {count_statuses(val_data)}")
    print(f"  Test:  {count_statuses(test_data)}")

    # Calculate class weights to handle imbalance
    # Weight is inversely proportional to class frequency
    # This makes the model pay more attention to underrepresented classes
    status_order = ['green', 'yellow', 'red']  # Matches labels 0, 1, 2
    total_train = len(train_data)
    class_weights = []

    for status in status_order:
        count = train_counts.get(status, 1)  # Avoid division by zero
        weight = total_train / (len(status_order) * count)
        class_weights.append(weight)

    # Normalize so the smallest weight (most common class) has weight 1.0
    # This makes the penalty for minority classes stronger
    min_weight = min(class_weights)
    class_weights = [w / min_weight for w in class_weights]

    # Hard negative mining: Boost yellow weight even more
    # Yellow is the hardest class (often confused with green)
    YELLOW_BOOST_FACTOR = 1.5  # Additional 50% boost for yellow
    class_weights[1] *= YELLOW_BOOST_FACTOR  # yellow is index 1

    print("\nClass weights (to handle imbalance):")
    for status, weight in zip(status_order, class_weights):
        print(f"  {status:6s}: {weight:.2f}x")
    print("  (Higher weight = model pays more attention to this class)")
    print(f"  Note: Yellow has additional {YELLOW_BOOST_FACTOR}x hard negative mining boost")

    # Convert to tensor for PyTorch
    class_weight_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

    # Load model and processor
    print("\n[2/6] Loading pre-trained BLIP model...")
    model_name = "Salesforce/blip-image-captioning-base"
    processor = BlipProcessor.from_pretrained(model_name, use_fast=True)
    base_model = BlipForConditionalGeneration.from_pretrained(model_name)

    # Create custom model with classification head
    model = MuniClassifier(base_model)
    model = model.to(device)

    # Create datasets
    print("\n[3/6] Preparing datasets...")
    train_dataset = MuniDataset(train_data, processor, augment_minority_classes=True)
    val_dataset = MuniDataset(val_data, processor, augment_minority_classes=False)
    test_dataset = MuniDataset(test_data, processor, augment_minority_classes=False)
    print("Data augmentation enabled for minority classes (yellow/red) in training set")

    # Create sample weights for hard example mining
    # Combine class balancing with reviewed-sample boost
    print("\nHard Example Mining (Reviewed Samples):")
    REVIEWED_BOOST = 2.0  # 2x weight for samples marked as reviewed

    sample_weights = []
    reviewed_count = 0
    for item in train_data:
        status = item.get('status', '')
        status_idx = train_dataset.status_to_label.get(status, 0)

        # Base weight from class balancing
        base_weight = class_weights[status_idx]

        # Additional boost for reviewed samples (hard examples from previous runs)
        reviewed_boost = REVIEWED_BOOST if item.get('reviewed', False) else 1.0
        if item.get('reviewed', False):
            reviewed_count += 1

        # Combined weight
        sample_weights.append(base_weight * reviewed_boost)

    print(f"  Training samples with 'reviewed' flag: {reviewed_count}/{len(train_data)} ({reviewed_count/len(train_data)*100:.1f}%)")
    print(f"  Reviewed samples get {REVIEWED_BOOST}x additional weight")
    print(f"  Effect: Model will see reviewed (hard) examples ~2x more often per epoch")

    # Create weighted sampler for training
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True  # Allow same sample to appear multiple times
    )

    # Create data loaders with hardware-optimized settings
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,  # Use weighted sampler instead of shuffle
        num_workers=config['num_workers'],
        pin_memory=config['pin_memory']
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=config['pin_memory']
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=config['pin_memory']
    )

    # Setup optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    # Loss function for classification with class weights
    classification_criterion = nn.CrossEntropyLoss(weight=class_weight_tensor, ignore_index=-1)

    # Setup AMP scaler for mixed precision training (GPU only)
    scaler = torch.amp.GradScaler('cuda') if use_amp else None

    # Training loop
    print(f"\n[4/6] Training model for {EPOCHS} epochs...")
    print()

    best_val_loss = float('inf')

    for epoch in range(EPOCHS):
        # Training
        model.train()
        train_loss = 0
        train_caption_loss = 0
        train_status_loss = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
        for batch in pbar:
            # Move batch to device
            pixel_values = batch['pixel_values'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            status_labels = batch['status_label'].to(device)

            # Forward pass with optional AMP
            if use_amp:
                with torch.amp.autocast('cuda'):
                    caption_outputs, status_logits = model(
                        pixel_values=pixel_values,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=input_ids
                    )
                    caption_loss = caption_outputs.loss
                    status_loss = classification_criterion(status_logits, status_labels)
                    loss = caption_loss + 0.5 * status_loss

                # Backward pass with scaler
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard forward pass (CPU or no AMP)
                caption_outputs, status_logits = model(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=input_ids
                )
                caption_loss = caption_outputs.loss
                status_loss = classification_criterion(status_logits, status_labels)
                loss = caption_loss + 0.5 * status_loss

                # Standard backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            scheduler.step()

            # Track losses
            train_loss += loss.item()
            train_caption_loss += caption_loss.item()
            train_status_loss += status_loss.item()

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'cap': f'{caption_loss.item():.4f}',
                'status': f'{status_loss.item():.4f}'
            })

        avg_train_loss = train_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        val_caption_loss = 0
        val_status_loss = 0
        correct_status = 0
        total_status = 0

        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]  ")
            for batch in pbar:
                pixel_values = batch['pixel_values'].to(device)
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                status_labels = batch['status_label'].to(device)

                # Use AMP for validation too (faster inference)
                if use_amp:
                    with torch.amp.autocast('cuda'):
                        caption_outputs, status_logits = model(
                            pixel_values=pixel_values,
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=input_ids
                        )
                        caption_loss = caption_outputs.loss
                        status_loss = classification_criterion(status_logits, status_labels)
                else:
                    caption_outputs, status_logits = model(
                        pixel_values=pixel_values,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=input_ids
                    )
                    caption_loss = caption_outputs.loss
                    status_loss = classification_criterion(status_logits, status_labels)

                loss = caption_loss + 0.5 * status_loss

                val_loss += loss.item()
                val_caption_loss += caption_loss.item()
                val_status_loss += status_loss.item()

                # Calculate status accuracy (ignore -1 labels)
                predictions = torch.argmax(status_logits, dim=1)
                mask = status_labels != -1
                if mask.sum() > 0:
                    correct_status += (predictions[mask] == status_labels[mask]).sum().item()
                    total_status += mask.sum().item()

        avg_val_loss = val_loss / len(val_loader)
        status_accuracy = correct_status / total_status if total_status > 0 else 0

        print(f"\nEpoch {epoch+1}/{EPOCHS}:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}")
        print(f"  Status Accuracy: {status_accuracy:.2%}")
        print()

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print(f"  + New best model! Saving...")
            os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)

            # Save model and processor
            model.base_model.save_pretrained(MODEL_OUTPUT_DIR)
            processor.save_pretrained(MODEL_OUTPUT_DIR)

            # Save status classifier head
            torch.save(
                model.status_classifier.state_dict(),
                os.path.join(MODEL_OUTPUT_DIR, 'status_classifier.pt')
            )

            # Save label mappings
            with open(os.path.join(MODEL_OUTPUT_DIR, 'config.json'), 'w') as f:
                json.dump({
                    'status_to_label': train_dataset.status_to_label,
                    'label_to_status': train_dataset.label_to_status
                }, f, indent=2)

    print("\n[5/6] Evaluating on test set (holdout)...")

    # Generate run ID for this training session
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Run ID: {run_id}")

    # Load best model for testing
    model.eval()
    test_loss = 0
    test_caption_loss = 0
    test_status_loss = 0
    correct_status = 0
    total_status = 0

    # Track predictions for outlier detection and metrics
    test_results = []
    all_predictions = []
    all_true_labels = []

    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Testing")
        batch_idx = 0
        for batch in pbar:
            pixel_values = batch['pixel_values'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            status_labels = batch['status_label'].to(device)

            # Use AMP for testing too
            if use_amp:
                with torch.amp.autocast('cuda'):
                    caption_outputs, status_logits = model(
                        pixel_values=pixel_values,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=input_ids
                    )
                    caption_loss = caption_outputs.loss
                    status_loss = classification_criterion(status_logits, status_labels)
            else:
                caption_outputs, status_logits = model(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=input_ids
                )
                caption_loss = caption_outputs.loss
                status_loss = classification_criterion(status_logits, status_labels)

            loss = caption_loss + 0.5 * status_loss

            test_loss += loss.item()
            test_caption_loss += caption_loss.item()
            test_status_loss += status_loss.item()

            # Calculate status accuracy and track per-sample results
            predictions = torch.argmax(status_logits, dim=1)
            probabilities = torch.softmax(status_logits, dim=1)

            mask = status_labels != -1
            if mask.sum() > 0:
                correct_status += (predictions[mask] == status_labels[mask]).sum().item()
                total_status += mask.sum().item()

            # Store results for each sample in batch
            for i in range(len(status_labels)):
                sample_idx = batch_idx * batch_size + i
                if sample_idx < len(test_data):
                    true_label = status_labels[i].item()
                    pred_label = predictions[i].item()
                    confidence = probabilities[i, pred_label].item()

                    # Collect for per-class metrics (skip unlabeled)
                    if true_label >= 0:
                        all_true_labels.append(true_label)
                        all_predictions.append(pred_label)

                    test_results.append({
                        'image_path': test_data[sample_idx]['image_path'],
                        'true_status': test_dataset.label_to_status.get(true_label, 'unknown'),
                        'predicted_status': test_dataset.label_to_status.get(pred_label, 'unknown'),
                        'confidence': confidence,
                        'correct': true_label == pred_label,
                        'probabilities': {
                            'green': probabilities[i, 0].item(),
                            'yellow': probabilities[i, 1].item(),
                            'red': probabilities[i, 2].item()
                        }
                    })

            batch_idx += 1

    avg_test_loss = test_loss / len(test_loader)
    test_accuracy = correct_status / total_status if total_status > 0 else 0

    # Calculate per-class metrics
    print("\nCalculating per-class metrics...")
    precision, recall, f1, support = precision_recall_fscore_support(
        all_true_labels,
        all_predictions,
        labels=[0, 1, 2],  # green, yellow, red
        zero_division=0
    )

    per_class_metrics = {}
    status_names = ['green', 'yellow', 'red']
    for i, status in enumerate(status_names):
        per_class_metrics[status] = {
            'precision': float(precision[i]),
            'recall': float(recall[i]),
            'f1': float(f1[i]),
            'support': int(support[i])
        }

    print(f"\nTest Set Results:")
    print(f"  Test Loss: {avg_test_loss:.4f}")
    print(f"  Test Status Accuracy: {test_accuracy:.2%}")
    print()
    print("Per-class metrics:")
    for status, metrics in per_class_metrics.items():
        print(f"  {status.upper():6s}: Precision={metrics['precision']:.3f}, "
              f"Recall={metrics['recall']:.3f}, F1={metrics['f1']:.3f}, "
              f"Support={metrics['support']}")
    print()

    # Calculate and display confusion matrix
    cm = confusion_matrix(all_true_labels, all_predictions, labels=[0, 1, 2])
    print("Confusion Matrix:")
    print("  (Rows = True labels, Columns = Predictions)")
    print()
    print(f"              {'Pred GREEN':<12} {'Pred YELLOW':<13} {'Pred RED':<10}")
    print(f"  True GREEN  {cm[0][0]:>10}   {cm[0][1]:>11}   {cm[0][2]:>8}")
    print(f"  True YELLOW {cm[1][0]:>10}   {cm[1][1]:>11}   {cm[1][2]:>8}")
    print(f"  True RED    {cm[2][0]:>10}   {cm[2][1]:>11}   {cm[2][2]:>8}")
    print()

    # Analyze key confusions
    yellow_to_green = cm[1][0]
    yellow_to_red = cm[1][2]
    green_to_yellow = cm[0][1]
    print("Key Confusion Analysis:")
    print(f"  Yellow → Green misclassifications: {yellow_to_green} ({yellow_to_green/cm[1].sum()*100:.1f}% of yellows)")
    print(f"  Yellow → Red misclassifications: {yellow_to_red} ({yellow_to_red/cm[1].sum()*100:.1f}% of yellows)")
    print(f"  Green → Yellow false positives: {green_to_yellow} ({green_to_yellow/cm[0].sum()*100:.1f}% of greens)")
    print()

    # Production validation: metrics on false_positive flagged images
    fp_results = [r for r in test_results if any(
        item.get('false_positive', False) and item['image_path'] == r['image_path']
        for item in test_data
    )]

    production_validation_metrics = None
    if fp_results:
        print("=" * 80)
        print("PRODUCTION VALIDATION (images marked as 'false_positive')")
        print("=" * 80)
        print(f"  Total: {len(fp_results)} images")

        fp_correct = sum(1 for r in fp_results if r['correct'])
        fp_accuracy = fp_correct / len(fp_results) if fp_results else 0
        print(f"  Accuracy: {fp_accuracy:.1%} ({fp_correct}/{len(fp_results)})")

        # Count by true label
        fp_by_true = {}
        fp_correct_by_true = {}
        for r in fp_results:
            true_status = r['true_status']
            fp_by_true[true_status] = fp_by_true.get(true_status, 0) + 1
            if r['correct']:
                fp_correct_by_true[true_status] = fp_correct_by_true.get(true_status, 0) + 1

        print("\n  By true label:")
        for status in ['green', 'yellow', 'red']:
            if status in fp_by_true:
                correct = fp_correct_by_true.get(status, 0)
                total = fp_by_true[status]
                acc = correct / total if total > 0 else 0
                print(f"    {status.upper():6s}: {acc:.1%} ({correct}/{total})")

        # Show specific failures on false_positive images
        fp_wrong = [r for r in fp_results if not r['correct']]
        if fp_wrong:
            print(f"\n  ❌ Still failing on {len(fp_wrong)} previously problematic images:")
            for r in fp_wrong[:5]:
                print(f"    {os.path.basename(r['image_path'])}: predicted {r['predicted_status']}, should be {r['true_status']}")
            if len(fp_wrong) > 5:
                print(f"    ... and {len(fp_wrong) - 5} more")
        else:
            print(f"\n  ✅ All previously problematic images now classified correctly!")

        print()

        production_validation_metrics = {
            'total': len(fp_results),
            'correct': fp_correct,
            'accuracy': fp_accuracy,
            'by_true_label': {
                status: {
                    'total': fp_by_true.get(status, 0),
                    'correct': fp_correct_by_true.get(status, 0)
                } for status in ['green', 'yellow', 'red']
            },
            'failures': [
                {'image': os.path.basename(r['image_path']), 'predicted': r['predicted_status'], 'true': r['true_status']}
                for r in fp_wrong
            ]
        }

    # Generate outlier report
    print("\n[6/6] Generating outlier analysis...")
    print()

    # Find incorrect predictions
    incorrect = [r for r in test_results if not r['correct']]

    # Find low confidence predictions (even if correct)
    low_confidence = sorted(test_results, key=lambda x: x['confidence'])[:10]

    # Find high confidence but wrong predictions (possible mislabels)
    high_conf_wrong = sorted(
        [r for r in test_results if not r['correct']],
        key=lambda x: x['confidence'],
        reverse=True
    )[:10]

    print("=" * 80)
    print("OUTLIER ANALYSIS REPORT")
    print("=" * 80)
    print()

    print(f"Total test samples: {len(test_results)}")
    print(f"Incorrect predictions: {len(incorrect)} ({len(incorrect)/len(test_results)*100:.1f}%)")
    print()

    if incorrect:
        print("-" * 80)
        print("MISCLASSIFIED IMAGES (All incorrect predictions):")
        print("-" * 80)
        for i, result in enumerate(incorrect, 1):
            print(f"\n{i}. {os.path.basename(result['image_path'])}")
            print(f"   True: {result['true_status'].upper()}")
            print(f"   Predicted: {result['predicted_status'].upper()} (confidence: {result['confidence']:.1%})")
            print(f"   Probabilities: G={result['probabilities']['green']:.2%} "
                  f"Y={result['probabilities']['yellow']:.2%} "
                  f"R={result['probabilities']['red']:.2%}")
        print()

    print("-" * 80)
    print("LOW CONFIDENCE PREDICTIONS (Top 10 least confident, may need review):")
    print("-" * 80)
    for i, result in enumerate(low_confidence, 1):
        status_icon = "+" if result['correct'] else "x"
        print(f"\n{i}. {status_icon} {os.path.basename(result['image_path'])}")
        print(f"   True: {result['true_status'].upper()}")
        print(f"   Predicted: {result['predicted_status'].upper()} (confidence: {result['confidence']:.1%})")
        print(f"   Probabilities: G={result['probabilities']['green']:.2%} "
              f"Y={result['probabilities']['yellow']:.2%} "
              f"R={result['probabilities']['red']:.2%}")
    print()

    if high_conf_wrong:
        print("-" * 80)
        print("HIGH CONFIDENCE ERRORS (Possible mislabeled images):")
        print("-" * 80)
        for i, result in enumerate(high_conf_wrong, 1):
            print(f"\n{i}. {os.path.basename(result['image_path'])}")
            print(f"   Labeled as: {result['true_status'].upper()}")
            print(f"   Model thinks: {result['predicted_status'].upper()} (confidence: {result['confidence']:.1%})")
            print(f"   Probabilities: G={result['probabilities']['green']:.2%} "
                  f"Y={result['probabilities']['yellow']:.2%} "
                  f"R={result['probabilities']['red']:.2%}")
            print(f"   -> Review this image - model is very confident it's mislabeled!")
        print()

    # Save detailed report to file
    report_path = os.path.join(MODEL_OUTPUT_DIR, 'outlier_report.json')
    with open(report_path, 'w') as f:
        json.dump({
            'summary': {
                'total_samples': len(test_results),
                'incorrect': len(incorrect),
                'accuracy': test_accuracy
            },
            'misclassified': incorrect,
            'low_confidence': low_confidence,
            'high_confidence_errors': high_conf_wrong
        }, f, indent=2)

    print("=" * 80)
    print(f"Detailed outlier report saved to: {report_path}")
    print("=" * 80)
    print()

    # Prepare run data for history and snapshot
    git_info = get_git_info()

    run_data = {
        'run_id': run_id,
        'timestamp': datetime.now().isoformat(),
        'git_info': git_info,
        'dataset': {
            'total_samples': len(all_data),
            'train_samples': len(train_data),
            'val_samples': len(val_data),
            'test_samples': len(test_data),
            'class_distribution': train_counts
        },
        'hyperparameters': {
            'epochs': EPOCHS,
            'learning_rate': LEARNING_RATE,
            'batch_size': batch_size,
            'train_split': TRAIN_SPLIT,
            'val_split': VAL_SPLIT,
            'test_split': TEST_SPLIT,
            'max_length': MAX_LENGTH,
            'class_weights': [float(w) for w in class_weights],
            'augment_minority_classes': True
        },
        'training_config': {
            'device': str(device),
            'use_amp': use_amp,
            'num_workers': config['num_workers'],
            'pin_memory': config['pin_memory']
        },
        'final_metrics': {
            'best_val_loss': float(best_val_loss),
            'test_loss': float(avg_test_loss),
            'test_accuracy': float(test_accuracy)
        },
        'per_class_metrics': per_class_metrics,
        'confusion_matrix': {
            'matrix': cm.tolist(),  # Convert numpy array to list for JSON
            'labels': ['green', 'yellow', 'red'],
            'analysis': {
                'yellow_to_green': int(yellow_to_green),
                'yellow_to_red': int(yellow_to_red),
                'green_to_yellow': int(green_to_yellow),
                'yellow_to_green_percent': float(yellow_to_green/cm[1].sum()*100) if cm[1].sum() > 0 else 0.0,
                'yellow_to_red_percent': float(yellow_to_red/cm[1].sum()*100) if cm[1].sum() > 0 else 0.0,
                'green_to_yellow_percent': float(green_to_yellow/cm[0].sum()*100) if cm[0].sum() > 0 else 0.0
            }
        },
        'outliers': {
            'total_test_samples': len(test_results),
            'misclassified': len(incorrect),
            'low_confidence_count': len(low_confidence),
            'high_confidence_errors_count': len(high_conf_wrong)
        },
        'production_validation': production_validation_metrics,  # Metrics on false_positive flagged images
        'model_path': MODEL_OUTPUT_DIR,
        'snapshot_path': None  # Will be updated after creating snapshot
    }

    # Save training history
    print("\nSaving training history...")
    save_training_history(run_data)

    # Create snapshot
    print("\nCreating reproducible snapshot...")
    snapshot_path = create_snapshot(run_id, model, processor, train_dataset, run_data)

    # Update run_data with snapshot path and re-save
    run_data['snapshot_path'] = snapshot_path
    save_training_history(run_data)

    print("\nTraining complete!")
    print(f"Model saved to: {MODEL_OUTPUT_DIR}")
    print(f"Snapshot saved to: {snapshot_path}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Final test accuracy: {test_accuracy:.2%}")
    print(f"\nRun ID: {run_id}")
    print("Use view_training_history.py to compare with previous runs")


if __name__ == "__main__":
    train_model()
