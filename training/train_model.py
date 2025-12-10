#!/usr/bin/env python3
"""
Train a vision-language model to classify Muni subway status and generate descriptions.

This script:
1. Loads labeled training data from training_labels.json
2. Fine-tunes a pre-trained vision-language model (BLIP-2)
3. Trains both status classification (green/yellow/red) and description generation
4. Saves the trained model for inference
"""

import json
import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    get_linear_schedule_with_warmup
)
from tqdm import tqdm
import numpy as np

# Configuration
LABELS_FILE = "../artifacts/training_data/labels.json"
MODEL_OUTPUT_DIR = "../artifacts/models/v1"
BATCH_SIZE = 4
EPOCHS = 10
LEARNING_RATE = 5e-5
TRAIN_SPLIT = 0.7   # 70% train
VAL_SPLIT = 0.15    # 15% validation
TEST_SPLIT = 0.15   # 15% test (holdout set, never used during training)
MAX_LENGTH = 128


class MuniDataset(Dataset):
    """Dataset for Muni subway status images and labels."""

    def __init__(self, data, processor):
        self.data = data
        self.processor = processor

        # Map status to numeric labels
        self.status_to_label = {
            'green': 0,
            'yellow': 1,
            'red': 2,
            '': -1  # No status
        }
        self.label_to_status = {v: k for k, v in self.status_to_label.items() if v >= 0}

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


def split_data(data, train_ratio=0.7, val_ratio=0.15):
    """Split data into train, validation, and test sets with stratification.

    Args:
        data: List of labeled samples
        train_ratio: Proportion for training (default 0.7)
        val_ratio: Proportion for validation (default 0.15)

    Returns:
        train_data, val_data, test_data (remaining samples go to test)
    """
    # Stratify by status to ensure balanced representation
    status_groups = {}
    for item in data:
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

    # Check for GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data
    print("\n[1/6] Loading training data...")
    all_data = load_training_data()

    if len(all_data) == 0:
        print("ERROR: No training data found. Please label some images first.")
        return

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
    print(f"  Train: {count_statuses(train_data)}")
    print(f"  Val:   {count_statuses(val_data)}")
    print(f"  Test:  {count_statuses(test_data)}")

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
    train_dataset = MuniDataset(train_data, processor)
    val_dataset = MuniDataset(val_data, processor)
    test_dataset = MuniDataset(test_data, processor)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Setup optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    # Loss function for classification
    classification_criterion = nn.CrossEntropyLoss(ignore_index=-1)

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

            # Forward pass
            caption_outputs, status_logits = model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids
            )

            # Calculate losses
            caption_loss = caption_outputs.loss
            status_loss = classification_criterion(status_logits, status_labels)

            # Combined loss (weighted)
            loss = caption_loss + 0.5 * status_loss

            # Backward pass
            optimizer.zero_grad()
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
            print(f"  ✓ New best model! Saving...")
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

    # Load best model for testing
    model.eval()
    test_loss = 0
    test_caption_loss = 0
    test_status_loss = 0
    correct_status = 0
    total_status = 0

    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Testing")
        for batch in pbar:
            pixel_values = batch['pixel_values'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            status_labels = batch['status_label'].to(device)

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

            # Calculate status accuracy
            predictions = torch.argmax(status_logits, dim=1)
            mask = status_labels != -1
            if mask.sum() > 0:
                correct_status += (predictions[mask] == status_labels[mask]).sum().item()
                total_status += mask.sum().item()

    avg_test_loss = test_loss / len(test_loader)
    test_accuracy = correct_status / total_status if total_status > 0 else 0

    print(f"\nTest Set Results:")
    print(f"  Test Loss: {avg_test_loss:.4f}")
    print(f"  Test Status Accuracy: {test_accuracy:.2%}")
    print()

    print("\n[6/6] Training complete!")
    print(f"Model saved to: {MODEL_OUTPUT_DIR}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Final test accuracy: {test_accuracy:.2%}")


if __name__ == "__main__":
    train_model()
