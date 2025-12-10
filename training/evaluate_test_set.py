#!/usr/bin/env python3
"""
Evaluate the trained model on the holdout test set.

Loads the best trained model and evaluates it on the test set
that was never used during training.
"""

import json
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from train_model import MuniDataset, MuniClassifier, load_training_data, split_data
from transformers import BlipProcessor, BlipForConditionalGeneration
from tqdm import tqdm

# Configuration
LABELS_FILE = "../data/training_labels.json"
MODEL_DIR = "../models/trained_model"
BATCH_SIZE = 4
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15

def evaluate_test_set():
    """Evaluate the trained model on the holdout test set."""

    print("=" * 60)
    print("Muni Model Test Set Evaluation")
    print("=" * 60)
    print()

    # Check for GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data and split
    print("\n[1/4] Loading and splitting data...")
    all_data = load_training_data()
    train_data, val_data, test_data = split_data(all_data, TRAIN_SPLIT, VAL_SPLIT)

    print(f"Test samples: {len(test_data)}")

    # Show test set class distribution
    test_counts = {}
    for item in test_data:
        status = item.get('status', 'unknown')
        test_counts[status] = test_counts.get(status, 0) + 1
    print(f"Test set distribution: {test_counts}")

    # Load trained model
    print(f"\n[2/4] Loading trained model from {MODEL_DIR}...")
    processor = BlipProcessor.from_pretrained(MODEL_DIR, use_fast=True)
    base_model = BlipForConditionalGeneration.from_pretrained(MODEL_DIR)

    # Create model with classification head
    model = MuniClassifier(base_model)

    # Load status classifier weights
    classifier_path = os.path.join(MODEL_DIR, 'status_classifier.pt')
    model.status_classifier.load_state_dict(torch.load(classifier_path, map_location=device))

    model = model.to(device)
    model.eval()

    # Create test dataset
    print("\n[3/4] Preparing test dataset...")
    test_dataset = MuniDataset(test_data, processor)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Loss function
    classification_criterion = nn.CrossEntropyLoss(ignore_index=-1)

    # Evaluate
    print("\n[4/4] Evaluating on test set...")
    test_loss = 0
    test_caption_loss = 0
    test_status_loss = 0
    correct_status = 0
    total_status = 0

    # Track per-class accuracy
    class_correct = {0: 0, 1: 0, 2: 0}  # green, yellow, red
    class_total = {0: 0, 1: 0, 2: 0}

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
                correct = (predictions[mask] == status_labels[mask])
                correct_status += correct.sum().item()
                total_status += mask.sum().item()

                # Per-class accuracy
                for i, label in enumerate(status_labels):
                    if label != -1:
                        label_val = label.item()
                        class_total[label_val] += 1
                        if predictions[i] == label:
                            class_correct[label_val] += 1

    # Calculate metrics
    avg_test_loss = test_loss / len(test_loader)
    avg_caption_loss = test_caption_loss / len(test_loader)
    avg_status_loss = test_status_loss / len(test_loader)
    test_accuracy = correct_status / total_status if total_status > 0 else 0

    # Print results
    print("\n" + "=" * 60)
    print("TEST SET RESULTS (Holdout Set - Never Seen During Training)")
    print("=" * 60)
    print(f"\nOverall Metrics:")
    print(f"  Total Loss:        {avg_test_loss:.4f}")
    print(f"  Caption Loss:      {avg_caption_loss:.4f}")
    print(f"  Status Loss:       {avg_status_loss:.4f}")
    print(f"  Status Accuracy:   {test_accuracy:.2%} ({correct_status}/{total_status})")

    print(f"\nPer-Class Accuracy:")
    label_names = {0: 'Green', 1: 'Yellow', 2: 'Red'}
    for label_id, name in label_names.items():
        if class_total[label_id] > 0:
            class_acc = class_correct[label_id] / class_total[label_id]
            print(f"  {name:7s}: {class_acc:.2%} ({class_correct[label_id]}/{class_total[label_id]})")
        else:
            print(f"  {name:7s}: N/A (no samples)")

    print("\n" + "=" * 60)
    print()


if __name__ == "__main__":
    evaluate_test_set()
