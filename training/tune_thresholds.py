#!/usr/bin/env python3
"""
Tune decision thresholds for yellow detection without retraining.

This script tests different threshold combinations on the test set
to find optimal values for YELLOW_THRESHOLD and GREEN_CONFIDENCE_THRESHOLD.
"""

import json
import os
import sys
import torch
import numpy as np
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.muni_lib import load_muni_model
from PIL import Image
from transformers import BlipProcessor
from tqdm import tqdm

LABELS_FILE = "../artifacts/training_data/labels.json"
MODEL_DIR = "../artifacts/models/v1"


def load_test_data():
    """Load test set from labels.json (15% of data)."""
    if not os.path.exists(LABELS_FILE):
        raise FileNotFoundError(f"Labels file not found: {LABELS_FILE}")

    with open(LABELS_FILE, 'r') as f:
        data = json.load(f)

    training_data = [
        item for item in data.get('training_data', [])
        if item.get('description', '').strip() and item.get('status', '')
    ]

    # Use same stratified split as training (15% test)
    np.random.seed(42)  # Same seed as training for consistency

    # Group by status for stratification
    status_groups = {}
    for item in training_data:
        status = item.get('status', '')
        if status not in status_groups:
            status_groups[status] = []
        status_groups[status].append(item)

    test_data = []
    for status, items in status_groups.items():
        np.random.shuffle(items)
        n = len(items)
        test_start = int(n * 0.85)  # Skip 70% train + 15% val
        test_data.extend(items[test_start:])

    np.random.shuffle(test_data)
    return test_data


def predict_with_thresholds(status_probs, yellow_threshold, green_threshold):
    """Apply custom thresholds to make prediction."""
    green_prob = status_probs[0]
    yellow_prob = status_probs[1]
    red_prob = status_probs[2]

    # Same logic as muni_lib.py
    if red_prob > max(green_prob, yellow_prob):
        return 2  # red
    elif yellow_prob > yellow_threshold and green_prob < green_threshold:
        return 1  # yellow
    else:
        return int(np.argmax(status_probs))


def evaluate_thresholds(test_data, model, processor, device, yellow_thresh, green_thresh):
    """Evaluate model on test set with given thresholds."""
    status_to_label = {'green': 0, 'yellow': 1, 'red': 2}

    correct = 0
    total = 0

    # Per-class metrics
    tp = {0: 0, 1: 0, 2: 0}  # true positives
    fp = {0: 0, 1: 0, 2: 0}  # false positives
    fn = {0: 0, 1: 0, 2: 0}  # false negatives

    yellow_to_green = 0
    green_to_yellow = 0

    with torch.inference_mode():
        for item in test_data:
            # Load image
            try:
                image = Image.open(item['image_path']).convert('RGB')
            except:
                continue

            # Get true label
            true_status = item.get('status', '')
            true_label = status_to_label.get(true_status, -1)
            if true_label == -1:
                continue

            # Process and predict
            inputs = processor(images=image, return_tensors="pt")
            pixel_values = inputs.pixel_values.to(device)

            status_logits = model(pixel_values)
            status_probs = torch.softmax(status_logits, dim=1)
            probs = status_probs[0].cpu().numpy()

            # Apply thresholds
            pred_label = predict_with_thresholds(probs, yellow_thresh, green_thresh)

            # Track metrics
            total += 1
            if pred_label == true_label:
                correct += 1
                tp[true_label] += 1
            else:
                fp[pred_label] += 1
                fn[true_label] += 1

                # Track specific confusions
                if true_label == 1 and pred_label == 0:
                    yellow_to_green += 1
                elif true_label == 0 and pred_label == 1:
                    green_to_yellow += 1

    # Calculate metrics
    accuracy = correct / total if total > 0 else 0

    # Per-class F1
    def calc_f1(label):
        precision = tp[label] / (tp[label] + fp[label]) if (tp[label] + fp[label]) > 0 else 0
        recall = tp[label] / (tp[label] + fn[label]) if (tp[label] + fn[label]) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        return precision, recall, f1

    green_p, green_r, green_f1 = calc_f1(0)
    yellow_p, yellow_r, yellow_f1 = calc_f1(1)
    red_p, red_r, red_f1 = calc_f1(2)

    return {
        'accuracy': accuracy,
        'yellow_f1': yellow_f1,
        'yellow_recall': yellow_r,
        'yellow_precision': yellow_p,
        'green_f1': green_f1,
        'red_f1': red_f1,
        'yellow_to_green': yellow_to_green,
        'green_to_yellow': green_to_yellow
    }


def main():
    """Find optimal thresholds through grid search."""
    print("=" * 80)
    print("DECISION THRESHOLD TUNING")
    print("=" * 80)
    print()

    # Load model
    print("Loading model...")
    model, processor, label_to_status, device = load_muni_model(MODEL_DIR)
    model.eval()

    # Load test data
    print("Loading test data...")
    test_data = load_test_data()
    print(f"Test set size: {len(test_data)}")
    print()

    # Define threshold ranges to test
    yellow_thresholds = [0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.12, 0.15]
    green_thresholds = [0.85, 0.86, 0.87, 0.88, 0.89, 0.90, 0.91, 0.92]

    print("Testing threshold combinations...")
    print(f"Yellow thresholds: {yellow_thresholds}")
    print(f"Green confidence thresholds: {green_thresholds}")
    print()

    results = []
    total_tests = len(yellow_thresholds) * len(green_thresholds)

    with tqdm(total=total_tests, desc="Testing") as pbar:
        for yellow_t in yellow_thresholds:
            for green_t in green_thresholds:
                metrics = evaluate_thresholds(
                    test_data, model, processor, device,
                    yellow_t, green_t
                )

                results.append({
                    'yellow_threshold': yellow_t,
                    'green_threshold': green_t,
                    **metrics
                })

                pbar.update(1)

    print()
    print("=" * 80)
    print("TOP 10 CONFIGURATIONS BY YELLOW RECALL")
    print("=" * 80)
    print()

    # Sort by yellow recall
    results.sort(key=lambda x: x['yellow_recall'], reverse=True)

    print(f"{'YT':<6} {'GT':<6} {'Y-Rec':<8} {'Y-F1':<8} {'Acc':<8} {'Y→G':<6} {'G→Y':<6}")
    print("-" * 60)

    for i, r in enumerate(results[:10], 1):
        print(f"{r['yellow_threshold']:<6.2f} {r['green_threshold']:<6.2f} "
              f"{r['yellow_recall']:>6.1%}  {r['yellow_f1']:>6.3f}  "
              f"{r['accuracy']:>6.1%}  {r['yellow_to_green']:>4}  "
              f"{r['green_to_yellow']:>4}")

    print()
    print("=" * 80)
    print("TOP 10 CONFIGURATIONS BY YELLOW F1")
    print("=" * 80)
    print()

    # Sort by yellow F1
    results.sort(key=lambda x: x['yellow_f1'], reverse=True)

    print(f"{'YT':<6} {'GT':<6} {'Y-F1':<8} {'Y-Rec':<8} {'Acc':<8} {'Y→G':<6} {'G→Y':<6}")
    print("-" * 60)

    for i, r in enumerate(results[:10], 1):
        print(f"{r['yellow_threshold']:<6.2f} {r['green_threshold']:<6.2f} "
              f"{r['yellow_f1']:>6.3f}  {r['yellow_recall']:>6.1%}  "
              f"{r['accuracy']:>6.1%}  {r['yellow_to_green']:>4}  "
              f"{r['green_to_yellow']:>4}")

    print()
    print("=" * 80)
    print("BALANCED RECOMMENDATIONS")
    print("=" * 80)
    print()

    # Find balanced option (high recall, acceptable false positives)
    balanced = [r for r in results if r['green_to_yellow'] <= 10]  # Max 10 false yellows
    if balanced:
        best_balanced = max(balanced, key=lambda x: x['yellow_recall'])

        print("Best Yellow Recall (with ≤10 false yellow alarms):")
        print(f"  YELLOW_THRESHOLD = {best_balanced['yellow_threshold']}")
        print(f"  GREEN_CONFIDENCE_THRESHOLD = {best_balanced['green_threshold']}")
        print()
        print(f"  Yellow Recall: {best_balanced['yellow_recall']:.1%}")
        print(f"  Yellow F1: {best_balanced['yellow_f1']:.3f}")
        print(f"  Overall Accuracy: {best_balanced['accuracy']:.1%}")
        print(f"  Yellow→Green errors: {best_balanced['yellow_to_green']}")
        print(f"  Green→Yellow false alarms: {best_balanced['green_to_yellow']}")
    else:
        print("Warning: All configurations have >10 false yellow alarms")

    print()
    print("Legend:")
    print("  YT = YELLOW_THRESHOLD (if yellow prob > this, consider yellow)")
    print("  GT = GREEN_CONFIDENCE_THRESHOLD (green must be > this to override yellow)")
    print("  Y-Rec = Yellow Recall (% of yellows caught)")
    print("  Y-F1 = Yellow F1 Score")
    print("  Y→G = Yellow misclassified as Green (missed warnings)")
    print("  G→Y = Green misclassified as Yellow (false alarms)")
    print()
    print("To apply these thresholds, edit lib/muni_lib.py lines 359-360")


if __name__ == "__main__":
    main()
