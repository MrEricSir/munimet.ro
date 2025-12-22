#!/usr/bin/env python3
"""
View and analyze training history.

This script displays all training runs with their metrics,
allowing you to compare performance over time.
"""

import json
import os
from datetime import datetime
from pathlib import Path

TRAINING_HISTORY_FILE = "../artifacts/models/training_history.json"


def load_history():
    """Load training history from JSON file."""
    if not os.path.exists(TRAINING_HISTORY_FILE):
        print(f"No training history found at {TRAINING_HISTORY_FILE}")
        print("Train a model first using train_model.py")
        return None

    with open(TRAINING_HISTORY_FILE, 'r') as f:
        return json.load(f)


def format_timestamp(iso_timestamp):
    """Format ISO timestamp to readable string."""
    try:
        dt = datetime.fromisoformat(iso_timestamp)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except:
        return iso_timestamp


def print_summary_table(runs):
    """Print a summary table of all training runs."""
    if not runs:
        print("No training runs found.")
        return

    print("=" * 120)
    print("TRAINING HISTORY SUMMARY")
    print("=" * 120)
    print()

    # Header
    print(f"{'#':<3} {'Run ID':<17} {'Date':<20} {'Test Acc':<10} {'Yellow F1':<11} "
          f"{'Outliers':<10} {'Samples':<9} {'Git':<15}")
    print("-" * 120)

    # Find best accuracy
    best_acc_idx = max(range(len(runs)), key=lambda i: runs[i]['final_metrics']['test_accuracy'])

    # Rows
    for i, run in enumerate(runs):
        run_num = i + 1
        run_id = run['run_id']
        timestamp = format_timestamp(run['timestamp'])
        test_acc = run['final_metrics']['test_accuracy']

        # Yellow F1 (important for minority class)
        yellow_f1 = run['per_class_metrics']['yellow']['f1']

        # Outlier count
        outlier_count = run['outliers']['misclassified']

        # Dataset size
        samples = run['dataset']['total_samples']

        # Git info
        git = run['git_info']['commit_hash'][:7]
        if run['git_info']['has_uncommitted_changes']:
            git += "*"

        # Highlight best run
        marker = "⭐" if i == best_acc_idx else "  "

        print(f"{marker}{run_num:<3} {run_id:<17} {timestamp:<20} "
              f"{test_acc:>8.2%}  {yellow_f1:>9.3f}  "
              f"{outlier_count:>8}  {samples:>8}  {git:<15}")

    print("-" * 120)
    print(f"\nTotal runs: {len(runs)}")
    print(f"Best accuracy: Run #{best_acc_idx + 1} ({runs[best_acc_idx]['run_id']}) "
          f"with {runs[best_acc_idx]['final_metrics']['test_accuracy']:.2%}")
    print()


def print_detailed_run(run, run_number):
    """Print detailed information about a specific run."""
    print("=" * 80)
    print(f"RUN #{run_number}: {run['run_id']}")
    print("=" * 80)
    print()

    print(f"Timestamp: {format_timestamp(run['timestamp'])}")
    print(f"Snapshot: {run.get('snapshot_path', 'N/A')}")
    print()

    # Git info
    git = run['git_info']
    print(f"Git Branch: {git['branch']}")
    print(f"Git Commit: {git['commit_hash']}")
    if git['has_uncommitted_changes']:
        print("⚠ WARNING: Training run had uncommitted changes!")
    print()

    # Dataset
    dataset = run['dataset']
    print("Dataset:")
    print(f"  Total samples: {dataset['total_samples']}")
    print(f"  Train: {dataset['train_samples']}, Val: {dataset['val_samples']}, Test: {dataset['test_samples']}")
    print(f"  Class distribution: {dataset['class_distribution']}")
    print()

    # Hyperparameters
    hp = run['hyperparameters']
    print("Hyperparameters:")
    print(f"  Epochs: {hp['epochs']}, Learning rate: {hp['learning_rate']}, Batch size: {hp['batch_size']}")
    print(f"  Data splits: Train={hp['train_split']}, Val={hp['val_split']}, Test={hp['test_split']}")
    print(f"  Class weights: {[f'{w:.2f}' for w in hp['class_weights']]}")
    print(f"  Minority class augmentation: {hp.get('augment_minority_classes', False)}")
    print()

    # Final metrics
    metrics = run['final_metrics']
    print("Final Metrics:")
    print(f"  Test Accuracy: {metrics['test_accuracy']:.2%}")
    print(f"  Test Loss: {metrics['test_loss']:.4f}")
    print(f"  Best Val Loss: {metrics['best_val_loss']:.4f}")
    print()

    # Per-class metrics
    print("Per-Class Metrics:")
    print(f"  {'Class':<8} {'Precision':<12} {'Recall':<10} {'F1':<8} {'Support':<10}")
    print("  " + "-" * 55)
    for status, class_metrics in run['per_class_metrics'].items():
        print(f"  {status.upper():<8} {class_metrics['precision']:>10.3f}  "
              f"{class_metrics['recall']:>8.3f}  {class_metrics['f1']:>6.3f}  "
              f"{class_metrics['support']:>8}")
    print()

    # Outliers
    outliers = run['outliers']
    print("Outlier Analysis:")
    print(f"  Total test samples: {outliers['total_test_samples']}")
    print(f"  Misclassified: {outliers['misclassified']} ({outliers['misclassified']/outliers['total_test_samples']*100:.1f}%)")
    print(f"  Low confidence: {outliers['low_confidence_count']}")
    print(f"  High confidence errors: {outliers['high_confidence_errors_count']}")
    print()


def compare_runs(runs):
    """Compare key metrics across runs."""
    if len(runs) < 2:
        print("Need at least 2 runs to compare.")
        return

    print("=" * 80)
    print("PERFORMANCE TRENDS")
    print("=" * 80)
    print()

    # Test accuracy trend
    accuracies = [r['final_metrics']['test_accuracy'] for r in runs]
    print("Test Accuracy Trend:")
    for i, acc in enumerate(accuracies):
        change = ""
        if i > 0:
            diff = acc - accuracies[i-1]
            if diff > 0:
                change = f" (↑ {diff:+.2%})"
            elif diff < 0:
                change = f" (↓ {diff:+.2%})"
            else:
                change = " (→ no change)"
        print(f"  Run {i+1}: {acc:.2%}{change}")
    print()

    # Yellow F1 trend (critical for minority class)
    yellow_f1s = [r['per_class_metrics']['yellow']['f1'] for r in runs]
    print("Yellow Detection F1 Trend (minority class):")
    for i, f1 in enumerate(yellow_f1s):
        change = ""
        if i > 0:
            diff = f1 - yellow_f1s[i-1]
            if diff > 0:
                change = f" (↑ {diff:+.3f})"
            elif diff < 0:
                change = f" (↓ {diff:+.3f})"
            else:
                change = " (→ no change)"
        print(f"  Run {i+1}: {f1:.3f}{change}")
    print()

    # Outlier trend
    outliers = [r['outliers']['misclassified'] for r in runs]
    print("Misclassified Count Trend:")
    for i, count in enumerate(outliers):
        change = ""
        if i > 0:
            diff = count - outliers[i-1]
            if diff > 0:
                change = f" (↑ +{diff})"
            elif diff < 0:
                change = f" (↓ {diff})"
            else:
                change = " (→ no change)"
        print(f"  Run {i+1}: {count}{change}")
    print()


def main():
    """Main function."""
    history = load_history()
    if not history:
        return

    runs = history.get('training_runs', [])
    if not runs:
        print("No training runs in history.")
        return

    # Print summary table
    print_summary_table(runs)

    # Print trends
    if len(runs) > 1:
        compare_runs(runs)

    # Offer detailed view
    print("=" * 80)
    print("DETAILED RUN INFORMATION")
    print("=" * 80)
    print()

    while True:
        choice = input(f"Enter run number (1-{len(runs)}) for details, or 'q' to quit: ").strip()

        if choice.lower() == 'q':
            break

        try:
            run_num = int(choice)
            if 1 <= run_num <= len(runs):
                print()
                print_detailed_run(runs[run_num - 1], run_num)
            else:
                print(f"Invalid run number. Please enter 1-{len(runs)}")
        except ValueError:
            print("Invalid input. Please enter a number or 'q'")


if __name__ == "__main__":
    main()
