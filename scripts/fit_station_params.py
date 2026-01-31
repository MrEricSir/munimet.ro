#!/usr/bin/env python3
"""
Fit station detection parameters using human-corrected evaluations.

Loads segment-level evaluations from station_evaluations.json, pre-computes
intermediate CV data for each image, then sweeps threshold parameters via
grid search to find the combination that best matches human corrections.

Usage:
    python training/fit_station_params.py
"""

import json
import os
import sys
from pathlib import Path

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.detect_stations import (
    analyze_image_detailed,
    ICON_COUNT_THRESHOLD, TOTAL_ICON_THRESHOLD, RED_TRACK_THRESHOLD,
)

EVALUATIONS_FILE = "artifacts/reference_data/station_evaluations.json"

# Parameter grid
ICON_COUNT_VALUES = [1, 2, 3, 4, 5]
TOTAL_ICON_VALUES = [4, 5, 6, 7, 8, 9, 10, 12, 15]
RED_THRESHOLD_VALUES = [0.02, 0.03, 0.04, 0.05, 0.07, 0.10, 0.15]


def load_evaluations():
    """Load human-corrected evaluations."""
    if not os.path.exists(EVALUATIONS_FILE):
        print(f"Error: Evaluations file not found: {EVALUATIONS_FILE}")
        print("Run scripts/evaluate_segments.py first to create evaluations.")
        sys.exit(1)

    with open(EVALUATIONS_FILE, 'r') as f:
        data = json.load(f)

    evaluations = data.get('evaluations', [])
    if not evaluations:
        print("Error: No evaluations found in file.")
        sys.exit(1)

    return evaluations


def precompute_intermediate_data(evaluations):
    """Run analyze_image_detailed() on all evaluated images and cache results.

    Returns list of dicts, each with:
      - 'image_path': str
      - 'human_delays': set of (from_name, to_name, direction) tuples
      - 'segments': list of per-segment dicts with icon_count, red_ratio, etc.
      - 'total_train_icons': int
    """
    cached = []
    errors = 0

    for i, evaluation in enumerate(evaluations):
        image_path = evaluation['image_path']
        print(f"  [{i+1}/{len(evaluations)}] {Path(image_path).name}...", end="")

        if not os.path.exists(image_path):
            print(" MISSING")
            errors += 1
            continue

        try:
            analysis = analyze_image_detailed(image_path)
        except Exception as e:
            print(f" ERROR: {e}")
            errors += 1
            continue

        # Build human delay set
        human_delays = set()
        for seg in evaluation.get('segments', []):
            human_delays.add((
                seg['from'], seg['to'], seg['direction']
            ))

        cached.append({
            'image_path': image_path,
            'human_delays': human_delays,
            'segments': analysis['segments'],
            'total_train_icons': analysis['total_train_icons'],
        })
        print(" OK")

    if errors:
        print(f"\n  {errors} image(s) skipped due to errors")

    return cached


def apply_thresholds(cached_data, icon_thresh, total_thresh, red_thresh):
    """Re-derive predictions from cached data using given thresholds.

    Returns per-segment predictions as a list of (image_idx, from, to, direction, predicted).
    """
    all_predictions = []

    for entry in cached_data:
        segments = entry['segments']
        total_icons = entry['total_train_icons']

        segment_delays = []
        for seg in segments:
            predicted = False
            if seg['icon_count'] >= icon_thresh:
                predicted = True
            elif seg['red_ratio'] >= red_thresh:
                predicted = True
            segment_delays.append((
                seg['from_name'], seg['to_name'], seg['direction'], predicted
            ))

        # Check for spread delay
        cluster_delays = [s for s in segment_delays if s[3]]
        if total_icons >= total_thresh and not cluster_delays:
            # System-wide spread delay
            all_predictions.append({
                'entry': entry,
                'segment_delays': segment_delays,
                'spread_delay': True,
            })
        else:
            all_predictions.append({
                'entry': entry,
                'segment_delays': segment_delays,
                'spread_delay': False,
            })

    return all_predictions


def score_predictions(all_predictions):
    """Score predictions against human corrections.

    Returns dict with precision, recall, F1, FP rate, and composite score.
    """
    tp = 0  # true positives (predicted delay, human agrees)
    fp = 0  # false positives (predicted delay, human says no)
    fn = 0  # false negatives (human says delay, not predicted)
    tn = 0  # true negatives (no delay predicted, human agrees)

    for pred in all_predictions:
        entry = pred['entry']
        human_delays = entry['human_delays']

        # Check segment-level predictions
        for from_name, to_name, direction, predicted in pred['segment_delays']:
            key = (from_name, to_name, direction)
            human_says_delay = key in human_delays

            if predicted and human_says_delay:
                tp += 1
            elif predicted and not human_says_delay:
                fp += 1
            elif not predicted and human_says_delay:
                fn += 1
            else:
                tn += 1

        # Handle spread delay
        if pred['spread_delay']:
            spread_key = ('Multiple', 'stations', 'Both')
            if spread_key in human_delays:
                tp += 1
            else:
                fp += 1
        else:
            spread_key = ('Multiple', 'stations', 'Both')
            if spread_key in human_delays:
                fn += 1

    total = tp + fp + fn + tn
    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)
    accuracy = (tp + tn) / total if total > 0 else 0.0
    fp_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    # Composite score: F1 penalized by false positive rate
    # (FP is worse than FN for user experience)
    composite = f1 - 0.3 * fp_rate

    return {
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy,
        'fp_rate': fp_rate,
        'composite': composite,
    }


def main():
    print("=" * 70)
    print("STATION DETECTION PARAMETER FITTING")
    print("=" * 70)
    print()

    # Load evaluations
    evaluations = load_evaluations()
    print(f"Loaded {len(evaluations)} evaluations")
    print()

    # Count total human-marked delays
    total_human_delays = sum(
        len(e.get('segments', [])) for e in evaluations
    )
    print(f"Total human-marked delay segments: {total_human_delays}")
    print()

    # Pre-compute intermediate data
    print("Pre-computing intermediate CV data...")
    cached_data = precompute_intermediate_data(evaluations)
    print(f"\nCached data for {len(cached_data)} images")
    print()

    if not cached_data:
        print("Error: No valid images to process.")
        sys.exit(1)

    # Grid search
    total_combos = (
        len(ICON_COUNT_VALUES) * len(TOTAL_ICON_VALUES)
        * len(RED_THRESHOLD_VALUES)
    )
    print(f"Testing {total_combos} parameter combinations...")
    print(f"  ICON_COUNT_THRESHOLD: {ICON_COUNT_VALUES}")
    print(f"  TOTAL_ICON_THRESHOLD: {TOTAL_ICON_VALUES}")
    print(f"  RED_TRACK_THRESHOLD: {RED_THRESHOLD_VALUES}")
    print()

    results = []
    for ict in ICON_COUNT_VALUES:
        for tit in TOTAL_ICON_VALUES:
            for rtt in RED_THRESHOLD_VALUES:
                predictions = apply_thresholds(cached_data, ict, tit, rtt)
                scores = score_predictions(predictions)
                results.append({
                    'icon_count_threshold': ict,
                    'total_icon_threshold': tit,
                    'red_track_threshold': rtt,
                    **scores,
                })

    # Sort by composite score
    results.sort(key=lambda x: x['composite'], reverse=True)

    # Report top 10
    print("=" * 70)
    print("TOP 10 BY COMPOSITE SCORE (F1 - 0.3 * FP_rate)")
    print("=" * 70)
    print()
    print(f"{'ICT':<5} {'TIT':<5} {'RTT':<7} {'Prec':>7} {'Rec':>7} "
          f"{'F1':>7} {'FP%':>7} {'Score':>7}")
    print("-" * 55)

    for r in results[:10]:
        print(f"{r['icon_count_threshold']:<5} "
              f"{r['total_icon_threshold']:<5} "
              f"{r['red_track_threshold']:<7.2f} "
              f"{r['precision']:>6.1%} "
              f"{r['recall']:>6.1%} "
              f"{r['f1']:>6.3f} "
              f"{r['fp_rate']:>6.1%} "
              f"{r['composite']:>6.3f}")

    # Find best by different criteria
    best_composite = results[0]
    best_f1 = max(results, key=lambda x: x['f1'])
    best_precision = max(results, key=lambda x: x['precision'])
    best_recall = max(results, key=lambda x: x['recall'])

    print()
    print("=" * 70)
    print("RECOMMENDED CONFIGURATION")
    print("=" * 70)
    print()

    r = best_composite
    print(f"  ICON_COUNT_THRESHOLD = {r['icon_count_threshold']}")
    print(f"  TOTAL_ICON_THRESHOLD = {r['total_icon_threshold']}")
    print(f"  RED_TRACK_THRESHOLD = {r['red_track_threshold']}")
    print()
    print(f"  Segment Precision: {r['precision']:.1%}")
    print(f"  Segment Recall:    {r['recall']:.1%}")
    print(f"  Segment F1:        {r['f1']:.3f}")
    print(f"  False Positive Rate: {r['fp_rate']:.1%}")
    print(f"  Composite Score:   {r['composite']:.3f}")
    print()

    # Compare with current values
    print("=" * 70)
    print("COMPARISON WITH CURRENT VALUES")
    print("=" * 70)
    print()

    current = None
    for r in results:
        if (r['icon_count_threshold'] == ICON_COUNT_THRESHOLD and
            r['total_icon_threshold'] == TOTAL_ICON_THRESHOLD and
            r['red_track_threshold'] == RED_TRACK_THRESHOLD):
            current = r
            break

    if current:
        print(f"  Current:     ICT={ICON_COUNT_THRESHOLD}, "
              f"TIT={TOTAL_ICON_THRESHOLD}, "
              f"RTT={RED_TRACK_THRESHOLD} -> "
              f"F1={current['f1']:.3f}, "
              f"FP%={current['fp_rate']:.1%}, "
              f"Score={current['composite']:.3f}")
        rec = best_composite
        print(f"  Recommended: ICT={rec['icon_count_threshold']}, "
              f"TIT={rec['total_icon_threshold']}, "
              f"RTT={rec['red_track_threshold']} -> "
              f"F1={rec['f1']:.3f}, "
              f"FP%={rec['fp_rate']:.1%}, "
              f"Score={rec['composite']:.3f}")

        if best_composite['composite'] > current['composite']:
            delta = best_composite['composite'] - current['composite']
            print(f"\n  Improvement: +{delta:.3f} composite score")
        else:
            print(f"\n  Current values are already optimal!")
    else:
        print(f"  Current values (ICT={ICON_COUNT_THRESHOLD}, "
              f"TIT={TOTAL_ICON_THRESHOLD}, "
              f"RTT={RED_TRACK_THRESHOLD}) not in search grid")

    print()
    print("To apply recommended values, update scripts/detect_stations.py:")
    print(f"  ICON_COUNT_THRESHOLD = {best_composite['icon_count_threshold']}")
    print(f"  TOTAL_ICON_THRESHOLD = {best_composite['total_icon_threshold']}")
    print(f"  RED_TRACK_THRESHOLD = {best_composite['red_track_threshold']}")


if __name__ == "__main__":
    main()
