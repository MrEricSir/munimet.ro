#!/usr/bin/env python3
"""
DEPRECATED: This script is no longer needed.

The system now uses OpenCV-based detection instead of ML models.
No model management is required.

---

(Legacy) Manage ML models for MuniMetro.

Commands:
    list        List available model snapshots in GCS
    current     Show currently deployed model version
    switch      Switch to a different model version
    info        Show detailed info about a model version

Usage:
    python scripts/manage-models.py list
    python scripts/manage-models.py current
    python scripts/manage-models.py switch 20251223_224331
    python scripts/manage-models.py info 20251223_224331

Environment:
    GCP_PROJECT_ID  - GCP project (default: munimetro)
    GCP_REGION      - GCP region (default: us-west1)
"""

import argparse
import json
import os
import subprocess
import sys

# Configuration
GCP_PROJECT_ID = os.getenv('GCP_PROJECT_ID', 'munimetro')
GCP_REGION = os.getenv('GCP_REGION', 'us-west1')
GCS_BUCKET = 'gs://munimetro-annex'
MODELS_PATH = f'{GCS_BUCKET}/models/snapshots'

API_SERVICE = 'munimetro-api'
CHECKER_JOB = 'munimetro-checker'


def run_cmd(cmd, capture=True):
    """Run a shell command and return output."""
    result = subprocess.run(cmd, shell=True, capture_output=capture, text=True)
    if capture:
        return result.stdout.strip(), result.stderr.strip(), result.returncode
    return None, None, result.returncode


def list_models():
    """List available model snapshots."""
    print("Available model snapshots:")
    print("=" * 70)

    # Get list of snapshots
    stdout, stderr, rc = run_cmd(f'gsutil ls {MODELS_PATH}/')
    if rc != 0:
        print(f"Error listing models: {stderr}")
        return

    snapshots = [line.strip().rstrip('/').split('/')[-1]
                 for line in stdout.strip().split('\n') if line.strip()]

    # Get current deployed version for comparison
    current = get_current_model()

    for snapshot in sorted(snapshots, reverse=True):
        # Try to get metadata
        meta_cmd = f'gsutil cat {MODELS_PATH}/{snapshot}/run_metadata.json 2>/dev/null'
        stdout, _, rc = run_cmd(meta_cmd)

        marker = " <-- DEPLOYED" if snapshot == current else ""

        if rc == 0:
            try:
                meta = json.loads(stdout)
                accuracy = meta.get('final_metrics', {}).get('test_accuracy', 0) * 100
                red_recall = meta.get('per_class_metrics', {}).get('red', {}).get('recall', 0) * 100
                yellow_prec = meta.get('per_class_metrics', {}).get('yellow', {}).get('precision', 0) * 100

                # Production validation metrics (false positive flagged images)
                prod_val = meta.get('production_validation')
                prod_str = ""
                if prod_val and prod_val.get('total', 0) > 0:
                    prod_acc = prod_val.get('accuracy', 0) * 100
                    prod_str = f"  FP-val:{prod_acc:.0f}%"

                print(f"  {snapshot}  acc:{accuracy:.1f}%  red-R:{red_recall:.1f}%  yel-P:{yellow_prec:.1f}%{prod_str}{marker}")
            except json.JSONDecodeError:
                print(f"  {snapshot}  (no metrics){marker}")
        else:
            print(f"  {snapshot}  (no metrics){marker}")

    print()
    print("Use 'switch <version>' to deploy a different model.")


def get_current_model():
    """Get currently deployed model version."""
    # Check the checker job (it's the one that does predictions)
    # Note: Cloud Run Jobs have an extra 'template' level in the spec path
    cmd = f'gcloud run jobs describe {CHECKER_JOB} --region={GCP_REGION} --project={GCP_PROJECT_ID} --format="value(spec.template.spec.template.spec.containers[0].env)" 2>/dev/null'
    stdout, stderr, rc = run_cmd(cmd)

    if rc != 0 or not stdout:
        return None

    # Parse environment variables
    # Format is like: {'name': 'MODEL_VERSION', 'value': '20251223_224331'};{'name': 'OTHER', ...}
    for part in stdout.split(';'):
        if 'MODEL_VERSION' in part and "'value'" in part:
            # Extract value using string parsing (avoid eval for safety)
            # Look for 'value': 'XXXXX' pattern
            import re
            match = re.search(r"'value':\s*'([^']+)'", part)
            if match:
                return match.group(1)

    return "(baked into image)"


def show_current():
    """Show currently deployed model version."""
    current = get_current_model()
    print(f"Currently deployed model: {current}")

    if current and current != "(baked into image)":
        print()
        show_info(current)


def show_info(version):
    """Show detailed info about a model version."""
    meta_cmd = f'gsutil cat {MODELS_PATH}/{version}/run_metadata.json 2>/dev/null'
    stdout, stderr, rc = run_cmd(meta_cmd)

    if rc != 0:
        print(f"Error: Could not find model version '{version}'")
        print(f"Use 'list' to see available versions.")
        return

    try:
        meta = json.loads(stdout)
    except json.JSONDecodeError:
        print(f"Error: Could not parse metadata for '{version}'")
        return

    print(f"Model: {version}")
    print("=" * 50)
    print(f"Timestamp: {meta.get('timestamp', 'unknown')}")
    print(f"Git commit: {meta.get('git_info', {}).get('commit_hash', 'unknown')[:8]}")
    print()

    # Dataset info
    dataset = meta.get('dataset', {})
    print(f"Training data: {dataset.get('total_samples', '?')} samples")
    dist = dataset.get('class_distribution', {})
    print(f"  Green: {dist.get('green', '?')}, Yellow: {dist.get('yellow', '?')}, Red: {dist.get('red', '?')}")
    print()

    # Metrics
    final = meta.get('final_metrics', {})
    print(f"Test accuracy: {final.get('test_accuracy', 0) * 100:.1f}%")
    print()

    print("Per-class metrics:")
    for cls in ['green', 'yellow', 'red']:
        m = meta.get('per_class_metrics', {}).get(cls, {})
        p = m.get('precision', 0) * 100
        r = m.get('recall', 0) * 100
        f1 = m.get('f1', 0) * 100
        print(f"  {cls:6s}: P={p:.1f}%  R={r:.1f}%  F1={f1:.1f}%")

    # Production validation (false positive flagged images)
    prod_val = meta.get('production_validation')
    if prod_val and prod_val.get('total', 0) > 0:
        print()
        print(f"Production validation ({prod_val['total']} false-positive flagged images):")
        print(f"  Accuracy: {prod_val['accuracy']*100:.1f}% ({prod_val['correct']}/{prod_val['total']})")
        failures = prod_val.get('failures', [])
        if failures:
            print(f"  Still failing: {len(failures)} images")
        else:
            print(f"  All previously problematic images now correct!")


def switch_model(version):
    """Switch to a different model version."""
    # Verify the model exists
    check_cmd = f'gsutil ls {MODELS_PATH}/{version}/model/status_classifier.pt 2>/dev/null'
    _, _, rc = run_cmd(check_cmd)

    if rc != 0:
        print(f"Error: Model version '{version}' not found in GCS.")
        print(f"Use 'list' to see available versions.")
        return False

    current = get_current_model()
    if current == version:
        print(f"Model '{version}' is already deployed.")
        return True

    print(f"Switching model: {current} -> {version}")
    print()

    # Update both the API service and checker job
    services = [
        (f'gcloud run services update {API_SERVICE}', 'API service'),
        (f'gcloud run jobs update {CHECKER_JOB}', 'Checker job'),
    ]

    for base_cmd, name in services:
        print(f"Updating {name}...")
        cmd = f'{base_cmd} --region={GCP_REGION} --project={GCP_PROJECT_ID} --update-env-vars=MODEL_VERSION={version}'
        _, stderr, rc = run_cmd(cmd, capture=True)

        if rc != 0:
            print(f"  Error: {stderr}")
            return False
        print(f"  Done")

    print()
    print(f"Successfully switched to model '{version}'")
    print("Services will load the new model on next request/execution.")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Manage MuniMetro ML models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # list command
    subparsers.add_parser('list', help='List available model snapshots')

    # current command
    subparsers.add_parser('current', help='Show currently deployed model')

    # info command
    info_parser = subparsers.add_parser('info', help='Show detailed model info')
    info_parser.add_argument('version', help='Model version (e.g., 20251223_224331)')

    # switch command
    switch_parser = subparsers.add_parser('switch', help='Switch to a different model')
    switch_parser.add_argument('version', help='Model version to switch to')

    args = parser.parse_args()

    if args.command == 'list':
        list_models()
    elif args.command == 'current':
        show_current()
    elif args.command == 'info':
        show_info(args.version)
    elif args.command == 'switch':
        switch_model(args.version)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
