#!/usr/bin/env python3
"""
Download latest Muni snapshot and detect its status.

Usage:
    python check_status.py                    # Single check
    python check_status.py --continuous       # Keep checking every 30 seconds
    python check_status.py --write-cache      # Single check, write to cache
    python check_status.py --continuous --write-cache --interval 60  # Cache mode with custom interval
    python check_status.py --generate-reports # Generate analytics reports only

In continuous mode with --write-cache, analytics reports are auto-generated at midnight.
"""

import sys
import time
from datetime import datetime
from pathlib import Path

# Path resolution - get absolute paths relative to project root
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

# Add parent directory to path for lib imports
sys.path.insert(0, str(PROJECT_ROOT))
from lib.muni_lib import download_muni_image, detect_muni_status, read_cache, write_cache, write_cached_image, write_cached_badge, calculate_best_status
from lib.detection import apply_status_hysteresis
from lib.notifiers import notify_status_change
from lib.analytics import log_status_check
from lib.image_archive import archive_image, should_archive_baseline

# Configuration
SNAPSHOT_DIR = str(PROJECT_ROOT / "artifacts" / "runtime" / "downloads")
DEFAULT_INTERVAL = 30  # seconds (local development)

def _get_check_interval():
    """Get the appropriate check interval based on environment."""
    import os
    if os.getenv('CLOUD_RUN'):
        from lib.config import CLOUD_CHECK_INTERVAL
        return CLOUD_CHECK_INTERVAL  # 180 seconds for Cloud Run
    return DEFAULT_INTERVAL  # 30 seconds for local


def _record_check_failure(error_message, failure_type):
    """
    Record a check failure in the cache without clearing existing status data.

    This preserves the last known good status while tracking failure info
    for staleness detection and monitoring.

    Args:
        error_message: Description of the error
        failure_type: Type of failure ('download' or 'detection')
    """
    cache_data = read_cache()
    if cache_data is None:
        cache_data = {}

    # Increment consecutive failures
    consecutive_failures = cache_data.get('consecutive_failures', 0) + 1

    # Update failure tracking fields (preserve existing status data)
    cache_data['consecutive_failures'] = consecutive_failures
    cache_data['last_error'] = {
        'message': error_message,
        'type': failure_type,
        'timestamp': datetime.now().isoformat()
    }
    # Don't update cached_at - that tracks when we last had good data
    # Don't update last_successful_check - that's the point

    if write_cache(cache_data):
        print(f"  Recorded failure #{consecutive_failures}: {failure_type}")
    else:
        print(f"  Failed to record failure in cache")


def check_status(should_write_cache=False, interval_seconds=None):
    """Download image and detect status.

    Args:
        should_write_cache: If True, write result to cache file
        interval_seconds: Check interval in seconds for analytics logging

    Returns:
        bool: True if successful, False otherwise
    """
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]")
    print("-" * 60)

    # Download image
    print("Downloading latest snapshot...")
    result = download_muni_image(output_folder=SNAPSHOT_DIR, validate_dimensions=True)

    if not result['success']:
        print(f"Download failed: {result['error']}")
        if result.get('retried'):
            print(f"  (failed after {result.get('attempts', 1)} attempts)")

        # Update cache with failure info (increment consecutive_failures)
        if should_write_cache:
            _record_check_failure(result.get('error', 'Unknown error'), 'download')

        return False

    print(f"Downloaded: {result['filepath']}")
    print(f"  Dimensions: {result['width']} x {result['height']}")
    if result.get('retried'):
        print(f"  (succeeded after {result.get('attempts', 1)} attempts)")
    print()

    # Detect status using OpenCV
    print("Analyzing image...")
    try:
        detection = detect_muni_status(result['filepath'])
    except Exception as e:
        print(f"Detection failed: {e}")
        # Update cache with failure info
        if should_write_cache:
            _record_check_failure(str(e), 'detection')
        return False

    # Display results
    status_emoji = {
        'green': '[GREEN]',
        'yellow': '[YELLOW]',
        'red': '[RED]'
    }
    emoji = status_emoji.get(detection['status'], '[?]')

    print()
    print(f"Status: {emoji} {detection['status'].upper()}")
    print(f"Description: {detection['description']}")
    print()

    # Display detection details
    det = detection.get('detection', {})
    trains = det.get('trains', [])
    delays_platforms = det.get('delays_platforms', [])
    delays_segments = det.get('delays_segments', [])
    delays_bunching = det.get('delays_bunching', [])

    print(f"Trains detected: {len(trains)}")
    if delays_platforms:
        print(f"Platforms in hold: {len(delays_platforms)}")
        for d in delays_platforms:
            print(f"  - {d['name']} ({d['direction']})")
    if delays_segments:
        print(f"Track segments disabled: {len(delays_segments)}")
        for d in delays_segments:
            print(f"  - {d['from']} to {d['to']} ({d['direction']})")
    if delays_bunching:
        print(f"Train bunching: {len(delays_bunching)}")
        for d in delays_bunching:
            print(f"  - {d['train_count']} trains at {d['station']} ({d['direction']})")

    # Write to cache if requested
    if should_write_cache:
        # Create new status entry
        now = datetime.now()
        new_status = {
            'status': detection['status'],
            'description': detection['description'],
            'confidence': detection['status_confidence'],
            'probabilities': detection['probabilities'],
            'detection': detection.get('detection', {}),
            'image_path': result['filepath'],
            'image_dimensions': {
                'width': result['width'],
                'height': result['height']
            },
            'timestamp': now.isoformat()
        }

        # Read existing cache to get previous statuses and hysteresis state
        statuses = []
        previous_reported_status = None
        pending_status = None
        pending_streak = 0
        cache_data = read_cache()
        if cache_data:
            if 'statuses' in cache_data:
                statuses = cache_data['statuses'][:]
            if 'reported_status' in cache_data:
                previous_reported_status = cache_data['reported_status']
            elif 'best_status' in cache_data:
                # Backward compatibility: use best_status if reported_status not present
                previous_reported_status = cache_data['best_status']
            pending_status = cache_data.get('pending_status')
            pending_streak = cache_data.get('pending_streak', 0)
            cached_at = cache_data.get('cached_at', 'unknown')
            prev_statuses_str = ', '.join(s['status'] for s in statuses) if statuses else '(empty)'
            reported_str = previous_reported_status['status'] if previous_reported_status else 'none'
            print(f"\nCache read: {len(statuses)} previous statuses [{prev_statuses_str}], reported={reported_str}, cached_at={cached_at}")
        else:
            print(f"\nCache read: No existing cache found (starting fresh)")

        # Add new status at the front
        statuses.insert(0, new_status)

        # Calculate best status using shared function (first smoothing pass)
        best_status = calculate_best_status(statuses, window_size=3)

        # Apply hysteresis (second smoothing pass)
        # This prevents rapid status flips by requiring consistent agreement
        hysteresis_result = apply_status_hysteresis(
            best_status=best_status,
            reported_status=previous_reported_status,
            pending_status=pending_status,
            pending_streak=pending_streak,
            timestamp=new_status['timestamp']
        )

        reported_status = hysteresis_result['reported_status']
        pending_status = hysteresis_result['pending_status']
        pending_streak = hysteresis_result['pending_streak']

        # Log the smoothing calculation
        all_statuses_str = ', '.join(s['status'] for s in statuses[:3])
        best_str = best_status['status']
        reported_str = reported_status['status']
        if best_str != reported_str:
            print(f"Smoothing: [{all_statuses_str}] -> best={best_str}, reported={reported_str} (pending: {pending_status} x{pending_streak})")
        else:
            print(f"Smoothing: [{all_statuses_str}] -> {reported_str}")

        # Keep only last 3 statuses (~1.5 min window at 30s intervals)
        statuses = statuses[:3]

        # Preserve last_notified_status from previous cache
        previous_last_notified = cache_data.get('last_notified_status') if cache_data else None

        # Write cache with status history, hysteresis state, and tracking fields
        cache_data = {
            'statuses': statuses,
            'best_status': best_status,  # Keep for backward compatibility
            'reported_status': reported_status,  # Hysteresis-smoothed status
            'pending_status': pending_status,
            'pending_streak': pending_streak,
            'cached_at': now.isoformat(),
            # Track last successful check separately (for staleness detection)
            'last_successful_check': now.isoformat(),
            # Reset failure counter on success
            'consecutive_failures': 0,
            'last_error': None,
            # Track last successfully notified status for recovery
            'last_notified_status': previous_last_notified,
        }

        if write_cache(cache_data):
            print(f"\nCache updated")
            if len(statuses) > 1:
                history = ' -> '.join(s['status'] for s in statuses)
                print(f"  History: [{history}], Reported: {reported_status['status']}")
            # Also cache the image and badge for the dashboard
            if write_cached_image(result['filepath']):
                print(f"  Image cached")
            else:
                print(f"  Image cache failed")
            if write_cached_badge(reported_status['status']):
                print(f"  Badge cached")
            else:
                print(f"  Badge cache failed")

            # Notify all channels — done early so notifications aren't lost to timeouts.
            # Determines whether to notify due to a new transition or a missed previous notification.
            should_notify = False
            current_reported = reported_status['status']
            previous_reported = None

            if hysteresis_result['status_changed'] and previous_reported_status is not None:
                should_notify = True
                previous_reported = previous_reported_status['status']
                print(f"\nReported status changed: {previous_reported} -> {current_reported}")
            elif previous_last_notified is not None and previous_last_notified != current_reported:
                should_notify = True
                previous_reported = previous_last_notified
                print(f"\nRecovering missed notification: {previous_reported} -> {current_reported}")

            if should_notify:
                delay_summaries = reported_status.get('detection', {}).get('delay_summaries', [])
                notify_results = notify_status_change(
                    status=current_reported,
                    previous_status=previous_reported,
                    delay_summaries=delay_summaries,
                    timestamp=reported_status['timestamp']
                )
                any_failed = False
                for channel, notify_result in notify_results.items():
                    if notify_result['success']:
                        print(f"  {channel}: OK")
                    elif 'Not configured' in str(notify_result.get('error', '')):
                        pass  # Unconfigured channels are not failures
                    else:
                        print(f"  {channel}: Failed - {notify_result.get('error', 'Unknown error')}")
                        any_failed = True
                if not any_failed:
                    cache_data['last_notified_status'] = current_reported
                    write_cache(cache_data)

            # Archive image for debugging/auditing (cloud only, best-effort)
            archive_reasons = []
            if hysteresis_result.get('status_changed') and previous_reported_status is not None:
                archive_reasons.append('transition')
            if detection['status'] != reported_status['status']:
                archive_reasons.append('override')
            if should_archive_baseline(cache_data, reported_status['status']):
                archive_reasons.append('baseline')
                cache_data['last_baseline_archive'] = now.isoformat()
                write_cache(cache_data)
            for reason in archive_reasons:
                raw = detection['status'] if reason == 'override' else None
                if archive_image(result['filepath'], new_status['timestamp'], reason, raw_status=raw):
                    print(f"  Image archived ({reason})")
        else:
            print(f"\nCache write failed")

        # Log to analytics database
        try:
            from lib.analytics import check_database_health
            check_id = log_status_check(
                status=detection['status'],
                best_status=reported_status['status'],  # Use hysteresis-smoothed status
                detection_data=detection.get('detection', {}),
                timestamp=new_status['timestamp'],
                interval_seconds=interval_seconds
            )
            print(f"  Analytics logged (check_id={check_id}, raw={detection['status']}, reported={reported_status['status']})")

            # Verify database health after logging
            health = check_database_health()
            print(f"  DB health: {health['check_count']} checks, exists={health['exists']}, has_data={health['has_data']}")
            if health.get('error'):
                print(f"  DB health error: {health['error']}")
        except Exception as e:
            print(f"  Analytics log failed: {e}")
            import traceback
            traceback.print_exc()

    return True


def generate_analytics_reports():
    """Generate all analytics reports."""
    from lib.analytics import generate_all_reports

    print("\nGenerating analytics reports...")
    try:
        results = generate_all_reports()
        for days, result in results.items():
            if result['success']:
                print(f"  {days}-day report: {result['total_minutes']:.1f} min monitored, {result['delayed_minutes']:.1f} min delays")
            else:
                print(f"  {days}-day report: FAILED")
        return True
    except Exception as e:
        print(f"  Error generating reports: {e}")
        return False


def main():
    # Parse arguments
    continuous = '--continuous' in sys.argv or '-c' in sys.argv
    should_write_cache = '--write-cache' in sys.argv
    generate_reports_only = '--generate-reports' in sys.argv

    # Handle report generation mode
    if generate_reports_only:
        print("=" * 60)
        print("Analytics Report Generator")
        print("=" * 60)
        generate_analytics_reports()
        return

    # Parse interval (use environment-appropriate default)
    interval = _get_check_interval()
    if '--interval' in sys.argv:
        try:
            idx = sys.argv.index('--interval')
            interval = int(sys.argv[idx + 1])
        except (IndexError, ValueError):
            print(f"Warning: Invalid --interval value, using default ({interval} seconds)")

    print("=" * 60)
    print("Muni Status Checker (OpenCV Detection)")
    if continuous:
        print("Mode: Continuous (Ctrl+C to stop)")
        print(f"Interval: {interval} seconds")
    else:
        print("Mode: Single check")
    if should_write_cache:
        from lib.muni_lib import get_cache_path
        print(f"Cache: Enabled ({get_cache_path()})")
    print("=" * 60)

    if not continuous:
        # Single check
        check_status(should_write_cache=should_write_cache, interval_seconds=interval)
    else:
        # Continuous checking
        count = 0
        successful = 0
        failed = 0
        last_report_hour = -1  # Track when we last generated reports

        try:
            while True:
                count += 1
                print(f"\n{'='*60}")
                print(f"Check #{count}")
                print(f"{'='*60}")

                if check_status(should_write_cache=should_write_cache, interval_seconds=interval):
                    successful += 1
                else:
                    failed += 1

                print(f"\nStats: {successful} successful, {failed} failed")

                # Generate analytics reports at midnight (hour 0)
                current_hour = datetime.now().hour
                if current_hour == 0 and last_report_hour != 0:
                    generate_analytics_reports()
                last_report_hour = current_hour

                if count > 1:
                    print(f"\nWaiting {interval} seconds until next check...")

                time.sleep(interval)

        except KeyboardInterrupt:
            print("\n\n" + "=" * 60)
            print("Stopped by user")
            print(f"Total checks: {count}")
            print(f"Successful: {successful}")
            print(f"Failed: {failed}")
            print("=" * 60)


if __name__ == "__main__":
    main()
