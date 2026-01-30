"""
Pytest configuration for train detection tests.
Adds visual pass/fail summary at the end of test runs.
"""

import pytest


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Print visual summary at the end of test run."""
    # Get stats from terminalreporter
    stats = terminalreporter.stats

    passed = stats.get('passed', [])
    failed = stats.get('failed', [])
    skipped = stats.get('skipped', [])

    if not passed and not failed and not skipped:
        return

    # Group by test class
    results_by_class = {}

    for report in passed:
        class_name = _get_class_name(report)
        if class_name not in results_by_class:
            results_by_class[class_name] = {"passed": 0, "failed": 0, "skipped": 0, "failures": []}
        results_by_class[class_name]["passed"] += 1

    for report in failed:
        class_name = _get_class_name(report)
        if class_name not in results_by_class:
            results_by_class[class_name] = {"passed": 0, "failed": 0, "skipped": 0, "failures": []}
        results_by_class[class_name]["failed"] += 1
        results_by_class[class_name]["failures"].append(report.head_line.split("::")[-1])

    for report in skipped:
        class_name = _get_class_name(report)
        if class_name not in results_by_class:
            results_by_class[class_name] = {"passed": 0, "failed": 0, "skipped": 0, "failures": []}
        results_by_class[class_name]["skipped"] += 1

    # Print summary
    terminalreporter.write_line("")
    terminalreporter.write_line("=" * 60)
    terminalreporter.write_line("  TRAIN DETECTION TEST SUMMARY")
    terminalreporter.write_line("=" * 60)

    total_passed = 0
    total_failed = 0
    total_skipped = 0

    for class_name, counts in sorted(results_by_class.items()):
        total_passed += counts["passed"]
        total_failed += counts["failed"]
        total_skipped += counts["skipped"]

        total_in_class = counts["passed"] + counts["failed"] + counts["skipped"]

        # Class-level indicator
        if counts["failed"] > 0:
            indicator = "\033[91m✗\033[0m"  # Red X
        elif counts["skipped"] == total_in_class:
            indicator = "\033[93m○\033[0m"  # Yellow circle
        else:
            indicator = "\033[92m✓\033[0m"  # Green checkmark

        terminalreporter.write_line(f"  {indicator} {class_name}: {counts['passed']}/{total_in_class} passed")

        # Show individual failures
        for failure in counts["failures"]:
            terminalreporter.write_line(f"      \033[91m✗\033[0m {failure}")

    terminalreporter.write_line("-" * 60)

    # Overall summary
    total = total_passed + total_failed + total_skipped
    if total_failed == 0:
        status = "\033[92m✓ ALL TESTS PASSED\033[0m"
    else:
        status = f"\033[91m✗ {total_failed} FAILED\033[0m"

    terminalreporter.write_line(f"  {status}  ({total_passed} passed, {total_failed} failed, {total_skipped} skipped)")
    terminalreporter.write_line("=" * 60)
    terminalreporter.write_line("")


def _get_class_name(report):
    """Extract class name from test report."""
    # nodeid format: tests/test_file.py::TestClass::test_method
    parts = report.nodeid.split("::")
    if len(parts) >= 2:
        # Check if second part is a class name (starts with Test)
        if parts[1].startswith("Test"):
            return parts[1]
    return "Other"
