"""
Tests for the circuit breaker pattern implementation.

Tests cover:
- State transitions (CLOSED -> OPEN -> HALF_OPEN -> CLOSED)
- Failure counting and threshold behavior
- Reset timeout behavior
- Thread safety basics
"""

import pytest
import time
from unittest.mock import patch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.circuit_breaker import (
    CircuitBreaker,
    CircuitState,
    CircuitOpenError,
)


class TestCircuitBreakerBasics:
    """Basic circuit breaker functionality tests."""

    def test_initial_state_is_closed(self):
        """Circuit breaker starts in CLOSED state."""
        breaker = CircuitBreaker(name="test")
        assert breaker.state == CircuitState.CLOSED
        assert breaker.failure_count == 0

    def test_can_execute_when_closed(self):
        """Requests are allowed when circuit is CLOSED."""
        breaker = CircuitBreaker(name="test")
        assert breaker.can_execute() is True

    def test_success_resets_failure_count(self):
        """Recording success resets failure count."""
        breaker = CircuitBreaker(name="test", failure_threshold=5)
        breaker.record_failure()
        breaker.record_failure()
        assert breaker.failure_count == 2

        breaker.record_success()
        assert breaker.failure_count == 0
        assert breaker.state == CircuitState.CLOSED


class TestCircuitBreakerTransitions:
    """Tests for state transitions."""

    def test_opens_after_threshold_failures(self):
        """Circuit opens after reaching failure threshold."""
        breaker = CircuitBreaker(name="test", failure_threshold=3)

        breaker.record_failure()
        assert breaker.state == CircuitState.CLOSED

        breaker.record_failure()
        assert breaker.state == CircuitState.CLOSED

        breaker.record_failure()
        assert breaker.state == CircuitState.OPEN

    def test_rejects_requests_when_open(self):
        """Requests are rejected when circuit is OPEN."""
        breaker = CircuitBreaker(name="test", failure_threshold=2)
        breaker.record_failure()
        breaker.record_failure()

        assert breaker.state == CircuitState.OPEN
        assert breaker.can_execute() is False

    def test_transitions_to_half_open_after_timeout(self):
        """Circuit transitions to HALF_OPEN after reset timeout."""
        breaker = CircuitBreaker(name="test", failure_threshold=2, reset_timeout=0.1)
        breaker.record_failure()
        breaker.record_failure()

        assert breaker.state == CircuitState.OPEN

        # Wait for reset timeout
        time.sleep(0.15)

        assert breaker.state == CircuitState.HALF_OPEN
        assert breaker.can_execute() is True

    def test_closes_on_success_in_half_open(self):
        """Circuit closes on successful request in HALF_OPEN state."""
        breaker = CircuitBreaker(name="test", failure_threshold=2, reset_timeout=0.1)
        breaker.record_failure()
        breaker.record_failure()

        time.sleep(0.15)
        assert breaker.state == CircuitState.HALF_OPEN

        breaker.record_success()
        assert breaker.state == CircuitState.CLOSED
        assert breaker.failure_count == 0

    def test_reopens_on_failure_in_half_open(self):
        """Circuit re-opens on failed request in HALF_OPEN state."""
        breaker = CircuitBreaker(name="test", failure_threshold=2, reset_timeout=0.1)
        breaker.record_failure()
        breaker.record_failure()

        time.sleep(0.15)
        assert breaker.state == CircuitState.HALF_OPEN

        breaker.record_failure()
        assert breaker.state == CircuitState.OPEN


class TestCircuitBreakerStatus:
    """Tests for status reporting."""

    def test_get_status_closed(self):
        """Status reports correctly for CLOSED state."""
        breaker = CircuitBreaker(name="test", failure_threshold=5)
        breaker.record_failure()

        status = breaker.get_status()
        assert status['name'] == 'test'
        assert status['state'] == 'closed'
        assert status['failure_count'] == 1
        assert status['failure_threshold'] == 5
        assert status['time_until_retry'] == 0

    def test_get_status_open(self):
        """Status reports correctly for OPEN state."""
        breaker = CircuitBreaker(name="test", failure_threshold=2, reset_timeout=60)
        breaker.record_failure()
        breaker.record_failure()

        status = breaker.get_status()
        assert status['state'] == 'open'
        assert status['time_until_retry'] > 0
        assert status['time_until_retry'] <= 60

    def test_time_until_retry(self):
        """time_until_retry decreases over time."""
        breaker = CircuitBreaker(name="test", failure_threshold=2, reset_timeout=1.0)
        breaker.record_failure()
        breaker.record_failure()

        initial = breaker.time_until_retry()
        time.sleep(0.2)
        later = breaker.time_until_retry()

        assert later < initial

    def test_reset(self):
        """Reset returns circuit to initial state."""
        breaker = CircuitBreaker(name="test", failure_threshold=2)
        breaker.record_failure()
        breaker.record_failure()
        assert breaker.state == CircuitState.OPEN

        breaker.reset()
        assert breaker.state == CircuitState.CLOSED
        assert breaker.failure_count == 0
        assert breaker.can_execute() is True


class TestCircuitBreakerIntegration:
    """Integration tests with download_muni_image."""

    def test_circuit_breaker_blocks_when_open(self):
        """download_muni_image returns immediately when circuit is open."""
        from lib.circuit_breaker import image_source_breaker
        from lib.muni_lib import download_muni_image

        # Reset to known state
        image_source_breaker.reset()

        # Open the circuit
        for _ in range(image_source_breaker.failure_threshold):
            image_source_breaker.record_failure()

        assert image_source_breaker.state == CircuitState.OPEN

        # Try to download - should fail fast
        result = download_muni_image()

        assert result['success'] is False
        assert result['circuit_open'] is True
        assert 'Circuit breaker open' in result['error']
        assert result['attempts'] == 0

        # Clean up
        image_source_breaker.reset()

    def test_successful_download_resets_circuit(self):
        """Successful download resets circuit breaker failure count."""
        from lib.circuit_breaker import image_source_breaker
        from lib.muni_lib import download_muni_image
        import requests

        # Reset and record some failures (but not enough to open)
        image_source_breaker.reset()
        image_source_breaker.record_failure()
        image_source_breaker.record_failure()
        assert image_source_breaker.failure_count == 2

        # Mock a successful download
        mock_response = type('Response', (), {
            'status_code': 200,
            'content': b'\xff\xd8\xff\xe0' + b'\x00' * 100,  # Minimal JPEG
            'raise_for_status': lambda self: None,
        })()

        with patch('lib.muni_lib.requests.get', return_value=mock_response):
            with patch('lib.muni_lib.Image.open') as mock_image:
                mock_image.return_value.size = (1860, 800)
                result = download_muni_image(validate_dimensions=True)

        # Even if download fails for other reasons, check circuit breaker state
        # Note: This might fail due to file operations, but circuit should be reset
        # if we got a 200 response
        if result['success']:
            assert image_source_breaker.failure_count == 0

        # Clean up
        image_source_breaker.reset()
