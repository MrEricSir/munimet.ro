"""
Circuit Breaker pattern implementation for external API calls.

The circuit breaker prevents repeated calls to a failing service by "opening"
after a threshold of consecutive failures, then periodically allowing probe
requests to check if the service has recovered.

States:
    CLOSED: Normal operation, requests pass through
    OPEN: Too many failures, requests are immediately rejected
    HALF_OPEN: After reset timeout, allow one probe request to test recovery

Usage:
    breaker = CircuitBreaker(failure_threshold=5, reset_timeout=60)

    if breaker.can_execute():
        try:
            result = external_api_call()
            breaker.record_success()
        except Exception as e:
            breaker.record_failure()
            raise
    else:
        # Circuit is open, fail fast
        raise CircuitOpenError("Service temporarily unavailable")
"""

import logging
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, rejecting requests
    HALF_OPEN = "half_open"  # Testing if service recovered


class CircuitOpenError(Exception):
    """Raised when circuit breaker is open and rejecting requests."""
    def __init__(self, message: str, time_until_retry: float = 0):
        super().__init__(message)
        self.time_until_retry = time_until_retry


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior."""
    failure_threshold: int = 5      # Consecutive failures before opening
    reset_timeout: float = 60.0     # Seconds before trying again (half-open)
    half_open_max_calls: int = 1    # Max concurrent calls in half-open state


class CircuitBreaker:
    """
    Thread-safe circuit breaker for protecting external API calls.

    The circuit breaker tracks consecutive failures and state transitions:

    1. CLOSED (normal): Requests pass through
       - On success: Reset failure count
       - On failure: Increment failure count
       - If failures >= threshold: Transition to OPEN

    2. OPEN (failing): Requests rejected immediately
       - After reset_timeout: Transition to HALF_OPEN

    3. HALF_OPEN (testing): Allow limited probe requests
       - On success: Transition to CLOSED
       - On failure: Transition back to OPEN
    """

    def __init__(
        self,
        name: str = "default",
        failure_threshold: int = 5,
        reset_timeout: float = 60.0,
    ):
        """
        Initialize circuit breaker.

        Args:
            name: Identifier for logging
            failure_threshold: Consecutive failures before opening circuit
            reset_timeout: Seconds to wait before allowing probe request
        """
        self.name = name
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time: Optional[float] = None
        self._lock = threading.RLock()  # Reentrant lock to allow nested calls

        self._logger = logging.getLogger(__name__)

    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        with self._lock:
            return self._get_current_state()

    @property
    def failure_count(self) -> int:
        """Get current failure count."""
        with self._lock:
            return self._failure_count

    def _get_current_state(self) -> CircuitState:
        """
        Get current state, checking if OPEN should transition to HALF_OPEN.
        Must be called with lock held.
        """
        if self._state == CircuitState.OPEN:
            if self._last_failure_time is not None:
                elapsed = time.time() - self._last_failure_time
                if elapsed >= self.reset_timeout:
                    self._state = CircuitState.HALF_OPEN
                    self._logger.info(
                        f"Circuit breaker '{self.name}' transitioning to HALF_OPEN "
                        f"after {elapsed:.1f}s"
                    )
        return self._state

    def can_execute(self) -> bool:
        """
        Check if a request can be executed.

        Returns:
            True if request should proceed, False if circuit is open
        """
        with self._lock:
            state = self._get_current_state()
            return state in (CircuitState.CLOSED, CircuitState.HALF_OPEN)

    def time_until_retry(self) -> float:
        """
        Get seconds until circuit may allow retry (transitions to HALF_OPEN).

        Returns:
            Seconds until retry possible, or 0 if circuit is not OPEN
        """
        with self._lock:
            if self._state != CircuitState.OPEN or self._last_failure_time is None:
                return 0
            elapsed = time.time() - self._last_failure_time
            remaining = self.reset_timeout - elapsed
            return max(0, remaining)

    def record_success(self) -> None:
        """
        Record a successful request.

        In CLOSED state: Resets failure count
        In HALF_OPEN state: Transitions to CLOSED
        """
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._logger.info(
                    f"Circuit breaker '{self.name}' closing after successful probe"
                )
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._last_failure_time = None

    def record_failure(self) -> None:
        """
        Record a failed request.

        In CLOSED state: Increments failure count, may transition to OPEN
        In HALF_OPEN state: Transitions back to OPEN
        """
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()

            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.OPEN
                self._logger.warning(
                    f"Circuit breaker '{self.name}' re-opening after failed probe"
                )
            elif self._state == CircuitState.CLOSED:
                if self._failure_count >= self.failure_threshold:
                    self._state = CircuitState.OPEN
                    self._logger.warning(
                        f"Circuit breaker '{self.name}' opening after "
                        f"{self._failure_count} consecutive failures"
                    )

    def reset(self) -> None:
        """Reset circuit breaker to initial closed state."""
        with self._lock:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._last_failure_time = None

    def get_status(self) -> dict:
        """
        Get circuit breaker status for monitoring.

        Returns:
            Dict with state, failure_count, and time_until_retry
        """
        with self._lock:
            state = self._get_current_state()
            return {
                'name': self.name,
                'state': state.value,
                'failure_count': self._failure_count,
                'failure_threshold': self.failure_threshold,
                'time_until_retry': self.time_until_retry() if state == CircuitState.OPEN else 0,
            }


# Global circuit breaker instance for sfmunicentral.com
# Configured with reasonable defaults for a status checking service:
# - 5 consecutive failures before opening (covers brief hiccups)
# - 60 second reset timeout (quick recovery detection)
image_source_breaker = CircuitBreaker(
    name="sfmunicentral",
    failure_threshold=5,
    reset_timeout=60.0,
)
