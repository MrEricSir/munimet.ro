# Proposed Improvements

Remaining items from the improvement brainstorm. Items are organized by priority.

## Medium Priority

### 2. Detection Confidence Scoring
- Current detection returns binary confidence (1.0 for deterministic)
- Could add nuanced confidence based on:
  - OCR quality scores
  - Number of trains detected vs expected
  - Platform color detection clarity
- Would help identify edge cases for retraining/tuning

### 3. Rate Limiting for External APIs
- Add rate limiting for Bluesky/Mastodon posting
- Prevent accidental spam if status flaps rapidly
- Could use token bucket or sliding window algorithm

## Low Priority

### 4. Performance Monitoring
- Add timing metrics for detection pipeline stages
- Track OCR performance over time
- Monitor GCS latency in production
- Could integrate with Cloud Monitoring or simple logging

### 5. Documentation Updates
- Add architecture diagram
- Document the detection pipeline visually
- Add troubleshooting guide for common issues
- Document hysteresis behavior for operators

### 6. Code Deduplication
- `_read_items()` and `_write_items()` in RSS have similar GCS/local patterns as other modules
- Could extract a common `StorageBackend` abstraction
- Low value given current codebase size

---

## Completed

- [x] Configuration Centralization (`lib/config.py`)
- [x] API Integration Tests (`tests/test_api.py`)
- [x] Notification System Tests (`tests/test_notifiers.py`)
- [x] Analytics Module Tests (`tests/test_analytics.py`)
- [x] Error Recovery Improvements
  - Circuit breaker pattern for sfmunicentral.com (`lib/circuit_breaker.py`)
  - Graceful degradation when image source is unavailable (`api/api.py`)
  - Retry with exponential backoff for GCS operations (`lib/gcs_utils.py`)
