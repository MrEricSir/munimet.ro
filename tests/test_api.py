"""
Tests for the API endpoints.

Tests HTTP responses, JSON structure, caching behavior,
and error handling for all API endpoints.
"""

import json
import os
import pytest
import sys
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import Falcon testing utilities
try:
    from falcon import testing
    FALCON_AVAILABLE = True
except ImportError:
    FALCON_AVAILABLE = False
    testing = None


@pytest.fixture
def client():
    """Create a test client for the API."""
    if not FALCON_AVAILABLE:
        pytest.skip("Falcon not available")

    # Import inside fixture to avoid import errors when falcon isn't installed
    from api.api import falcon_app
    return testing.TestClient(falcon_app)


class TestStatusEndpoint:
    """Tests for the /status endpoint."""

    def test_status_returns_json(self, client):
        """Test that /status returns valid JSON."""
        # Mock the cache to return valid data
        mock_cache = {
            'cached_at': datetime.now().isoformat(),
            'last_successful_check': datetime.now().isoformat(),
            'reported_status': {
                'status': 'green',
                'description': 'Normal operation',
                'confidence': 1.0,
                'probabilities': {'green': 1.0, 'yellow': 0.0, 'red': 0.0},
                'timestamp': datetime.now().isoformat(),
            },
            'statuses': [{
                'status': 'green',
                'description': 'Normal operation',
                'confidence': 1.0,
                'probabilities': {'green': 1.0, 'yellow': 0.0, 'red': 0.0},
                'timestamp': datetime.now().isoformat(),
            }],
        }

        with patch('api.api.read_cache', return_value=mock_cache):
            response = client.simulate_get('/status')

        assert response.status_code == 200
        assert response.headers.get('content-type') == 'application/json'

        data = response.json
        assert 'status' in data
        assert 'confidence' in data
        assert 'timestamp' in data

    def test_status_green_response(self, client):
        """Test /status response for green status."""
        mock_cache = {
            'cached_at': datetime.now().isoformat(),
            'last_successful_check': datetime.now().isoformat(),
            'reported_status': {
                'status': 'green',
                'description': 'Muni Metro is running normally',
                'confidence': 1.0,
                'probabilities': {'green': 1.0, 'yellow': 0.0, 'red': 0.0},
                'timestamp': datetime.now().isoformat(),
            },
        }

        with patch('api.api.read_cache', return_value=mock_cache):
            response = client.simulate_get('/status')

        data = response.json
        assert data['status'] == 'green'
        assert data['confidence'] == 1.0
        assert data['cached'] is True

    def test_status_yellow_response(self, client):
        """Test /status response for yellow status with delay details."""
        mock_cache = {
            'cached_at': datetime.now().isoformat(),
            'last_successful_check': datetime.now().isoformat(),
            'reported_status': {
                'status': 'yellow',
                'description': 'Delays detected',
                'confidence': 1.0,
                'probabilities': {'green': 0.0, 'yellow': 1.0, 'red': 0.0},
                'timestamp': datetime.now().isoformat(),
                'detection': {
                    'delays_platforms': [
                        {'station': 'PO', 'name': 'Powell', 'direction': 'Westbound'}
                    ]
                }
            },
        }

        with patch('api.api.read_cache', return_value=mock_cache):
            response = client.simulate_get('/status')

        data = response.json
        assert data['status'] == 'yellow'
        assert 'detection' in data

    def test_status_staleness_levels(self, client):
        """Test that staleness levels are reported correctly."""
        # Fresh data (< 5 min)
        mock_cache = {
            'cached_at': datetime.now().isoformat(),
            'last_successful_check': datetime.now().isoformat(),
            'reported_status': {
                'status': 'green',
                'description': 'Normal',
                'confidence': 1.0,
                'probabilities': {'green': 1.0, 'yellow': 0.0, 'red': 0.0},
                'timestamp': datetime.now().isoformat(),
            },
        }

        with patch('api.api.read_cache', return_value=mock_cache):
            response = client.simulate_get('/status')

        data = response.json
        assert data['staleness'] == 'fresh'
        assert data['staleness_message'] is None

    def test_status_aging_data(self, client):
        """Test staleness level for aging data (5-15 min)."""
        old_time = (datetime.now() - timedelta(minutes=10)).isoformat()
        mock_cache = {
            'cached_at': datetime.now().isoformat(),
            'last_successful_check': old_time,
            'reported_status': {
                'status': 'green',
                'description': 'Normal',
                'confidence': 1.0,
                'probabilities': {'green': 1.0, 'yellow': 0.0, 'red': 0.0},
                'timestamp': old_time,
            },
        }

        with patch('api.api.read_cache', return_value=mock_cache):
            response = client.simulate_get('/status')

        data = response.json
        assert data['staleness'] == 'aging'
        assert 'minutes old' in data['staleness_message']

    def test_status_cache_miss_fallback_disabled(self, client):
        """Test /status when cache is empty and fallback is disabled."""
        with patch('api.api.read_cache', return_value=None):
            with patch('api.api.ENABLE_FALLBACK', False):
                response = client.simulate_get('/status')

        assert response.status_code == 503
        data = response.json
        assert 'error' in data
        assert 'Cache unavailable' in data['error']

    def test_status_includes_consecutive_failures(self, client):
        """Test that consecutive failures are reported."""
        mock_cache = {
            'cached_at': datetime.now().isoformat(),
            'last_successful_check': datetime.now().isoformat(),
            'consecutive_failures': 3,
            'last_error': 'Connection timeout',
            'reported_status': {
                'status': 'green',
                'description': 'Normal',
                'confidence': 1.0,
                'probabilities': {'green': 1.0, 'yellow': 0.0, 'red': 0.0},
                'timestamp': datetime.now().isoformat(),
            },
        }

        with patch('api.api.read_cache', return_value=mock_cache):
            response = client.simulate_get('/status')

        data = response.json
        assert data['consecutive_failures'] == 3
        assert data['last_error'] == 'Connection timeout'

    def test_status_graceful_degradation_returns_stale_cache(self, client):
        """Test graceful degradation: returns stale cache when download fails."""
        from lib.config import CACHE_MAX_AGE, STALENESS_EXPIRED

        # Create stale cache (older than CACHE_MAX_AGE for fallback trigger,
        # and older than STALENESS_EXPIRED for 'expired' staleness level)
        stale_time = (datetime.now() - timedelta(seconds=STALENESS_EXPIRED + 60)).isoformat()
        mock_cache = {
            'cached_at': stale_time,
            'last_successful_check': stale_time,
            'reported_status': {
                'status': 'yellow',
                'description': 'Platform hold at Powell',
                'confidence': 1.0,
                'probabilities': {'green': 0.0, 'yellow': 1.0, 'red': 0.0},
                'timestamp': stale_time,
            },
        }

        # Mock download failure
        mock_download_result = {
            'success': False,
            'error': 'Connection timeout',
            'filepath': None,
        }

        with patch('api.api.read_cache', return_value=mock_cache):
            with patch('api.api.ENABLE_FALLBACK', True):
                with patch('api.api.download_muni_image', return_value=mock_download_result):
                    response = client.simulate_get('/status')

        # Should return 200 with stale data, not an error
        assert response.status_code == 200
        data = response.json
        assert data['status'] == 'yellow'
        assert data['source_unavailable'] is True
        assert 'Connection timeout' in data['source_error']
        assert data['staleness'] == 'expired'

    def test_status_graceful_degradation_no_cache_returns_503(self, client):
        """Test that 503 is returned when download fails and no cache exists."""
        # Mock download failure with no cache
        mock_download_result = {
            'success': False,
            'error': 'Connection refused',
            'filepath': None,
        }

        with patch('api.api.read_cache', return_value=None):
            with patch('api.api.ENABLE_FALLBACK', True):
                with patch('api.api.download_muni_image', return_value=mock_download_result):
                    response = client.simulate_get('/status')

        # Should return 503 since no cached data to fall back to
        assert response.status_code == 503
        data = response.json
        assert 'error' in data
        assert 'Service temporarily unavailable' in data['error']
        assert 'Connection refused' in data['details']

    def test_status_graceful_degradation_detection_failure(self, client):
        """Test graceful degradation when detection fails after successful download."""
        from lib.config import CACHE_MAX_AGE, STALENESS_EXPIRED

        # Create stale cache (older than CACHE_MAX_AGE)
        stale_time = (datetime.now() - timedelta(seconds=STALENESS_EXPIRED + 60)).isoformat()
        mock_cache = {
            'cached_at': stale_time,
            'last_successful_check': stale_time,
            'reported_status': {
                'status': 'green',
                'description': 'Normal operation',
                'confidence': 1.0,
                'probabilities': {'green': 1.0, 'yellow': 0.0, 'red': 0.0},
                'timestamp': stale_time,
            },
        }

        # Mock successful download but failed detection
        mock_download_result = {
            'success': True,
            'filepath': '/tmp/test.jpg',
            'width': 1860,
            'height': 800,
        }

        with patch('api.api.read_cache', return_value=mock_cache):
            with patch('api.api.ENABLE_FALLBACK', True):
                with patch('api.api.download_muni_image', return_value=mock_download_result):
                    with patch('api.api.detect_muni_status', side_effect=Exception('OCR failed')):
                        response = client.simulate_get('/status')

        # Should return 200 with stale data, not an error
        assert response.status_code == 200
        data = response.json
        assert data['status'] == 'green'
        assert data['source_unavailable'] is True
        assert 'OCR failed' in data['source_error']


class TestHealthEndpoint:
    """Tests for the /health endpoint."""

    def test_health_returns_json(self, client):
        """Test that /health returns valid JSON."""
        mock_cache = {
            'cached_at': datetime.now().isoformat(),
            'reported_status': {'status': 'green'},
        }

        with patch('lib.muni_lib.read_cache', return_value=mock_cache):
            with patch('lib.analytics.init_db'):
                with patch('lib.analytics.get_db_connection') as mock_conn:
                    mock_cursor = MagicMock()
                    mock_cursor.fetchone.return_value = {'count': 100}
                    mock_conn.return_value.cursor.return_value = mock_cursor
                    response = client.simulate_get('/health')

        assert response.status_code == 200
        data = response.json
        assert 'status' in data
        assert 'components' in data
        assert 'timestamp' in data

    def test_health_healthy_status(self, client):
        """Test health check response structure with cache available."""
        mock_cache = {
            'cached_at': datetime.now().isoformat(),
            'reported_status': {'status': 'green'},
        }

        with patch('lib.muni_lib.read_cache', return_value=mock_cache):
            with patch('lib.analytics.init_db'):
                with patch('lib.analytics.get_db_connection') as mock_conn:
                    mock_cursor = MagicMock()
                    mock_cursor.fetchone.return_value = {'count': 100}
                    mock_conn.return_value.cursor.return_value = mock_cursor
                    response = client.simulate_get('/health')

        data = response.json
        # Status should be healthy or degraded (depends on analytics component)
        assert data['status'] in ['healthy', 'degraded']
        # Cache with fresh data should be healthy
        assert 'cache' in data['components']

    def test_health_degraded_stale_cache(self, client):
        """Test health check with stale cache."""
        old_time = (datetime.now() - timedelta(minutes=10)).isoformat()
        mock_cache = {
            'cached_at': old_time,
            'reported_status': {'status': 'green'},
        }

        with patch('lib.muni_lib.read_cache', return_value=mock_cache):
            with patch('lib.analytics.init_db'):
                with patch('lib.analytics.get_db_connection') as mock_conn:
                    mock_cursor = MagicMock()
                    mock_cursor.fetchone.return_value = {'count': 100}
                    mock_conn.return_value.cursor.return_value = mock_cursor
                    response = client.simulate_get('/health')

        data = response.json
        assert data['status'] == 'degraded'
        assert data['components']['cache']['status'] == 'degraded'
        assert data['components']['cache']['is_stale'] is True

    def test_health_no_cache(self, client):
        """Test health check when cache is unavailable."""
        with patch('lib.muni_lib.read_cache', return_value=None):
            with patch('lib.analytics.init_db'):
                with patch('lib.analytics.get_db_connection') as mock_conn:
                    mock_cursor = MagicMock()
                    mock_cursor.fetchone.return_value = {'count': 0}
                    mock_conn.return_value.cursor.return_value = mock_cursor
                    response = client.simulate_get('/health')

        data = response.json
        # When cache is unavailable, overall status should be degraded
        assert data['status'] == 'degraded'
        # Cache component should indicate a problem (unhealthy or degraded)
        assert data['components']['cache']['status'] in ['unhealthy', 'degraded']


class TestLatestImageEndpoint:
    """Tests for the /latest-image endpoint."""

    def test_latest_image_from_cache(self, client, tmp_path):
        """Test serving image from cache path."""
        # Create a test image file
        test_image = tmp_path / "test.jpg"
        test_image.write_bytes(b'\xff\xd8\xff\xe0' + b'\x00' * 100)  # Fake JPEG header

        mock_cache = {
            'statuses': [{'image_path': str(test_image)}]
        }

        with patch('api.api.read_cache', return_value=mock_cache):
            response = client.simulate_get('/latest-image')

        assert response.status_code == 200
        assert response.headers.get('content-type') == 'image/jpeg'

    def test_latest_image_not_found(self, client):
        """Test 404 when no image is available."""
        with patch('api.api.read_cache', return_value=None):
            with patch('api.api.read_cached_image', return_value=None):
                with patch('api.api.SNAPSHOT_DIR', '/nonexistent'):
                    response = client.simulate_get('/latest-image')

        assert response.status_code == 404


class TestRSSFeedEndpoint:
    """Tests for the /feed.xml endpoint."""

    def test_rss_feed_returns_xml(self, client):
        """Test that /feed.xml returns valid XML."""
        mock_feed = '''<?xml version="1.0" encoding="UTF-8"?>
        <rss version="2.0">
            <channel>
                <title>SF Muni Metro Status</title>
                <link>https://munimet.ro</link>
                <description>Status updates</description>
            </channel>
        </rss>'''

        with patch('lib.notifiers.rss.read_rss_feed', return_value=mock_feed):
            response = client.simulate_get('/feed.xml')

        assert response.status_code == 200
        assert 'application/rss+xml' in response.headers.get('content-type')
        assert '<?xml' in response.text

    def test_rss_feed_empty(self, client):
        """Test that empty feed is returned when no history."""
        with patch('lib.notifiers.rss.read_rss_feed', return_value=None):
            with patch('lib.notifiers.rss.generate_empty_feed', return_value='<?xml version="1.0"?><rss></rss>'):
                response = client.simulate_get('/feed.xml')

        assert response.status_code == 200
        assert 'rss' in response.text

    def test_rss_feed_has_cache_header(self, client):
        """Test that RSS feed has appropriate cache headers."""
        mock_feed = '<?xml version="1.0"?><rss version="2.0"><channel></channel></rss>'

        with patch('lib.notifiers.rss.read_rss_feed', return_value=mock_feed):
            response = client.simulate_get('/feed.xml')

        cache_control = response.headers.get('Cache-Control')
        assert cache_control is not None
        assert 'max-age' in cache_control


class TestAnalyticsEndpoint:
    """Tests for the /analytics-data endpoint."""

    def test_analytics_returns_json(self, client):
        """Test that /analytics-data returns valid JSON."""
        mock_report = {
            'period_days': 7,
            'frequency': {
                'total_minutes': 5000.0,
                'delayed_minutes': 250.0,
                'delay_rate': 0.05,
                'by_status': {'green': 4500.0, 'yellow': 250.0, 'red': 250.0}
            },
            'by_station': [],
            'by_time': {'by_hour': {}, 'by_day': {}},
            'generated_at': datetime.now().isoformat(),
        }

        with patch('lib.analytics.get_analytics_report', return_value=mock_report):
            response = client.simulate_get('/analytics-data')

        assert response.status_code == 200
        data = response.json
        assert 'frequency' in data
        assert 'by_station' in data
        assert 'by_time' in data

    def test_analytics_days_parameter(self, client):
        """Test that days query parameter is passed correctly."""
        mock_report = {
            'period_days': 30,
            'frequency': {'total_minutes': 0.0, 'delayed_minutes': 0.0, 'delay_rate': 0.0, 'by_status': {}},
            'by_station': [],
            'by_time': {'by_hour': {}, 'by_day': {}},
            'generated_at': datetime.now().isoformat(),
        }

        with patch('lib.analytics.get_analytics_report', return_value=mock_report) as mock_get:
            response = client.simulate_get('/analytics-data', params={'days': '30'})

        # Check that get_analytics_report was called with days=30
        mock_get.assert_called_once_with(30)

    def test_analytics_invalid_days(self, client):
        """Test that invalid days parameter defaults to 7."""
        mock_report = {
            'period_days': 7,
            'frequency': {'total_minutes': 0.0, 'delayed_minutes': 0.0, 'delay_rate': 0.0, 'by_status': {}},
            'by_station': [],
            'by_time': {'by_hour': {}, 'by_day': {}},
            'generated_at': datetime.now().isoformat(),
        }

        with patch('lib.analytics.get_analytics_report', return_value=mock_report) as mock_get:
            response = client.simulate_get('/analytics-data', params={'days': 'invalid'})

        mock_get.assert_called_once_with(7)

    def test_analytics_has_cache_header(self, client):
        """Test that analytics has cache headers."""
        mock_report = {
            'period_days': 7,
            'frequency': {'total_minutes': 0.0, 'delayed_minutes': 0.0, 'delay_rate': 0.0, 'by_status': {}},
            'by_station': [],
            'by_time': {'by_hour': {}, 'by_day': {}},
            'generated_at': datetime.now().isoformat(),
        }

        with patch('lib.analytics.get_analytics_report', return_value=mock_report):
            response = client.simulate_get('/analytics-data')

        cache_control = response.headers.get('Cache-Control')
        assert cache_control is not None
        assert 'max-age' in cache_control


class TestStaticPages:
    """Tests for static page endpoints."""

    def test_index_page(self, client):
        """Test that index page is served."""
        response = client.simulate_get('/')

        # Should return 200 if file exists, or 404 if not
        # In a test environment, we just check it doesn't crash
        assert response.status_code in [200, 404]

    def test_dashboard_page(self, client):
        """Test that dashboard page is served."""
        response = client.simulate_get('/dashboard')
        assert response.status_code in [200, 404]

    def test_about_page(self, client):
        """Test that about page is served."""
        response = client.simulate_get('/about')
        assert response.status_code in [200, 404]

    def test_analytics_page(self, client):
        """Test that analytics page is served."""
        response = client.simulate_get('/analytics')
        assert response.status_code in [200, 404]


class TestCacheHeaders:
    """Tests for cache header behavior."""

    def test_status_no_cache_header(self, client):
        """Test that /status doesn't cache for too long."""
        mock_cache = {
            'cached_at': datetime.now().isoformat(),
            'last_successful_check': datetime.now().isoformat(),
            'reported_status': {
                'status': 'green',
                'description': 'Normal',
                'confidence': 1.0,
                'probabilities': {'green': 1.0, 'yellow': 0.0, 'red': 0.0},
                'timestamp': datetime.now().isoformat(),
            },
        }

        with patch('api.api.read_cache', return_value=mock_cache):
            response = client.simulate_get('/status')

        # Status endpoint should not have long cache
        # (it's real-time data)
        cache_control = response.headers.get('Cache-Control')
        # If there's a cache header, it should be short
        if cache_control:
            assert 'max-age=86400' not in cache_control  # Not a full day


class TestErrorHandling:
    """Tests for error handling in API."""

    def test_invalid_endpoint(self, client):
        """Test that invalid endpoints return 404."""
        response = client.simulate_get('/nonexistent-endpoint')
        assert response.status_code == 404

    def test_status_corrupted_cache(self, client):
        """Test handling of corrupted cache data."""
        # Cache with missing required fields
        mock_cache = {
            'cached_at': datetime.now().isoformat(),
            # Missing reported_status, best_status, statuses
        }

        with patch('api.api.read_cache', return_value=mock_cache):
            with patch('api.api.ENABLE_FALLBACK', False):
                response = client.simulate_get('/status')

        # Should handle gracefully (return error or fallback)
        assert response.status_code in [200, 500, 503]


class TestResponseStructure:
    """Tests for API response structure consistency."""

    def test_status_response_fields(self, client):
        """Test that status response has all expected fields."""
        mock_cache = {
            'cached_at': datetime.now().isoformat(),
            'last_successful_check': datetime.now().isoformat(),
            'reported_status': {
                'status': 'green',
                'description': 'Normal operation',
                'confidence': 1.0,
                'probabilities': {'green': 1.0, 'yellow': 0.0, 'red': 0.0},
                'timestamp': datetime.now().isoformat(),
            },
        }

        with patch('api.api.read_cache', return_value=mock_cache):
            response = client.simulate_get('/status')

        data = response.json

        # Required fields
        assert 'status' in data
        assert 'description' in data
        assert 'confidence' in data
        assert 'probabilities' in data
        assert 'timestamp' in data
        assert 'cached' in data

        # Probability structure
        assert 'green' in data['probabilities']
        assert 'yellow' in data['probabilities']
        assert 'red' in data['probabilities']

    def test_health_response_fields(self, client):
        """Test that health response has all expected fields."""
        mock_cache = {
            'cached_at': datetime.now().isoformat(),
            'reported_status': {'status': 'green'},
        }

        with patch('lib.muni_lib.read_cache', return_value=mock_cache):
            with patch('lib.analytics.init_db'):
                with patch('lib.analytics.get_db_connection') as mock_conn:
                    mock_cursor = MagicMock()
                    mock_cursor.fetchone.return_value = {'count': 100}
                    mock_conn.return_value.cursor.return_value = mock_cursor
                    response = client.simulate_get('/health')

        data = response.json

        assert 'status' in data
        assert 'service' in data
        assert 'timestamp' in data
        assert 'components' in data

        # Components structure
        assert 'cache' in data['components']
