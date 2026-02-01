#!/usr/bin/env python3
"""
Muni Status API - Web service for checking SF Muni subway status.

Endpoints:
    GET /status - Return current status (from cache or live detection)
    GET /feed.xml - RSS feed of status updates
    GET /health - Health check

Usage:
    gunicorn api:app --bind 0.0.0.0:8000
"""

import falcon
import hashlib
import os
import sys
from datetime import datetime
from pathlib import Path
from whitenoise import WhiteNoise

# Path resolution - get absolute paths relative to project root
API_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = API_DIR.parent

# Add parent directory to path for lib imports
sys.path.insert(0, str(PROJECT_ROOT))
from lib.muni_lib import download_muni_image, detect_muni_status, read_cache, read_cached_image

# Configuration
SNAPSHOT_DIR = str(PROJECT_ROOT / "artifacts" / "runtime" / "downloads")
CACHE_MAX_AGE = 300  # seconds (5 minutes) - fallback if cache is stale
ENABLE_FALLBACK = os.getenv('ENABLE_FALLBACK', 'true').lower() == 'true'

print("Muni Status API starting...")
if ENABLE_FALLBACK:
    print("Fallback mode enabled - will perform live detection if cache is stale")
else:
    print("Fallback mode disabled - API will only serve cached results")
    print("Set ENABLE_FALLBACK=true to enable live detection fallback")


class StatusResource:
    """API endpoint for checking current Muni status."""

    def on_get(self, req, resp):
        """
        Handle GET request to /status

        Returns JSON with:
        - status: green/yellow/red
        - description: status description
        - confidence: detection confidence (1.0 for deterministic detection)
        - probabilities: breakdown by status
        - timestamp: when check was performed
        - image_path: path to downloaded image
        - cached: whether result came from cache
        - cache_age: age of cache in seconds (if cached)
        - detection: detailed detection data (trains, delays, etc.)
        """
        timestamp = datetime.now().isoformat()

        # Try to read from cache first
        cache_data = read_cache()

        if cache_data:
            try:
                # Check cache age
                cached_at = datetime.fromisoformat(cache_data['cached_at'])
                cache_age = (datetime.now() - cached_at).total_seconds()

                # Use cache if it's fresh enough
                if cache_age < CACHE_MAX_AGE:
                    # Get best status (most optimistic of last 2)
                    best = cache_data.get('best_status', cache_data.get('statuses', [{}])[0])

                    # Build response with best status
                    response_data = {
                        'status': best['status'],
                        'description': best['description'],
                        'confidence': round(best['confidence'], 4),
                        'probabilities': {
                            'green': round(best['probabilities']['green'], 4),
                            'yellow': round(best['probabilities']['yellow'], 4),
                            'red': round(best['probabilities']['red'], 4)
                        },
                        'image_path': best.get('image_path'),
                        'image_dimensions': best.get('image_dimensions'),
                        'timestamp': best['timestamp'],
                        'cached': True,
                        'cache_age': round(cache_age, 1)
                    }

                    # Add detection details if available
                    if 'detection' in best:
                        response_data['detection'] = best['detection']

                    # Add status history info if available
                    if 'statuses' in cache_data and len(cache_data['statuses']) > 1:
                        response_data['status_history'] = [
                            {
                                'status': s['status'],
                                'timestamp': s['timestamp']
                            } for s in cache_data['statuses']
                        ]
                        response_data['is_best_of_two'] = True

                    resp.status = falcon.HTTP_200
                    resp.media = response_data
                    return
            except (KeyError, ValueError) as e:
                # Cache is corrupted, fall through to download + detect
                print(f"Cache read failed: {e}")

        # Cache miss or stale - check if fallback is enabled
        if not ENABLE_FALLBACK:
            resp.status = falcon.HTTP_503
            resp.media = {
                'error': 'Cache unavailable and fallback mode disabled',
                'details': 'Set ENABLE_FALLBACK=true to enable live detection',
                'timestamp': timestamp
            }
            return

        # Fallback mode: download and detect
        download_result = download_muni_image(
            output_folder=SNAPSHOT_DIR,
            validate_dimensions=True
        )

        if not download_result['success']:
            resp.status = falcon.HTTP_500
            resp.media = {
                'error': 'Failed to download image',
                'details': download_result['error'],
                'timestamp': timestamp
            }
            return

        # Detect status using OpenCV
        try:
            detection = detect_muni_status(download_result['filepath'])
        except Exception as e:
            resp.status = falcon.HTTP_500
            resp.media = {
                'error': 'Failed to detect status',
                'details': str(e),
                'image_path': download_result['filepath'],
                'timestamp': timestamp
            }
            return

        # Return successful response
        resp.status = falcon.HTTP_200
        resp.media = {
            'status': detection['status'],
            'description': detection['description'],
            'confidence': round(detection['status_confidence'], 4),
            'probabilities': {
                'green': round(detection['probabilities']['green'], 4),
                'yellow': round(detection['probabilities']['yellow'], 4),
                'red': round(detection['probabilities']['red'], 4)
            },
            'image_path': download_result['filepath'],
            'image_dimensions': {
                'width': download_result['width'],
                'height': download_result['height']
            },
            'timestamp': timestamp,
            'cached': False,
            'detection': detection.get('detection', {})
        }


class HealthResource:
    """Health check endpoint."""

    def on_get(self, req, resp):
        """Handle GET request to /health"""
        resp.status = falcon.HTTP_200
        resp.media = {
            'status': 'ok',
            'service': 'muni-status-api',
            'timestamp': datetime.now().isoformat()
        }


class LatestImageResource:
    """Serve the latest downloaded status image."""

    def on_get(self, req, resp):
        """Handle GET request to /latest-image"""
        image_path = None

        # Try to get image path from cache (works for local development)
        cache_data = read_cache()
        if cache_data:
            statuses = cache_data.get('statuses', [])
            if statuses and 'image_path' in statuses[0]:
                image_path = statuses[0]['image_path']

        # Check if cached path exists locally
        if image_path and os.path.exists(image_path):
            with open(image_path, 'rb') as f:
                resp.data = f.read()
            resp.status = falcon.HTTP_200
            resp.content_type = 'image/jpeg'
            resp.set_header('Cache-Control', 'public, max-age=30')
            return

        # Fallback: find most recent image in downloads directory (local dev)
        downloads_dir = Path(SNAPSHOT_DIR)
        if downloads_dir.exists():
            images = sorted(downloads_dir.glob('muni_snapshot_*.jpg'), reverse=True)
            if images:
                with open(images[0], 'rb') as f:
                    resp.data = f.read()
                resp.status = falcon.HTTP_200
                resp.content_type = 'image/jpeg'
                resp.set_header('Cache-Control', 'public, max-age=30')
                return

        # Cloud Run: read cached image from GCS
        # (The checker job uploads analyzed images to GCS)
        cached_image = read_cached_image()
        if cached_image:
            resp.data = cached_image
            resp.status = falcon.HTTP_200
            resp.content_type = 'image/jpeg'
            resp.set_header('Cache-Control', 'public, max-age=30')
            return

        resp.status = falcon.HTTP_404
        resp.media = {'error': 'No image available'}


class RSSFeedResource:
    """Serve the RSS feed of status updates."""

    def on_get(self, req, resp):
        """Handle GET request to /feed.xml"""
        from lib.notifiers import read_rss_feed, generate_empty_feed

        feed_content = read_rss_feed()

        if feed_content:
            resp.status = falcon.HTTP_200
            resp.content_type = 'application/rss+xml; charset=utf-8'
            resp.text = feed_content
        else:
            # Return empty feed if no history yet
            resp.status = falcon.HTTP_200
            resp.content_type = 'application/rss+xml; charset=utf-8'
            resp.text = generate_empty_feed()

        # Cache for 1 minute - feeds update when status changes
        resp.set_header('Cache-Control', 'public, max-age=60')


class StaticResource:
    """Serve the frontend HTML files with proper caching."""
    def __init__(self, filename, content_type):
        self.filename = filename
        self.content_type = content_type

    def on_get(self, req, resp):
        """Handle GET request with cache headers"""
        index_path = str(API_DIR / 'html' / self.filename)
        try:
            with open(index_path, 'rb') as f:
                content = f.read()

            resp.status = falcon.HTTP_200
            resp.content_type = self.content_type
            resp.text = content

            # Add cache headers for HTML files (shorter cache than static assets)
            # Cache for 5 minutes, but allow revalidation
            resp.set_header('Cache-Control', 'public, max-age=300, must-revalidate')

            # Add ETag based on file content for efficient revalidation
            etag = hashlib.md5(content).hexdigest()
            resp.set_header('ETag', f'"{etag}"')

        except FileNotFoundError:
            resp.status = falcon.HTTP_404
            resp.text = '<h1>Not found</h1><p>Something\'s missing</p>'


class TextResource(StaticResource):
    def __init__(self, filename):
        super().__init__(filename, 'text/html; charset=utf-8')


class IconResource(StaticResource):
    def __init__(self, filename):
        super().__init__(filename, 'image/x-icon')


# Create Falcon app
falcon_app = falcon.App()

# Add routes
falcon_app.add_route('/', TextResource('index.html'))
falcon_app.add_route('/dashboard', TextResource('dashboard.html'))
falcon_app.add_route('/about', TextResource('about.html'))
falcon_app.add_route('/favicon.ico', IconResource('static/favicon.ico'))
falcon_app.add_route('/status', StatusResource())
falcon_app.add_route('/health', HealthResource())
falcon_app.add_route('/latest-image', LatestImageResource())
falcon_app.add_route('/feed.xml', RSSFeedResource())

# Wrap with WhiteNoise for efficient static file serving with compression and caching
# WhiteNoise automatically compresses files (gzip/brotli) and adds proper headers
app = WhiteNoise(
    falcon_app,
    root=str(API_DIR / 'html' / 'static'),
    prefix='/static/',
    # Cache static files aggressively (1 year)
    max_age=31536000,  # 1 year in seconds
    # Add immutable directive for better caching
    immutable_file_test=lambda path, url: True  # All static files are immutable
)
