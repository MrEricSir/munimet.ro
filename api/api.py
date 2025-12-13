#!/usr/bin/env python3
"""
Muni Status API - Web service for checking SF Muni subway status.

Endpoints:
    GET /status - Download latest image and return predicted status

Usage:
    gunicorn api:app --bind 0.0.0.0:8000
"""

import falcon
import os
import sys
from datetime import datetime
from pathlib import Path

# Path resolution - get absolute paths relative to project root
API_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = API_DIR.parent

# Add parent directory to path for lib imports
sys.path.insert(0, str(PROJECT_ROOT))
from lib.muni_lib import download_muni_image, predict_muni_status, load_muni_model, read_cache

# Configuration
SNAPSHOT_DIR = str(PROJECT_ROOT / "artifacts" / "runtime" / "downloads")
CACHE_MAX_AGE = 300  # seconds (5 minutes) - fallback if cache is stale
ENABLE_FALLBACK = os.getenv('ENABLE_FALLBACK', 'true').lower() == 'true'

# Pre-load model at startup only if fallback mode is enabled
# If using cache writer, set ENABLE_FALLBACK=false to make API lightweight
MODEL = PROCESSOR = LABEL_TO_STATUS = DEVICE = None

if ENABLE_FALLBACK:
    print("Loading ML model for fallback mode...")
    MODEL, PROCESSOR, LABEL_TO_STATUS, DEVICE = load_muni_model()
    print(f"Model loaded successfully on {DEVICE}")
else:
    print("Fallback mode disabled - API will only serve cached results")
    print("Set ENABLE_FALLBACK=true to enable live download+predict fallback")


class StatusResource:
    """API endpoint for checking current Muni status."""

    def on_get(self, req, resp):
        """
        Handle GET request to /status

        Returns JSON with:
        - status: green/yellow/red
        - description: AI-generated description
        - confidence: prediction confidence (0-1)
        - probabilities: breakdown by status
        - timestamp: when check was performed
        - image_path: path to downloaded image
        - cached: whether result came from cache
        - cache_age: age of cache in seconds (if cached)
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
                        'image_path': best['image_path'],
                        'image_dimensions': best['image_dimensions'],
                        'timestamp': best['timestamp'],
                        'cached': True,
                        'cache_age': round(cache_age, 1)
                    }

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
                # Cache is corrupted, fall through to download + predict
                print(f"Cache read failed: {e}")

        # Cache miss or stale - check if fallback is enabled
        if not ENABLE_FALLBACK:
            resp.status = falcon.HTTP_503
            resp.media = {
                'error': 'Cache unavailable and fallback mode disabled',
                'details': 'Set ENABLE_FALLBACK=true to enable live predictions',
                'timestamp': timestamp
            }
            return

        # Fallback mode: download and predict
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

        # Predict status using pre-loaded model
        try:
            prediction = predict_muni_status(
                download_result['filepath'],
                model=MODEL,
                processor=PROCESSOR,
                label_to_status=LABEL_TO_STATUS,
                device=DEVICE
            )
        except Exception as e:
            resp.status = falcon.HTTP_500
            resp.media = {
                'error': 'Failed to predict status',
                'details': str(e),
                'image_path': download_result['filepath'],
                'timestamp': timestamp
            }
            return

        # Return successful response
        resp.status = falcon.HTTP_200
        resp.media = {
            'status': prediction['status'],
            'description': prediction['description'],
            'confidence': round(prediction['status_confidence'], 4),
            'probabilities': {
                'green': round(prediction['probabilities']['green'], 4),
                'yellow': round(prediction['probabilities']['yellow'], 4),
                'red': round(prediction['probabilities']['red'], 4)
            },
            'image_path': download_result['filepath'],
            'image_dimensions': {
                'width': download_result['width'],
                'height': download_result['height']
            },
            'timestamp': timestamp,
            'cached': False
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


class StaticResource:
    """Serve the frontend files."""
    def __init__(self, filename):
        self.filename = filename

    def on_get(self, req, resp):
        """Handle GET request"""
        resp.status = falcon.HTTP_200
        resp.content_type = 'text/html'

        # Return file data
        index_path = str(API_DIR / 'html' / self.filename)
        try:
            with open(index_path, 'rb') as f:
                resp.text = f.read()
        except FileNotFoundError:
            resp.status = falcon.HTTP_404
            resp.text = '<h1>Not found</h1><p>Something\'s missing</p>'


# Create Falcon app
app = falcon.App()

# Add routes
app.add_route('/', StaticResource('index.html'))
app.add_route('/dashboard', StaticResource('dashboard.html'))
app.add_route('/about', StaticResource('about.html'))
app.add_static_route('/static', str(API_DIR / 'html' / 'static'))
app.add_route('/status', StatusResource())
app.add_route('/health', HealthResource())
