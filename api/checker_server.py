#!/usr/bin/env python3
"""
Simple HTTP server for Cloud Run that triggers status checks.
Cloud Scheduler sends GET requests to this server to trigger checks.
"""

import os
import sys
from pathlib import Path
import falcon

# Add parent directory to path for imports
API_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = API_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from api.check_status import check_status  # noqa: E402


class CheckerResource:
    """HTTP endpoint that triggers a status check."""

    def on_get(self, req, resp):
        """Handle GET request from Cloud Scheduler."""
        try:
            # Run status check with cache writing enabled
            check_status(should_write_cache=True)

            resp.status = falcon.HTTP_200
            resp.media = {
                'success': True,
                'message': 'Status check completed successfully'
            }
        except Exception as e:
            print(f"❌ Error during status check: {e}", file=sys.stderr)
            resp.status = falcon.HTTP_500
            resp.media = {
                'success': False,
                'error': str(e)
            }


class HealthResource:
    """Health check endpoint for Cloud Run."""

    def on_get(self, req, resp):
        """Simple health check."""
        resp.status = falcon.HTTP_200
        resp.media = {'status': 'healthy'}


# Create Falcon app
app = falcon.App()
app.add_route('/', CheckerResource())
app.add_route('/health', HealthResource())


if __name__ == '__main__':
    # For local testing
    from wsgiref.simple_server import make_server
    port = int(os.getenv('PORT', '8000'))
    with make_server('', port, app) as httpd:
        print(f'Checker server listening on port {port}...')
        httpd.serve_forever()
