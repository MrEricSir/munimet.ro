"""
Notification dispatcher module.

Dispatches status change notifications to all enabled channels.
"""

import os
from . import bluesky, rss
from .messages import STATUS_MESSAGES


def notify_status_change(status, previous_status, delay_summaries=None, timestamp=None):
    """
    Dispatch status change notification to all enabled channels.

    Args:
        status: Current status ('green', 'yellow', 'red')
        previous_status: Previous status for context
        delay_summaries: List of delay summary strings (optional)
        timestamp: ISO timestamp string (optional)

    Returns:
        dict: Results from each notifier channel
            {
                'bluesky': {'success': bool, 'uri': str, 'error': str},
                'rss': {'success': bool, 'path': str, 'error': str}
            }
    """
    results = {}

    # Bluesky (if credentials configured)
    if os.getenv('BLUESKY_HANDLE') and os.getenv('BLUESKY_APP_PASSWORD'):
        results['bluesky'] = bluesky.post_to_bluesky(
            status=status,
            previous_status=previous_status,
            delay_summaries=delay_summaries
        )
    else:
        results['bluesky'] = {
            'success': False,
            'uri': None,
            'error': 'Not configured (BLUESKY_HANDLE/BLUESKY_APP_PASSWORD not set)'
        }

    # RSS feed (always enabled - no credentials needed)
    # Use same status message as Bluesky
    description = STATUS_MESSAGES.get(status, f'Status: {status}')
    results['rss'] = rss.update_rss_feed(
        status=status,
        description=description,
        delay_summaries=delay_summaries,
        timestamp=timestamp
    )

    return results
