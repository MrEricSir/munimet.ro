"""
Notification dispatcher module.

Dispatches status change notifications to all enabled channels.
"""

import os
from . import bluesky, mastodon, rss, webhooks
from .messages import STATUS_MESSAGES

VALID_STATUSES = {'green', 'yellow', 'red'}
ALL_CHANNELS = {'bluesky', 'mastodon', 'rss', 'webhooks'}


def notify_status_change(status, previous_status, delay_summaries=None,
                         timestamp=None, channels=None):
    """
    Dispatch status change notification to all enabled channels.

    Args:
        status: Current status ('green', 'yellow', 'red')
        previous_status: Previous status for context
        delay_summaries: List of delay summary strings (optional)
        timestamp: ISO timestamp string (optional)
        channels: Set of channel names to dispatch to (optional).
                  When None, dispatches to all channels. When provided,
                  only dispatches to channels in the intersection with ALL_CHANNELS.

    Returns:
        dict: Results from each notifier channel
            {
                'bluesky': {'success': bool, 'skipped': bool, 'uri': str, 'error': str},
                'mastodon': {'success': bool, 'skipped': bool, 'url': str, 'error': str},
                'rss': {'success': bool, 'skipped': bool, 'path': str, 'error': str},
                'webhooks': {'success': bool, 'skipped': bool, 'sent': int, 'failed': int, 'error': str}
            }
            'skipped' is True when the channel is not configured (no credentials/URLs set).

    Raises:
        ValueError: If status is not one of VALID_STATUSES
    """
    if not isinstance(status, str) or status not in VALID_STATUSES:
        raise ValueError(f"Invalid status {status!r}; must be one of {sorted(VALID_STATUSES)}")

    dispatch_channels = ALL_CHANNELS if channels is None else channels & ALL_CHANNELS
    results = {}

    # Bluesky (if credentials configured)
    if 'bluesky' not in dispatch_channels:
        pass
    elif os.getenv('BLUESKY_HANDLE') and os.getenv('BLUESKY_APP_PASSWORD'):
        results['bluesky'] = bluesky.post_to_bluesky(
            status=status,
            previous_status=previous_status,
            delay_summaries=delay_summaries
        )
    else:
        results['bluesky'] = {
            'success': False,
            'skipped': True,
            'uri': None,
            'error': 'Not configured (BLUESKY_HANDLE/BLUESKY_APP_PASSWORD not set)'
        }

    # Mastodon (if credentials configured)
    if 'mastodon' not in dispatch_channels:
        pass
    elif os.getenv('MASTODON_INSTANCE') and os.getenv('MASTODON_ACCESS_TOKEN'):
        results['mastodon'] = mastodon.post_to_mastodon(
            status=status,
            previous_status=previous_status,
            delay_summaries=delay_summaries
        )
    else:
        results['mastodon'] = {
            'success': False,
            'skipped': True,
            'url': None,
            'error': 'Not configured (MASTODON_INSTANCE/MASTODON_ACCESS_TOKEN not set)'
        }

    # RSS feed (always enabled - no credentials needed)
    if 'rss' in dispatch_channels:
        # Use same status message as Bluesky/Mastodon
        description = STATUS_MESSAGES.get(status, f'Status: {status}')
        results['rss'] = rss.update_rss_feed(
            status=status,
            description=description,
            delay_summaries=delay_summaries,
            timestamp=timestamp
        )

    # Webhooks (if URLs configured)
    if 'webhooks' not in dispatch_channels:
        pass
    else:
        results['webhooks'] = webhooks.send_webhooks(
            status=status,
            previous_status=previous_status,
            delay_summaries=delay_summaries,
            timestamp=timestamp
        )

    return results
