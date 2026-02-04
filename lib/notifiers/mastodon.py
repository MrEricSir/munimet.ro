"""
Mastodon notification module.

Posts status updates to Mastodon social network.
"""

import os
from .messages import STATUS_MESSAGES


def post_to_mastodon(status, previous_status=None, delay_summaries=None):
    """
    Post a status update to Mastodon.

    Requires environment variables:
    - MASTODON_INSTANCE: The instance URL (e.g., 'https://mastodon.social')
    - MASTODON_ACCESS_TOKEN: An access token for the account

    Args:
        status: Current status ('green', 'yellow', 'red')
        previous_status: Previous status for context (optional)
        delay_summaries: List of delay summary strings for yellow status (optional)

    Returns:
        dict: {'success': bool, 'url': str or None, 'error': str or None}
    """
    MASTODON_CHAR_LIMIT = 500

    instance = os.getenv('MASTODON_INSTANCE')
    access_token = os.getenv('MASTODON_ACCESS_TOKEN')

    if not instance or not access_token:
        return {
            'success': False,
            'url': None,
            'error': 'MASTODON_INSTANCE and MASTODON_ACCESS_TOKEN environment variables required'
        }

    message = STATUS_MESSAGES.get(status, f'Status: {status}')

    # Add delay details for yellow status
    if status == 'yellow' and delay_summaries:
        details = '\n'.join(f'• {s}' for s in delay_summaries)
        full_message = f'{message}\n\n{details}'

        # Truncate if over limit
        if len(full_message) > MASTODON_CHAR_LIMIT:
            # Leave room for "..."
            truncated = full_message[:MASTODON_CHAR_LIMIT - 3].rsplit('\n', 1)[0]
            message = truncated + '...'
        else:
            message = full_message

    try:
        from mastodon import Mastodon

        client = Mastodon(
            access_token=access_token,
            api_base_url=instance
        )

        # Post the status (called "toot" in Mastodon)
        result = client.status_post(message)

        return {
            'success': True,
            'url': result.get('url'),
            'error': None
        }
    except Exception as e:
        return {
            'success': False,
            'url': None,
            'error': str(e)
        }
