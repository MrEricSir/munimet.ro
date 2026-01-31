"""
Bluesky notification module.

Posts status updates to Bluesky social network.
"""

import os

# Status messages matching the frontend (api/html/index.html)
STATUS_MESSAGES = {
    'green': '🟢 All aboard: Muni is on track',
    'yellow': "🟡 Uh oh: Muni's not feeling well",
    'red': '🔴 Muni is taking a nap',
}


def post_to_bluesky(status, previous_status=None, delay_summaries=None):
    """
    Post a status update to Bluesky.

    Requires environment variables:
    - BLUESKY_HANDLE: The account handle (e.g., 'munimetro.bsky.social')
    - BLUESKY_APP_PASSWORD: An app password for the account

    Args:
        status: Current status ('green', 'yellow', 'red')
        previous_status: Previous status for context (optional)
        delay_summaries: List of delay summary strings for yellow status (optional)

    Returns:
        dict: {'success': bool, 'uri': str or None, 'error': str or None}
    """
    BLUESKY_CHAR_LIMIT = 300

    handle = os.getenv('BLUESKY_HANDLE')
    app_password = os.getenv('BLUESKY_APP_PASSWORD')

    if not handle or not app_password:
        return {
            'success': False,
            'uri': None,
            'error': 'BLUESKY_HANDLE and BLUESKY_APP_PASSWORD environment variables required'
        }

    message = STATUS_MESSAGES.get(status, f'Status: {status}')

    # Add delay details for yellow status
    if status == 'yellow' and delay_summaries:
        details = '\n'.join(f'• {s}' for s in delay_summaries)
        full_message = f'{message}\n\n{details}'

        # Truncate if over limit
        if len(full_message) > BLUESKY_CHAR_LIMIT:
            # Leave room for "..."
            truncated = full_message[:BLUESKY_CHAR_LIMIT - 3].rsplit('\n', 1)[0]
            message = truncated + '...'
        else:
            message = full_message

    try:
        from atproto import Client

        client = Client()
        client.login(handle, app_password)

        post = client.send_post(text=message)

        return {
            'success': True,
            'uri': post.uri,
            'error': None
        }
    except Exception as e:
        return {
            'success': False,
            'uri': None,
            'error': str(e)
        }
