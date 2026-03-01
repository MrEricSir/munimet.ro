"""
Webhook notification module.

Sends status change notifications to registered webhook URLs.
Auto-detects Slack, Discord, and Microsoft Teams URLs and formats
payloads accordingly. All other URLs receive a generic JSON payload.

Configure via the WEBHOOK_URLS environment variable (comma-separated).
"""

import logging
import os

import requests

from .messages import STATUS_MESSAGES

logger = logging.getLogger(__name__)

# Timeout for outgoing webhook requests
_WEBHOOK_TIMEOUT = 10  # seconds

# Status colors for rich embeds
_STATUS_COLORS = {
    'green': 0x44CC11,   # #4c1 approximation
    'yellow': 0xDFB317,
    'red': 0xE05D44,
}

_STATUS_COLORS_HEX = {
    'green': '44CC11',
    'yellow': 'DFB317',
    'red': 'E05D44',
}

_SITE_URL = 'https://munimet.ro'
_ICON_URL = f'{_SITE_URL}/static/android-chrome-192x192.png'
_BADGE_URL = f'{_SITE_URL}/badge.svg'


def get_webhook_urls():
    """
    Get configured webhook URLs from environment.

    Returns:
        list[str]: List of webhook URLs, or empty list if not configured.
    """
    raw = os.getenv('WEBHOOK_URLS', '').strip()
    if not raw:
        return []
    return [url.strip() for url in raw.split(',') if url.strip()]


def _detect_webhook_type(url):
    """
    Detect the webhook platform from the URL.

    Returns:
        str: One of 'slack', 'discord', 'teams', or 'generic'.
    """
    if 'hooks.slack.com' in url:
        return 'slack'
    if 'discord.com/api/webhooks' in url or 'discordapp.com/api/webhooks' in url:
        return 'discord'
    if 'webhook.office.com' in url or '.logic.azure.com' in url:
        return 'teams'
    return 'generic'


def _build_slack_payload(status, previous_status, delay_summaries):
    """Build a Slack incoming webhook payload."""
    message = STATUS_MESSAGES.get(status, f'Status: {status}')

    if status == 'yellow' and delay_summaries:
        details = '\n'.join(f'• {s}' for s in delay_summaries)
        message = f'{message}\n{details}'

    return {
        'text': message,
        'username': 'MuniMetro',
        'icon_url': _ICON_URL,
    }


def _build_discord_payload(status, previous_status, delay_summaries):
    """Build a Discord webhook payload with a rich embed."""
    message = STATUS_MESSAGES.get(status, f'Status: {status}')
    color = _STATUS_COLORS.get(status, 0x9F9F9F)

    embed = {
        'title': 'MuniMetro Status Update',
        'description': message,
        'color': color,
        'url': _SITE_URL,
    }

    if status == 'yellow' and delay_summaries:
        embed['fields'] = [{
            'name': 'Delays',
            'value': '\n'.join(f'• {s}' for s in delay_summaries),
        }]

    return {
        'username': 'MuniMetro',
        'avatar_url': _ICON_URL,
        'embeds': [embed],
    }


def _build_teams_payload(status, previous_status, delay_summaries):
    """Build a Microsoft Teams incoming webhook payload (MessageCard)."""
    message = STATUS_MESSAGES.get(status, f'Status: {status}')
    color = _STATUS_COLORS_HEX.get(status, '9F9F9F')

    card = {
        '@type': 'MessageCard',
        '@context': 'https://schema.org/extensions',
        'summary': f'MuniMetro: {message}',
        'themeColor': color,
        'title': 'MuniMetro Status Update',
        'text': message,
    }

    if status == 'yellow' and delay_summaries:
        card['text'] += '\n\n' + '\n'.join(f'- {s}' for s in delay_summaries)

    card['potentialAction'] = [{
        '@type': 'OpenUri',
        'name': 'View Dashboard',
        'targets': [{'os': 'default', 'uri': _SITE_URL}],
    }]

    return card


def _build_generic_payload(status, previous_status, delay_summaries, timestamp):
    """Build a generic JSON webhook payload."""
    return {
        'status': status,
        'previous_status': previous_status,
        'description': STATUS_MESSAGES.get(status, f'Status: {status}'),
        'delay_summaries': delay_summaries or [],
        'timestamp': timestamp,
        'url': _SITE_URL,
        'badge_url': _BADGE_URL,
    }


def send_webhooks(status, previous_status=None, delay_summaries=None, timestamp=None):
    """
    Send status change notifications to all configured webhooks.

    Auto-detects the platform (Slack, Discord, Teams) from the URL
    and formats the payload accordingly.

    Args:
        status: Current status ('green', 'yellow', 'red')
        previous_status: Previous status for context (optional)
        delay_summaries: List of delay summary strings (optional)
        timestamp: ISO timestamp string (optional)

    Returns:
        dict: {
            'success': bool (True if all webhooks succeeded),
            'sent': int (number of webhooks attempted),
            'failed': int (number of failed webhooks),
            'error': str or None (first error message, if any),
        }
    """
    urls = get_webhook_urls()
    if not urls:
        return {
            'success': False,
            'sent': 0,
            'failed': 0,
            'error': 'Not configured (WEBHOOK_URLS not set)',
        }

    sent = 0
    failed = 0
    first_error = None

    for url in urls:
        webhook_type = _detect_webhook_type(url)

        if webhook_type == 'slack':
            payload = _build_slack_payload(status, previous_status, delay_summaries)
        elif webhook_type == 'discord':
            payload = _build_discord_payload(status, previous_status, delay_summaries)
        elif webhook_type == 'teams':
            payload = _build_teams_payload(status, previous_status, delay_summaries)
        else:
            payload = _build_generic_payload(status, previous_status, delay_summaries, timestamp)

        try:
            response = requests.post(url, json=payload, timeout=_WEBHOOK_TIMEOUT)
            response.raise_for_status()
            sent += 1
            logger.info(f"Webhook sent ({webhook_type}): {response.status_code}")
        except Exception as e:
            failed += 1
            error_msg = f'{webhook_type} webhook failed: {e}'
            logger.error(error_msg)
            if first_error is None:
                first_error = error_msg

    return {
        'success': failed == 0 and sent > 0,
        'sent': sent,
        'failed': failed,
        'error': first_error,
    }
