"""
Notification channels for Muni Metro status updates.

Provides a unified interface for publishing status updates to multiple channels:
- Bluesky (social network)
- RSS feed

Usage:
    from lib.notifiers import notify_status_change

    results = notify_status_change(
        status='yellow',
        previous_status='green',
        delay_summaries=['Westbound delay at Powell']
    )
"""

from .dispatcher import notify_status_change
from .bluesky import post_to_bluesky
from .rss import update_rss_feed, read_rss_feed, generate_empty_feed

__all__ = [
    'notify_status_change',
    'post_to_bluesky',
    'update_rss_feed',
    'read_rss_feed',
    'generate_empty_feed',
]
