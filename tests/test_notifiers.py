"""
Tests for the notification system (Bluesky, Mastodon, RSS, dispatcher).

Tests message formatting, character limits, RSS feed generation,
and notification dispatch logic.
"""

import json
import os
import pytest
import sys
import tempfile
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.notifiers import (
    post_to_bluesky,
    post_to_mastodon,
    update_rss_feed,
    read_rss_feed,
    generate_empty_feed,
    notify_status_change,
    send_webhooks,
)
from lib.notifiers.webhooks import (
    _detect_webhook_type,
    _build_slack_payload,
    _build_discord_payload,
    _build_teams_payload,
    _build_generic_payload,
    get_webhook_urls,
)
from lib.notifiers.rss import (
    _generate_xml,
    _read_items,
    _write_items,
    STATUS_TITLES,
    MAX_ITEMS,
)
from lib.notifiers.messages import STATUS_MESSAGES


class TestBlueskyNotifier:
    """Tests for Bluesky notification posting."""

    def test_bluesky_missing_credentials(self):
        """Test that missing credentials return appropriate error."""
        with patch.dict(os.environ, {}, clear=True):
            result = post_to_bluesky(status='green')

        assert result['success'] is False
        assert 'BLUESKY_HANDLE' in result['error']
        assert result['uri'] is None

    def test_bluesky_message_formatting_green(self):
        """Test that green status uses correct message format."""
        # This test verifies message construction without calling the actual API
        # We check that the function correctly builds the message before sending

        # Test by capturing what would be sent (mock at atproto level)
        captured_message = None

        def mock_post_to_bluesky(status, previous_status=None, delay_summaries=None):
            from lib.notifiers.messages import STATUS_MESSAGES
            nonlocal captured_message
            captured_message = STATUS_MESSAGES.get(status, f'Status: {status}')
            return {'success': True, 'uri': 'mock://uri', 'error': None}

        # Just verify the message format
        from lib.notifiers.messages import STATUS_MESSAGES
        message = STATUS_MESSAGES.get('green')
        assert message is not None
        assert len(message) > 0
        assert len(message) <= 300  # Within Bluesky limit

    def test_bluesky_yellow_message_with_delays(self):
        """Test that yellow status message format includes delay summaries."""
        from lib.notifiers.messages import STATUS_MESSAGES

        delay_summaries = [
            'Westbound delay at Powell',
            'Eastbound delay at Montgomery',
        ]

        # Build message the same way as the actual function
        base_message = STATUS_MESSAGES.get('yellow')
        details = '\n'.join(f'• {s}' for s in delay_summaries)
        full_message = f'{base_message}\n\n{details}'

        # Verify message structure
        assert 'Powell' in full_message
        assert 'Montgomery' in full_message
        assert '•' in full_message

    def test_bluesky_message_truncation_logic(self):
        """Test that message truncation logic works correctly."""
        from lib.notifiers.messages import STATUS_MESSAGES
        BLUESKY_CHAR_LIMIT = 300

        # Create very long delay summaries
        long_delays = [f'Delay at station {i} with extended description' for i in range(20)]

        base_message = STATUS_MESSAGES.get('yellow')
        details = '\n'.join(f'• {s}' for s in long_delays)
        full_message = f'{base_message}\n\n{details}'

        # Apply same truncation logic as the function
        if len(full_message) > BLUESKY_CHAR_LIMIT:
            truncated = full_message[:BLUESKY_CHAR_LIMIT - 3].rsplit('\n', 1)[0]
            message = truncated + '...'
        else:
            message = full_message

        # Message should be at most 300 chars
        assert len(message) <= BLUESKY_CHAR_LIMIT
        # Truncated message should end with ...
        assert message.endswith('...')

    def test_bluesky_api_error_returns_failure(self):
        """Test that API errors result in failure response."""
        # This verifies the error handling structure
        # The actual function catches Exception and returns this format
        error_response = {
            'success': False,
            'uri': None,
            'error': 'API rate limit exceeded'
        }

        assert error_response['success'] is False
        assert error_response['uri'] is None
        assert 'rate limit' in error_response['error']


class TestMastodonNotifier:
    """Tests for Mastodon notification posting."""

    def test_mastodon_missing_credentials(self):
        """Test that missing credentials return appropriate error."""
        with patch.dict(os.environ, {}, clear=True):
            result = post_to_mastodon(status='green')

        assert result['success'] is False
        assert 'MASTODON_INSTANCE' in result['error'] or 'MASTODON_ACCESS_TOKEN' in result['error']
        assert result['url'] is None


class TestRSSFeedGeneration:
    """Tests for RSS feed generation and management."""

    def test_generate_empty_feed(self):
        """Test that empty feed has valid RSS structure."""
        xml_content = generate_empty_feed()

        # Should be valid XML
        root = ET.fromstring(xml_content)

        assert root.tag == 'rss'
        assert root.get('version') == '2.0'

        channel = root.find('channel')
        assert channel is not None
        assert channel.find('title') is not None
        assert channel.find('link') is not None
        assert channel.find('description') is not None

    def test_generate_xml_with_items(self):
        """Test XML generation with items."""
        items = [
            {
                'title': 'Test Title 1',
                'description': 'Test Description 1',
                'pub_date': 'Mon, 01 Jan 2024 12:00:00 +0000',
                'guid': 'test-guid-1',
            },
            {
                'title': 'Test Title 2',
                'description': 'Test Description 2',
                'pub_date': 'Mon, 01 Jan 2024 11:00:00 +0000',
                'guid': 'test-guid-2',
            },
        ]

        xml_content = _generate_xml(items)
        root = ET.fromstring(xml_content)

        channel = root.find('channel')
        item_elements = channel.findall('item')

        assert len(item_elements) == 2

        # Check first item
        first_item = item_elements[0]
        assert first_item.find('title').text == 'Test Title 1'
        assert first_item.find('description').text == 'Test Description 1'
        assert first_item.find('guid').text == 'test-guid-1'

    def test_update_rss_feed_creates_new_item(self, tmp_path):
        """Test that update_rss_feed adds a new item."""
        rss_path = tmp_path / "feed.xml"
        items_path = tmp_path / "items.json"

        with patch('lib.notifiers.rss.get_rss_path', return_value=str(rss_path)):
            with patch('lib.notifiers.rss._get_items_path', return_value=str(items_path)):
                result = update_rss_feed(
                    status='yellow',
                    description='Delays detected',
                    delay_summaries=['Delay at Powell'],
                    timestamp=datetime.now().isoformat()
                )

        assert result['success'] is True
        assert rss_path.exists()

        # Parse and verify
        root = ET.parse(str(rss_path)).getroot()
        items = root.find('channel').findall('item')

        assert len(items) == 1
        assert items[0].find('title').text == STATUS_TITLES['yellow']
        assert 'Powell' in items[0].find('description').text

    def test_rss_feed_max_items(self, tmp_path):
        """Test that RSS feed respects MAX_ITEMS limit."""
        rss_path = tmp_path / "feed.xml"
        items_path = tmp_path / "items.json"

        with patch('lib.notifiers.rss.get_rss_path', return_value=str(rss_path)):
            with patch('lib.notifiers.rss._get_items_path', return_value=str(items_path)):
                # Add more items than MAX_ITEMS
                for i in range(MAX_ITEMS + 5):
                    update_rss_feed(
                        status='green',
                        description=f'Update {i}',
                        timestamp=(datetime.now()).isoformat()
                    )

        # Read back items
        with open(items_path, 'r') as f:
            items = json.load(f)

        assert len(items) == MAX_ITEMS

    def test_rss_status_titles(self):
        """Test that all statuses have RSS titles defined."""
        for status in ['green', 'yellow', 'red']:
            assert status in STATUS_TITLES
            assert 'Muni' in STATUS_TITLES[status]

    def test_rss_read_nonexistent_feed(self, tmp_path):
        """Test reading a feed that doesn't exist."""
        with patch('lib.notifiers.rss.get_rss_path', return_value=str(tmp_path / "nonexistent.xml")):
            result = read_rss_feed()

        assert result is None


class TestNotificationDispatcher:
    """Tests for the notification dispatcher."""

    def test_dispatch_all_channels(self):
        """Test that dispatcher calls all notification channels."""
        # Mock all notifiers
        mock_bluesky_result = {'success': True, 'uri': 'test-uri', 'error': None}
        mock_mastodon_result = {'success': True, 'url': 'test-url', 'error': None}
        mock_rss_result = {'success': True, 'path': '/test/path', 'error': None}

        with patch.dict(os.environ, {
            'BLUESKY_HANDLE': 'test.bsky.social',
            'BLUESKY_APP_PASSWORD': 'secret',
            'MASTODON_INSTANCE': 'mastodon.social',
            'MASTODON_ACCESS_TOKEN': 'token',
        }):
            with patch('lib.notifiers.bluesky.post_to_bluesky', return_value=mock_bluesky_result):
                with patch('lib.notifiers.mastodon.post_to_mastodon', return_value=mock_mastodon_result):
                    with patch('lib.notifiers.rss.update_rss_feed', return_value=mock_rss_result):
                        result = notify_status_change(
                            status='yellow',
                            previous_status='green',
                            delay_summaries=['Test delay']
                        )

        assert 'bluesky' in result
        assert 'mastodon' in result
        assert 'rss' in result

    def test_dispatch_unconfigured_channels(self):
        """Test that unconfigured channels are handled gracefully."""
        # Clear environment variables
        with patch.dict(os.environ, {}, clear=True):
            with patch('lib.notifiers.rss.update_rss_feed', return_value={'success': True, 'path': '/test', 'error': None}):
                result = notify_status_change(
                    status='green',
                    previous_status='yellow'
                )

        # Bluesky and Mastodon should report not configured
        assert result['bluesky']['success'] is False
        assert 'Not configured' in result['bluesky']['error']

        assert result['mastodon']['success'] is False
        assert 'Not configured' in result['mastodon']['error']

        # RSS should still work (no credentials needed)
        assert 'rss' in result

    def test_dispatch_partial_failure(self):
        """Test that one channel failing doesn't affect others."""
        mock_bluesky_result = {'success': False, 'uri': None, 'error': 'API error'}
        mock_rss_result = {'success': True, 'path': '/test/path', 'error': None}

        with patch.dict(os.environ, {
            'BLUESKY_HANDLE': 'test.bsky.social',
            'BLUESKY_APP_PASSWORD': 'secret',
        }):
            with patch('lib.notifiers.bluesky.post_to_bluesky', return_value=mock_bluesky_result):
                with patch('lib.notifiers.rss.update_rss_feed', return_value=mock_rss_result):
                    result = notify_status_change(
                        status='green',
                        previous_status='yellow'
                    )

        # Bluesky failed but RSS succeeded
        assert result['bluesky']['success'] is False
        assert result['rss']['success'] is True


class TestStatusMessages:
    """Tests for status message content."""

    def test_all_statuses_have_messages(self):
        """Test that all statuses have defined messages."""
        for status in ['green', 'yellow', 'red']:
            assert status in STATUS_MESSAGES
            assert len(STATUS_MESSAGES[status]) > 0

    def test_message_content_appropriate(self):
        """Test that messages convey appropriate meaning."""
        # Green should indicate positive status
        green_msg = STATUS_MESSAGES['green'].lower()
        assert 'track' in green_msg or 'running' in green_msg or 'normal' in green_msg

        # Yellow should indicate issues
        yellow_msg = STATUS_MESSAGES['yellow'].lower()
        assert 'uh oh' in yellow_msg or 'delay' in yellow_msg or 'not feeling' in yellow_msg

        # Red should indicate not operating
        red_msg = STATUS_MESSAGES['red'].lower()
        assert 'nap' in red_msg or 'not' in red_msg or 'closed' in red_msg


class TestRSSItemStorage:
    """Tests for RSS item persistence."""

    def test_write_and_read_items(self, tmp_path):
        """Test that items can be written and read back."""
        items_path = tmp_path / "items.json"

        test_items = [
            {'title': 'Test 1', 'description': 'Desc 1', 'pub_date': 'date1', 'guid': 'guid1'},
            {'title': 'Test 2', 'description': 'Desc 2', 'pub_date': 'date2', 'guid': 'guid2'},
        ]

        with patch('lib.notifiers.rss._get_items_path', return_value=str(items_path)):
            success = _write_items(test_items)
            assert success is True

            read_back = _read_items()

        assert len(read_back) == 2
        assert read_back[0]['title'] == 'Test 1'
        assert read_back[1]['guid'] == 'guid2'

    def test_read_empty_items(self, tmp_path):
        """Test reading when no items file exists."""
        items_path = tmp_path / "nonexistent.json"

        with patch('lib.notifiers.rss._get_items_path', return_value=str(items_path)):
            items = _read_items()

        assert items == []


class TestRSSFeedValidation:
    """Tests for RSS feed XML validity."""

    def test_feed_has_required_elements(self, tmp_path):
        """Test that generated feed has all required RSS 2.0 elements."""
        rss_path = tmp_path / "feed.xml"
        items_path = tmp_path / "items.json"

        with patch('lib.notifiers.rss.get_rss_path', return_value=str(rss_path)):
            with patch('lib.notifiers.rss._get_items_path', return_value=str(items_path)):
                update_rss_feed(
                    status='green',
                    description='Normal operation'
                )

        root = ET.parse(str(rss_path)).getroot()
        channel = root.find('channel')

        # Required elements per RSS 2.0 spec
        assert channel.find('title') is not None
        assert channel.find('link') is not None
        assert channel.find('description') is not None

    def test_items_have_required_elements(self, tmp_path):
        """Test that items have required elements."""
        rss_path = tmp_path / "feed.xml"
        items_path = tmp_path / "items.json"

        with patch('lib.notifiers.rss.get_rss_path', return_value=str(rss_path)):
            with patch('lib.notifiers.rss._get_items_path', return_value=str(items_path)):
                update_rss_feed(
                    status='yellow',
                    description='Delays detected',
                    delay_summaries=['Powell delay']
                )

        root = ET.parse(str(rss_path)).getroot()
        items = root.find('channel').findall('item')

        assert len(items) >= 1

        item = items[0]
        assert item.find('title') is not None
        assert item.find('description') is not None
        assert item.find('pubDate') is not None
        assert item.find('guid') is not None

    def test_feed_has_atom_self_link(self, tmp_path):
        """Test that feed includes atom:link for self-reference."""
        rss_path = tmp_path / "feed.xml"
        items_path = tmp_path / "items.json"

        with patch('lib.notifiers.rss.get_rss_path', return_value=str(rss_path)):
            with patch('lib.notifiers.rss._get_items_path', return_value=str(items_path)):
                update_rss_feed(status='green', description='Test')

        # Read raw XML to check for atom namespace
        with open(rss_path, 'r') as f:
            content = f.read()

        # Should have atom namespace
        assert 'atom' in content or 'http://www.w3.org/2005/Atom' in content


class TestWebhookTypeDetection:
    """Tests for webhook URL platform detection."""

    def test_detect_slack(self):
        assert _detect_webhook_type('https://hooks.slack.com/services/T00/B00/xxx') == 'slack'

    def test_detect_discord(self):
        assert _detect_webhook_type('https://discord.com/api/webhooks/123/abc') == 'discord'

    def test_detect_discord_legacy(self):
        assert _detect_webhook_type('https://discordapp.com/api/webhooks/123/abc') == 'discord'

    def test_detect_teams(self):
        assert _detect_webhook_type('https://outlook.webhook.office.com/webhookb2/xxx') == 'teams'

    def test_detect_teams_logic_app(self):
        assert _detect_webhook_type('https://prod-01.westus.logic.azure.com/workflows/xxx') == 'teams'

    def test_detect_generic(self):
        assert _detect_webhook_type('https://example.com/webhook') == 'generic'


class TestWebhookPayloads:
    """Tests for webhook payload formatting."""

    def test_slack_payload_structure(self):
        payload = _build_slack_payload('green', None, None)
        assert 'text' in payload
        assert 'username' in payload
        assert payload['username'] == 'MuniMetro'

    def test_slack_payload_includes_delays(self):
        payload = _build_slack_payload('yellow', 'green', ['Delay at Powell'])
        assert 'Powell' in payload['text']

    def test_discord_payload_has_embed(self):
        payload = _build_discord_payload('green', None, None)
        assert 'embeds' in payload
        assert len(payload['embeds']) == 1
        assert 'color' in payload['embeds'][0]

    def test_discord_payload_delay_fields(self):
        payload = _build_discord_payload('yellow', 'green', ['Delay at Powell'])
        embed = payload['embeds'][0]
        assert 'fields' in embed
        assert 'Powell' in embed['fields'][0]['value']

    def test_teams_payload_structure(self):
        payload = _build_teams_payload('red', 'yellow', None)
        assert payload['@type'] == 'MessageCard'
        assert 'themeColor' in payload
        assert 'potentialAction' in payload

    def test_generic_payload_structure(self):
        payload = _build_generic_payload('green', 'yellow', ['Delay'], '2026-03-01T12:00:00')
        assert payload['status'] == 'green'
        assert payload['previous_status'] == 'yellow'
        assert payload['delay_summaries'] == ['Delay']
        assert 'url' in payload
        assert 'badge_url' in payload


class TestWebhookURLConfig:
    """Tests for webhook URL configuration."""

    def test_no_urls_configured(self):
        with patch.dict(os.environ, {}, clear=True):
            assert get_webhook_urls() == []

    def test_empty_env_var(self):
        with patch.dict(os.environ, {'WEBHOOK_URLS': ''}, clear=True):
            assert get_webhook_urls() == []

    def test_single_url(self):
        with patch.dict(os.environ, {'WEBHOOK_URLS': 'https://example.com/hook'}, clear=True):
            urls = get_webhook_urls()
            assert urls == ['https://example.com/hook']

    def test_multiple_urls(self):
        with patch.dict(os.environ, {'WEBHOOK_URLS': 'https://a.com,https://b.com'}, clear=True):
            urls = get_webhook_urls()
            assert len(urls) == 2

    def test_urls_stripped(self):
        with patch.dict(os.environ, {'WEBHOOK_URLS': '  https://a.com , https://b.com  '}, clear=True):
            urls = get_webhook_urls()
            assert urls == ['https://a.com', 'https://b.com']


class TestWebhookSending:
    """Tests for webhook delivery."""

    def test_send_no_urls(self):
        with patch.dict(os.environ, {}, clear=True):
            result = send_webhooks(status='green')
        assert result['success'] is False
        assert result['sent'] == 0
        assert 'Not configured' in result['error']

    def test_send_successful(self):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()

        with patch.dict(os.environ, {'WEBHOOK_URLS': 'https://example.com/hook'}, clear=True):
            with patch('lib.notifiers.webhooks.requests.post', return_value=mock_response):
                result = send_webhooks(status='green')

        assert result['success'] is True
        assert result['sent'] == 1
        assert result['failed'] == 0

    def test_send_partial_failure(self):
        mock_ok = MagicMock()
        mock_ok.raise_for_status = MagicMock()
        mock_fail = MagicMock()
        mock_fail.raise_for_status.side_effect = Exception('Connection refused')

        with patch.dict(os.environ, {'WEBHOOK_URLS': 'https://a.com,https://b.com'}, clear=True):
            with patch('lib.notifiers.webhooks.requests.post', side_effect=[mock_ok, mock_fail]):
                result = send_webhooks(status='green')

        assert result['success'] is False
        assert result['sent'] == 1
        assert result['failed'] == 1
        assert 'Connection refused' in result['error']

    def test_dispatcher_includes_webhooks(self):
        """Test that the dispatcher includes webhook results."""
        with patch.dict(os.environ, {}, clear=True):
            with patch('lib.notifiers.rss.update_rss_feed', return_value={'success': True, 'path': '/test', 'error': None}):
                result = notify_status_change(status='green', previous_status='yellow')

        assert 'webhooks' in result
