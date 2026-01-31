"""
RSS feed notification module.

Generates and maintains an RSS 2.0 feed of Muni status updates.
"""

import os
import json
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from email.utils import formatdate

# Path resolution
LIB_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = LIB_DIR.parent

# Feed configuration
FEED_TITLE = "SF Muni Metro Status"
FEED_LINK = "https://munimet.ro"
FEED_URL = "https://munimet.ro/feed.xml"
FEED_DESCRIPTION = "Real-time SF Muni Metro subway status updates"
MAX_ITEMS = 20

# Register atom namespace to avoid ns0: prefix
ET.register_namespace('atom', 'http://www.w3.org/2005/Atom')
ATOM_NS = '{http://www.w3.org/2005/Atom}'

# Status titles for RSS items
STATUS_TITLES = {
    'green': 'Muni Status: Normal Operation',
    'yellow': 'Muni Status: Delays Detected',
    'red': 'Muni Status: Not Operating',
}


def get_rss_path():
    """
    Get RSS feed file path based on environment.

    Returns:
        str: RSS feed path (local file path or gs:// URL)
    """
    if os.getenv('CLOUD_RUN'):
        bucket = os.getenv('GCS_BUCKET', 'munimetro-cache')
        return f'gs://{bucket}/rss_feed.xml'
    else:
        return str(PROJECT_ROOT / "artifacts" / "runtime" / "cache" / "rss_feed.xml")


def _get_items_path():
    """Get path for RSS items JSON storage."""
    if os.getenv('CLOUD_RUN'):
        bucket = os.getenv('GCS_BUCKET', 'munimetro-cache')
        return f'gs://{bucket}/rss_items.json'
    else:
        return str(PROJECT_ROOT / "artifacts" / "runtime" / "cache" / "rss_items.json")


def _read_items():
    """Read RSS items from JSON storage."""
    items_path = _get_items_path()

    try:
        if items_path.startswith('gs://'):
            from google.cloud import storage

            parts = items_path[5:].split('/', 1)
            bucket_name = parts[0]
            blob_name = parts[1]

            client = storage.Client()
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(blob_name)

            if not blob.exists():
                return []

            return json.loads(blob.download_as_string())
        else:
            if not os.path.exists(items_path):
                return []

            with open(items_path, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        print(f"Error reading RSS items: {e}")
        return []


def _write_items(items):
    """Write RSS items to JSON storage."""
    items_path = _get_items_path()

    try:
        if items_path.startswith('gs://'):
            from google.cloud import storage

            parts = items_path[5:].split('/', 1)
            bucket_name = parts[0]
            blob_name = parts[1]

            client = storage.Client()
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(blob_name)

            blob.upload_from_string(
                json.dumps(items, indent=2),
                content_type='application/json'
            )
            return True
        else:
            os.makedirs(os.path.dirname(items_path), exist_ok=True)
            with open(items_path, 'w', encoding='utf-8') as f:
                json.dump(items, f, indent=2)
            return True
    except Exception as e:
        print(f"Error writing RSS items: {e}")
        return False


def _generate_xml(items):
    """Generate RSS XML from items list using ElementTree."""
    now = formatdate(usegmt=True)

    # Create root RSS element (namespace is added by register_namespace)
    rss = ET.Element('rss', version='2.0')

    # Create channel
    channel = ET.SubElement(rss, 'channel')
    ET.SubElement(channel, 'title').text = FEED_TITLE
    ET.SubElement(channel, 'link').text = FEED_LINK
    ET.SubElement(channel, 'description').text = FEED_DESCRIPTION

    # Add atom:link for self-reference (required for valid RSS)
    atom_link = ET.SubElement(channel, f'{ATOM_NS}link')
    atom_link.set('href', FEED_URL)
    atom_link.set('rel', 'self')
    atom_link.set('type', 'application/rss+xml')

    # Add channel image
    image = ET.SubElement(channel, 'image')
    ET.SubElement(image, 'url').text = f'{FEED_LINK}/static/favicon.png'
    ET.SubElement(image, 'title').text = FEED_TITLE
    ET.SubElement(image, 'link').text = FEED_LINK

    ET.SubElement(channel, 'lastBuildDate').text = now

    # Add items
    for item_data in items:
        item = ET.SubElement(channel, 'item')
        ET.SubElement(item, 'title').text = item_data['title']
        ET.SubElement(item, 'description').text = item_data['description']
        ET.SubElement(item, 'pubDate').text = item_data['pub_date']
        guid = ET.SubElement(item, 'guid')
        guid.text = item_data['guid']
        guid.set('isPermaLink', 'false')

    # Convert to string with XML declaration
    xml_str = ET.tostring(rss, encoding='unicode', xml_declaration=False)
    return f'<?xml version="1.0" encoding="UTF-8"?>\n{xml_str}'


def read_rss_feed():
    """
    Read RSS feed XML from cache.

    Returns:
        str: RSS XML content, or None if not found
    """
    rss_path = get_rss_path()

    try:
        if rss_path.startswith('gs://'):
            from google.cloud import storage

            parts = rss_path[5:].split('/', 1)
            bucket_name = parts[0]
            blob_name = parts[1] if len(parts) > 1 else 'rss_feed.xml'

            client = storage.Client()
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(blob_name)

            if not blob.exists():
                return None

            return blob.download_as_string().decode('utf-8')
        else:
            if not os.path.exists(rss_path):
                return None

            with open(rss_path, 'r', encoding='utf-8') as f:
                return f.read()
    except Exception as e:
        print(f"Error reading RSS feed: {e}")
        return None


def write_rss_feed(xml_content):
    """
    Write RSS feed XML to cache.

    Args:
        xml_content: RSS XML string

    Returns:
        bool: True if successful, False otherwise
    """
    rss_path = get_rss_path()

    try:
        if rss_path.startswith('gs://'):
            from google.cloud import storage

            parts = rss_path[5:].split('/', 1)
            bucket_name = parts[0]
            blob_name = parts[1] if len(parts) > 1 else 'rss_feed.xml'

            client = storage.Client()
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(blob_name)

            blob.upload_from_string(
                xml_content,
                content_type='application/rss+xml'
            )
            return True
        else:
            os.makedirs(os.path.dirname(rss_path), exist_ok=True)
            with open(rss_path, 'w', encoding='utf-8') as f:
                f.write(xml_content)
            return True
    except Exception as e:
        print(f"Error writing RSS feed: {e}")
        return False


def generate_empty_feed():
    """
    Generate an empty RSS feed with just the channel info.

    Returns:
        str: RSS XML content
    """
    return _generate_xml([])


def update_rss_feed(status, description=None, delay_summaries=None, timestamp=None):
    """
    Add a new status entry to the RSS feed.

    Args:
        status: Current status ('green', 'yellow', 'red')
        description: Status description (optional)
        delay_summaries: List of delay summary strings (optional)
        timestamp: ISO timestamp string (optional, defaults to now)

    Returns:
        dict: {'success': bool, 'path': str or None, 'error': str or None}
    """
    try:
        # Read existing items
        items = _read_items()

        # Build item content
        title = STATUS_TITLES.get(status, f'Muni Status: {status.title()}')

        # Build description with full details (no truncation)
        desc_parts = []
        if description:
            desc_parts.append(description)
        if delay_summaries:
            if desc_parts:
                desc_parts.append('')  # blank line
            desc_parts.extend(f'• {s}' for s in delay_summaries)

        item_description = '\n'.join(desc_parts) if desc_parts else title

        # Parse or generate timestamp
        if timestamp:
            try:
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            except ValueError:
                dt = datetime.now()
        else:
            dt = datetime.now()

        pub_date = formatdate(dt.timestamp(), usegmt=True)
        guid = f"status-{dt.isoformat()}"

        # Create new item
        new_item = {
            'title': title,
            'description': item_description,
            'pub_date': pub_date,
            'guid': guid
        }

        # Prepend new item and limit to MAX_ITEMS
        items = [new_item] + items[:MAX_ITEMS - 1]

        # Save items
        if not _write_items(items):
            return {
                'success': False,
                'path': None,
                'error': 'Failed to write RSS items'
            }

        # Generate and write XML
        xml_content = _generate_xml(items)
        if write_rss_feed(xml_content):
            return {
                'success': True,
                'path': get_rss_path(),
                'error': None
            }
        else:
            return {
                'success': False,
                'path': None,
                'error': 'Failed to write RSS feed'
            }

    except Exception as e:
        return {
            'success': False,
            'path': None,
            'error': str(e)
        }
