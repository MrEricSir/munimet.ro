#!/usr/bin/env python3
"""
Manage webhook URLs for MuniMetro status notifications.

Supports two storage backends:
- Local: Reads/writes WEBHOOK_URLS in .env file
- Google Cloud: Reads/writes WEBHOOK_URLS in Secret Manager

Usage:
    python scripts/setup/manage-webhooks.py             # Local (.env file)
    python scripts/setup/manage-webhooks.py --cloud      # Google Cloud Secret Manager
    python scripts/setup/manage-webhooks.py --add URL    # Add a URL non-interactively
    python scripts/setup/manage-webhooks.py --remove URL # Remove a URL non-interactively
    python scripts/setup/manage-webhooks.py --list       # List current webhooks
    python scripts/setup/manage-webhooks.py --test       # Send test to all webhooks
"""

import argparse
import os
import sys
from pathlib import Path

# Get project root
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
ENV_FILE = PROJECT_ROOT / ".env"
ENV_KEY = "WEBHOOK_URLS"

# GCP configuration
GCP_PROJECT_ID = os.getenv('GCP_PROJECT_ID', 'munimetro')

# Add project root to path for lib imports
sys.path.insert(0, str(PROJECT_ROOT))


def _detect_type(url):
    """Detect webhook platform from URL."""
    if 'hooks.slack.com' in url:
        return 'Slack'
    if 'discord.com/api/webhooks' in url or 'discordapp.com/api/webhooks' in url:
        return 'Discord'
    if 'webhook.office.com' in url or '.logic.azure.com' in url:
        return 'Teams'
    return 'Generic'


# ---------------------------------------------------------------------------
# .env backend
# ---------------------------------------------------------------------------

def _load_env():
    """Load all key=value pairs from .env."""
    env = {}
    if ENV_FILE.exists():
        with open(ENV_FILE, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, _, value = line.partition('=')
                    env[key.strip()] = value.strip()
    return env


def _save_env(env):
    """Write all key=value pairs back to .env, preserving comments."""
    lines = []
    written_keys = set()

    # Preserve existing lines, updating values in place
    if ENV_FILE.exists():
        with open(ENV_FILE, 'r') as f:
            for line in f:
                stripped = line.strip()
                if stripped and not stripped.startswith('#') and '=' in stripped:
                    key, _, _ = stripped.partition('=')
                    key = key.strip()
                    if key in env:
                        lines.append(f"{key}={env[key]}\n")
                        written_keys.add(key)
                    # Drop keys no longer in env (removed)
                else:
                    lines.append(line if line.endswith('\n') else line + '\n')

    # Append any new keys not already in the file
    for key, value in env.items():
        if key not in written_keys:
            lines.append(f"{key}={value}\n")

    with open(ENV_FILE, 'w') as f:
        f.writelines(lines)


def load_urls_local():
    """Load webhook URLs from .env."""
    env = _load_env()
    raw = env.get(ENV_KEY, '').strip()
    if not raw:
        return []
    return [u.strip() for u in raw.split(',') if u.strip()]


def save_urls_local(urls):
    """Save webhook URLs to .env."""
    env = _load_env()
    if urls:
        env[ENV_KEY] = ','.join(urls)
    else:
        env.pop(ENV_KEY, None)
    _save_env(env)


# ---------------------------------------------------------------------------
# GCP Secret Manager backend
# ---------------------------------------------------------------------------

def load_urls_cloud():
    """Load webhook URLs from Secret Manager."""
    try:
        from google.cloud import secretmanager
        client = secretmanager.SecretManagerServiceClient()
        name = f"projects/{GCP_PROJECT_ID}/secrets/{ENV_KEY}/versions/latest"
        response = client.access_secret_version(request={"name": name})
        raw = response.payload.data.decode("UTF-8").strip()
        if not raw:
            return []
        return [u.strip() for u in raw.split(',') if u.strip()]
    except Exception:
        return []


def save_urls_cloud(urls):
    """Save webhook URLs to Secret Manager."""
    from google.cloud import secretmanager

    client = secretmanager.SecretManagerServiceClient()
    parent = f"projects/{GCP_PROJECT_ID}"
    secret_path = f"{parent}/secrets/{ENV_KEY}"

    # Create secret if it doesn't exist
    try:
        client.get_secret(request={"name": secret_path})
    except Exception:
        client.create_secret(
            request={
                "parent": parent,
                "secret_id": ENV_KEY,
                "secret": {"replication": {"automatic": {}}},
            }
        )

    value = ','.join(urls) if urls else ''
    client.add_secret_version(
        request={
            "parent": secret_path,
            "payload": {"data": value.encode("UTF-8")},
        }
    )


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------

def print_urls(urls):
    """Print the current list of webhook URLs."""
    if not urls:
        print("  (none configured)")
        return
    for i, url in enumerate(urls, 1):
        platform = _detect_type(url)
        # Mask the URL token portion for security
        if len(url) > 40:
            display = url[:35] + '...' + url[-8:]
        else:
            display = url
        print(f"  {i}. {display} ({platform})")


def add_url(urls, new_url=None):
    """Add a webhook URL. Prompts if new_url is None."""
    if new_url is None:
        print()
        new_url = input("Webhook URL: ").strip()

    if not new_url:
        print("No URL provided.")
        return urls

    if not new_url.startswith('https://'):
        print("Warning: webhook URLs should use HTTPS.")
        confirm = input("Add anyway? [y/N]: ").strip().lower()
        if confirm != 'y':
            return urls

    if new_url in urls:
        print(f"Already registered: {new_url}")
        return urls

    platform = _detect_type(new_url)
    urls.append(new_url)
    print(f"Added {platform} webhook.")
    return urls


def remove_url(urls, target_url=None):
    """Remove a webhook URL. Prompts for index if target_url is None."""
    if not urls:
        print("No webhooks to remove.")
        return urls

    if target_url:
        if target_url in urls:
            urls.remove(target_url)
            print(f"Removed: {target_url}")
        else:
            print(f"Not found: {target_url}")
        return urls

    print()
    print_urls(urls)
    print()
    choice = input("Number to remove (or 'c' to cancel): ").strip()

    if choice.lower() == 'c':
        return urls

    try:
        idx = int(choice) - 1
        if 0 <= idx < len(urls):
            removed = urls.pop(idx)
            print(f"Removed: {removed}")
        else:
            print("Invalid number.")
    except ValueError:
        print("Invalid input.")

    return urls


def test_webhooks(urls):
    """Send a test notification to all configured webhooks."""
    if not urls:
        print("No webhooks configured.")
        return

    from lib.notifiers.webhooks import send_webhooks

    # Temporarily set the env var so send_webhooks picks up these URLs
    old_val = os.environ.get('WEBHOOK_URLS')
    os.environ['WEBHOOK_URLS'] = ','.join(urls)

    try:
        print(f"Sending test to {len(urls)} webhook(s)...")
        result = send_webhooks(
            status='green',
            previous_status='green',
            delay_summaries=[],
            timestamp='(test notification)',
        )
        print(f"  Sent: {result['sent']}, Failed: {result['failed']}")
        if result['error']:
            print(f"  Error: {result['error']}")
    finally:
        # Restore original env
        if old_val is not None:
            os.environ['WEBHOOK_URLS'] = old_val
        else:
            os.environ.pop('WEBHOOK_URLS', None)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def interactive(load_fn, save_fn, backend_name):
    """Run the interactive webhook manager."""
    print("=" * 50)
    print(f"MuniMetro Webhook Manager ({backend_name})")
    print("=" * 50)
    print()

    urls = load_fn()
    print("Current webhooks:")
    print_urls(urls)

    while True:
        print()
        print("[a] Add  [r] Remove  [t] Test  [l] List  [q] Save & quit")
        choice = input("> ").strip().lower()

        if choice == 'a':
            urls = add_url(urls)
        elif choice == 'r':
            urls = remove_url(urls)
        elif choice == 't':
            test_webhooks(urls)
        elif choice == 'l':
            print()
            print("Current webhooks:")
            print_urls(urls)
        elif choice == 'q':
            break
        else:
            print("Unknown option.")

    save_fn(urls)
    print()
    print(f"Saved {len(urls)} webhook(s).")


def main():
    parser = argparse.ArgumentParser(
        description="Manage MuniMetro webhook URLs."
    )
    parser.add_argument(
        "--cloud", action="store_true",
        help="Use Google Cloud Secret Manager instead of local .env file",
    )
    parser.add_argument(
        "--add", metavar="URL",
        help="Add a webhook URL (non-interactive)",
    )
    parser.add_argument(
        "--remove", metavar="URL",
        help="Remove a webhook URL (non-interactive)",
    )
    parser.add_argument(
        "--list", action="store_true", dest="list_urls",
        help="List current webhook URLs",
    )
    parser.add_argument(
        "--test", action="store_true",
        help="Send a test notification to all webhooks",
    )
    args = parser.parse_args()

    # Pick backend
    if args.cloud:
        load_fn, save_fn, name = load_urls_cloud, save_urls_cloud, "Cloud"
    else:
        load_fn, save_fn, name = load_urls_local, save_urls_local, "Local"

    # Non-interactive modes
    if args.list_urls:
        urls = load_fn()
        print_urls(urls)
        return

    if args.add:
        urls = load_fn()
        urls = add_url(urls, args.add)
        save_fn(urls)
        return

    if args.remove:
        urls = load_fn()
        urls = remove_url(urls, args.remove)
        save_fn(urls)
        return

    if args.test:
        urls = load_fn()
        test_webhooks(urls)
        return

    # Interactive mode
    interactive(load_fn, save_fn, name)


if __name__ == "__main__":
    main()
