# Notification System

Developer reference for the notification subsystem. For user-facing setup (webhook URLs, RSS endpoint, social media links), see [README.md](../README.md). For cloud credential provisioning, see [CONFIGURATION.md](../CONFIGURATION.md).

---

## Overview

When the reported system status changes (e.g. green to yellow), `check_status.py` dispatches notifications to all configured channels:

| Channel | Module | Credentials required | Notes |
|---|---|---|---|
| Bluesky | `lib/notifiers/bluesky.py` | Yes | Posts via AT Protocol |
| Mastodon | `lib/notifiers/mastodon.py` | Yes | Posts via Mastodon API |
| RSS | `lib/notifiers/rss.py` | No | Always enabled; writes XML to cache |
| Webhooks | `lib/notifiers/webhooks.py` | Yes (URLs) | Auto-detects Slack, Discord, Teams |

All channels use shared status messages from `lib/notifiers/messages.py`.

---

## Architecture

### Call Flow

```
check_status.py
  ├── detect status, apply hysteresis → reported_status
  ├── determine should_notify (see Notification Triggers below)
  │
  ├── notify_status_change(status, previous_status, delay_summaries, timestamp)
  │     ├── bluesky.post_to_bluesky()        # if BLUESKY_* env vars set
  │     ├── mastodon.post_to_mastodon()       # if MASTODON_* env vars set
  │     ├── rss.update_rss_feed()             # always
  │     └── webhooks.send_webhooks()          # if WEBHOOK_URLS set
  │
  ├── check results for failures
  └── if no failures: cache_data['last_notified_status'] = current_reported
```

The dispatcher (`lib/notifiers/dispatcher.py`) checks environment variables for each channel. Unconfigured channels return a `skipped` result immediately without attempting any network calls.

### Notification Triggers

`check_status.py` sends notifications in two cases:

1. **Hysteresis transition** — `apply_status_hysteresis()` reports `status_changed=True` and there was a previous reported status. This is the normal path: the smoothed status has changed.

2. **Missed notification recovery** — The hysteresis status did not change, but `last_notified_status` differs from the current reported status. This catches cases where a previous notification attempt failed partway through (some channels succeeded, some didn't), so the next successful check retries the notification.

```python
# Case 1: hysteresis transition
if hysteresis_result['status_changed'] and previous_reported_status is not None:
    should_notify = True

# Case 2: missed notification recovery
elif previous_last_notified is not None and previous_last_notified != current_reported:
    should_notify = True
```

---

## Channel Return Format

Every channel function returns a dict with this common structure:

```python
{
    'success': bool,    # True if the notification was delivered
    'skipped': bool,    # True if channel is not configured (credentials missing)
    'error': str,       # Error message, or None on success
    # ... plus channel-specific fields (see below)
}
```

### Tri-state logic

| `success` | `skipped` | Meaning |
|---|---|---|
| `True` | `False` | Delivered successfully |
| `False` | `True` | Channel not configured — not a failure |
| `False` | `False` | Attempted but failed (network error, auth error, etc.) |

`check_status.py` treats `skipped` results as non-failures. Only `success=False, skipped=False` results count as failures and prevent `last_notified_status` from being updated.

### Channel-specific fields

| Channel | Extra fields |
|---|---|
| Bluesky | `uri` — AT Protocol URI of the created post |
| Mastodon | `url` — URL of the created post |
| RSS | `path` — file path or `gs://` URL where the feed was written |
| Webhooks | `sent` — count of successful webhooks, `failed` — count of failed webhooks |

---

## Duplicate Prevention

### `last_notified_status`

The cache stores `last_notified_status` — the status string (`'green'`, `'yellow'`, `'red'`) that was last successfully sent to all channels.

**Update rule:** `last_notified_status` is only updated when *no channel fails*. If any configured channel fails (not skipped, but actually fails), the value is left unchanged so the next check cycle will retry via the missed notification recovery path.

**Flow:**

```
1. Read previous_last_notified from cache
2. Determine should_notify (transition or recovery)
3. Dispatch to all channels
4. If any_failed:
     last_notified_status stays unchanged → next cycle retries
5. If no failures:
     last_notified_status = current_reported → no retry needed
```

### Interaction with hysteresis

The hysteresis system (`apply_status_hysteresis()`) prevents rapid status flips by requiring consistent readings before changing `reported_status`. Notifications are gated behind hysteresis — a notification is only sent when `reported_status` actually changes, not on every raw detection fluctuation.

---

## Configuration

### Environment Variables

| Variable | Channel | Description |
|---|---|---|
| `BLUESKY_HANDLE` | Bluesky | Account handle (e.g. `munimetro.bsky.social`) |
| `BLUESKY_APP_PASSWORD` | Bluesky | App password for the account |
| `MASTODON_INSTANCE` | Mastodon | Instance URL (e.g. `https://mastodon.social`) |
| `MASTODON_ACCESS_TOKEN` | Mastodon | Access token for the account |
| `WEBHOOK_URLS` | Webhooks | Comma-separated list of webhook URLs |

RSS requires no environment variables — it writes to the local cache directory (or GCS when `CLOUD_RUN` is set).

For cloud credential setup and secret management, see [CONFIGURATION.md](../CONFIGURATION.md).

### Webhook Platform Detection

`send_webhooks()` auto-detects the platform from the URL and formats the payload accordingly:

| URL pattern | Platform | Payload format |
|---|---|---|
| `hooks.slack.com` | Slack | Slack incoming webhook |
| `discord.com/api/webhooks` | Discord | Discord embed |
| `webhook.office.com` / `.logic.azure.com` | Teams | MessageCard |
| Anything else | Generic | JSON with `status`, `previous_status`, `description`, `delay_summaries`, `timestamp` |

---

## Key Files

| File | Purpose |
|---|---|
| `lib/notifiers/__init__.py` | Public API: re-exports `notify_status_change` and channel functions |
| `lib/notifiers/dispatcher.py` | `notify_status_change()` — dispatches to all channels, checks env vars |
| `lib/notifiers/bluesky.py` | `post_to_bluesky()` — AT Protocol client |
| `lib/notifiers/mastodon.py` | `post_to_mastodon()` — Mastodon API client |
| `lib/notifiers/rss.py` | `update_rss_feed()`, `read_rss_feed()` — RSS 2.0 feed generation |
| `lib/notifiers/webhooks.py` | `send_webhooks()` — multi-platform webhook delivery |
| `lib/notifiers/messages.py` | `STATUS_MESSAGES` — shared message templates |
| `api/check_status.py` | Notification trigger logic, `last_notified_status` management |
