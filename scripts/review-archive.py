#!/usr/bin/env python3
"""
Download archived snapshot images, run detection on each, and open an HTML report
focused on train detection quality. Flag problematic images to add to test suite.

Usage:
    python scripts/review-archive.py                         # Today's overrides
    python scripts/review-archive.py --date 2026-02-21       # Specific date
    python scripts/review-archive.py --from 2026-02-01 --to 2026-02-07  # Date range
    python scripts/review-archive.py --reason transition      # Filter by reason
    python scripts/review-archive.py --all-reasons            # Show all reasons
    python scripts/review-archive.py --skip-download          # Re-run on already-downloaded images
"""

import argparse
import base64
import html
import json
import os
import re
import subprocess
import sys
import webbrowser
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from lib.config import DEFAULT_ARCHIVE_BUCKET, REVENUE_SINGLE_SUFFIXES, REVENUE_DOUBLE_SUFFIXES
from lib.detection import detect_system_status

DEFAULT_ARCHIVE_DIR = PROJECT_ROOT / 'artifacts' / 'runtime' / 'archive'
DEFAULT_REPORT_PATH = PROJECT_ROOT / 'artifacts' / 'runtime' / 'archive_review.html'

# Matches both old format (reason only) and new format (reason_rawStatus)
FILENAME_PATTERN = re.compile(
    r'muni_snapshot_(\d{8})_(\d{6})_(\w+?)(?:_raw([A-Z]\w+))?\.jpg'
)


def parse_image_filename(filename: str) -> dict | None:
    """Extract timestamp, reason, and optional raw_status from an archive filename."""
    m = FILENAME_PATTERN.match(filename)
    if not m:
        return None
    date_str, time_str, reason, raw_status = m.groups()
    dt = datetime.strptime(f'{date_str}_{time_str}', '%Y%m%d_%H%M%S')
    return {
        'datetime': dt,
        'timestamp': dt.strftime('%Y-%m-%d %H:%M:%S'),
        'reason': reason,
        'raw_status': raw_status.lower() if raw_status else None,
    }


def run_download(args) -> Path:
    """Run download-archive-images.py and return the output directory."""
    output_dir = Path(args.output_dir)
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / 'scripts' / 'download-archive-images.py'),
        '--output-dir', str(output_dir),
        '--bucket', args.bucket,
    ]
    if args.from_date and args.to_date:
        cmd += ['--from', args.from_date, '--to', args.to_date]
    elif args.date:
        cmd += ['--date', args.date]
    if args.reason:
        cmd += ['--reason', args.reason]

    print('Downloading archived images...')
    subprocess.run(cmd, check=True)
    print()
    return output_dir


def collect_images(directory: Path, reason_filter: str | None = None) -> list[dict]:
    """Collect and parse archive image files from a directory."""
    entries = []
    for path in sorted(directory.glob('muni_snapshot_*.jpg')):
        info = parse_image_filename(path.name)
        if not info:
            continue
        if reason_filter and info['reason'] != reason_filter:
            continue
        info['path'] = path
        entries.append(info)
    entries.sort(key=lambda e: e['datetime'])
    return entries


def analyze_image(path: Path) -> dict:
    """Run detection on an image and return results."""
    try:
        return detect_system_status(str(path))
    except Exception as e:
        return {'system_status': 'error', 'error': str(e),
                'trains': [], 'delay_summaries': [],
                'delays_platforms': [], 'delays_segments': [], 'delays_bunching': []}


def image_to_data_uri(path: Path) -> str:
    """Encode an image as a base64 data URI."""
    data = path.read_bytes()
    b64 = base64.b64encode(data).decode('ascii')
    return f'data:image/jpeg;base64,{b64}'


def is_revenue(train: dict) -> bool:
    """Check if a train is in revenue service based on route suffix."""
    route = train.get('route', '')
    return route in REVENUE_DOUBLE_SUFFIXES or route in REVENUE_SINGLE_SUFFIXES


def build_html(entries: list[dict]) -> str:
    """Generate a self-contained HTML report with train detection focus and flagging."""
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Compute stats
    reason_counts = {}
    status_counts = {}
    low_conf_total = 0
    for e in entries:
        det = e['detection']
        reason_counts[e['reason']] = reason_counts.get(e['reason'], 0) + 1
        s = det['system_status']
        status_counts[s] = status_counts.get(s, 0) + 1
        low_conf_total += sum(1 for t in det.get('trains', []) if t.get('confidence') != 'high')

    # Build per-card HTML
    cards_html = []
    for i, e in enumerate(entries):
        det = e['detection']
        status = det['system_status']
        reason = e['reason']
        raw_status = e.get('raw_status')
        trains = det.get('trains', [])
        summaries = det.get('delay_summaries', [])

        # Override label
        override_html = ''
        if reason == 'override' and raw_status:
            override_html = (
                f'<span class="override-label">Raw: '
                f'<span class="status-dot {html.escape(raw_status)}"></span>{html.escape(raw_status)}'
                f' &rarr; Reported: '
                f'<span class="status-dot {html.escape(status)}"></span>{html.escape(status)}'
                f'</span>'
            )

        # Train table
        upper = sorted([t for t in trains if t.get('track') == 'upper'], key=lambda t: t['x'])
        lower = sorted([t for t in trains if t.get('track') == 'lower'], key=lambda t: t['x'])
        revenue_count = sum(1 for t in trains if is_revenue(t))

        def train_rows(track_trains):
            rows = []
            for t in track_trains:
                conf = t.get('confidence', 'high')
                cls = 'low-conf' if conf != 'high' else ''
                rev = 'Y' if is_revenue(t) else 'N'
                rows.append(
                    f'<tr class="{cls}">'
                    f'<td class="train-id">{html.escape(t["id"])}</td>'
                    f'<td>x={t["x"]}</td>'
                    f'<td>{html.escape(conf)}</td>'
                    f'<td>{rev}</td>'
                    f'</tr>'
                )
            return ''.join(rows)

        train_table = f'''<div class="train-summary">{len(trains)} trains ({revenue_count} revenue)</div>
<table class="train-table">'''
        if upper:
            train_table += f'<tr class="track-header"><td colspan="4">Upper track ({len(upper)})</td></tr>'
            train_table += train_rows(upper)
        if lower:
            train_table += f'<tr class="track-header"><td colspan="4">Lower track ({len(lower)})</td></tr>'
            train_table += train_rows(lower)
        train_table += '</table>'

        # Delay summaries
        delay_html = ''
        if summaries:
            items = ''.join(f'<li>{html.escape(s)}</li>' for s in summaries)
            delay_html = f'<ul class="delays">{items}</ul>'

        # Baseline cards start collapsed
        collapsed_cls = ' collapsed' if reason == 'baseline' else ''

        # JSON blob for export (embedded in data attribute)
        card_data = json.dumps({
            'filename': e['path'].name,
            'source_dir': str(e['path'].parent),
            'detected_status': status,
            'detected_trains': [[t['id'], t['x']] for t in trains],
            'detected_delay_summaries': summaries,
        })

        cards_html.append(f'''
        <div class="card{collapsed_cls}" id="card-{i}" data-detection='{html.escape(card_data, quote=True)}'>
            <div class="card-header {status}" onclick="toggleCard({i})">
                <span class="time">{html.escape(e['timestamp'])}</span>
                {override_html}
                <span class="badge reason-{reason}">{reason}</span>
                <span class="badge status-{status}">{status}</span>
                <span class="collapse-indicator"></span>
            </div>
            <div class="card-body">
                <img src="{e['data_uri']}" alt="Snapshot {i+1}" loading="lazy" onclick="this.classList.toggle('expanded')">
                <div class="info">
                    {train_table}
                    {delay_html}
                    <div class="flag-buttons">
                        <button class="flag-btn" onclick="toggleFlag({i},'missing_trains')">Missing Trains</button>
                        <button class="flag-btn" onclick="toggleFlag({i},'extra_trains')">Extra Trains</button>
                        <button class="flag-btn" onclick="toggleFlag({i},'wrong_status')">Wrong Status</button>
                    </div>
                    <textarea class="flag-notes" id="notes-{i}" placeholder="Notes (e.g. M2031KK near Church not detected)" style="display:none"></textarea>
                </div>
            </div>
        </div>''')

    # Summary stats
    reason_parts = []
    for r in ['override', 'transition', 'baseline']:
        if r in reason_counts:
            reason_parts.append(f'{reason_counts[r]} {r}')
    status_parts = [f'{c} {s}' for s, c in sorted(status_counts.items())]

    return f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Archive Review - {now}</title>
<style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        background: #1a1a2e;
        color: #eee;
        padding: 20px 20px 80px;
    }}
    .header {{ text-align: center; margin-bottom: 24px; }}
    .header h1 {{ font-size: 1.5em; margin-bottom: 6px; }}
    .header .meta {{ color: #888; font-size: 0.85em; line-height: 1.6; }}
    .header .stat-highlight {{ color: #f6f; }}

    .card {{
        background: #262640;
        border-radius: 10px;
        margin-bottom: 12px;
        overflow: hidden;
        max-width: 1100px;
        margin-left: auto;
        margin-right: auto;
    }}
    .card.flagged {{ outline: 2px solid #f6f; }}
    .card.collapsed .card-body {{ display: none; }}
    .card.collapsed .collapse-indicator::after {{ content: "\\25B6"; }}
    .collapse-indicator::after {{ content: "\\25BC"; color: #666; font-size: 0.75em; }}

    .card-header {{
        padding: 10px 16px;
        display: flex;
        align-items: center;
        gap: 10px;
        border-bottom: 2px solid transparent;
        cursor: pointer;
        user-select: none;
    }}
    .card-header.green  {{ border-color: #2a9a2a; }}
    .card-header.yellow {{ border-color: #c0a000; }}
    .card-header.red    {{ border-color: #cc3030; }}
    .card-header.error  {{ border-color: #666; }}

    .time {{ font-weight: 600; font-size: 0.95em; }}
    .override-label {{
        font-size: 0.82em; color: #ccc; margin-right: auto;
    }}
    .status-dot {{
        display: inline-block; width: 10px; height: 10px;
        border-radius: 50%; margin: 0 3px; vertical-align: middle;
    }}
    .status-dot.green  {{ background: #2a2; }}
    .status-dot.yellow {{ background: #cc0; }}
    .status-dot.red    {{ background: #c33; }}

    .badge {{
        padding: 2px 10px; border-radius: 12px;
        font-size: 0.78em; font-weight: 600;
        text-transform: uppercase; letter-spacing: 0.03em;
        margin-left: auto;
    }}
    .badge + .badge {{ margin-left: 0; }}
    .status-green  {{ background: #1a5e1a; color: #6f6; }}
    .status-yellow {{ background: #5e4e00; color: #ff0; }}
    .status-red    {{ background: #5e1a1a; color: #f66; }}
    .status-error  {{ background: #333; color: #999; }}
    .reason-transition {{ background: #1a3a5e; color: #6af; }}
    .reason-override   {{ background: #5e1a4e; color: #f6f; }}
    .reason-baseline   {{ background: #2a2a3a; color: #aaa; }}

    .card-body {{
        display: flex; gap: 16px;
        padding: 12px 16px; align-items: flex-start;
    }}
    .card-body img {{
        width: 480px; min-width: 480px;
        border-radius: 6px; cursor: pointer; transition: width 0.2s;
    }}
    .card-body img.expanded {{ width: 100%; min-width: 100%; }}

    .info {{ flex: 1; font-size: 0.85em; }}
    .train-summary {{ color: #aaa; margin-bottom: 6px; font-size: 0.95em; }}
    .train-table {{
        width: 100%; border-collapse: collapse;
        margin-bottom: 8px; font-size: 0.9em;
    }}
    .train-table td {{ padding: 2px 8px 2px 0; }}
    .train-table .track-header td {{
        color: #888; font-weight: 600; padding-top: 6px;
        border-bottom: 1px solid #333;
    }}
    .train-table .train-id {{ font-family: monospace; }}
    .train-table .low-conf {{ color: #fa4; }}
    .train-table .low-conf .train-id {{ color: #fa4; font-weight: 700; }}

    .delays {{ list-style: none; padding: 0; margin-bottom: 8px; }}
    .delays li {{ padding: 2px 0; color: #dda; }}
    .delays li::before {{ content: "\\25B8  "; color: #886; }}

    .flag-buttons {{ display: flex; gap: 6px; margin-top: 8px; flex-wrap: wrap; }}
    .flag-btn {{
        background: #3a3a5a; color: #aaa; border: 1px solid #555;
        border-radius: 6px; padding: 4px 12px; cursor: pointer;
        font-size: 0.85em; transition: all 0.15s;
    }}
    .flag-btn:hover {{ background: #4a4a6a; color: #eee; }}
    .flag-btn.active {{ background: #5e1a4e; color: #f6f; border-color: #f6f; }}

    .flag-notes {{
        width: 100%; margin-top: 6px; padding: 6px 10px;
        background: #1e1e38; color: #eee; border: 1px solid #555;
        border-radius: 6px; font-size: 0.85em; resize: vertical;
        min-height: 36px; font-family: inherit;
    }}

    .footer {{
        position: fixed; bottom: 0; left: 0; right: 0;
        background: #16162a; border-top: 1px solid #333;
        padding: 10px 24px; display: flex; align-items: center;
        justify-content: space-between; z-index: 100;
    }}
    .footer .count {{ color: #aaa; font-size: 0.9em; }}
    .footer .count strong {{ color: #f6f; }}
    .export-btn {{
        background: #5e1a4e; color: #fff; border: none;
        border-radius: 8px; padding: 8px 20px; cursor: pointer;
        font-size: 0.9em; font-weight: 600;
    }}
    .export-btn:hover {{ background: #7e2a6e; }}
    .export-btn:disabled {{ background: #333; color: #666; cursor: default; }}

    @media (max-width: 900px) {{
        .card-body {{ flex-direction: column; }}
        .card-body img {{ width: 100%; min-width: 100%; }}
    }}
</style>
</head>
<body>
    <div class="header">
        <h1>Archive Review</h1>
        <div class="meta">
            Generated {html.escape(now)}<br>
            {len(entries)} images: {html.escape(', '.join(reason_parts))}<br>
            Detection: {html.escape(', '.join(status_parts))}
            {f'<br><span class="stat-highlight">{low_conf_total} low-confidence trains</span>' if low_conf_total else ''}
        </div>
    </div>
    {''.join(cards_html) if cards_html else '<p style="text-align:center;color:#888;">No images to display.</p>'}
    <div class="footer">
        <span class="count"><strong id="flag-count">0</strong> images flagged</span>
        <button class="export-btn" id="export-btn" disabled onclick="exportFlagged()">Export Flagged</button>
    </div>
<script>
const flags = {{}};

function toggleCard(i) {{
    document.getElementById('card-' + i).classList.toggle('collapsed');
}}

function toggleFlag(cardIdx, flagType) {{
    const card = document.getElementById('card-' + cardIdx);
    const btns = card.querySelectorAll('.flag-btn');
    const notes = document.getElementById('notes-' + cardIdx);

    // Find the clicked button
    let clickedBtn = null;
    btns.forEach(btn => {{
        if (btn.textContent.toLowerCase().replace(/ /g, '_') === flagType) clickedBtn = btn;
    }});

    if (flags[cardIdx] === flagType) {{
        // Deselect
        delete flags[cardIdx];
        clickedBtn.classList.remove('active');
        card.classList.remove('flagged');
        notes.style.display = 'none';
    }} else {{
        // Select (deselect others first)
        btns.forEach(btn => btn.classList.remove('active'));
        flags[cardIdx] = flagType;
        clickedBtn.classList.add('active');
        card.classList.add('flagged');
        notes.style.display = 'block';
        notes.focus();
    }}
    updateFooter();
}}

function updateFooter() {{
    const count = Object.keys(flags).length;
    document.getElementById('flag-count').textContent = count;
    document.getElementById('export-btn').disabled = count === 0;
}}

function exportFlagged() {{
    const images = [];
    for (const [idx, flagType] of Object.entries(flags)) {{
        const card = document.getElementById('card-' + idx);
        const data = JSON.parse(card.dataset.detection);
        const notes = document.getElementById('notes-' + idx).value;
        images.push({{
            filename: data.filename,
            source_dir: data.source_dir,
            flag: flagType,
            notes: notes,
            detected_status: data.detected_status,
            detected_trains: data.detected_trains,
            detected_delay_summaries: data.detected_delay_summaries,
        }});
    }}
    const exported = {{
        exported_at: new Date().toISOString(),
        images: images,
    }};
    const blob = new Blob([JSON.stringify(exported, null, 2)], {{type: 'application/json'}});
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'flagged-images.json';
    a.click();
    URL.revokeObjectURL(url);
}}
</script>
</body>
</html>'''


def main():
    parser = argparse.ArgumentParser(
        description='Download archived images, run detection, and open an HTML report.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--date', help='Specific date (YYYY-MM-DD)')
    parser.add_argument('--from', dest='from_date', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--to', dest='to_date', help='End date (YYYY-MM-DD)')
    parser.add_argument('--reason', choices=['transition', 'override', 'baseline'],
                        help='Filter by archive reason (default: override)')
    parser.add_argument('--all-reasons', action='store_true',
                        help='Show all reasons instead of defaulting to override')
    parser.add_argument('--skip-download', action='store_true',
                        help='Use already-downloaded images instead of re-downloading')
    parser.add_argument('--output-dir', default=str(DEFAULT_ARCHIVE_DIR),
                        help=f'Image directory (default: {DEFAULT_ARCHIVE_DIR})')
    parser.add_argument('--report', default=str(DEFAULT_REPORT_PATH),
                        help=f'Report output path (default: {DEFAULT_REPORT_PATH})')
    parser.add_argument('--bucket', default=DEFAULT_ARCHIVE_BUCKET,
                        help=f'GCS bucket (default: {DEFAULT_ARCHIVE_BUCKET})')
    parser.add_argument('--no-open', action='store_true',
                        help='Generate report without opening in browser')

    args = parser.parse_args()

    # Default to override reason unless --all-reasons or explicit --reason
    effective_reason = args.reason
    if not args.reason and not args.all_reasons:
        effective_reason = 'override'

    # For downloading, pass the effective reason
    if not args.skip_download:
        args.reason = effective_reason
        run_download(args)

    # Collect images
    image_dir = Path(args.output_dir)
    if not image_dir.exists():
        print(f'No images directory found at {image_dir}')
        sys.exit(1)

    filter_reason = effective_reason if args.skip_download else None
    entries = collect_images(image_dir, reason_filter=filter_reason)
    if not entries:
        print('No archive images found.')
        sys.exit(0)

    print(f'Analyzing {len(entries)} images...')

    # Run detection and encode images
    for i, entry in enumerate(entries):
        name = entry['path'].name
        print(f'  [{i+1}/{len(entries)}] {name}', end='', flush=True)
        entry['detection'] = analyze_image(entry['path'])
        entry['data_uri'] = image_to_data_uri(entry['path'])
        status = entry['detection']['system_status']
        print(f'  -> {status}')

    # Generate report
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(build_html(entries))
    print(f'\nReport saved to {report_path}')

    if not args.no_open:
        webbrowser.open(report_path.as_uri())


if __name__ == '__main__':
    main()
