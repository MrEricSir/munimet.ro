#!/usr/bin/env python3
"""
Flask web UI for train detection diagnostics.

Displays detection results with interactive overlays. Click the image
to inspect pixel HSV values. Expand the contour table to see per-contour
metrics from the symbol detection pipeline.

Usage:
    python scripts/diagnostic_viewer.py                          # test images
    python scripts/diagnostic_viewer.py path/to/image.jpg        # single image
    python scripts/diagnostic_viewer.py --folder path/to/images/  # folder
    python scripts/diagnostic_viewer.py --archive                 # archive
"""

import argparse
import base64
import json
import shutil
import sys
import threading
import webbrowser
from pathlib import Path

from flask import Flask, render_template_string, jsonify, request

sys.path.insert(0, str(Path(__file__).parent.parent))
from lib.baseline_writer import update_baseline_trains, update_baseline_delays, write_baseline_entries
from lib.diagnostics import run_diagnostic, get_pixel_info, encode_mask_overlay

app = Flask(__name__)

IMAGE_FOLDER = Path(__file__).parent.parent / 'tests' / 'images'
image_files = []
single_image_mode = False
_diag_cache = {}


def load_image_files():
    global image_files
    if IMAGE_FOLDER.exists():
        image_files = sorted(
            [f for f in IMAGE_FOLDER.glob('*.jpg')]
            + [f for f in IMAGE_FOLDER.glob('*.JPG')]
        )
    return image_files


def get_cached_diagnostic(image_path):
    key = str(image_path)
    if key not in _diag_cache:
        _diag_cache[key] = run_diagnostic(key)
    return _diag_cache[key]


def image_to_base64(image_path):
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')


HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Detection Diagnostics</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, monospace;
            background: #1a1a2e;
            color: #ddd;
            overflow-x: hidden;
        }

        /* Header bar */
        .header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 8px 16px;
            background: #16213e;
            border-bottom: 1px solid #333;
        }
        .header h1 { font-size: 1.1em; color: #8af; }
        .header .filename { color: #888; font-size: 0.85em; margin-left: 10px; }
        .nav-buttons button {
            background: #4a4a6a; color: white; border: none;
            padding: 5px 12px; margin: 0 2px; cursor: pointer;
            border-radius: 4px; font-size: 0.85em;
        }
        .nav-buttons button:hover { background: #6a6a8a; }
        .nav-buttons button:disabled { opacity: 0.4; cursor: not-allowed; }

        /* Image area — takes most of the screen */
        .image-area {
            position: relative;
            display: flex;
            align-items: flex-start;
            justify-content: center;
            padding: 8px;
            /* Fill available space between header and summary bar */
            height: calc(100vh - 80px);
        }
        .image-area img {
            max-width: 100%;
            max-height: 100%;
            display: block;
            cursor: crosshair;
        }
        .overlay {
            position: absolute;
            top: 8px;
            left: 50%;
            transform: translateX(-50%);
            pointer-events: none;
        }
        .overlay svg { pointer-events: all; cursor: crosshair; }
        .mask-overlay {
            position: absolute;
            top: 8px;
            left: 50%;
            transform: translateX(-50%);
            pointer-events: none;
            opacity: 0.5;
        }

        /* Floating pixel tooltip */
        .pixel-tooltip {
            display: none;
            position: absolute;
            background: rgba(20, 20, 40, 0.95);
            border: 1px solid #555;
            border-radius: 6px;
            padding: 8px 10px;
            font-size: 0.8em;
            line-height: 1.5;
            pointer-events: none;
            z-index: 100;
            min-width: 200px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.5);
        }
        .pixel-tooltip .coord { color: #8af; font-weight: bold; }
        .pixel-swatch {
            display: inline-block; width: 12px; height: 12px;
            border: 1px solid #666; vertical-align: middle; margin-right: 4px;
        }
        .fp { color: #4f4; }
        .ff { color: #f66; }

        /* Summary bar */
        .summary-bar {
            display: flex;
            align-items: center;
            gap: 16px;
            padding: 6px 16px;
            background: #16213e;
            border-top: 1px solid #333;
            font-size: 0.82em;
            flex-wrap: wrap;
        }
        .summary-item { display: flex; align-items: center; gap: 4px; }
        .summary-label { color: #888; }
        .summary-value { color: #ccc; font-weight: bold; }

        .status-dot {
            display: inline-block; width: 10px; height: 10px;
            border-radius: 50%; margin-right: 2px;
        }
        .status-dot.green { background: #4f4; }
        .status-dot.yellow { background: #ff0; }
        .status-dot.red { background: #f44; }

        .baseline-pass { color: #4f4; }
        .baseline-fail { color: #f44; }

        /* Toggle buttons in summary bar */
        .toggle-btn {
            background: #2a2a4a; color: #aaa; border: 1px solid #444;
            padding: 3px 10px; border-radius: 3px; cursor: pointer;
            font-size: 1em;
        }
        .toggle-btn:hover { background: #3a3a5a; color: #ddd; }
        .toggle-btn.active { background: #3a3a6a; color: #fff; border-color: #8af; }

        .summary-sep { color: #333; }

        /* Overlay controls — compact */
        .overlay-controls {
            display: flex; gap: 10px; align-items: center;
            margin-left: auto;
        }
        .overlay-controls label {
            display: flex; align-items: center; gap: 3px;
            cursor: pointer; color: #888;
        }
        .overlay-controls label:hover { color: #ccc; }
        .overlay-controls select {
            background: #2a2a4a; color: #aaa; border: 1px solid #444;
            border-radius: 3px; padding: 2px; font-size: 1em;
        }

        /* Expandable contour drawer */
        .contour-drawer {
            display: none;
            background: #16213e;
            border-top: 1px solid #444;
            max-height: 40vh;
            overflow: hidden;
            flex-direction: column;
        }
        .contour-drawer.open { display: flex; }

        .contour-toolbar {
            display: flex; justify-content: space-between; align-items: center;
            padding: 6px 14px; border-bottom: 1px solid #333;
        }
        .contour-toolbar h3 {
            font-size: 0.82em; color: #8af;
            text-transform: uppercase; letter-spacing: 0.5px;
        }
        .contour-filters { display: flex; gap: 6px; }
        .contour-filters button {
            background: #2a2a4a; color: #aaa; border: 1px solid #444;
            padding: 2px 8px; border-radius: 3px; cursor: pointer; font-size: 0.8em;
        }
        .contour-filters button.active { background: #3a3a6a; color: #fff; border-color: #8af; }
        .contour-filters button:hover { background: #3a3a5a; }

        .contour-table-wrap { overflow: auto; flex: 1; }
        .contour-table {
            width: 100%; border-collapse: collapse; font-size: 0.78em;
        }
        .contour-table th {
            position: sticky; top: 0; background: #1a1a3e; color: #8af;
            padding: 4px 6px; text-align: left; cursor: pointer;
            user-select: none; white-space: nowrap; border-bottom: 1px solid #444;
        }
        .contour-table th:hover { background: #2a2a4e; }
        .contour-table td {
            padding: 3px 6px; white-space: nowrap; border-bottom: 1px solid #222;
        }
        .contour-table tr.accepted { background: rgba(0, 100, 0, 0.12); }
        .contour-table tr.rejected { background: rgba(100, 0, 0, 0.08); }
        .contour-table tr:hover { background: rgba(100, 100, 200, 0.15) !important; }
        .contour-table tr.selected { background: rgba(100, 100, 255, 0.25) !important; }
        .contour-table .pass { color: #4f4; }
        .contour-table .fail { color: #f66; }
        .contour-table .reason { color: #f96; }

        /* Annotation mode buttons */
        .anno-btn {
            background: #2a2a4a; color: #aaa; border: 1px solid #444;
            padding: 3px 10px; border-radius: 3px; cursor: pointer;
            font-size: 1em;
        }
        .anno-btn:hover { background: #3a3a5a; color: #ddd; }
        .anno-btn.active-add { background: #2a4a2a; color: #4f4; border-color: #4f4; }
        .anno-btn.active-flag { background: #4a2a2a; color: #f66; border-color: #f66; }

        /* Corrections panel */
        .corrections-panel {
            position: fixed; right: 0; top: 40px; bottom: 40px;
            width: 280px; background: #16213e; border-left: 1px solid #444;
            display: flex; flex-direction: column; z-index: 200;
            transform: translateX(100%); transition: transform 0.2s;
        }
        .corrections-panel.open { transform: translateX(0); }
        .corrections-panel h3 {
            padding: 10px 14px; font-size: 0.85em; color: #8af;
            border-bottom: 1px solid #333; flex-shrink: 0;
        }
        .corrections-list {
            flex: 1; overflow-y: auto; padding: 6px 10px;
        }
        .corr-item {
            display: flex; align-items: center; justify-content: space-between;
            padding: 5px 6px; margin-bottom: 4px; border-radius: 4px;
            font-size: 0.82em;
        }
        .corr-item.add { background: rgba(0, 100, 0, 0.2); color: #4f4; }
        .corr-item.remove { background: rgba(100, 0, 0, 0.2); color: #f66; }
        .corr-item button {
            background: none; border: none; color: #888; cursor: pointer;
            font-size: 1.1em; padding: 0 4px;
        }
        .corr-item button:hover { color: #fff; }
        .corrections-actions {
            padding: 10px; border-top: 1px solid #333; flex-shrink: 0;
        }
        .corrections-actions button {
            width: 100%; padding: 7px; margin-bottom: 6px; border: none;
            border-radius: 4px; cursor: pointer; font-size: 0.85em;
        }
        .btn-save { background: #2a6a2a; color: #fff; }
        .btn-save:hover { background: #3a8a3a; }
        .btn-clear { background: #4a4a6a; color: #ccc; }
        .btn-clear:hover { background: #6a6a8a; }
        .btn-copy { background: #2a4a6a; color: #ccc; margin-top: 4px; }
        .btn-copy:hover { background: #3a6a8a; }

        .save-result {
            padding: 10px; font-size: 0.82em; color: #ccc;
        }
        .save-result .cmd-box {
            background: #111; padding: 8px; border-radius: 4px;
            margin: 6px 0; font-family: monospace; font-size: 0.95em;
            word-break: break-all; color: #8af;
        }

        /* Add-train inline form */
        .add-form {
            position: absolute; z-index: 150;
            background: rgba(20, 20, 40, 0.95); border: 1px solid #4f4;
            border-radius: 6px; padding: 8px 10px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.5);
        }
        .add-form input {
            background: #1a1a3e; border: 1px solid #555; color: #fff;
            padding: 4px 6px; border-radius: 3px; width: 110px;
            font-family: monospace; font-size: 0.9em;
        }
        .add-form button {
            background: #2a4a2a; color: #4f4; border: 1px solid #4f4;
            padding: 3px 8px; border-radius: 3px; cursor: pointer;
            margin-left: 4px; font-size: 0.85em;
        }
        .add-form .cancel-btn { background: #4a2a2a; color: #f66; border-color: #f66; }

        /* Bounding-box annotation flow */
        .add-status { color: #f90; font-size: 0.85em; margin-left: 8px; }
        .add-form .diag-info { font-size: 0.75em; color: #888; margin: 6px 0; }
        .add-form .diag-info pre { margin: 2px 0; white-space: pre-wrap; }
        .add-form .ocr-preview { max-width: 120px; border: 1px solid #555; margin: 4px 0; }
        .add-form .redraw-btn {
            font-size: 0.75em; color: #8af; cursor: pointer;
            background: none; border: none; text-decoration: underline;
            padding: 0; margin-right: 8px;
        }
        .add-form .redraw-btn:hover { color: #bdf; }
        .add-form .field-row { margin: 4px 0; display: flex; align-items: center; gap: 6px; }
        .add-form .field-row label { color: #888; font-size: 0.8em; min-width: 50px; }
        .image-area.dragging { user-select: none; -webkit-user-select: none; }
    </style>
</head>
<body>

    <!-- Header -->
    <div class="header">
        <div style="display:flex;align-items:center;">
            <h1>Diagnostics</h1>
            <span class="filename">{{ filename }}{% if not single_image_mode %} ({{ index + 1 }}/{{ total }}){% endif %}</span>
        </div>
        {% if not single_image_mode %}
        <div class="nav-buttons">
            <button onclick="navigate(-10)" {{ 'disabled' if index < 10 else '' }}>&lt;&lt;</button>
            <button onclick="navigate(-1)" {{ 'disabled' if index == 0 else '' }}>&lt; Prev</button>
            <button onclick="navigate(1)" {{ 'disabled' if index >= total - 1 else '' }}>Next &gt;</button>
            <button onclick="navigate(10)" {{ 'disabled' if index >= total - 10 else '' }}>&gt;&gt;</button>
        </div>
        {% endif %}
    </div>

    <!-- Image (fills the screen) -->
    <div class="image-area" id="imageArea">
        <img src="data:image/jpeg;base64,{{ image_data }}" id="mainImage"
             onload="setupOverlay()">
        <img id="maskOverlay" class="mask-overlay" style="display:none;">
        <div class="overlay" id="overlay"></div>
        <div class="pixel-tooltip" id="pixelTooltip"></div>
        <div style="position:absolute;bottom:12px;left:50%;transform:translateX(-50%);
            background:rgba(20,20,40,0.92);padding:10px 16px;border-radius:6px;
            font-size:0.8em;color:#ccc;pointer-events:none;line-height:1.7;
            border:1px solid #444;max-width:480px;" id="helpHint">
            <div style="color:#8af;font-weight:bold;margin-bottom:4px;">How to use</div>
            <div><span style="color:#fc4;">Extra train?</span> Click the train label to see why it was detected</div>
            <div><span style="color:#fc4;">Missing train?</span> Use "+ Add Missing" then draw boxes around the icon and label</div>
            <div><span style="color:#fc4;">Curious about a spot?</span> Click it to see its HSV values and filter status</div>
        </div>
    </div>

    <!-- Summary bar -->
    <div class="summary-bar">
        <div class="summary-item">
            <span class="status-dot {{ diag.detection.system_status }}"></span>
            <span class="summary-value">{{ diag.detection.system_status | upper }}</span>
        </div>
        <span class="summary-sep">|</span>
        <div class="summary-item">
            <span class="summary-label">Trains:</span>
            <span class="summary-value">{{ diag.pipeline.merged_train_count }}</span>
            <span class="summary-label" style="margin-left:4px;">({{ diag.pipeline.detection_method }})</span>
        </div>

        {% if diag.baseline_comparison %}
        <span class="summary-sep">|</span>
        <div class="summary-item">
            <span class="summary-label">Baseline:</span>
            {% if diag.baseline_comparison.type == 'train_free' %}
            <span class="summary-value {{ 'baseline-pass' if diag.baseline_comparison.passes else 'baseline-fail' }}">
                {{ 'PASS' if diag.baseline_comparison.passes else 'FAIL' }}
                ({{ diag.baseline_comparison.detected_count }}/{{ diag.baseline_comparison.expected_max_false_positives }} FP)
            </span>
            {% else %}
            <span class="summary-value {{ 'baseline-pass' if not diag.baseline_comparison.missing and not diag.baseline_comparison.extra else 'baseline-fail' }}">
                {{ diag.baseline_comparison.matched | length }}/{{ diag.baseline_comparison.expected_count }} matched
                {% if diag.baseline_comparison.missing %}, {{ diag.baseline_comparison.missing | length }} missing{% endif %}
                {% if diag.baseline_comparison.extra %}, {{ diag.baseline_comparison.extra | length }} extra{% endif %}
            </span>
            {% endif %}
        </div>
        {% endif %}

        {% if diag.detection.delay_summaries %}
        <span class="summary-sep">|</span>
        <div class="summary-item">
            <span class="summary-label">Delays:</span>
            <span class="summary-value" style="color:#fc8;">{{ diag.detection.delay_summaries | length }}</span>
        </div>
        {% endif %}

        <span class="summary-sep">|</span>
        <div class="summary-item">
            <span class="summary-label">Contours:</span>
            <span class="summary-value">{{ diag.summary_stats.accepted }}/{{ diag.summary_stats.total }}</span>
            <button class="toggle-btn" id="contourToggle" onclick="toggleContours()">Show</button>
        </div>

        <span class="summary-sep">|</span>
        <button class="anno-btn" id="btnAddMissing" onclick="setAnnotationMode('add')">+ Add Missing</button>
        <span class="add-status" id="addStatus"></span>
        <button class="anno-btn" id="btnFlagExtra" onclick="setAnnotationMode('flag')">Flag Extra</button>
        <button class="toggle-btn" id="btnCorrections" onclick="toggleCorrections()">Corrections (0)</button>

        <!-- Overlay controls (right-aligned) -->
        <div class="overlay-controls">
            <label><input type="checkbox" id="showTrains" checked onchange="updateOverlay()"> Trains</label>
            <label><input type="checkbox" id="showRejected" onchange="updateOverlay()"> Rejected</label>
            <select id="maskSelect" onchange="toggleMask()">
                <option value="">No mask</option>
                <option value="saturation">Saturation</option>
                <option value="pre_close">Pre-close</option>
                <option value="post_close">Post-close</option>
                <option value="cyan">Cyan excl.</option>
                <option value="red">Red excl.</option>
            </select>
        </div>
    </div>

    <!-- Corrections panel (right side) -->
    <div class="corrections-panel" id="correctionsPanel">
        <h3>Train Corrections</h3>
        <div class="corrections-list" id="correctionsList"></div>
        <div class="corrections-actions" id="correctionsActions">
            <button class="btn-save" onclick="saveCorrections()">Save to Baselines</button>
            <button class="btn-clear" onclick="clearCorrections()">Clear All</button>
        </div>
        <div class="save-result" id="saveResult" style="display:none;"></div>
    </div>

    <!-- Contour drawer (hidden by default) -->
    <div class="contour-drawer" id="contourDrawer">
        <div class="contour-toolbar">
            <h3>Symbol Contours ({{ diag.summary_stats.accepted }} accepted, {{ diag.summary_stats.total - diag.summary_stats.accepted }} rejected)</h3>
            <div class="contour-filters">
                <button class="active" onclick="filterContours('all', this)">All</button>
                <button onclick="filterContours('accepted', this)">Accepted</button>
                <button onclick="filterContours('rejected', this)">Rejected</button>
            </div>
        </div>
        <div class="contour-table-wrap">
            <table class="contour-table" id="contourTable">
                <thead>
                    <tr>
                        <th onclick="sortTable(0)">#</th>
                        <th onclick="sortTable(1)">X</th>
                        <th onclick="sortTable(2)">Y</th>
                        <th onclick="sortTable(3)">W x H</th>
                        <th onclick="sortTable(4)">Area</th>
                        <th onclick="sortTable(5)">Aspect</th>
                        <th onclick="sortTable(6)">Rect</th>
                        <th onclick="sortTable(7)">Fill</th>
                        <th onclick="sortTable(8)">Track</th>
                        <th onclick="sortTable(9)">Status</th>
                        <th>Reason</th>
                    </tr>
                </thead>
                <tbody>
                    {% for c in diag.symbol_contours %}
                    <tr class="{{ 'accepted' if c.accepted else 'rejected' }}"
                        data-accepted="{{ c.accepted | lower }}"
                        data-x="{{ c.x }}" data-y="{{ c.y }}"
                        data-w="{{ c.width }}" data-h="{{ c.height }}"
                        onclick="selectContour(this, {{ c.x }}, {{ c.y }}, {{ c.width }}, {{ c.height }})">
                        <td>{{ loop.index }}</td>
                        <td>{{ c.x }}</td>
                        <td>{{ c.y }}</td>
                        <td>{{ c.width }}x{{ c.height }}</td>
                        <td>{{ c.area | int }}</td>
                        <td>{{ '%.2f' % c.aspect_ratio }}</td>
                        <td>{{ '%.2f' % c.rectangularity }}</td>
                        <td>{{ '%.2f' % c.fill_ratio }}</td>
                        <td>{{ c.track }}</td>
                        <td class="{{ 'pass' if c.accepted else 'fail' }}">{{ 'PASS' if c.accepted else 'FAIL' }}</td>
                        <td class="reason">{{ c.rejection_reason }}</td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>
    </div>

    <script>
        const diagData = {{ diag_json | safe }};
        let imgWidth, imgHeight;
        let selectedContour = null;
        let contourOpen = false;
        let annotationMode = null;  // null, 'add', or 'flag'
        let corrections = [];       // [{type: 'add'|'remove', id, x, y}, ...]
        let addFormEl = null;       // current inline add form element

        // Bounding-box add flow state
        let addStep = null;         // null, 'icon', 'label'
        let dragStart = null;       // {x, y} in image coords
        let iconBox = null;         // {x, y, w, h} in image coords
        let labelBox = null;        // {x, y, w, h} in image coords
        let isDragging = false;

        let _setupRetries = 0;
        function setupOverlay() {
            const img = document.getElementById('mainImage');
            // Firefox may fire onload before layout is complete for data URIs;
            // poll until the image has a rendered size (up to ~500ms)
            if (!img.clientWidth && _setupRetries < 30) {
                _setupRetries++;
                setTimeout(setupOverlay, 16);
                return;
            }
            imgWidth = img.naturalWidth;
            imgHeight = img.naturalHeight;

            const overlay = document.getElementById('overlay');
            overlay.style.width = img.clientWidth + 'px';
            overlay.style.height = img.clientHeight + 'px';

            const maskOv = document.getElementById('maskOverlay');
            maskOv.style.width = img.clientWidth + 'px';
            maskOv.style.height = img.clientHeight + 'px';

            updateOverlay();
        }

        function updateOverlay(dragCurrent) {
            const showTrains = document.getElementById('showTrains').checked;
            const showRejected = document.getElementById('showRejected').checked;
            const pipeline = diagData.pipeline;

            let svg = `<svg viewBox="0 0 ${imgWidth} ${imgHeight}" preserveAspectRatio="xMidYMid meet" width="100%" height="100%">`;

            // Track band outline
            if (pipeline.track_band) {
                svg += `<rect x="0" y="${pipeline.track_band[0]}" width="${imgWidth}" height="${pipeline.track_band[1] - pipeline.track_band[0]}"
                    fill="none" stroke="rgba(100,100,255,0.25)" stroke-width="1" stroke-dasharray="8,4"/>`;
            }

            // Rejected contours (red outlines)
            if (showRejected) {
                diagData.symbol_contours.forEach((c, i) => {
                    if (c.accepted) return;
                    svg += `<rect x="${c.x}" y="${c.y}" width="${c.width}" height="${c.height}"
                        fill="rgba(255,50,50,0.15)" stroke="rgba(255,80,80,0.6)" stroke-width="1"
                        style="cursor:pointer" onclick="selectContourByIndex(${i})"/>`;
                });
            }

            // Accepted symbol contours (green outlines)
            diagData.symbol_contours.forEach((c, i) => {
                if (!c.accepted) return;
                svg += `<rect x="${c.x}" y="${c.y}" width="${c.width}" height="${c.height}"
                    fill="rgba(50,255,50,0.15)" stroke="rgba(80,255,80,0.7)" stroke-width="1.5"
                    style="cursor:pointer" onclick="selectContourByIndex(${i})"/>`;
            });

            // Train labels
            if (showTrains && diagData.detection.trains) {
                diagData.detection.trains.forEach(t => {
                    let fill, stroke;
                    if (t.confidence === 'high') { fill = 'rgba(0,100,0,0.85)'; stroke = '#4f4'; }
                    else if (t.confidence === 'medium') { fill = 'rgba(100,100,0,0.85)'; stroke = '#fc4'; }
                    else { fill = 'rgba(100,50,0,0.85)'; stroke = '#f84'; }
                    const label = t.id.length > 9 ? t.id.substring(0, 9) : t.id;
                    svg += `<g style="cursor:pointer">
                        <rect x="${t.x - 28}" y="${t.y - 9}" width="56" height="18" rx="3"
                              fill="${fill}" stroke="${stroke}" stroke-width="1"/>
                        <text x="${t.x}" y="${t.y + 4}" text-anchor="middle" fill="#fff"
                              font-size="9" font-weight="bold">${label}</text>
                    </g>`;
                });
            }

            // Baseline-only trains (in baseline but not detected — includes manually-added)
            if (showTrains && diagData.baseline_comparison && diagData.baseline_comparison.missing) {
                const detTrains = diagData.detection.trains || [];
                const bc = diagData.baseline_comparison;
                const bandTop = pipeline.track_band ? pipeline.track_band[0] : imgHeight * 0.48;
                const bandBot = pipeline.track_band ? pipeline.track_band[1] : imgHeight * 0.62;
                const upperY = bandTop - 40;
                const lowerY = bandBot + 40;

                // Build set of detected trains already matched to baseline entries
                const matchedDetX = new Set((bc.matched || []).map(m => m.detected_x));

                diagData.baseline_comparison.missing.forEach(t => {
                    // Find nearest UNMATCHED (extra) detected train to infer track
                    let labelY = null;
                    let bestDx = Infinity;
                    detTrains.forEach(dt => {
                        if (matchedDetX.has(dt.x)) return;  // skip matched
                        const dx = Math.abs(dt.x - t.x);
                        if (dx < bestDx && dx <= 60) { bestDx = dx; labelY = dt.y; }
                    });

                    // Fall back: find nearest contour to infer track
                    if (labelY === null) {
                        bestDx = Infinity;
                        diagData.symbol_contours.forEach(c => {
                            if (!c.accepted) return;
                            const cx = c.x + c.width / 2;
                            const dx = Math.abs(cx - t.x);
                            if (dx < bestDx && dx <= 60) {
                                bestDx = dx;
                                const cy = c.y + c.height / 2;
                                labelY = cy < (bandTop + bandBot) / 2 ? upperY : lowerY;
                            }
                        });
                    }

                    // Last resort: upper track (arbitrary, but at least consistent)
                    if (labelY === null) labelY = upperY;

                    // Place label near inferred position, offset to avoid overlap
                    const aboveBand = labelY < (bandTop + bandBot) / 2;
                    const offsetY = aboveBand ? labelY - 16 : labelY + 16;
                    const lineStart = aboveBand ? offsetY + 9 : offsetY - 9;
                    const lineEnd = aboveBand ? labelY : labelY;

                    svg += `<g style="cursor:pointer">
                        <line x1="${t.x}" y1="${lineEnd}" x2="${t.x}" y2="${lineStart}"
                              stroke="rgba(160,100,255,0.4)" stroke-width="1" stroke-dasharray="2,2"/>
                        <rect x="${t.x - 28}" y="${offsetY - 9}" width="56" height="18" rx="3"
                              fill="rgba(80,40,120,0.7)" stroke="#a6f" stroke-width="1" stroke-dasharray="4,2"/>
                        <text x="${t.x}" y="${offsetY + 3}" text-anchor="middle" fill="#caf"
                              font-size="8" font-weight="bold">${t.id}</text>
                        <text x="${t.x}" y="${offsetY + 14}" text-anchor="middle" fill="#86a"
                              font-size="7">baseline</text>
                    </g>`;
                });
            }

            // Correction overlays
            corrections.forEach(c => {
                if (c.type === 'add') {
                    // Orange diamond marker + label
                    const cy = c.y || imgHeight * 0.55;
                    svg += `<polygon points="${c.x},${cy-10} ${c.x+8},${cy} ${c.x},${cy+10} ${c.x-8},${cy}"
                        fill="rgba(255,160,0,0.7)" stroke="#fc4" stroke-width="1.5"/>`;
                    svg += `<text x="${c.x}" y="${cy-14}" text-anchor="middle" fill="#fc4"
                        font-size="9" font-weight="bold">+ ${c.id}</text>`;
                } else if (c.type === 'remove') {
                    // Red strikethrough over the train label
                    const train = (diagData.detection.trains || []).find(t => t.id === c.id && t.x === c.x);
                    if (train) {
                        svg += `<rect x="${train.x - 30}" y="${train.y - 10}" width="60" height="20" rx="3"
                            fill="rgba(180,0,0,0.5)" stroke="#f44" stroke-width="1.5"/>`;
                        svg += `<line x1="${train.x - 28}" y1="${train.y}" x2="${train.x + 28}" y2="${train.y}"
                            stroke="#f44" stroke-width="2"/>`;
                    }
                }
            });

            // Selected contour highlight
            if (selectedContour) {
                const c = selectedContour;
                svg += `<rect x="${c.x - 2}" y="${c.y - 2}" width="${c.w + 4}" height="${c.h + 4}"
                    fill="none" stroke="#ff0" stroke-width="2" stroke-dasharray="4,2"/>`;
            }

            // Bounding-box annotation overlays
            if (iconBox) {
                svg += `<rect x="${iconBox.x}" y="${iconBox.y}" width="${iconBox.w}" height="${iconBox.h}"
                    fill="rgba(255,160,0,0.1)" stroke="#f90" stroke-width="2" stroke-dasharray="6,3"/>`;
            }
            if (labelBox) {
                svg += `<rect x="${labelBox.x}" y="${labelBox.y}" width="${labelBox.w}" height="${labelBox.h}"
                    fill="rgba(80,140,255,0.1)" stroke="#58f" stroke-width="2" stroke-dasharray="6,3"/>`;
            }
            // Connector line between icon and label boxes
            if (iconBox && labelBox) {
                const icx = iconBox.x + iconBox.w / 2, icy = iconBox.y + iconBox.h / 2;
                const lcx = labelBox.x + labelBox.w / 2, lcy = labelBox.y + labelBox.h / 2;
                svg += `<line x1="${icx}" y1="${icy}" x2="${lcx}" y2="${lcy}"
                    stroke="rgba(200,200,200,0.3)" stroke-width="1" stroke-dasharray="4,4"/>`;
            }

            // Live drag preview rectangle
            if (dragStart && isDragging && dragCurrent) {
                const bx = Math.min(dragStart.x, dragCurrent.x);
                const by = Math.min(dragStart.y, dragCurrent.y);
                const bw = Math.abs(dragCurrent.x - dragStart.x);
                const bh = Math.abs(dragCurrent.y - dragStart.y);
                const color = addStep === 'icon' ? '#f90' : '#58f';
                svg += `<rect x="${bx}" y="${by}" width="${bw}" height="${bh}"
                    fill="none" stroke="${color}" stroke-width="2" stroke-dasharray="4,4" opacity="0.8"/>`;
            }

            svg += '</svg>';
            document.getElementById('overlay').innerHTML = svg;
        }

        function findNearby(x, y, radius) {
            // Find ALL contours and trains within radius of click, sorted by distance
            const contours = [];
            diagData.symbol_contours.forEach((c, i) => {
                const cx = c.x + c.width / 2, cy = c.y + c.height / 2;
                const d = Math.sqrt((cx - x) ** 2 + (cy - y) ** 2);
                if (d <= radius) contours.push({...c, _idx: i, _dist: d});
            });
            contours.sort((a, b) => a._dist - b._dist);

            const trains = [];
            (diagData.detection.trains || []).forEach(t => {
                const d = Math.sqrt((t.x - x) ** 2 + (t.y - y) ** 2);
                if (d <= radius) trains.push({...t, _dist: d});
            });
            trains.sort((a, b) => a._dist - b._dist);

            return { contours, trains };
        }

        function getImageCoords(event) {
            const img = document.getElementById('mainImage');
            const imgRect = img.getBoundingClientRect();
            const sx = imgWidth / img.clientWidth;
            const sy = imgHeight / img.clientHeight;
            return {
                x: Math.round((event.clientX - imgRect.left) * sx),
                y: Math.round((event.clientY - imgRect.top) * sy),
            };
        }

        function clampBox(box) {
            const x = Math.max(0, Math.min(box.x, imgWidth - 1));
            const y = Math.max(0, Math.min(box.y, imgHeight - 1));
            const w = Math.min(box.w, imgWidth - x);
            const h = Math.min(box.h, imgHeight - y);
            return {x, y, w, h};
        }

        function onImageMouseDown(event) {
            // Ignore if clicking on form elements
            if (addFormEl && addFormEl.contains(event.target)) return;

            const coords = getImageCoords(event);
            if (coords.x < 0 || coords.x >= imgWidth || coords.y < 0 || coords.y >= imgHeight) return;

            // Hide help hint
            const hint = document.getElementById('helpHint');
            if (hint) hint.style.display = 'none';

            if (annotationMode === 'add' && (addStep === 'icon' || addStep === 'label')) {
                event.preventDefault();
                dragStart = coords;
                isDragging = false;
                return;
            }
        }

        function onImageMouseMove(event) {
            if (!dragStart || annotationMode !== 'add') return;

            const coords = getImageCoords(event);
            const dx = Math.abs(coords.x - dragStart.x);
            const dy = Math.abs(coords.y - dragStart.y);

            if (dx > 3 || dy > 3) {
                isDragging = true;
                // Update drag preview in overlay
                updateOverlay(coords);
            }
        }

        function onImageMouseUp(event) {
            if (annotationMode === 'add' && dragStart) {
                const coords = getImageCoords(event);
                const dx = Math.abs(coords.x - dragStart.x);
                const dy = Math.abs(coords.y - dragStart.y);

                if (dx < 5 && dy < 5) {
                    // Too small — treat as tap
                    dragStart = null;
                    isDragging = false;
                    showTapHint(event);
                    return;
                }

                // Finalize the box
                const box = clampBox({
                    x: Math.min(dragStart.x, coords.x),
                    y: Math.min(dragStart.y, coords.y),
                    w: Math.abs(coords.x - dragStart.x),
                    h: Math.abs(coords.y - dragStart.y),
                });

                if (addStep === 'icon') {
                    iconBox = box;
                    addStep = 'label';
                    updateAddStatus();
                } else if (addStep === 'label') {
                    labelBox = box;
                    addStep = 'confirm';
                    updateAddStatus();
                    showBBoxConfirmForm(event);
                }

                dragStart = null;
                isDragging = false;
                updateOverlay();
                return;
            }

            // Non-add-drag clicks: dispatch normally
            if (dragStart) { dragStart = null; isDragging = false; return; }
            handleClick(event);
        }

        function showTapHint(event) {
            const tooltip = document.getElementById('pixelTooltip');
            const area = document.getElementById('imageArea');
            const areaRect = area.getBoundingClientRect();
            tooltip.style.left = (event.clientX - areaRect.left + 16) + 'px';
            tooltip.style.top = (event.clientY - areaRect.top + 8) + 'px';
            tooltip.style.display = 'block';
            tooltip.innerHTML = '<span style="color:#f90;">Drag to draw a box (click and hold, then drag)</span>';
            setTimeout(() => { tooltip.style.display = 'none'; }, 2000);
        }

        function handleClick(event) {
            const coords = getImageCoords(event);
            const x = coords.x, y = coords.y;
            if (x < 0 || x >= imgWidth || y < 0 || y >= imgHeight) return;

            // Dismiss any open add form if clicking elsewhere
            if (addFormEl && !addFormEl.contains(event.target)) {
                addFormEl.remove();
                addFormEl = null;
            }

            if (annotationMode === 'flag') {
                flagNearestTrain(x, y);
                return;
            }

            // Default: pixel inspection
            const tooltip = document.getElementById('pixelTooltip');
            const area = document.getElementById('imageArea');
            const areaRect = area.getBoundingClientRect();
            let tipLeft = event.clientX - areaRect.left + 16;
            if (tipLeft + 220 > area.clientWidth) {
                tipLeft = event.clientX - areaRect.left - 230;
            }
            tooltip.style.left = tipLeft + 'px';
            tooltip.style.top = (event.clientY - areaRect.top + 8) + 'px';
            tooltip.style.display = 'block';

            // Find nearby context (client-side, instant)
            const nearby = findNearby(x, y, 150);
            let contextHtml = '';

            if (nearby.trains.length > 0) {
                contextHtml += `<div style="margin-top:6px;padding-top:5px;border-top:1px solid #444;">
                    <b style="color:#4f4;">Detected train${nearby.trains.length > 1 ? 's' : ''} nearby</b>`;
                nearby.trains.forEach(t => {
                    const confColor = t.confidence === 'high' ? '#4f4' : t.confidence === 'medium' ? '#fc4' : '#f84';
                    contextHtml += `<br>${t.id} <span style="color:${confColor}">(${t.confidence})</span> at x=${t.x}`;
                });
                contextHtml += '</div>';
            }

            if (nearby.contours.length > 0) {
                const c = nearby.contours[0];
                if (c.accepted) {
                    contextHtml += `<div style="margin-top:6px;padding-top:5px;border-top:1px solid #444;">
                        <b style="color:#4f4;">Symbol contour (accepted)</b><br>
                        ${c.width}x${c.height} at (${c.x},${c.y}), fill=${c.fill_ratio.toFixed(2)}
                    </div>`;
                } else {
                    contextHtml += `<div style="margin-top:6px;padding-top:5px;border-top:1px solid #444;">
                        <b style="color:#f66;">Rejected contour nearby</b><br>
                        ${c.width}x${c.height} at (${c.x},${c.y})<br>
                        <span style="color:#f96;">Rejected: ${c.rejection_reason}</span><br>
                        <span style="color:#888;">fill=${c.fill_ratio.toFixed(2)}, aspect=${c.aspect_ratio.toFixed(2)}, area=${Math.round(c.area)}</span>
                    </div>`;
                }
                selectedContour = {x: c.x, y: c.y, w: c.width, h: c.height};
                updateOverlay();
            }

            if (nearby.trains.length === 0 && nearby.contours.length === 0) {
                contextHtml += `<div style="margin-top:6px;padding-top:5px;border-top:1px solid #444;color:#888;">
                    No detected trains or contours within 150px.<br>
                    If a train should be here, it was likely missed by OCR.
                    Check if the pixel above is a candidate &mdash; if not, that's why.
                </div>`;
            }

            tooltip.innerHTML = '<span style="color:#888">Loading...</span>';
            fetch(`/api/pixel/{{ index }}?x=${x}&y=${y}`)
                .then(r => r.json())
                .then(p => {
                    if (p.error) { tooltip.innerHTML = `<span class="ff">${p.error}</span>`; return; }
                    const ic = v => v ? 'fp' : 'ff';
                    const ck = v => v ? '\\u2713' : '\\u2717';
                    const reason = !p.in_track_band ? 'outside track band' :
                        !p.passes_saturation ? 'saturation too low (' + p.hsv.s + ' < 150)' :
                        p.in_cyan_range ? 'excluded as cyan UI element' :
                        p.in_red_range ? 'excluded as red UI element' : '';
                    tooltip.innerHTML = `
                        <div><span class="coord">(${p.x}, ${p.y})</span>
                            <span class="pixel-swatch" style="background:${p.hex};"></span>
                            HSV: ${p.hsv.h}, ${p.hsv.s}, ${p.hsv.v}</div>
                        <div class="${ic(p.would_be_candidate)}" style="font-weight:bold">
                            ${ck(p.would_be_candidate)} ${p.would_be_candidate ? 'Detector sees this pixel' : 'Detector ignores this pixel'}
                            ${reason ? '<br><span style="font-weight:normal;color:#f96;">' + reason + '</span>' : ''}
                        </div>
                        ${contextHtml}`;
                });
        }

        // --- Annotation mode ---

        function setAnnotationMode(mode) {
            // Clean up any in-progress add flow
            resetAddFlow();

            if (annotationMode === mode) {
                annotationMode = null;
            } else {
                annotationMode = mode;
            }
            document.getElementById('btnAddMissing').className = 'anno-btn' + (annotationMode === 'add' ? ' active-add' : '');
            document.getElementById('btnFlagExtra').className = 'anno-btn' + (annotationMode === 'flag' ? ' active-flag' : '');
            // Change cursor
            const img = document.getElementById('mainImage');
            const svgs = document.querySelectorAll('.overlay svg');
            const area = document.getElementById('imageArea');
            if (annotationMode === 'add') {
                img.style.cursor = 'crosshair';
                svgs.forEach(s => s.style.cursor = 'crosshair');
                area.classList.add('dragging');
                addStep = 'icon';
                updateAddStatus();
            } else if (annotationMode === 'flag') {
                img.style.cursor = 'not-allowed';
                svgs.forEach(s => s.style.cursor = 'not-allowed');
                area.classList.remove('dragging');
            } else {
                img.style.cursor = 'crosshair';
                svgs.forEach(s => s.style.cursor = 'crosshair');
                area.classList.remove('dragging');
            }
            // Dismiss tooltip
            document.getElementById('pixelTooltip').style.display = 'none';
        }

        function resetAddFlow() {
            addStep = null;
            dragStart = null;
            iconBox = null;
            labelBox = null;
            isDragging = false;
            if (addFormEl) { addFormEl.remove(); addFormEl = null; }
            document.getElementById('addStatus').textContent = '';
            updateOverlay();
        }

        function updateAddStatus() {
            const el = document.getElementById('addStatus');
            if (addStep === 'icon') {
                el.textContent = 'Step 1: Draw a box around the train icon on the track';
            } else if (addStep === 'label') {
                el.textContent = 'Step 2: Draw a box around the train\\'s text label';
            } else if (addStep === 'confirm') {
                el.textContent = 'Step 3: Review and confirm';
            } else {
                el.textContent = '';
            }
        }

        function showBBoxConfirmForm(event) {
            if (addFormEl) { addFormEl.remove(); addFormEl = null; }

            const area = document.getElementById('imageArea');
            const areaRect = area.getBoundingClientRect();
            const img = document.getElementById('mainImage');
            const imgRect = img.getBoundingClientRect();
            const scaleX = img.clientWidth / imgWidth;

            // Position form near the label box
            const formX = imgRect.left - areaRect.left + (labelBox.x + labelBox.w) * scaleX + 12;
            const formY = imgRect.top - areaRect.top + labelBox.y * (img.clientHeight / imgHeight) - 10;

            const trackX = iconBox.x + Math.round(iconBox.w / 2);
            const trackY = iconBox.y + Math.round(iconBox.h / 2);

            const form = document.createElement('div');
            form.className = 'add-form';
            form.style.left = Math.min(formX, areaRect.width - 280) + 'px';
            form.style.top = Math.max(10, formY) + 'px';
            form.style.minWidth = '240px';
            form.innerHTML = `
                <div style="color:#4f4;font-size:0.8em;margin-bottom:6px;font-weight:bold;">Add Missing Train</div>
                <div class="field-row">
                    <label>Train ID:</label>
                    <input type="text" id="addTrainId" placeholder="Loading OCR..." style="width:120px;">
                </div>
                <div class="field-row">
                    <label>Track x:</label>
                    <input type="text" id="addTrackX" value="${trackX}" style="width:70px;">
                </div>
                <div class="diag-info" id="addDiagInfo">
                    <div style="color:#888;">Running OCR on label region...</div>
                </div>
                <div style="margin-top:6px;">
                    <button onclick="confirmBBoxAdd()">OK</button>
                    <button class="cancel-btn" onclick="cancelBBoxAdd()">Cancel</button>
                    <span style="margin-left:8px;">
                        <button class="redraw-btn" onclick="redrawStep('icon')">Redraw Icon</button>
                        <button class="redraw-btn" onclick="redrawStep('label')">Redraw Label</button>
                    </span>
                </div>
            `;
            form.onclick = e => e.stopPropagation();
            form.onmousedown = e => e.stopPropagation();
            area.appendChild(form);
            addFormEl = form;

            // Run OCR on label region
            fetch(`/api/ocr-region/{{ index }}`, {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(labelBox),
            })
            .then(r => r.json())
            .then(data => {
                const input = document.getElementById('addTrainId');
                if (!input) return;

                // Pre-fill with first OCR result
                if (data.train_ids && data.train_ids.length > 0) {
                    input.value = data.train_ids[0];
                } else {
                    input.placeholder = 'Type train ID';
                }
                input.focus();
                input.select();

                // Show diagnostics
                const diagEl = document.getElementById('addDiagInfo');
                if (!diagEl) return;
                let html = '';
                if (data.raw_text) {
                    html += `<div>OCR raw: <code>${data.raw_text}</code></div>`;
                }
                if (data.train_ids && data.train_ids.length > 1) {
                    html += `<div>All IDs found: ${data.train_ids.join(', ')}</div>`;
                }
                if (data.binarized) {
                    html += `<div>Binarized: <img class="ocr-preview" src="data:image/png;base64,${data.binarized}"></div>`;
                }
                diagEl.innerHTML = html || '<div>No OCR text detected</div>';
            })
            .catch(() => {
                const diagEl = document.getElementById('addDiagInfo');
                if (diagEl) diagEl.innerHTML = '<div style="color:#f66;">OCR request failed</div>';
                const input = document.getElementById('addTrainId');
                if (input) { input.placeholder = 'Type train ID'; input.focus(); }
            });

            // Also fetch icon diagnostics
            fetch(`/api/icon-diagnostics/{{ index }}`, {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(iconBox),
            })
            .then(r => r.json())
            .then(data => {
                const diagEl = document.getElementById('addDiagInfo');
                if (!diagEl) return;
                if (data.contours && data.contours.length > 0) {
                    let html = diagEl.innerHTML;
                    html += `<div style="margin-top:4px;border-top:1px solid #444;padding-top:4px;">`;
                    html += `<b>Contours near icon (${data.contours.length}):</b>`;
                    data.contours.forEach(c => {
                        const status = c.accepted ? '<span style="color:#4f4">accepted</span>' : `<span style="color:#f66">rejected: ${c.rejection_reason}</span>`;
                        html += `<div style="margin-left:8px;">${c.width}x${c.height} at (${c.x},${c.y}) fill=${c.fill_ratio.toFixed(2)} ${status}</div>`;
                    });
                    html += '</div>';
                    diagEl.innerHTML = html;
                }
            })
            .catch(() => {});

            // Keyboard handlers
            setTimeout(() => {
                const input = document.getElementById('addTrainId');
                if (input) {
                    input.addEventListener('keydown', e => {
                        if (e.key === 'Enter') confirmBBoxAdd();
                        if (e.key === 'Escape') cancelBBoxAdd();
                    });
                }
            }, 100);
        }

        function confirmBBoxAdd() {
            const idInput = document.getElementById('addTrainId');
            const xInput = document.getElementById('addTrackX');
            const id = idInput ? idInput.value.trim().replace(/[*]/g, '') : '';
            const x = xInput ? parseInt(xInput.value, 10) : (iconBox.x + Math.round(iconBox.w / 2));
            if (!id) { if (idInput) idInput.focus(); return; }

            const y = iconBox.y + Math.round(iconBox.h / 2);
            corrections.push({type: 'add', id, x, y});
            if (addFormEl) { addFormEl.remove(); addFormEl = null; }

            // Reset add flow and go back to icon step for next train
            iconBox = null;
            labelBox = null;
            addStep = 'icon';
            updateAddStatus();
            updateCorrectionsUI();
            updateOverlay();
        }

        function cancelBBoxAdd() {
            resetAddFlow();
            annotationMode = null;
            document.getElementById('btnAddMissing').className = 'anno-btn';
            document.getElementById('imageArea').classList.remove('dragging');
            const img = document.getElementById('mainImage');
            img.style.cursor = 'crosshair';
            document.querySelectorAll('.overlay svg').forEach(s => s.style.cursor = 'crosshair');
        }

        function redrawStep(step) {
            if (step === 'icon') {
                iconBox = null;
                labelBox = null;
                addStep = 'icon';
            } else {
                labelBox = null;
                addStep = 'label';
            }
            if (addFormEl) { addFormEl.remove(); addFormEl = null; }
            updateAddStatus();
            updateOverlay();
        }

        function flagNearestTrain(x, y) {
            // Search both detected trains and baseline-only (missing) trains
            const candidates = [];
            (diagData.detection.trains || []).forEach(t => {
                const d = Math.sqrt((t.x - x) ** 2 + ((t.y || imgHeight * 0.55) - y) ** 2);
                if (d <= 80) candidates.push({id: t.id, x: t.x, y: t.y || 0, _dist: d});
            });
            if (diagData.baseline_comparison && diagData.baseline_comparison.missing) {
                // Use a generous radius — baseline labels may be offset from where user clicks
                diagData.baseline_comparison.missing.forEach(t => {
                    // Match by x proximity only (within tolerance) since y is approximate
                    const dx = Math.abs(t.x - x);
                    if (dx <= 40) candidates.push({id: t.id, x: t.x, y: 0, _dist: dx});
                });
            }
            candidates.sort((a, b) => a._dist - b._dist);
            if (candidates.length === 0) return;

            const t = candidates[0];
            // Don't flag the same train twice
            if (corrections.some(c => c.type === 'remove' && c.id === t.id && c.x === t.x)) return;

            corrections.push({type: 'remove', id: t.id, x: t.x, y: t.y});
            updateCorrectionsUI();
            updateOverlay();
        }

        function removeCorrection(idx) {
            corrections.splice(idx, 1);
            updateCorrectionsUI();
            updateOverlay();
        }

        function clearCorrections() {
            corrections = [];
            updateCorrectionsUI();
            updateOverlay();
            document.getElementById('saveResult').style.display = 'none';
            document.getElementById('correctionsActions').style.display = '';
        }

        function updateCorrectionsUI() {
            const btn = document.getElementById('btnCorrections');
            btn.textContent = 'Corrections (' + corrections.length + ')';
            if (corrections.length > 0) btn.classList.add('active');
            else btn.classList.remove('active');

            const list = document.getElementById('correctionsList');
            if (corrections.length === 0) {
                list.innerHTML = '<div style="color:#666;padding:20px 10px;text-align:center;font-size:0.82em;">No corrections yet.<br>Use "+ Add Missing" or "Flag Extra" to annotate.</div>';
                return;
            }

            list.innerHTML = corrections.map((c, i) => {
                const prefix = c.type === 'add' ? '+' : '\u2212';
                return `<div class="corr-item ${c.type}">
                    <span>${prefix} ${c.id} at x=${c.x}</span>
                    <button onclick="removeCorrection(${i})" title="Remove">\u00d7</button>
                </div>`;
            }).join('');
        }

        function toggleCorrections() {
            const panel = document.getElementById('correctionsPanel');
            panel.classList.toggle('open');
        }

        function saveCorrections() {
            if (corrections.length === 0) return;
            const btn = document.querySelector('.btn-save');
            btn.textContent = 'Saving...';
            btn.disabled = true;

            fetch('/api/save-corrections/{{ index }}', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({corrections}),
            })
            .then(r => r.json())
            .then(data => {
                if (data.error) {
                    alert('Error: ' + data.error);
                    btn.textContent = 'Save to Baselines';
                    btn.disabled = false;
                    return;
                }

                // Show success
                document.getElementById('correctionsActions').style.display = 'none';
                const result = document.getElementById('saveResult');
                result.style.display = 'block';
                result.innerHTML = `
                    <div style="color:#4f4;font-weight:bold;">Saved ${data.trains_saved} trains</div>
                    ${data.image_copied ? '<div style="color:#ccc;">Image copied to tests/images/</div>' : ''}
                    <div style="margin-top:8px;color:#888;">Run next:</div>
                    <div class="cmd-box" id="pytestCmd">${data.pytest_cmd}</div>
                    <button class="btn-copy" onclick="copyCommand()">Copy Command</button>
                    <button class="btn-clear" style="margin-top:6px;" onclick="location.reload()">New Corrections</button>
                `;
                corrections = [];
                updateCorrectionsUI();
            })
            .catch(err => {
                alert('Error: ' + err.message);
                btn.textContent = 'Save to Baselines';
                btn.disabled = false;
            });
        }

        function copyCommand() {
            const cmd = document.getElementById('pytestCmd');
            if (cmd) {
                navigator.clipboard.writeText(cmd.textContent).then(() => {
                    const btn = document.querySelector('.btn-copy');
                    btn.textContent = 'Copied!';
                    setTimeout(() => btn.textContent = 'Copy Command', 1500);
                });
            }
        }

        function toggleMask() {
            const name = document.getElementById('maskSelect').value;
            const maskImg = document.getElementById('maskOverlay');
            if (!name) { maskImg.style.display = 'none'; return; }
            fetch(`/api/mask/{{ index }}/${name}`)
                .then(r => r.json())
                .then(data => {
                    if (data.data) {
                        maskImg.src = 'data:image/png;base64,' + data.data;
                        maskImg.style.display = 'block';
                    }
                });
        }

        function toggleContours() {
            contourOpen = !contourOpen;
            const drawer = document.getElementById('contourDrawer');
            const btn = document.getElementById('contourToggle');
            const area = document.getElementById('imageArea');
            drawer.classList.toggle('open', contourOpen);
            btn.classList.toggle('active', contourOpen);
            btn.textContent = contourOpen ? 'Hide' : 'Show';
            // Shrink image area when drawer is open
            area.style.height = contourOpen ? 'calc(100vh - 80px - 40vh)' : 'calc(100vh - 80px)';
            setupOverlay();
        }

        function selectContour(row, x, y, w, h) {
            document.querySelectorAll('.contour-table tr.selected').forEach(r => r.classList.remove('selected'));
            row.classList.add('selected');
            selectedContour = {x, y, w, h};
            updateOverlay();
        }

        function selectContourByIndex(idx) {
            // Open drawer if closed
            if (!contourOpen) toggleContours();
            const rows = document.querySelectorAll('.contour-table tbody tr');
            const c = diagData.symbol_contours[idx];
            for (const row of rows) {
                if (row.style.display === 'none') continue;
                if (parseInt(row.dataset.x) === c.x && parseInt(row.dataset.y) === c.y) {
                    selectContour(row, c.x, c.y, c.width, c.height);
                    row.scrollIntoView({block: 'nearest'});
                    return;
                }
            }
        }

        function filterContours(filter, btn) {
            document.querySelectorAll('.contour-filters button').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            document.querySelectorAll('.contour-table tbody tr').forEach(row => {
                const accepted = row.dataset.accepted === 'true';
                if (filter === 'all') row.style.display = '';
                else if (filter === 'accepted') row.style.display = accepted ? '' : 'none';
                else row.style.display = accepted ? 'none' : '';
            });
        }

        function sortTable(colIdx) {
            const tbody = document.querySelector('.contour-table tbody');
            const rows = Array.from(tbody.querySelectorAll('tr'));
            const val = (row, i) => { const t = row.cells[i].textContent.trim(); const n = parseFloat(t); return isNaN(n) ? t : n; };
            const th = document.querySelectorAll('.contour-table th')[colIdx];
            const asc = th.dataset.sort !== 'asc';
            document.querySelectorAll('.contour-table th').forEach(h => delete h.dataset.sort);
            th.dataset.sort = asc ? 'asc' : 'desc';
            rows.sort((a, b) => {
                const va = val(a, colIdx), vb = val(b, colIdx);
                if (typeof va === 'number' && typeof vb === 'number') return asc ? va - vb : vb - va;
                return asc ? String(va).localeCompare(String(vb)) : String(vb).localeCompare(String(va));
            });
            rows.forEach(row => tbody.appendChild(row));
        }

        function navigate(delta) { window.location.href = '/?index=' + ({{ index }} + delta); }

        // Register mouse event handlers on image area
        const imageArea = document.getElementById('imageArea');
        imageArea.addEventListener('mousedown', onImageMouseDown);
        imageArea.addEventListener('mousemove', onImageMouseMove);
        imageArea.addEventListener('mouseup', onImageMouseUp);

        // Dismiss tooltip on click outside image
        document.addEventListener('click', e => {
            if (!document.getElementById('imageArea').contains(e.target))
                document.getElementById('pixelTooltip').style.display = 'none';
        });

        // Escape key: cancel add flow or annotation mode
        document.addEventListener('keydown', e => {
            if (e.key === 'Escape') {
                if (annotationMode === 'add') {
                    cancelBBoxAdd();
                } else if (annotationMode === 'flag') {
                    setAnnotationMode('flag'); // toggles off
                }
            }
        });

        window.addEventListener('resize', setupOverlay);

        // Navigation guard: warn before leaving with unsaved corrections
        window.addEventListener('beforeunload', e => {
            if (corrections.length > 0) {
                e.preventDefault();
                e.returnValue = '';
            }
        });

        // Initialize corrections UI
        updateCorrectionsUI();
    </script>
</body>
</html>
'''


@app.route('/')
def index():
    if not image_files:
        load_image_files()
    if not image_files:
        return 'No images found in ' + str(IMAGE_FOLDER)

    idx = request.args.get('index', 0, type=int)
    idx = max(0, min(idx, len(image_files) - 1))

    image_path = image_files[idx]
    diag = get_cached_diagnostic(image_path)
    image_data = image_to_base64(image_path)

    return render_template_string(
        HTML_TEMPLATE,
        filename=image_path.name,
        index=idx,
        total=len(image_files),
        image_data=image_data,
        diag=diag,
        diag_json=json.dumps(diag, default=str),
        single_image_mode=single_image_mode,
    )


@app.route('/api/diagnose/<int:idx>')
def api_diagnose(idx):
    if not image_files:
        load_image_files()
    if idx < 0 or idx >= len(image_files):
        return jsonify({'error': 'Invalid index'}), 400
    return jsonify(get_cached_diagnostic(image_files[idx]))


@app.route('/api/pixel/<int:idx>')
def api_pixel(idx):
    if not image_files:
        load_image_files()
    if idx < 0 or idx >= len(image_files):
        return jsonify({'error': 'Invalid index'}), 400
    x = request.args.get('x', type=int)
    y = request.args.get('y', type=int)
    if x is None or y is None:
        return jsonify({'error': 'x and y required'}), 400
    return jsonify(get_pixel_info(str(image_files[idx]), x, y))


@app.route('/api/mask/<int:idx>/<mask_name>')
def api_mask(idx, mask_name):
    if not image_files:
        load_image_files()
    if idx < 0 or idx >= len(image_files):
        return jsonify({'error': 'Invalid index'}), 400
    data = encode_mask_overlay(str(image_files[idx]), mask_name)
    if not data:
        return jsonify({'error': f'Unknown mask: {mask_name}'}), 400
    return jsonify({'data': data})


@app.route('/api/ocr-region/<int:idx>', methods=['POST'])
def api_ocr_region(idx):
    """Run OCR on a user-selected image region."""
    import cv2
    try:
        import pytesseract
    except ImportError:
        return jsonify({'error': 'pytesseract not installed', 'train_ids': [], 'raw_text': ''})

    if not image_files:
        load_image_files()
    if idx < 0 or idx >= len(image_files):
        return jsonify({'error': 'Invalid index'}), 400

    body = request.get_json()
    x, y, w, h = body['x'], body['y'], body['w'], body['h']

    img = cv2.imread(str(image_files[idx]))
    if img is None:
        return jsonify({'error': 'Could not read image', 'train_ids': [], 'raw_text': ''})

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_h, img_w = gray.shape[:2]
    # Clamp to image bounds
    x, y = max(0, x), max(0, y)
    w = min(w, img_w - x)
    h = min(h, img_h - y)
    roi = gray[y:y+h, x:x+w]

    # Run OCR using TrainDetector
    from lib.train_detector import TrainDetector
    td = TrainDetector()
    train_ids = td._ocr_roi(roi, 'upper')

    # Also get raw OCR text for diagnostics
    scale = 4
    roi_large = cv2.resize(roi, None, fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)
    _, roi_bin = cv2.threshold(roi_large, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    try:
        raw_text = pytesseract.image_to_string(roi_bin, config='--oem 3 --psm 6')
    except Exception:
        raw_text = ''

    # Encode binarized image for preview
    _, buffer = cv2.imencode('.png', roi_bin)
    binarized_b64 = base64.b64encode(buffer).decode()

    return jsonify({
        'train_ids': train_ids,
        'raw_text': raw_text.strip(),
        'binarized': binarized_b64,
    })


@app.route('/api/icon-diagnostics/<int:idx>', methods=['POST'])
def api_icon_diagnostics(idx):
    """Get diagnostic info about a user-selected icon region."""
    if not image_files:
        load_image_files()
    if idx < 0 or idx >= len(image_files):
        return jsonify({'error': 'Invalid index'}), 400

    body = request.get_json()
    x, y, w, h = body['x'], body['y'], body['w'], body['h']
    cx = x + w // 2

    diag = get_cached_diagnostic(image_files[idx])
    nearby = []
    for c in diag.get('symbol_contours', []):
        c_cx = c['x'] + c['width'] // 2
        if abs(c_cx - cx) < 40:
            nearby.append(c)

    return jsonify({
        'contours': nearby,
        'center_x': cx,
        'center_y': y + h // 2,
    })


@app.route('/api/images')
def api_images():
    if not image_files:
        load_image_files()
    return jsonify([{'index': i, 'name': f.name} for i, f in enumerate(image_files)])


@app.route('/api/save-corrections/<int:idx>', methods=['POST'])
def api_save_corrections(idx):
    if not image_files:
        load_image_files()
    if idx < 0 or idx >= len(image_files):
        return jsonify({'error': 'Invalid index'}), 400

    body = request.get_json()
    corrections = body.get('corrections', [])
    if not corrections:
        return jsonify({'error': 'No corrections provided'}), 400

    image_path = image_files[idx]
    image_name = image_path.name

    # Get current detection results (for fallback and delay info)
    diag = get_cached_diagnostic(image_path)
    detected = diag['detection'].get('trains', [])
    status = diag['detection'].get('system_status', 'green')
    summaries = diag['detection'].get('delay_summaries', [])

    # Start from existing baseline if available, otherwise use detection
    # Preserve full entries (including override/optional metadata)
    baseline_comp = diag.get('baseline_comparison')
    tests_dir = Path(__file__).resolve().parent.parent / 'tests'
    if baseline_comp and baseline_comp.get('type') == 'trains_expected':
        baseline_data = json.loads((tests_dir / 'baseline_trains.json').read_text())
        entries = list(baseline_data.get('images_with_trains', {}).get(image_name, []))
    else:
        entries = [[t['id'], t['x']] for t in detected]

    # Apply corrections (remove matches by x proximity since baseline
    # and detection x-coords may differ slightly)
    tolerance = 30
    for corr in corrections:
        if corr['type'] == 'remove':
            entries = [e for e in entries if abs(e[1] - corr['x']) > tolerance]
        elif corr['type'] == 'add':
            entries.append([corr['id'], corr['x']])

    entries.sort(key=lambda e: e[1])

    # Copy image to tests/images/ if not already there
    tests_images = Path(__file__).resolve().parent.parent / 'tests' / 'images'
    dest = tests_images / image_name
    image_copied = False
    if not dest.exists():
        shutil.copy2(str(image_path), str(dest))
        image_copied = True

    # Write train baseline (preserving override/optional metadata)
    count = write_baseline_entries(image_name, entries)

    # Only write delay baseline if no entry exists yet
    delay_data = json.loads(
        (Path(__file__).resolve().parent.parent / 'tests' / 'baseline_delay_summaries.json').read_text()
    )
    if image_name not in delay_data.get('images', {}):
        update_baseline_delays(image_name, status, summaries)

    # Invalidate diagnostic cache so next load reflects changes
    _diag_cache.pop(str(image_path), None)

    pytest_cmd = '.venv/bin/python -m pytest tests/test_train_detection.py -v -k ' + image_name.replace('.jpg', '')

    return jsonify({
        'success': True,
        'trains_saved': count,
        'image_copied': image_copied,
        'pytest_cmd': pytest_cmd,
    })


def open_browser(port):
    import time
    time.sleep(0.5)
    webbrowser.open(f'http://localhost:{port}')


def main():
    global IMAGE_FOLDER, image_files, single_image_mode

    parser = argparse.ArgumentParser(
        description='Train detection diagnostic viewer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('image', nargs='?', help='Path to a single image file')
    parser.add_argument('--folder', '-f', help='Folder containing images to browse')
    parser.add_argument('--archive', '-a', action='store_true',
                        help='Browse archive images (artifacts/runtime/archive/)')
    parser.add_argument('--port', '-p', type=int, default=5002, help='Server port (default: 5002)')
    parser.add_argument('--no-browser', action='store_true', help="Don't auto-open browser")

    args = parser.parse_args()

    if args.image:
        image_path = Path(args.image).resolve()
        if not image_path.exists():
            print(f'Error: Image not found: {image_path}')
            sys.exit(1)
        image_files = [image_path]
        single_image_mode = True
        print(f'Viewing: {image_path.name}')
    elif args.folder:
        IMAGE_FOLDER = Path(args.folder).resolve()
        if not IMAGE_FOLDER.exists():
            print(f'Error: Folder not found: {IMAGE_FOLDER}')
            sys.exit(1)
        load_image_files()
        print(f'Found {len(image_files)} images in {IMAGE_FOLDER}')
    elif args.archive:
        IMAGE_FOLDER = Path(__file__).parent.parent / 'artifacts' / 'runtime' / 'archive'
        if not IMAGE_FOLDER.exists():
            print(f'Error: Archive folder not found: {IMAGE_FOLDER}')
            sys.exit(1)
        load_image_files()
        print(f'Found {len(image_files)} images in {IMAGE_FOLDER}')
    else:
        load_image_files()
        print(f'Found {len(image_files)} images in {IMAGE_FOLDER}')

    if not image_files:
        print('No images found!')
        sys.exit(1)

    url = f'http://localhost:{args.port}'
    print(f'Starting diagnostic viewer at {url}')

    if not args.no_browser:
        threading.Thread(target=open_browser, args=(args.port,), daemon=True).start()

    app.run(host='localhost', port=args.port, debug=args.no_browser)


if __name__ == '__main__':
    main()
