#!/usr/bin/env python3
"""
Simple Flask web UI for station detection visualization.

Run with: python scripts/station_viewer.py
Then open: http://localhost:5001
"""

import os
import sys
import json
import base64
from pathlib import Path
from io import BytesIO

import cv2
import numpy as np
from flask import Flask, render_template_string, jsonify, request

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from scripts.detect_stations import STATION_ORDER, INTERNAL_STATIONS
    from scripts.station_detector import (
        StationDetector, STATION_X_POSITIONS,
        UPPER_TRACK_Y_PCT, LOWER_TRACK_Y_PCT, TRACK_HEIGHT_PCT,
        UPPER_LABEL_Y_PCT, LOWER_LABEL_Y_PCT,
        REFERENCE_IMAGE_WIDTH, REFERENCE_IMAGE_HEIGHT,
    )
except ModuleNotFoundError:
    from detect_stations import STATION_ORDER, INTERNAL_STATIONS
    from station_detector import (
        StationDetector, STATION_X_POSITIONS,
        UPPER_TRACK_Y_PCT, LOWER_TRACK_Y_PCT, TRACK_HEIGHT_PCT,
        UPPER_LABEL_Y_PCT, LOWER_LABEL_Y_PCT,
        REFERENCE_IMAGE_WIDTH, REFERENCE_IMAGE_HEIGHT,
    )

app = Flask(__name__)

# Configuration
IMAGE_FOLDER = Path(__file__).parent.parent / "artifacts" / "training_data" / "images"
detector = StationDetector()

# Global state
current_image_index = 0
image_files = []


def load_image_files():
    """Load list of image files."""
    global image_files
    if IMAGE_FOLDER.exists():
        image_files = sorted([
            f for f in IMAGE_FOLDER.glob("*.jpg")
        ] + [
            f for f in IMAGE_FOLDER.glob("*.JPG")
        ])
    return image_files


def get_detection_data(image_path):
    """Get station detection data for an image."""
    img = cv2.imread(str(image_path))
    if img is None:
        return None

    h, w = img.shape[:2]

    # Get hardcoded positions scaled to this image
    positions = detector.get_hardcoded_positions(w, h, STATION_ORDER)

    # Build station markers
    stations = []
    for code, name in STATION_ORDER:
        if code in INTERNAL_STATIONS:
            continue

        station_data = positions['stations'].get(code, {})
        x = station_data.get('center_x', 0)
        upper_y = int(h * UPPER_LABEL_Y_PCT)
        lower_y = int(h * LOWER_LABEL_Y_PCT)

        stations.append({
            'code': code,
            'name': name,
            'x': x,
            'upper_y': upper_y,
            'lower_y': lower_y,
        })

    # Build track segments
    segments = []
    track_y_upper = int(h * UPPER_TRACK_Y_PCT)
    track_y_lower = int(h * LOWER_TRACK_Y_PCT)
    track_height = int(h * TRACK_HEIGHT_PCT)

    for seg_key, seg_data in positions['track_segments'].items():
        bounds = seg_data['bounds']
        segments.append({
            'key': seg_key,
            'from_code': seg_data['from_code'],
            'to_code': seg_data['to_code'],
            'direction': seg_data['direction'],
            'x_min': bounds['x_min'],
            'x_max': bounds['x_max'],
            'y_min': bounds['y_min'],
            'y_max': bounds['y_max'],
        })

    return {
        'width': w,
        'height': h,
        'stations': stations,
        'segments': segments,
        'track_y_upper': track_y_upper,
        'track_y_lower': track_y_lower,
        'track_height': track_height,
    }


def image_to_base64(image_path):
    """Convert image to base64 for embedding in HTML."""
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')


HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Station Detection Viewer</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: #1a1a2e;
            color: #eee;
            padding: 20px;
        }
        .header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
        }
        .header h1 { font-size: 1.5em; }
        .nav-buttons button {
            background: #4a4a6a;
            color: white;
            border: none;
            padding: 10px 20px;
            margin: 0 5px;
            cursor: pointer;
            border-radius: 4px;
        }
        .nav-buttons button:hover { background: #6a6a8a; }
        .nav-buttons button:disabled { opacity: 0.5; cursor: not-allowed; }

        .container {
            display: flex;
            gap: 20px;
        }

        .image-panel {
            flex: 1;
            position: relative;
        }

        .image-wrapper {
            position: relative;
            display: inline-block;
        }

        .image-wrapper img {
            max-width: 100%;
            display: block;
        }

        .overlay {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
        }

        .overlay svg {
            width: 100%;
            height: 100%;
        }

        /* Station markers */
        .station-marker {
            pointer-events: all;
            cursor: pointer;
        }
        .station-marker:hover circle {
            fill: #00ff00;
        }

        /* Segment regions */
        .segment-region {
            pointer-events: all;
            cursor: pointer;
            fill: rgba(200, 200, 200, 0.3);
            stroke: rgba(200, 200, 200, 0.8);
            stroke-width: 1;
        }
        .segment-region:hover {
            fill: rgba(100, 200, 255, 0.4);
            stroke: #00ffff;
            stroke-width: 2;
        }

        .info-panel {
            width: 350px;
            background: #2a2a4a;
            border-radius: 8px;
            padding: 15px;
        }

        .info-panel h2 {
            font-size: 1.1em;
            margin-bottom: 10px;
            color: #aaf;
        }

        .info-section {
            margin-bottom: 15px;
            padding: 10px;
            background: #1a1a2e;
            border-radius: 4px;
        }

        .info-section h3 {
            font-size: 0.9em;
            color: #8af;
            margin-bottom: 8px;
        }

        .station-list, .segment-list {
            font-size: 0.85em;
            max-height: 200px;
            overflow-y: auto;
        }

        .station-item, .segment-item {
            padding: 4px 8px;
            margin: 2px 0;
            background: #3a3a5a;
            border-radius: 3px;
            cursor: pointer;
        }
        .station-item:hover, .segment-item:hover {
            background: #4a4a7a;
        }

        .click-info {
            background: #3a5a3a;
            padding: 10px;
            border-radius: 4px;
            min-height: 60px;
        }

        .filename {
            color: #aaa;
            font-size: 0.9em;
        }

        .controls {
            margin-top: 15px;
        }
        .controls label {
            display: flex;
            align-items: center;
            gap: 8px;
            cursor: pointer;
        }
    </style>
</head>
<body>
    <div class="header">
        <div>
            <h1>Station Detection Viewer</h1>
            <span class="filename">{{ filename }} ({{ index + 1 }} / {{ total }})</span>
        </div>
        <div class="nav-buttons">
            <button onclick="navigate(-10)" {{ 'disabled' if index < 10 else '' }}>&lt;&lt; -10</button>
            <button onclick="navigate(-1)" {{ 'disabled' if index == 0 else '' }}>&lt; Prev</button>
            <button onclick="navigate(1)" {{ 'disabled' if index >= total - 1 else '' }}>Next &gt;</button>
            <button onclick="navigate(10)" {{ 'disabled' if index >= total - 10 else '' }}>+10 &gt;&gt;</button>
        </div>
    </div>

    <div class="container">
        <div class="image-panel">
            <div class="image-wrapper" id="imageWrapper">
                <img src="data:image/jpeg;base64,{{ image_data }}" id="mainImage" onload="setupOverlay()">
                <div class="overlay" id="overlay"></div>
            </div>
        </div>

        <div class="info-panel">
            <h2>Detection Info</h2>

            <div class="info-section">
                <h3>Click Info</h3>
                <div class="click-info" id="clickInfo">
                    Click on stations (green dots) or track segments (gray rectangles) to see details.
                </div>
            </div>

            <div class="info-section">
                <h3>Stations ({{ detection.stations|length }})</h3>
                <div class="station-list">
                    {% for s in detection.stations %}
                    <div class="station-item" onclick="showStation('{{ s.code }}')">
                        {{ s.code }} - {{ s.name }} (x={{ s.x }})
                    </div>
                    {% endfor %}
                </div>
            </div>

            <div class="info-section">
                <h3>Track Segments ({{ detection.segments|length }})</h3>
                <div class="segment-list">
                    {% for seg in detection.segments %}
                    <div class="segment-item" onclick="showSegment('{{ seg.key }}')">
                        {{ seg.from_code }}&rarr;{{ seg.to_code }} ({{ seg.direction[:2] }})
                    </div>
                    {% endfor %}
                </div>
            </div>

            <div class="controls">
                <label>
                    <input type="checkbox" id="showStations" checked onchange="updateOverlay()">
                    Show station markers
                </label>
                <label>
                    <input type="checkbox" id="showSegments" checked onchange="updateOverlay()">
                    Show track segments
                </label>
            </div>
        </div>
    </div>

    <script>
        const detection = {{ detection | tojson }};
        let imgWidth, imgHeight, scaleX, scaleY;

        function setupOverlay() {
            const img = document.getElementById('mainImage');
            const wrapper = document.getElementById('imageWrapper');

            // Get displayed size vs natural size
            imgWidth = img.naturalWidth;
            imgHeight = img.naturalHeight;
            scaleX = img.clientWidth / imgWidth;
            scaleY = img.clientHeight / imgHeight;

            updateOverlay();
        }

        function updateOverlay() {
            const showStations = document.getElementById('showStations').checked;
            const showSegments = document.getElementById('showSegments').checked;

            let svg = `<svg viewBox="0 0 ${imgWidth} ${imgHeight}" preserveAspectRatio="xMidYMid meet">`;

            // Draw segments first (behind stations)
            if (showSegments) {
                detection.segments.forEach(seg => {
                    svg += `<rect class="segment-region"
                        x="${seg.x_min}" y="${seg.y_min}"
                        width="${seg.x_max - seg.x_min}" height="${seg.y_max - seg.y_min}"
                        onclick="showSegment('${seg.key}')"
                        data-key="${seg.key}"/>`;
                });
            }

            // Draw station markers
            if (showStations) {
                detection.stations.forEach(s => {
                    // Upper station marker
                    svg += `<g class="station-marker" onclick="showStation('${s.code}')">
                        <circle cx="${s.x}" cy="${s.upper_y}" r="6" fill="#00cc00" stroke="#fff" stroke-width="1"/>
                        <text x="${s.x}" y="${s.upper_y - 10}" text-anchor="middle" fill="#0f0" font-size="10">${s.code}</text>
                    </g>`;
                    // Lower station marker
                    svg += `<g class="station-marker" onclick="showStation('${s.code}')">
                        <circle cx="${s.x}" cy="${s.lower_y}" r="6" fill="#00aa00" stroke="#fff" stroke-width="1"/>
                        <text x="${s.x}" y="${s.lower_y + 16}" text-anchor="middle" fill="#0a0" font-size="10">${s.code}</text>
                    </g>`;
                });
            }

            svg += '</svg>';
            document.getElementById('overlay').innerHTML = svg;
        }

        function showStation(code) {
            const s = detection.stations.find(st => st.code === code);
            if (s) {
                document.getElementById('clickInfo').innerHTML = `
                    <strong>Station: ${s.name}</strong><br>
                    Code: ${s.code}<br>
                    X position: ${s.x}px<br>
                    Upper Y: ${s.upper_y}px<br>
                    Lower Y: ${s.lower_y}px
                `;
            }
        }

        function showSegment(key) {
            const seg = detection.segments.find(s => s.key === key);
            if (seg) {
                document.getElementById('clickInfo').innerHTML = `
                    <strong>Segment: ${seg.from_code} &rarr; ${seg.to_code}</strong><br>
                    Direction: ${seg.direction}<br>
                    X: ${seg.x_min} - ${seg.x_max}px<br>
                    Y: ${seg.y_min} - ${seg.y_max}px<br>
                    Width: ${seg.x_max - seg.x_min}px
                `;
            }
        }

        function navigate(delta) {
            window.location.href = '/?index=' + ({{ index }} + delta);
        }

        // Handle window resize
        window.addEventListener('resize', setupOverlay);
    </script>
</body>
</html>
'''


@app.route('/')
def index():
    global current_image_index

    if not image_files:
        load_image_files()

    if not image_files:
        return "No images found in " + str(IMAGE_FOLDER)

    # Get index from query param
    idx = request.args.get('index', current_image_index, type=int)
    idx = max(0, min(idx, len(image_files) - 1))
    current_image_index = idx

    image_path = image_files[idx]
    detection = get_detection_data(image_path)
    image_data = image_to_base64(image_path)

    return render_template_string(
        HTML_TEMPLATE,
        filename=image_path.name,
        index=idx,
        total=len(image_files),
        image_data=image_data,
        detection=detection,
    )


@app.route('/api/detection/<int:idx>')
def get_detection(idx):
    """API endpoint for detection data."""
    if not image_files:
        load_image_files()

    if idx < 0 or idx >= len(image_files):
        return jsonify({'error': 'Invalid index'}), 400

    image_path = image_files[idx]
    detection = get_detection_data(image_path)
    return jsonify(detection)


if __name__ == '__main__':
    load_image_files()
    print(f"Found {len(image_files)} images in {IMAGE_FOLDER}")
    print("Starting server at http://localhost:5001")
    app.run(host='localhost', port=5001, debug=True)
