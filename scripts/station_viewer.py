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
        HSV_RANGES,
    )
    from scripts.train_detector import TrainDetector, TESSERACT_AVAILABLE
except ModuleNotFoundError:
    from detect_stations import STATION_ORDER, INTERNAL_STATIONS
    from station_detector import (
        StationDetector, STATION_X_POSITIONS,
        UPPER_TRACK_Y_PCT, LOWER_TRACK_Y_PCT, TRACK_HEIGHT_PCT,
        UPPER_LABEL_Y_PCT, LOWER_LABEL_Y_PCT,
        REFERENCE_IMAGE_WIDTH, REFERENCE_IMAGE_HEIGHT,
        HSV_RANGES,
    )
    from train_detector import TrainDetector, TESSERACT_AVAILABLE

app = Flask(__name__)

# Configuration
IMAGE_FOLDER = Path(__file__).parent.parent / "artifacts" / "training_data" / "images"
detector = StationDetector()
train_detector = TrainDetector() if TESSERACT_AVAILABLE else None

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


def detect_segment_color(hsv, bounds):
    """Detect if a track segment is cyan (normal) or red (off)."""
    x_min, x_max = bounds['x_min'], bounds['x_max']
    y_min, y_max = bounds['y_min'], bounds['y_max']

    h, w = hsv.shape[:2]
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(w, x_max)
    y_max = min(h, y_max)

    if x_max <= x_min or y_max <= y_min:
        return 'unknown', 0.0

    roi = hsv[y_min:y_max, x_min:x_max]

    # Detect red (wraps around hue 0/180)
    red_mask_low = cv2.inRange(roi,
        HSV_RANGES['track_red_low']['lower'],
        HSV_RANGES['track_red_low']['upper'])
    red_mask_high = cv2.inRange(roi,
        HSV_RANGES['track_red_high']['lower'],
        HSV_RANGES['track_red_high']['upper'])
    red_mask = red_mask_low | red_mask_high
    red_ratio = np.count_nonzero(red_mask) / max(roi.size, 1)

    # Detect cyan
    cyan_mask = cv2.inRange(roi,
        HSV_RANGES['track_cyan']['lower'],
        HSV_RANGES['track_cyan']['upper'])
    cyan_ratio = np.count_nonzero(cyan_mask) / max(roi.size, 1)

    if red_ratio > 0.05:
        return 'red', red_ratio
    elif cyan_ratio > 0.05:
        return 'cyan', cyan_ratio
    else:
        return 'unknown', 0.0


def detect_platform_color(hsv, x, y, size=25, height=25):
    """Detect if a platform area is blue (normal) or yellow (hold).

    Uses a detection region (50x25 pixels) centered on platform rectangles.
    """
    h_img, w_img = hsv.shape[:2]

    # Detection region: 50x25 pixels centered on (x, y)
    x_min = max(0, x - size)
    x_max = min(w_img, x + size)
    y_min = max(0, y - height // 2)
    y_max = min(h_img, y + height // 2)

    if x_max <= x_min or y_max <= y_min:
        return 'unknown', 0.0

    roi = hsv[y_min:y_max, x_min:x_max]

    # Detect yellow (holding) using the range from HSV_RANGES
    yellow_mask = cv2.inRange(roi,
        HSV_RANGES['platform_yellow']['lower'],
        HSV_RANGES['platform_yellow']['upper'])
    yellow_pixels = np.count_nonzero(yellow_mask)

    # Detect blue (normal)
    blue_mask = cv2.inRange(roi,
        HSV_RANGES['platform_blue']['lower'],
        HSV_RANGES['platform_blue']['upper'])
    blue_pixels = np.count_nonzero(blue_mask)

    # Compare which color is dominant
    # Yellow wins if it's close to or greater than blue (within 20%)
    # This accounts for X marks reducing visible yellow area
    min_threshold = 20

    if yellow_pixels >= blue_pixels * 0.8 and yellow_pixels >= min_threshold:
        return 'yellow', yellow_pixels
    elif blue_pixels >= min_threshold:
        return 'blue', blue_pixels
    elif yellow_pixels >= min_threshold:
        return 'yellow', yellow_pixels
    else:
        return 'unknown', 0


def get_detection_data(image_path):
    """Get station detection data for an image."""
    img = cv2.imread(str(image_path))
    if img is None:
        return None

    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Get hardcoded positions scaled to this image
    positions = detector.get_hardcoded_positions(w, h, STATION_ORDER)

    # Platform Y positions vary by station location
    # Western stations (WE-CH): upper at Y=380, lower at Y=500 in 800px image
    # Central stations (VN-EM): upper at Y=425, lower at Y=450 in 800px image
    WESTERN_STATIONS = {'WE', 'FH', 'CA', 'CH'}

    def get_platform_y(code):
        """Get platform Y positions for a station."""
        if code in WESTERN_STATIONS:
            upper_y = int(h * 0.475)   # ~380 in 800px
            lower_y = int(h * 0.625)   # ~500 in 800px
        else:
            upper_y = int(h * 0.53)    # ~424 in 800px (platform rect)
            lower_y = int(h * 0.5625)  # ~450 in 800px
        return upper_y, lower_y

    # Build station markers with color detection
    stations = []
    delays_platforms = []

    for code, name in STATION_ORDER:
        if code in INTERNAL_STATIONS:
            continue

        station_data = positions['stations'].get(code, {})
        x = station_data.get('center_x', 0)
        upper_y = int(h * UPPER_LABEL_Y_PCT)
        lower_y = int(h * LOWER_LABEL_Y_PCT)

        # Get station-specific platform Y positions
        upper_platform_y, lower_platform_y = get_platform_y(code)

        # Detect platform colors
        upper_color, upper_conf = detect_platform_color(hsv, x, upper_platform_y)
        lower_color, lower_conf = detect_platform_color(hsv, x, lower_platform_y)

        station_info = {
            'code': code,
            'name': name,
            'x': x,
            'upper_y': upper_y,
            'lower_y': lower_y,
            'upper_color': upper_color,
            'lower_color': lower_color,
        }
        stations.append(station_info)

        # Track delays (yellow = hold)
        # Special case: CT upper (CTL) is normally in hold mode for train turnaround
        # so we don't report it as a delay
        # Direction terminology: CT, US, YB are Northbound/Southbound; others are Westbound/Eastbound
        NORTH_SOUTH_STATIONS = {'CT', 'US', 'YB'}

        if upper_color == 'yellow':
            # Skip CT upper platform - it's normally in hold mode for turnaround
            if code != 'CT':
                direction = 'Northbound' if code in NORTH_SOUTH_STATIONS else 'Westbound'
                delays_platforms.append({
                    'station': code,
                    'name': name,
                    'track': 'upper',
                    'direction': direction
                })
        if lower_color == 'yellow':
            direction = 'Southbound' if code in NORTH_SOUTH_STATIONS else 'Eastbound'
            delays_platforms.append({
                'station': code,
                'name': name,
                'track': 'lower',
                'direction': direction
            })

    # Build track segments with color detection
    segments = []
    delays_segments = []
    track_y_upper = int(h * UPPER_TRACK_Y_PCT)
    track_y_lower = int(h * LOWER_TRACK_Y_PCT)
    track_height = int(h * TRACK_HEIGHT_PCT)

    for seg_key, seg_data in positions['track_segments'].items():
        bounds = seg_data['bounds']

        # Detect segment color
        color, confidence = detect_segment_color(hsv, bounds)

        segment_info = {
            'key': seg_key,
            'from_code': seg_data['from_code'],
            'to_code': seg_data['to_code'],
            'direction': seg_data['direction'],
            'x_min': bounds['x_min'],
            'x_max': bounds['x_max'],
            'y_min': bounds['y_min'],
            'y_max': bounds['y_max'],
            'color': color,
        }
        segments.append(segment_info)

        # Track delays (red = off)
        if color == 'red':
            delays_segments.append({
                'from': seg_data['from_code'],
                'to': seg_data['to_code'],
                'direction': seg_data['direction'],
                'key': seg_key
            })

    # Detect trains using OCR
    trains = []
    if train_detector is not None:
        trains = train_detector.detect_trains(img, hsv)

    return {
        'width': w,
        'height': h,
        'stations': stations,
        'segments': segments,
        'delays_platforms': delays_platforms,
        'delays_segments': delays_segments,
        'trains': trains,
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
            stroke-width: 3;
        }

        /* Segment regions */
        .segment-region {
            pointer-events: all;
            cursor: pointer;
            stroke-width: 1;
        }
        .segment-region:hover {
            stroke-width: 3;
        }

        /* Train markers */
        .train-marker {
            pointer-events: all;
        }
        .train-marker:hover rect {
            stroke-width: 2;
            fill: rgba(0, 150, 0, 0.9);
        }

        .info-panel {
            width: 350px;
            background: #2a2a4a;
            border-radius: 8px;
            padding: 15px;
            max-height: calc(100vh - 120px);
            overflow-y: auto;
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

        .section-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            cursor: pointer;
            user-select: none;
        }

        .section-header h3 {
            font-size: 0.9em;
            color: #8af;
            margin: 0;
        }

        .section-toggle {
            color: #8af;
            font-size: 0.8em;
        }

        .section-content {
            margin-top: 8px;
        }

        .section-content.collapsed {
            display: none;
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

        .delay-item {
            padding: 6px 10px;
            margin: 4px 0;
            border-radius: 4px;
            font-size: 0.9em;
        }

        .delay-red {
            background: #8b2020;
            border-left: 4px solid #ff4444;
        }

        .delay-yellow {
            background: #6b6b20;
            border-left: 4px solid #ffcc00;
        }

        .no-delays {
            color: #8a8;
            font-style: italic;
            padding: 8px;
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
            margin-bottom: 5px;
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
                <h3>Delays Detected</h3>
                <div id="delaysInfo">
                    {% if detection.delays_segments or detection.delays_platforms %}
                        {% for d in detection.delays_segments %}
                        <div class="delay-item delay-red" onclick="showSegment('{{ d.key }}')">
                            Track Off: {{ d.from }} &rarr; {{ d.to }} ({{ d.direction }})
                        </div>
                        {% endfor %}
                        {% for d in detection.delays_platforms %}
                        <div class="delay-item delay-yellow" onclick="showStation('{{ d.station }}')">
                            Hold: {{ d.name }} ({{ d.direction }})
                        </div>
                        {% endfor %}
                    {% else %}
                        <div class="no-delays">No delays detected</div>
                    {% endif %}
                </div>
            </div>

            <div class="info-section">
                <h3>Trains Detected ({{ detection.trains|length }})</h3>
                <div id="trainsInfo">
                    {% if detection.trains %}
                        {% for t in detection.trains %}
                        <div class="delay-item" style="background: #2a4a2a; border-left: 4px solid #44ff44; cursor: pointer;" onclick="showTrain('{{ t.id }}', {{ t.x }}, {{ t.y }}, '{{ t.track }}')">
                            {{ t.id }} ({{ t.track }})
                        </div>
                        {% endfor %}
                    {% else %}
                        <div class="no-delays">No trains detected</div>
                    {% endif %}
                </div>
            </div>

            <div class="info-section">
                <h3>Click Info</h3>
                <div class="click-info" id="clickInfo">
                    Click on stations or track segments to see details.
                </div>
            </div>

            <div class="info-section">
                <div class="section-header" onclick="toggleSection('stations')">
                    <h3>Stations ({{ detection.stations|length }})</h3>
                    <span class="section-toggle" id="stations-toggle">[+]</span>
                </div>
                <div class="section-content collapsed" id="stations-content">
                    <div class="station-list">
                        {% for s in detection.stations %}
                        <div class="station-item" onclick="showStation('{{ s.code }}')">
                            {{ s.code }} - {{ s.name }}
                        </div>
                        {% endfor %}
                    </div>
                </div>
            </div>

            <div class="info-section">
                <div class="section-header" onclick="toggleSection('segments')">
                    <h3>Track Segments ({{ detection.segments|length }})</h3>
                    <span class="section-toggle" id="segments-toggle">[+]</span>
                </div>
                <div class="section-content collapsed" id="segments-content">
                    <div class="segment-list">
                        {% for seg in detection.segments %}
                        <div class="segment-item" onclick="showSegment('{{ seg.key }}')">
                            {{ seg.from_code }}&rarr;{{ seg.to_code }} ({{ seg.direction[:2] }})
                        </div>
                        {% endfor %}
                    </div>
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
                <label>
                    <input type="checkbox" id="showTrains" checked onchange="updateOverlay()">
                    Show train markers
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
            const showTrains = document.getElementById('showTrains').checked;

            let svg = `<svg viewBox="0 0 ${imgWidth} ${imgHeight}" preserveAspectRatio="xMidYMid meet">`;

            // Draw segments first (behind stations)
            if (showSegments) {
                detection.segments.forEach(seg => {
                    let fillColor, strokeColor;
                    if (seg.color === 'red') {
                        fillColor = 'rgba(255, 50, 50, 0.4)';
                        strokeColor = '#ff4444';
                    } else if (seg.color === 'cyan') {
                        fillColor = 'rgba(50, 200, 200, 0.2)';
                        strokeColor = '#00cccc';
                    } else {
                        fillColor = 'rgba(150, 150, 150, 0.2)';
                        strokeColor = '#888888';
                    }
                    svg += `<rect class="segment-region"
                        x="${seg.x_min}" y="${seg.y_min}"
                        width="${seg.x_max - seg.x_min}" height="${seg.y_max - seg.y_min}"
                        fill="${fillColor}" stroke="${strokeColor}"
                        onclick="showSegment('${seg.key}')"
                        data-key="${seg.key}"/>`;
                });
            }

            // Draw station markers
            if (showStations) {
                detection.stations.forEach(s => {
                    // Upper station marker - color based on platform state
                    let upperFill = s.upper_color === 'yellow' ? '#ffcc00' : '#00cc00';
                    let upperStroke = s.upper_color === 'yellow' ? '#ff8800' : '#ffffff';
                    svg += `<g class="station-marker" onclick="showStation('${s.code}')">
                        <circle cx="${s.x}" cy="${s.upper_y}" r="6" fill="${upperFill}" stroke="${upperStroke}" stroke-width="1"/>
                        <text x="${s.x}" y="${s.upper_y - 10}" text-anchor="middle" fill="${upperFill}" font-size="10">${s.code}</text>
                    </g>`;

                    // Lower station marker
                    let lowerFill = s.lower_color === 'yellow' ? '#ffcc00' : '#00aa00';
                    let lowerStroke = s.lower_color === 'yellow' ? '#ff8800' : '#ffffff';
                    svg += `<g class="station-marker" onclick="showStation('${s.code}')">
                        <circle cx="${s.x}" cy="${s.lower_y}" r="6" fill="${lowerFill}" stroke="${lowerStroke}" stroke-width="1"/>
                        <text x="${s.x}" y="${s.lower_y + 16}" text-anchor="middle" fill="${lowerFill}" font-size="10">${s.code}</text>
                    </g>`;
                });
            }

            // Draw train identifiers
            if (showTrains && detection.trains) {
                detection.trains.forEach(t => {
                    svg += `<g class="train-marker" onclick="showTrain('${t.id}', ${t.x}, ${t.y}, '${t.track}')" style="cursor: pointer;">
                        <rect x="${t.x - 20}" y="${t.y - 8}" width="40" height="16" rx="3"
                              fill="rgba(0, 100, 0, 0.8)" stroke="#44ff44" stroke-width="1"/>
                        <text x="${t.x}" y="${t.y + 4}"
                              text-anchor="middle" fill="#ffffff" font-size="10"
                              font-weight="bold">${t.id}</text>
                    </g>`;
                });
            }

            svg += '</svg>';
            document.getElementById('overlay').innerHTML = svg;
        }

        function showStation(code) {
            const s = detection.stations.find(st => st.code === code);
            if (s) {
                let statusUpper = s.upper_color === 'yellow' ? '<span style="color:#ffcc00">HOLD</span>' :
                                  s.upper_color === 'blue' ? '<span style="color:#00cc00">Normal</span>' : 'Unknown';
                let statusLower = s.lower_color === 'yellow' ? '<span style="color:#ffcc00">HOLD</span>' :
                                  s.lower_color === 'blue' ? '<span style="color:#00cc00">Normal</span>' : 'Unknown';
                document.getElementById('clickInfo').innerHTML = `
                    <strong>Station: ${s.name}</strong><br>
                    Code: ${s.code}<br>
                    Upper platform: ${statusUpper}<br>
                    Lower platform: ${statusLower}
                `;
            }
        }

        function showSegment(key) {
            const seg = detection.segments.find(s => s.key === key);
            if (seg) {
                let status = seg.color === 'red' ? '<span style="color:#ff4444">OFF</span>' :
                            seg.color === 'cyan' ? '<span style="color:#00cccc">Normal</span>' : 'Unknown';
                document.getElementById('clickInfo').innerHTML = `
                    <strong>Segment: ${seg.from_code} &rarr; ${seg.to_code}</strong><br>
                    Direction: ${seg.direction}<br>
                    Status: ${status}
                `;
            }
        }

        function showTrain(id, x, y, track) {
            // Parse train ID: route letter + 4-digit number + suffix
            const route = id.charAt(0);
            const number = id.substring(1, 5);
            const suffix = id.substring(5);
            document.getElementById('clickInfo').innerHTML = `
                <strong>Train: ${id}</strong><br>
                Route: ${route}<br>
                Number: ${number}<br>
                Suffix: ${suffix}<br>
                Track: ${track}<br>
                Position: (${x}, ${y})
            `;
        }

        function toggleSection(section) {
            const content = document.getElementById(section + '-content');
            const toggle = document.getElementById(section + '-toggle');
            if (content.classList.contains('collapsed')) {
                content.classList.remove('collapsed');
                toggle.textContent = '[-]';
            } else {
                content.classList.add('collapsed');
                toggle.textContent = '[+]';
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
