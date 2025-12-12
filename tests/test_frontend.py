#!/usr/bin/env python3
"""
Frontend tests for Muni Status Dashboard.

Tests different status scenarios using a mock API server and Selenium.

Usage:
    python test_frontend.py
"""

import json
import time
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options

# Test configuration
TEST_PORT = 8888
BASE_URL = f"http://localhost:{TEST_PORT}"

# Mock API responses for different test scenarios
MOCK_RESPONSES = {
    "green_single": {
        "status": "green",
        "description": "normal",
        "confidence": 0.9998,
        "probabilities": {"green": 0.9998, "yellow": 0.0002, "red": 0.0},
        "image_path": "muni_snapshots/test.jpg",
        "image_dimensions": {"width": 1860, "height": 800},
        "timestamp": "2025-12-08T15:00:00",
        "cached": True,
        "cache_age": 5.2
    },
    "yellow_single": {
        "status": "yellow",
        "description": "minor delays",
        "confidence": 0.8765,
        "probabilities": {"green": 0.1000, "yellow": 0.8765, "red": 0.0235},
        "image_path": "muni_snapshots/test.jpg",
        "image_dimensions": {"width": 1860, "height": 800},
        "timestamp": "2025-12-08T15:00:00",
        "cached": True,
        "cache_age": 3.1
    },
    "red_single": {
        "status": "red",
        "description": "offline",
        "confidence": 0.9543,
        "probabilities": {"green": 0.0123, "yellow": 0.0334, "red": 0.9543},
        "image_path": "muni_snapshots/test.jpg",
        "image_dimensions": {"width": 1860, "height": 800},
        "timestamp": "2025-12-08T15:00:00",
        "cached": True,
        "cache_age": 8.7
    },
    "best_of_two_green_yellow": {
        "status": "green",
        "description": "normal",
        "confidence": 0.9998,
        "probabilities": {"green": 0.9998, "yellow": 0.0002, "red": 0.0},
        "image_path": "muni_snapshots/test.jpg",
        "image_dimensions": {"width": 1860, "height": 800},
        "timestamp": "2025-12-08T15:00:00",
        "cached": True,
        "cache_age": 5.2,
        "status_history": [
            {"status": "green", "timestamp": "2025-12-08T15:00:00"},
            {"status": "yellow", "timestamp": "2025-12-08T14:59:30"}
        ],
        "is_best_of_two": True
    },
    "best_of_two_green_red": {
        "status": "green",
        "description": "normal",
        "confidence": 0.9998,
        "probabilities": {"green": 0.9998, "yellow": 0.0002, "red": 0.0},
        "image_path": "muni_snapshots/test.jpg",
        "image_dimensions": {"width": 1860, "height": 800},
        "timestamp": "2025-12-08T15:00:00",
        "cached": True,
        "cache_age": 5.2,
        "status_history": [
            {"status": "green", "timestamp": "2025-12-08T15:00:00"},
            {"status": "red", "timestamp": "2025-12-08T14:59:30"}
        ],
        "is_best_of_two": True
    },
    "error_cache_unavailable": {
        "error": "Cache unavailable and fallback mode disabled",
        "details": "Set ENABLE_FALLBACK=true to enable live predictions",
        "timestamp": "2025-12-08T15:00:00"
    }
}

# Current mock response (changed by tests)
current_response = MOCK_RESPONSES["green_single"]


class MockAPIHandler(BaseHTTPRequestHandler):
    """Mock API server handler."""

    def log_message(self, format, *args):
        """Suppress logging."""
        pass

    def do_GET(self):
        """Handle GET requests."""
        if self.path == "/status":
            self.send_response(200 if "error" not in current_response else 503)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(current_response).encode())
        elif self.path == "/":
            # Serve the dashboard
            import os
            html_path = os.path.join(os.path.dirname(__file__), "..", "api", "html", "dashboard.html")
            with open(html_path, "r") as f:
                html = f.read()
                # Replace API URL to point to our test server
                html = html.replace("const API_URL = '/status';",
                                   f"const API_URL = '{BASE_URL}/status';")
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                self.wfile.write(html.encode())
        else:
            self.send_response(404)
            self.end_headers()


def start_mock_server():
    """Start the mock API server in a background thread."""
    server = HTTPServer(("localhost", TEST_PORT), MockAPIHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    time.sleep(1)  # Give server time to start
    return server


def setup_driver():
    """Create and configure Selenium WebDriver."""
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    driver = webdriver.Chrome(options=options)
    return driver


def test_green_status(driver):
    """Test green status display."""
    global current_response
    current_response = MOCK_RESPONSES["green_single"]

    driver.get(BASE_URL)
    wait = WebDriverWait(driver, 10)

    # Wait for status display to load (not just text)
    wait.until(
        EC.presence_of_element_located((By.CLASS_NAME, "status-display"))
    )
    time.sleep(0.5)  # Small delay to ensure rendering is complete

    # Check status text
    status_text = driver.find_element(By.CLASS_NAME, "status-text")
    assert "GREEN" in status_text.text.upper(), f"Expected GREEN, got {status_text.text}"
    assert "green" in status_text.get_attribute("class"), "Missing green class"

    # Check emoji (in page source since emoji might not be in .text)
    page_source = driver.page_source
    assert "🟢" in page_source, "Expected 🟢 emoji in page"

    # Check description
    description = driver.find_element(By.CLASS_NAME, "description")
    assert "normal" in description.text.lower(), f"Expected 'normal', got {description.text}"

    print("✓ Green status test passed")


def test_yellow_status(driver):
    """Test yellow status display."""
    global current_response
    current_response = MOCK_RESPONSES["yellow_single"]

    driver.get(BASE_URL)
    wait = WebDriverWait(driver, 10)

    wait.until(EC.presence_of_element_located((By.CLASS_NAME, "status-display")))
    time.sleep(0.5)

    status_text = driver.find_element(By.CLASS_NAME, "status-text")
    assert "YELLOW" in status_text.text.upper(), f"Expected YELLOW, got {status_text.text}"
    assert "yellow" in status_text.get_attribute("class"), "Missing yellow class"

    page_source = driver.page_source
    assert "🟡" in page_source, "Expected 🟡 emoji in page"

    print("✓ Yellow status test passed")


def test_red_status(driver):
    """Test red status display."""
    global current_response
    current_response = MOCK_RESPONSES["red_single"]

    driver.get(BASE_URL)
    wait = WebDriverWait(driver, 10)

    wait.until(EC.presence_of_element_located((By.CLASS_NAME, "status-display")))
    time.sleep(0.5)

    status_text = driver.find_element(By.CLASS_NAME, "status-text")
    assert "RED" in status_text.text.upper(), f"Expected RED, got {status_text.text}"
    assert "red" in status_text.get_attribute("class"), "Missing red class"

    page_source = driver.page_source
    assert "🔴" in page_source, "Expected 🔴 emoji in page"

    print("✓ Red status test passed")


def test_best_of_two_green_yellow(driver):
    """Test best of two: green + yellow = green."""
    global current_response
    current_response = MOCK_RESPONSES["best_of_two_green_yellow"]

    driver.get(BASE_URL)
    wait = WebDriverWait(driver, 10)

    wait.until(
        EC.presence_of_element_located((By.CLASS_NAME, "status-display"))
    )
    time.sleep(0.6)  # Wait for animations to complete (500ms fade + buffer)

    status_text = driver.find_element(By.CLASS_NAME, "status-text")
    # Should show green (best of two)
    assert "GREEN" in status_text.text.upper(), f"Expected GREEN (best of two), got {status_text.text}"

    # Check for "Best of 2" indicator
    page_source = driver.page_source
    assert "Best of 2" in page_source, "Missing 'Best of 2' indicator"

    # Check for emoji history (green → yellow)
    assert "🟢" in page_source and "🟡" in page_source, "Missing status history emojis"

    print("✓ Best of two (green + yellow) test passed")


def test_best_of_two_green_red(driver):
    """Test best of two: green + red = green."""
    global current_response
    current_response = MOCK_RESPONSES["best_of_two_green_red"]

    driver.get(BASE_URL)
    wait = WebDriverWait(driver, 10)

    wait.until(
        EC.presence_of_element_located((By.CLASS_NAME, "status-display"))
    )
    time.sleep(0.6)  # Wait for animations to complete (500ms fade + buffer)

    status_text = driver.find_element(By.CLASS_NAME, "status-text")
    # Should show green (best of two)
    assert "GREEN" in status_text.text.upper(), f"Expected GREEN (best of two), got {status_text.text}"

    # Check for emoji history (green → red)
    page_source = driver.page_source
    assert "🟢" in page_source and "🔴" in page_source, "Missing status history emojis"

    print("✓ Best of two (green + red) test passed")


def test_error_display(driver):
    """Test error message display."""
    global current_response
    current_response = MOCK_RESPONSES["error_cache_unavailable"]

    driver.get(BASE_URL)
    wait = WebDriverWait(driver, 10)

    # Wait for status display with error
    wait.until(
        EC.presence_of_element_located((By.CLASS_NAME, "status-display"))
    )

    # Check that error status is shown
    status_text = driver.find_element(By.CLASS_NAME, "status-text")
    assert "ERROR" in status_text.text.upper(), f"Expected ERROR status, got {status_text.text}"

    # Check that error message is in description
    description = driver.find_element(By.CLASS_NAME, "description")
    assert "Cache unavailable" in description.text, f"Expected error message, got {description.text}"

    print("✓ Error display test passed")


def test_probability_bars(driver):
    """Test probability bars are displayed correctly."""
    global current_response
    current_response = MOCK_RESPONSES["green_single"]

    driver.get(BASE_URL)
    wait = WebDriverWait(driver, 10)

    # Wait for probabilities section to load
    wait.until(EC.presence_of_element_located((By.CLASS_NAME, "probabilities")))

    # Check that probability bars exist
    prob_bars = driver.find_elements(By.CLASS_NAME, "prob-bar")
    assert len(prob_bars) == 3, f"Expected 3 probability bars, got {len(prob_bars)}"

    # Check for green, yellow, red fills
    fills = driver.find_elements(By.CLASS_NAME, "prob-fill")
    assert len(fills) == 3, f"Expected 3 probability fills, got {len(fills)}"

    print("✓ Probability bars test passed")


def run_tests():
    """Run all frontend tests."""
    print("=" * 60)
    print("Muni Status Dashboard - Frontend Tests")
    print("=" * 60)
    print()

    # Start mock API server
    print("Starting mock API server...")
    server = start_mock_server()
    print(f"✓ Mock server running on {BASE_URL}")
    print()

    # Setup Selenium driver
    print("Initializing Selenium WebDriver...")
    driver = setup_driver()
    print("✓ WebDriver ready")
    print()

    try:
        print("Running tests...")
        print("-" * 60)

        test_green_status(driver)
        test_yellow_status(driver)
        test_red_status(driver)
        test_best_of_two_green_yellow(driver)
        test_best_of_two_green_red(driver)
        test_error_display(driver)
        test_probability_bars(driver)

        print("-" * 60)
        print()
        print("=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)

    except AssertionError as e:
        print()
        print("=" * 60)
        print(f"✗ Test failed: {e}")
        print("=" * 60)
        return False
    except Exception as e:
        print()
        print("=" * 60)
        print(f"✗ Error running tests: {e}")
        print("=" * 60)
        return False
    finally:
        driver.quit()
        server.shutdown()

    return True


if __name__ == "__main__":
    import sys
    success = run_tests()
    sys.exit(0 if success else 1)
