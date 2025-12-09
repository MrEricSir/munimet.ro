# Testing Guide

Automated tests for the Muni Status Monitor web frontend.

## Setup

```bash
cd ~/Development/munimetro/tests

# Install test dependencies (requires Chrome)
pip install -r requirements.txt

# macOS - Install Chrome if not already installed
brew install --cask google-chrome
```

**Note**: These tests use a mock API server, so you **don't** need the actual API running. The tests are self-contained.

## Running Tests

```bash
# Run from the tests directory
cd ~/Development/munimetro/tests
python3 test_frontend.py
```

**Expected output:**
```
============================================================
Muni Status Dashboard - Frontend Tests
============================================================

Starting mock API server...
✓ Mock server running on http://localhost:8888

Initializing Selenium WebDriver...
✓ WebDriver ready

Running tests...
------------------------------------------------------------
✓ Green status test passed
✓ Yellow status test passed
✓ Red status test passed
✓ Best of two (green + yellow) test passed
✓ Best of two (green + red) test passed
✓ Error display test passed
✓ Probability bars test passed
------------------------------------------------------------

============================================================
✓ All tests passed!
============================================================
```

## What's Tested

The frontend integration tests (`test_frontend.py`) verify:

- ✓ Green status display with emoji 🟢
- ✓ Yellow status display with emoji 🟡
- ✓ Red status display with emoji 🔴
- ✓ Best of two: green + yellow = green
- ✓ Best of two: green + red = green
- ✓ Error message display
- ✓ Probability bars rendering
- ✓ Metadata display (timestamp, cache age)

## How It Works

Tests run a mock API server that serves predefined status responses, then uses Selenium to:
1. Load the web dashboard
2. Verify correct status indicators and colors
3. Check emoji rendering
4. Validate probability bar calculations
5. Test edge cases (errors, mixed statuses)

## Troubleshooting

**FileNotFoundError: index.html**

**Cause**: Running tests from wrong directory

**Solution**:
```bash
# Make sure you're in the tests directory
cd ~/Development/munimetro/tests
pwd  # Should show .../munimetro/tests

python3 test_frontend.py
```

**Chrome driver errors:**
```bash
# Update Chrome
brew update && brew upgrade google-chrome

# Or update selenium
pip install --upgrade selenium
```

**"WebDriver not found" or "chromedriver not found":**

**Solution**:
```bash
# Selenium 4+ automatically manages chromedriver
pip install --upgrade selenium

# If still having issues, explicitly install chromedriver
brew install chromedriver
```

**Port 8888 already in use:**

**Cause**: Another process using the test server port

**Solution**:
```bash
# Find and kill the process
lsof -ti:8888 | xargs kill

# Or change TEST_PORT in test_frontend.py
```

**Tests hang or timeout:**
- Check Chrome is installed: `which google-chrome-stable || which chrome`
- Try running in non-headless mode (comment out `--headless=new` in test file)
- Check firewall isn't blocking localhost connections

## Writing New Tests

Add new test methods to `test_frontend.py`:

```python
def test_new_feature(self):
    """Test description"""
    # Setup mock data
    mock_data = {...}

    # Load page
    self.driver.get(f"{self.base_url}?test_mode=true")

    # Verify behavior
    element = self.driver.find_element(By.ID, "element-id")
    self.assertEqual(element.text, "Expected Value")
```

## Next Steps

- Add API endpoint tests
- Add model prediction tests
- Add integration tests for cache writer
- Add performance/load tests
