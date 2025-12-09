# Testing Guide

Automated tests for the Muni Status Monitor web frontend.

## Setup

```bash
# Install dependencies (requires Chrome)
pip install selenium

# macOS - Install Chrome if not already installed
brew install --cask google-chrome
```

## Running Tests

```bash
# Make sure API is running first
cd ../api
docker-compose up -d

# Run frontend tests
cd ../tests
python test_frontend.py
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

**Chrome driver errors:**
```bash
# Update Chrome
brew update && brew upgrade google-chrome

# Or install specific chromedriver version
pip install --upgrade selenium
```

**API not running:**
```bash
cd ../api
docker-compose up -d
```

**Tests failing:**
- Check that port 8000 is available
- Verify API health: `curl http://localhost:8000/health`
- Check Docker containers are running: `docker-compose ps`

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
