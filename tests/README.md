# Test Suite Documentation

Automated integration tests for the web dashboard frontend.

## Prerequisites

- Python 3.13+
- Google Chrome browser
- ChromeDriver (automatically managed by Selenium 4+)

## Setup

```bash
cd tests

# Install test dependencies
pip install -r requirements.txt

# Install Chrome (if not installed)
# macOS
brew install --cask google-chrome

# Debian/Ubuntu
sudo apt-get install google-chrome-stable

# RHEL/CentOS
sudo yum install google-chrome-stable
```

**Note**: Tests use a self-contained mock API server. The production API does not need to be running.

## Running Tests

Execute tests from the `tests` directory:

```bash
cd tests
python3 test_frontend.py
```

### Expected Output

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

## Test Coverage

Frontend integration tests (`test_frontend.py`) verify:

### Status Display
- Green status indicator (🟢)
- Yellow status indicator (🟡)
- Red status indicator (🔴)
- Emoji rendering
- Status text formatting

### Best-of-Two Logic
- Green + Yellow → Green (optimistic)
- Green + Red → Green (tolerates temporary failures)
- Correct status history display

### Error Handling
- Error message presentation
- Graceful degradation
- User feedback

### Probability Visualization
- Probability bar rendering
- Percentage calculations
- Visual scaling

### Metadata Display
- Timestamp formatting
- Cache age calculation
- Status history

## Test Architecture

Tests implement the following pattern:

1. **Mock Server**: HTTP server serving predefined status responses
2. **Browser Automation**: Selenium WebDriver controls Chrome headlessly
3. **Assertions**: Verify DOM elements, styles, and content
4. **Cleanup**: Automatic teardown of server and browser

## Troubleshooting

### File Not Found: index.html

Cause: Test execution from incorrect directory

Solution:
```bash
# Navigate to tests directory
cd tests

# Verify location
pwd  # Should end with /tests

# Run tests
python3 test_frontend.py
```

### ChromeDriver Errors

Update Chrome and Selenium:

```bash
# macOS
brew update && brew upgrade google-chrome

# Upgrade Selenium
pip install --upgrade selenium
```

### WebDriver Not Found

Selenium 4+ manages ChromeDriver automatically:

```bash
# Upgrade Selenium
pip install --upgrade selenium

# Manual ChromeDriver installation (if needed)
brew install chromedriver

# Verify installation
chromedriver --version
```

### Port 8888 In Use

Cause: Process already bound to test server port

Solution:
```bash
# Find and terminate process
lsof -ti:8888 | xargs kill

# Alternative: Modify TEST_PORT in test_frontend.py
```

### Test Timeouts or Hangs

Debug steps:

```bash
# Verify Chrome installation
which google-chrome-stable || which chrome

# Run in visible mode (disable headless)
# Edit test_frontend.py: Comment out --headless=new option

# Check firewall rules
# Ensure localhost connections permitted
```

### Selenium Exceptions

Common issues:

```bash
# Verify Selenium version (4.0+ required)
pip show selenium

# Reinstall dependencies
pip uninstall selenium
pip install -r requirements.txt

# Check Python version
python3 --version  # Should be 3.13+
```

## Writing Additional Tests

Add test methods to `TestFrontend` class in `test_frontend.py`:

```python
def test_new_feature(self):
    """Test description"""
    # Configure mock response
    mock_data = {
        "status": "green",
        "description": "Test description",
        "confidence": 0.95
    }

    # Load dashboard
    self.driver.get(f"{self.base_url}?test_mode=true")

    # Assert expected behavior
    element = self.driver.find_element(By.ID, "element-id")
    self.assertEqual(element.text, "Expected Value")
```

## Continuous Integration

For CI/CD integration:

```bash
# Run tests in headless mode (default)
python3 test_frontend.py

# Exit code 0 = success, non-zero = failure
echo $?
```

## Future Enhancements

Planned test coverage expansion:

- **API Endpoint Tests**: Backend API validation
- **Model Prediction Tests**: ML inference correctness
- **Cache Writer Tests**: Background process integration
- **Load Tests**: Performance under concurrent requests
- **Cross-Browser Tests**: Firefox, Safari compatibility

## Related Documentation

- **Deployment**: [deploy/README.md](../deploy/README.md) - API deployment guide
- **Frontend**: [api/index.html](../api/index.html) - Dashboard implementation
- **Configuration**: [CONFIGURATION.md](../CONFIGURATION.md) - System configuration
