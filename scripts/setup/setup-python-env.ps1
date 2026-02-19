# Muni Metro - Python Environment Setup (Windows)

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Muni Metro - Python Environment Setup" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

# Get the project root directory (2 levels up from this script)
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent (Split-Path -Parent $scriptDir)

Write-Host "Project root: $projectRoot" -ForegroundColor White
Write-Host ""

Set-Location $projectRoot

# Create virtual environment if it doesn't exist
if (-not (Test-Path ".venv")) {
    Write-Host "Creating virtual environment..." -ForegroundColor White
    python -m venv .venv
} else {
    Write-Host "Virtual environment already exists." -ForegroundColor Green
}

# Activate virtual environment
& ".venv\Scripts\Activate.ps1"

# Upgrade pip
Write-Host "Upgrading pip..." -ForegroundColor White
python -m pip install --upgrade pip

# Install dependencies
Write-Host "Installing dependencies from api/requirements.txt..." -ForegroundColor White
pip install -r api/requirements.txt

# Deactivate virtual environment
deactivate

Write-Host "==========================================" -ForegroundColor Green
Write-Host "Python environment setup complete!" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Green
Write-Host ""
Write-Host "To activate the environment:" -ForegroundColor White
Write-Host "  .venv\Scripts\Activate.ps1"
Write-Host ""
Write-Host "Next steps:" -ForegroundColor White
Write-Host "1. Download reference data: .\scripts\sync-reference-data.ps1 download"
Write-Host "2. Follow SETUP.md for project-specific configuration"
Write-Host ""
