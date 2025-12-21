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

# Function to setup environment for a component
function Setup-Component {
    param (
        [string]$component
    )

    $componentPath = Join-Path $projectRoot $component

    Write-Host "Setting up $component environment..." -ForegroundColor Blue

    if (-not (Test-Path $componentPath)) {
        Write-Host "Error: $componentPath does not exist" -ForegroundColor Red
        return
    }

    Set-Location $componentPath

    # Check if requirements.txt exists
    if (-not (Test-Path "requirements.txt")) {
        Write-Host "No requirements.txt found in $component, skipping..." -ForegroundColor Yellow
        return
    }

    # Create virtual environment if it doesn't exist
    if (-not (Test-Path "venv")) {
        Write-Host "Creating virtual environment..." -ForegroundColor White
        python -m venv venv
    } else {
        Write-Host "Virtual environment already exists." -ForegroundColor Green
    }

    # Activate virtual environment
    & "venv\Scripts\Activate.ps1"

    # Upgrade pip
    Write-Host "Upgrading pip..." -ForegroundColor White
    python -m pip install --upgrade pip

    # Install dependencies
    Write-Host "Installing dependencies from requirements.txt..." -ForegroundColor White
    pip install -r requirements.txt

    # Deactivate virtual environment
    deactivate

    Write-Host "$component environment setup complete!" -ForegroundColor Green
    Write-Host ""
}

# Ask which components to set up
Write-Host "Which components would you like to set up?" -ForegroundColor White
Write-Host ""
Write-Host "1) All components (training, api, tests)"
Write-Host "2) Training only"
Write-Host "3) API only"
Write-Host "4) Tests only"
Write-Host "5) Custom selection"
Write-Host ""
$choice = Read-Host "Enter your choice [1-5]"

switch ($choice) {
    1 {
        Setup-Component "training"
        Setup-Component "api"
        Setup-Component "tests"
    }
    2 {
        Setup-Component "training"
    }
    3 {
        Setup-Component "api"
    }
    4 {
        Setup-Component "tests"
    }
    5 {
        $setupTraining = Read-Host "Setup training? [y/N]"
        $setupApi = Read-Host "Setup api? [y/N]"
        $setupTests = Read-Host "Setup tests? [y/N]"

        if ($setupTraining -eq 'y' -or $setupTraining -eq 'Y') { Setup-Component "training" }
        if ($setupApi -eq 'y' -or $setupApi -eq 'Y') { Setup-Component "api" }
        if ($setupTests -eq 'y' -or $setupTests -eq 'Y') { Setup-Component "tests" }
    }
    default {
        Write-Host "Invalid choice. Exiting." -ForegroundColor Red
        exit 1
    }
}

Write-Host "==========================================" -ForegroundColor Green
Write-Host "Python environment setup complete!" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Green
Write-Host ""
Write-Host "To activate an environment:" -ForegroundColor White
Write-Host "  cd $projectRoot\training; .\venv\Scripts\Activate.ps1"
Write-Host "  cd $projectRoot\api; .\venv\Scripts\Activate.ps1"
Write-Host "  cd $projectRoot\tests; .\venv\Scripts\Activate.ps1"
Write-Host ""
Write-Host "Next steps:" -ForegroundColor White
Write-Host "1. Download training data: .\scripts\sync-training-data.ps1 download"
Write-Host "2. Download models: .\scripts\sync-models.ps1 download"
Write-Host "3. Follow SETUP.md for project-specific configuration"
Write-Host ""
