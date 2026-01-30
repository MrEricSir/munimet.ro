# Muni Metro - Windows Setup Verification Script
# Verifies that all dependencies and configurations are working correctly

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Muni Metro - Setup Verification" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

$allGood = $true

# Check Git
Write-Host "[1/6] Checking Git..." -ForegroundColor White
$git = Get-Command git -ErrorAction SilentlyContinue
if ($git) {
    $gitVersion = git --version
    Write-Host "  OK Git installed: $gitVersion" -ForegroundColor Green
} else {
    Write-Host "  ERROR Git not found" -ForegroundColor Red
    $allGood = $false
}
Write-Host ""

# Check Python
Write-Host "[2/6] Checking Python..." -ForegroundColor White
$python = Get-Command python -ErrorAction SilentlyContinue
if ($python) {
    $pythonVersion = python --version
    Write-Host "  OK Python installed: $pythonVersion" -ForegroundColor Green
} else {
    Write-Host "  ERROR Python not found" -ForegroundColor Red
    $allGood = $false
}
Write-Host ""

# Check gcloud (required for data sync)
Write-Host "[3/6] Checking Google Cloud SDK..." -ForegroundColor White
$gcloud = Get-Command gcloud -ErrorAction SilentlyContinue
if ($gcloud) {
    $gcloudInfo = gcloud version 2>$null | Select-String "Google Cloud SDK" | Select-Object -First 1
    Write-Host "  OK $gcloudInfo" -ForegroundColor Green

    # Check authentication
    $account = gcloud auth list --filter=status:ACTIVE --format="value(account)" 2>$null
    if ($account) {
        Write-Host "  OK Authenticated as: $account" -ForegroundColor Green
    } else {
        Write-Host "  WARNING Not authenticated" -ForegroundColor Yellow
        Write-Host "    Run: gcloud auth login" -ForegroundColor Yellow
    }
} else {
    Write-Host "  WARNING gcloud not found (required for training data sync)" -ForegroundColor Yellow
    Write-Host "    Install with: scoop bucket add extras && scoop install gcloud" -ForegroundColor Yellow
}
Write-Host ""

# Check Git repository
Write-Host "[4/6] Checking Git repository..." -ForegroundColor White
if (Test-Path ".git") {
    Write-Host "  OK Git repository found" -ForegroundColor Green
} else {
    Write-Host "  WARNING Not in a git repository" -ForegroundColor Yellow
    Write-Host "    Navigate to the munimetro directory first" -ForegroundColor Yellow
}
Write-Host ""

# Check Python virtual environment
Write-Host "[5/5] Checking Python virtual environment..." -ForegroundColor White
if (Test-Path "training/venv") {
    Write-Host "  OK Training virtual environment exists" -ForegroundColor Green
} else {
    Write-Host "  WARNING Training virtual environment not found" -ForegroundColor Yellow
    Write-Host "    Run: .\scripts\setup\setup-python-env.ps1" -ForegroundColor Yellow
}

if (Test-Path "api/venv") {
    Write-Host "  OK API virtual environment exists" -ForegroundColor Green
} else {
    Write-Host "  WARNING API virtual environment not found" -ForegroundColor Yellow
    Write-Host "    Run: .\scripts\setup\setup-python-env.ps1" -ForegroundColor Yellow
}
Write-Host ""

# Summary
Write-Host "==========================================" -ForegroundColor Cyan
if ($allGood) {
    Write-Host "OK All required dependencies installed!" -ForegroundColor Green
} else {
    Write-Host "WARNING Some issues found - see above for fixes" -ForegroundColor Yellow
}
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "Next steps:" -ForegroundColor White
Write-Host "1. Run setup scripts if needed: .\scripts\setup\setup-python-env.ps1" -ForegroundColor White
Write-Host "2. Authenticate with Google Cloud: gcloud auth login" -ForegroundColor White
Write-Host "3. Download model files: .\scripts\sync-models.ps1 download" -ForegroundColor White
Write-Host "4. Download training data: .\scripts\sync-training-data.ps1 download" -ForegroundColor White
Write-Host ""
Write-Host "For detailed setup instructions, see SETUP.md" -ForegroundColor Cyan
Write-Host ""
