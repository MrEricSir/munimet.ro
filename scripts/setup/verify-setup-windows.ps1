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

# Check git-annex
Write-Host "[3/6] Checking git-annex..." -ForegroundColor White
$gitAnnex = Get-Command git-annex -ErrorAction SilentlyContinue
if ($gitAnnex) {
    $annexVersion = git-annex version --raw 2>$null
    Write-Host "  OK git-annex installed: $annexVersion" -ForegroundColor Green
} else {
    Write-Host "  ERROR git-annex not found" -ForegroundColor Red
    Write-Host "    Install with: scoop install git-annex" -ForegroundColor Yellow
    $allGood = $false
}
Write-Host ""

# Check gcloud (optional)
Write-Host "[4/8] Checking Google Cloud SDK (optional)..." -ForegroundColor White
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
        Write-Host "    Run: gcloud init" -ForegroundColor Yellow
    }
} else {
    Write-Host "  WARNING gcloud not found (optional for cloud deployment)" -ForegroundColor Yellow
    Write-Host "    Install with: scoop bucket add extras && scoop install gcloud" -ForegroundColor Yellow
}
Write-Host ""

# Check git-annex repository
Write-Host "[5/6] Checking git-annex repository..." -ForegroundColor White
if (Test-Path ".git") {
    $annexInit = git config --local annex.uuid 2>$null
    if ($annexInit) {
        Write-Host "  OK git-annex repository initialized" -ForegroundColor Green

        # Check for gcs remote
        $remoteExists = git annex info gcs 2>&1
        if ($remoteExists -notmatch "not found") {
            Write-Host "  OK gcs remote configured" -ForegroundColor Green
        } else {
            Write-Host "  WARNING gcs remote not enabled" -ForegroundColor Yellow
            Write-Host "    Run: git annex enableremote gcs" -ForegroundColor Yellow
        }
    } else {
        Write-Host "  WARNING git-annex not initialized" -ForegroundColor Yellow
        Write-Host "    Run: git annex init your-computer-name" -ForegroundColor Yellow
    }
} else {
    Write-Host "  WARNING Not in a git repository" -ForegroundColor Yellow
    Write-Host "    Navigate to the munimetro directory first" -ForegroundColor Yellow
}
Write-Host ""

# Check Python virtual environment
Write-Host "[6/6] Checking Python virtual environment..." -ForegroundColor White
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
Write-Host "2. Initialize git-annex: git annex init your-computer-name" -ForegroundColor White
Write-Host "3. For collaborators: Set S3 credentials (see GCS_SETUP.md)" -ForegroundColor White
Write-Host "4. Enable git-annex remote: git annex enableremote gcs" -ForegroundColor White
Write-Host "5. Download model files: git annex get artifacts/models/v1/" -ForegroundColor White
Write-Host ""
Write-Host "For detailed setup instructions, see SETUP.md" -ForegroundColor Cyan
Write-Host ""
