# Muni Metro - Windows Setup Verification Script
# Verifies that all dependencies and configurations are working correctly

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Muni Metro - Setup Verification" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

$allGood = $true

# Check Git
Write-Host "[1/8] Checking Git..." -ForegroundColor White
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
Write-Host "[2/8] Checking Python..." -ForegroundColor White
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
Write-Host "[3/8] Checking git-annex..." -ForegroundColor White
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

# Check rclone
Write-Host "[4/8] Checking rclone..." -ForegroundColor White
$rclone = Get-Command rclone -ErrorAction SilentlyContinue
if ($rclone) {
    $rcloneVersion = rclone version 2>&1 | Select-String "^rclone" | Select-Object -First 1
    Write-Host "  OK rclone installed: $rcloneVersion" -ForegroundColor Green

    # Check for rclone remotes
    $remotes = rclone listremotes 2>$null
    if ($remotes -match "munimetro-gcs:") {
        Write-Host "  OK rclone remote configured: munimetro-gcs" -ForegroundColor Green
    } else {
        Write-Host "  WARNING rclone remote not configured" -ForegroundColor Yellow
        Write-Host "    Run: rclone config create munimetro-gcs gcs" -ForegroundColor Yellow
    }
} else {
    Write-Host "  ERROR rclone not found" -ForegroundColor Red
    $allGood = $false
}
Write-Host ""

# Check rclone git-annex support
Write-Host "[5/8] Checking rclone git-annex support..." -ForegroundColor White
$rcloneGitAnnex = rclone gitannex --help 2>&1
if ($rcloneGitAnnex -match "gitannex") {
    Write-Host "  OK rclone has built-in git-annex support" -ForegroundColor Green
} else {
    Write-Host "  WARNING rclone git-annex subcommand not found" -ForegroundColor Yellow
    Write-Host "    Update rclone to version 1.50+ for git-annex support" -ForegroundColor Yellow
    Write-Host "    Run: scoop update rclone" -ForegroundColor Yellow
}
Write-Host ""

# Check gcloud (optional)
Write-Host "[6/8] Checking Google Cloud SDK (optional)..." -ForegroundColor White
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
Write-Host "[7/8] Checking git-annex repository..." -ForegroundColor White
if (Test-Path ".git") {
    $annexInit = git config --local annex.uuid 2>$null
    if ($annexInit) {
        Write-Host "  OK git-annex repository initialized" -ForegroundColor Green

        # Check for google-cloud remote
        $remoteExists = git annex info google-cloud 2>&1
        if ($remoteExists -notmatch "not found") {
            Write-Host "  OK google-cloud remote configured" -ForegroundColor Green
        } else {
            Write-Host "  WARNING google-cloud remote not enabled" -ForegroundColor Yellow
            Write-Host "    Run: git annex enableremote google-cloud" -ForegroundColor Yellow
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
Write-Host "[8/8] Checking Python virtual environment..." -ForegroundColor White
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
Write-Host "2. Configure rclone: rclone config create munimetro-gcs gcs" -ForegroundColor White
Write-Host "3. Initialize gcloud: gcloud init" -ForegroundColor White
Write-Host "4. Enable git-annex remote: git annex enableremote google-cloud" -ForegroundColor White
Write-Host "5. Download model files: git annex get artifacts/models/v1/" -ForegroundColor White
Write-Host ""
Write-Host "For detailed setup instructions, see SETUP.md" -ForegroundColor Cyan
Write-Host ""
