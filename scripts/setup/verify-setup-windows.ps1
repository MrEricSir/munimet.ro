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
    Write-Host "  ✓ Git installed: $gitVersion" -ForegroundColor Green
} else {
    Write-Host "  ✗ Git not found" -ForegroundColor Red
    $allGood = $false
}
Write-Host ""

# Check Python
Write-Host "[2/8] Checking Python..." -ForegroundColor White
$python = Get-Command python -ErrorAction SilentlyContinue
if ($python) {
    $pythonVersion = python --version
    Write-Host "  ✓ Python installed: $pythonVersion" -ForegroundColor Green
} else {
    Write-Host "  ✗ Python not found" -ForegroundColor Red
    $allGood = $false
}
Write-Host ""

# Check git-annex
Write-Host "[3/8] Checking git-annex..." -ForegroundColor White
$gitAnnex = Get-Command git-annex -ErrorAction SilentlyContinue
if ($gitAnnex) {
    $annexVersion = git-annex version --raw 2>$null
    Write-Host "  ✓ git-annex installed: $annexVersion" -ForegroundColor Green
} else {
    Write-Host "  ✗ git-annex not found" -ForegroundColor Red
    Write-Host "    Install with: scoop install git-annex" -ForegroundColor Yellow
    $allGood = $false
}
Write-Host ""

# Check rclone
Write-Host "[4/8] Checking rclone..." -ForegroundColor White
$rclone = Get-Command rclone -ErrorAction SilentlyContinue
if ($rclone) {
    $rcloneVersion = rclone version --check 2>&1 | Select-String "^rclone" | Select-Object -First 1
    Write-Host "  ✓ rclone installed: $rcloneVersion" -ForegroundColor Green

    # Check for rclone remotes
    $remotes = rclone listremotes 2>$null
    if ($remotes -match "munimetro-gcs:") {
        Write-Host "  ✓ rclone remote configured: munimetro-gcs" -ForegroundColor Green
    } else {
        Write-Host "  ⚠ rclone remote not configured" -ForegroundColor Yellow
        Write-Host "    Run: rclone config create munimetro-gcs gcs" -ForegroundColor Yellow
    }
} else {
    Write-Host "  ✗ rclone not found" -ForegroundColor Red
    $allGood = $false
}
Write-Host ""

# Check git-annex-remote-rclone
Write-Host "[5/8] Checking git-annex-remote-rclone..." -ForegroundColor White
$rcloneRemote = Get-Command git-annex-remote-rclone -ErrorAction SilentlyContinue
if ($rcloneRemote) {
    Write-Host "  ✓ git-annex-remote-rclone installed" -ForegroundColor Green
} else {
    Write-Host "  ✗ git-annex-remote-rclone not found in PATH" -ForegroundColor Red

    # Check if it's installed but not in PATH
    $scriptsDir = python -c "import sysconfig; print(sysconfig.get_path('scripts'))" 2>$null
    if (Test-Path "$scriptsDir\git-annex-remote-rclone.exe") {
        Write-Host "  ⚠ Found in Python Scripts but not in PATH" -ForegroundColor Yellow
        Write-Host "    Add to PATH: `$env:Path += ';$scriptsDir'" -ForegroundColor Yellow
    } else {
        Write-Host "    Install with: python -m pip install --user git-annex-remote-rclone" -ForegroundColor Yellow
        Write-Host "    Then add Scripts to PATH" -ForegroundColor Yellow
    }
    $allGood = $false
}
Write-Host ""

# Check gcloud (optional)
Write-Host "[6/8] Checking Google Cloud SDK (optional)..." -ForegroundColor White
$gcloud = Get-Command gcloud -ErrorAction SilentlyContinue
if ($gcloud) {
    $gcloudVersion = gcloud version --format="value(core.version)" 2>$null
    Write-Host "  ✓ gcloud installed: $gcloudVersion" -ForegroundColor Green

    # Check authentication
    $account = gcloud auth list --filter=status:ACTIVE --format="value(account)" 2>$null
    if ($account) {
        Write-Host "  ✓ Authenticated as: $account" -ForegroundColor Green
    } else {
        Write-Host "  ⚠ Not authenticated" -ForegroundColor Yellow
        Write-Host "    Run: gcloud init" -ForegroundColor Yellow
    }
} else {
    Write-Host "  ⚠ gcloud not found (optional for cloud deployment)" -ForegroundColor Yellow
    Write-Host "    Install with: scoop bucket add extras && scoop install gcloud" -ForegroundColor Yellow
}
Write-Host ""

# Check git-annex repository
Write-Host "[7/8] Checking git-annex repository..." -ForegroundColor White
if (Test-Path ".git") {
    $annexInit = git config --local annex.uuid 2>$null
    if ($annexInit) {
        Write-Host "  ✓ git-annex repository initialized" -ForegroundColor Green

        # Check for google-cloud remote
        $remoteExists = git annex info google-cloud 2>&1
        if ($remoteExists -notmatch "not found") {
            Write-Host "  ✓ google-cloud remote exists" -ForegroundColor Green
        } else {
            Write-Host "  ⚠ google-cloud remote not enabled" -ForegroundColor Yellow
            Write-Host "    Run: git annex enableremote google-cloud" -ForegroundColor Yellow
        }
    } else {
        Write-Host "  ⚠ git-annex not initialized" -ForegroundColor Yellow
        Write-Host "    Run: git annex init `"your-computer-name`"" -ForegroundColor Yellow
    }
} else {
    Write-Host "  ⚠ Not in a git repository" -ForegroundColor Yellow
    Write-Host "    Navigate to the munimetro directory first" -ForegroundColor Yellow
}
Write-Host ""

# Check Python virtual environment
Write-Host "[8/8] Checking Python virtual environment..." -ForegroundColor White
if (Test-Path "training/venv") {
    Write-Host "  ✓ Training virtual environment exists" -ForegroundColor Green
} else {
    Write-Host "  ⚠ Training virtual environment not found" -ForegroundColor Yellow
    Write-Host "    Run: .\scripts\setup\setup-python-env.ps1" -ForegroundColor Yellow
}

if (Test-Path "api/venv") {
    Write-Host "  ✓ API virtual environment exists" -ForegroundColor Green
} else {
    Write-Host "  ⚠ API virtual environment not found" -ForegroundColor Yellow
    Write-Host "    Run: .\scripts\setup\setup-python-env.ps1" -ForegroundColor Yellow
}
Write-Host ""

# Summary
Write-Host "==========================================" -ForegroundColor Cyan
if ($allGood) {
    Write-Host "✓ All required dependencies installed!" -ForegroundColor Green
} else {
    Write-Host "⚠ Some issues found - see above for fixes" -ForegroundColor Yellow
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
Write-Host "For detailed setup instructions, see: SETUP.md" -ForegroundColor Cyan
Write-Host ""
