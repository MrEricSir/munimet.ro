# Configure git-annex for automatic large file handling
# Run this after: git annex init "your-computer-name"

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Git-Annex Configuration" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

# Get project root
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Resolve-Path "$scriptDir\..\.."
Set-Location $projectRoot

# Check if git-annex is installed
$gitAnnex = Get-Command git-annex -ErrorAction SilentlyContinue
if (-not $gitAnnex) {
    Write-Host "ERROR: git-annex is not installed" -ForegroundColor Red
    Write-Host "Please install git-annex first (see SETUP.md)" -ForegroundColor Yellow
    exit 1
}

# Check if repository is initialized
$annexUuid = git config --local annex.uuid 2>$null
if (-not $annexUuid) {
    Write-Host "WARNING: Git-annex not initialized for this repository" -ForegroundColor Yellow
    Write-Host "Run: git annex init `"your-computer-name`"" -ForegroundColor White
    Write-Host ""
    $response = Read-Host "Would you like to initialize now? [y/N]"
    if ($response -eq 'y' -or $response -eq 'Y') {
        $computerName = Read-Host "Enter a descriptive name for this computer"
        git annex init $computerName
    } else {
        exit 1
    }
}

Write-Host "[1/6] Enabling adjusted branch for Windows..." -ForegroundColor White
# Windows doesn't handle symlinks well, so use adjusted branch with unlocked files
$currentBranch = git branch --show-current 2>$null
if ($currentBranch -notmatch "adjusted/") {
    Write-Host "Switching to adjusted/unlocked branch..." -ForegroundColor White
    git annex adjust --unlock 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "OK Adjusted branch enabled (files stored as regular files, not symlinks)" -ForegroundColor Green
    } else {
        Write-Host "WARNING Could not enable adjusted branch" -ForegroundColor Yellow
        Write-Host "  You may need to run: git annex adjust --unlock" -ForegroundColor Yellow
    }
} else {
    Write-Host "OK Already on adjusted branch" -ForegroundColor Green
}
Write-Host ""

Write-Host "[2/6] Configuring git-annex filter..." -ForegroundColor White
# Enable automatic annexing via git filter
git config filter.annex.process 'git-annex filter-process'
Write-Host "OK Git-annex filter enabled" -ForegroundColor Green
Write-Host ""

Write-Host "[3/6] Setting largefiles configuration..." -ForegroundColor White
# Configure which files should be automatically annexed
git annex config --set annex.largefiles 'largerthan=100kb or mimetype=image/*'
Write-Host "OK Files >100KB or images will be automatically annexed" -ForegroundColor Green
Write-Host ""

Write-Host "[4/6] Verifying .gitattributes..." -ForegroundColor White
if (Test-Path .gitattributes) {
    Write-Host "OK .gitattributes exists" -ForegroundColor Green
} else {
    Write-Host "WARNING: .gitattributes not found" -ForegroundColor Yellow
    Write-Host "This file should be committed to the repository" -ForegroundColor Yellow
}
Write-Host ""

Write-Host "[5/6] Enabling gcs remote..." -ForegroundColor White
# Check if remote exists
$remoteCheck = git annex info gcs 2>$null
if ($LASTEXITCODE -eq 0) {
    Write-Host "OK gcs remote is already available" -ForegroundColor Green
} else {
    Write-Host "Attempting to enable gcs remote..." -ForegroundColor White
    git annex enableremote gcs 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "OK gcs remote enabled successfully" -ForegroundColor Green
    } else {
        Write-Host "WARNING Could not enable remote (this is normal for first-time setup)" -ForegroundColor Yellow
        Write-Host "  The remote will be configured automatically when you clone" -ForegroundColor Yellow
    }
}
Write-Host ""

Write-Host "[6/6] Downloading pre-trained model..." -ForegroundColor White
# Check if model files are already present
$modelFile = "artifacts/models/v1/model.safetensors"
if ((Test-Path $modelFile) -and (-not (Get-Item $modelFile).LinkType)) {
    Write-Host "OK Model files already downloaded" -ForegroundColor Green
} else {
    $whereisCheck = git annex whereis artifacts/models/v1/ 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Downloading model files (856MB, this may take a few minutes)..." -ForegroundColor White
        git annex get artifacts/models/v1/ --jobs=4 2>$null
        if ($LASTEXITCODE -eq 0) {
            Write-Host "OK Model files downloaded successfully" -ForegroundColor Green
        } else {
            Write-Host "WARNING Could not download model files" -ForegroundColor Yellow
            Write-Host "  You can download them later with: git annex get artifacts/models/v1/" -ForegroundColor Yellow
        }
    } else {
        Write-Host "WARNING Model files not available yet (repository may need to be pushed)" -ForegroundColor Yellow
        Write-Host "  You can download them later with: git annex get artifacts/models/v1/" -ForegroundColor Yellow
    }
}
Write-Host ""

Write-Host "==========================================" -ForegroundColor Green
Write-Host "OK Git-annex configuration complete!" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Summary:" -ForegroundColor White
Write-Host "  Adjusted branch enabled (files as regular files, not symlinks)" -ForegroundColor White
Write-Host "  Automatic annexing enabled for files >100KB or images" -ForegroundColor White
Write-Host "  Model files downloaded (if available)" -ForegroundColor White
Write-Host ""
Write-Host "For collaborators with training data access:" -ForegroundColor Cyan
Write-Host "  1. Set `$env:AWS_ACCESS_KEY_ID and `$env:AWS_SECRET_ACCESS_KEY" -ForegroundColor White
Write-Host "  2. Run: git annex enableremote gcs" -ForegroundColor White
Write-Host "  3. Download training data: git annex get artifacts/training_data/" -ForegroundColor White
Write-Host ""
Write-Host "See GCS_SETUP.md for detailed instructions." -ForegroundColor Cyan
Write-Host ""
