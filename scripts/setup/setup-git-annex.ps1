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

Write-Host "[1/3] Configuring git-annex filter..." -ForegroundColor White
# Enable automatic annexing via git filter
git config filter.annex.process 'git-annex filter-process'
Write-Host "OK Git-annex filter enabled" -ForegroundColor Green
Write-Host ""

Write-Host "[2/3] Setting largefiles configuration..." -ForegroundColor White
# Configure which files should be automatically annexed
git annex config --set annex.largefiles 'largerthan=100kb or mimetype=image/*'
Write-Host "OK Files >100KB or images will be automatically annexed" -ForegroundColor Green
Write-Host ""

Write-Host "[3/3] Verifying .gitattributes..." -ForegroundColor White
if (Test-Path .gitattributes) {
    Write-Host "OK .gitattributes exists" -ForegroundColor Green
} else {
    Write-Host "WARNING: .gitattributes not found" -ForegroundColor Yellow
    Write-Host "This file should be committed to the repository" -ForegroundColor Yellow
}
Write-Host ""

Write-Host "==========================================" -ForegroundColor Green
Write-Host "OK Git-annex configuration complete!" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Automatic annexing is now enabled. When you run:" -ForegroundColor White
Write-Host "  git add artifacts/training_data/images/*.jpg" -ForegroundColor Cyan
Write-Host ""
Write-Host "Large files will be automatically annexed (symlinked)" -ForegroundColor White
Write-Host "instead of being added to regular git." -ForegroundColor White
Write-Host ""
Write-Host "The pre-commit hook will prevent accidental commits" -ForegroundColor White
Write-Host "of large files that weren't properly annexed." -ForegroundColor White
Write-Host ""
