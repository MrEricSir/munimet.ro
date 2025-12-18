# Muni Metro - Windows Setup Script
# Requires PowerShell 5.1+ and Administrator privileges for some operations

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Muni Metro - Windows Setup Script" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

# Check if running as Administrator
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
    Write-Host "WARNING: Not running as Administrator. Some installations may fail." -ForegroundColor Yellow
    Write-Host "Consider right-clicking PowerShell and selecting 'Run as Administrator'" -ForegroundColor Yellow
    Write-Host ""
}

# Detect and install package manager (winget or scoop)
Write-Host "Checking for package manager..." -ForegroundColor White

# Check for winget (Windows Package Manager)
$hasWinget = Get-Command winget -ErrorAction SilentlyContinue
if ($hasWinget) {
    Write-Host "Found winget (Windows Package Manager)" -ForegroundColor Green
    $pkgManager = "winget"
} else {
    # Check for scoop
    $hasScoop = Get-Command scoop -ErrorAction SilentlyContinue
    if ($hasScoop) {
        Write-Host "Found scoop" -ForegroundColor Green
        $pkgManager = "scoop"
    } else {
        # Offer to install scoop
        Write-Host "No package manager found." -ForegroundColor Yellow
        $installScoop = Read-Host "Would you like to install Scoop? (Y/N)"
        if ($installScoop -eq 'Y' -or $installScoop -eq 'y') {
            Write-Host "Installing Scoop..." -ForegroundColor Yellow
            Set-ExecutionPolicy RemoteSigned -Scope CurrentUser -Force
            Invoke-RestMethod get.scoop.sh | Invoke-Expression
            $pkgManager = "scoop"
        } else {
            Write-Host "ERROR: Cannot continue without a package manager." -ForegroundColor Red
            Write-Host "Please install winget or scoop manually and re-run this script." -ForegroundColor Red
            exit 1
        }
    }
}

Write-Host ""

# Function to install package based on available package manager
function Install-Package {
    param (
        [string]$packageName,
        [string]$wingetId,
        [string]$scoopName
    )

    Write-Host "Checking $packageName installation..." -ForegroundColor White

    if ($pkgManager -eq "winget") {
        # Check if already installed
        $installed = winget list --id $wingetId 2>&1 | Select-String $wingetId
        if ($installed) {
            Write-Host "$packageName is already installed." -ForegroundColor Green
        } else {
            Write-Host "Installing $packageName..." -ForegroundColor Yellow
            winget install --id $wingetId -e --accept-source-agreements --accept-package-agreements
        }
    } elseif ($pkgManager -eq "scoop") {
        $installed = scoop list | Select-String $scoopName
        if ($installed) {
            Write-Host "$packageName is already installed." -ForegroundColor Green
        } else {
            Write-Host "Installing $packageName..." -ForegroundColor Yellow
            scoop install $scoopName
        }
    }
    Write-Host ""
}

# Install Git
Install-Package -packageName "Git" -wingetId "Git.Git" -scoopName "git"

# Install Python 3.13+
Install-Package -packageName "Python 3.13" -wingetId "Python.Python.3.13" -scoopName "python"

# Refresh environment variables
$env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")

# Install git-annex (Windows port)
Write-Host "Checking git-annex installation..." -ForegroundColor White
$gitAnnexInstalled = Get-Command git-annex -ErrorAction SilentlyContinue
if (-not $gitAnnexInstalled) {
    Write-Host "Installing git-annex..." -ForegroundColor Yellow
    if ($pkgManager -eq "scoop") {
        scoop install git-annex
    } else {
        Write-Host "git-annex must be installed manually on Windows with winget." -ForegroundColor Yellow
        Write-Host "Visit: https://git-annex.branchable.com/install/Windows/" -ForegroundColor Yellow
        Write-Host "Or use scoop: scoop install git-annex" -ForegroundColor Yellow
    }
} else {
    Write-Host "git-annex is already installed." -ForegroundColor Green
}
Write-Host ""

# Note about tkinter
Write-Host "Note: tkinter should be included with Python on Windows." -ForegroundColor Cyan
Write-Host "If you encounter issues, reinstall Python with the 'tcl/tk' option enabled." -ForegroundColor Cyan
Write-Host ""

# Optional: Install Google Cloud SDK
$installGcloud = Read-Host "Do you want to install Google Cloud SDK? (required for cloud deployment) [y/N]"
if ($installGcloud -eq 'Y' -or $installGcloud -eq 'y') {
    if ($pkgManager -eq "scoop") {
        # gcloud requires the extras bucket
        Write-Host "Adding scoop extras bucket..." -ForegroundColor Yellow
        scoop bucket add extras 2>$null | Out-Null
        Write-Host "Installing Google Cloud SDK..." -ForegroundColor Yellow
        scoop install gcloud
    } else {
        Install-Package -packageName "Google Cloud SDK" -wingetId "Google.CloudSDK" -scoopName "gcloud"
    }
    Write-Host ""
    Write-Host "After installation completes, run: gcloud init" -ForegroundColor Cyan
}

# Optional: Install Docker Desktop
$installDocker = Read-Host "Do you want to install Docker Desktop? (required for containerized deployment) [y/N]"
if ($installDocker -eq 'Y' -or $installDocker -eq 'y') {
    Install-Package -packageName "Docker Desktop" -wingetId "Docker.DockerDesktop" -scoopName "docker"
    Write-Host "Please start Docker Desktop manually after installation." -ForegroundColor Yellow
}

# WSL2 recommendation
Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "RECOMMENDATION: Windows Subsystem for Linux" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "For the best development experience, consider using WSL2 (Windows Subsystem for Linux):" -ForegroundColor White
Write-Host ""
Write-Host "1. Install WSL2: wsl --install" -ForegroundColor Yellow
Write-Host "2. Install Ubuntu: wsl --install -d Ubuntu" -ForegroundColor Yellow
Write-Host "3. Run the Linux setup script from within WSL2" -ForegroundColor Yellow
Write-Host ""
Write-Host "WSL2 provides better compatibility with Linux-based development tools." -ForegroundColor White
Write-Host ""

Write-Host "==========================================" -ForegroundColor Green
Write-Host "Base dependencies installation complete!" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor White
Write-Host "1. Close and reopen PowerShell/Terminal to refresh environment variables" -ForegroundColor White
Write-Host "2. Run: .\scripts\setup\setup-python-env.ps1" -ForegroundColor White
Write-Host "3. Verify setup: .\scripts\setup\verify-setup-windows.ps1" -ForegroundColor White
Write-Host "4. Follow SETUP.md for git-annex and cloud storage configuration" -ForegroundColor White
Write-Host ""
Write-Host "For collaborators downloading training data:" -ForegroundColor Cyan
Write-Host "  - Set up HMAC credentials (ask project maintainer)" -ForegroundColor White
Write-Host "  - Enable git-annex remote: git annex enableremote gcs" -ForegroundColor White
Write-Host "  - Download files: git annex get artifacts/models/v1/" -ForegroundColor White
Write-Host "  - See GCS_SETUP.md for detailed instructions" -ForegroundColor White
Write-Host ""
