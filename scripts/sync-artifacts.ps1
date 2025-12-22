# Sync artifacts with Google Cloud Storage (Windows PowerShell version)
#
# Usage:
#   .\scripts\sync-artifacts.ps1 upload    # Upload local changes to GCS
#   .\scripts\sync-artifacts.ps1 download  # Download changes from GCS
#   .\scripts\sync-artifacts.ps1 both      # Sync both directions

param(
    [Parameter(Position=0)]
    [string]$Command = "both"
)

$ErrorActionPreference = "Stop"

$SCRIPT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path

Write-Host "=========================================="
Write-Host "Sync Artifacts with Google Cloud Storage"
Write-Host "=========================================="
Write-Host ""

# Check authentication
try {
    $account = gcloud auth list --filter=status:ACTIVE --format="value(account)" 2>&1
    if ([string]::IsNullOrWhiteSpace($account)) {
        throw "Not authenticated"
    }
} catch {
    Write-Host "Error: Not authenticated with Google Cloud" -ForegroundColor Red
    Write-Host "Run: gcloud auth login"
    exit 1
}

$cmd = $Command.ToLower()
if ($cmd -eq "upload" -or $cmd -eq "download" -or $cmd -eq "both") {
    Write-Host "[1/2] Syncing training data..."
    & "$SCRIPT_DIR\sync-training-data.ps1" $Command
    Write-Host ""

    Write-Host "[2/2] Syncing models..."
    & "$SCRIPT_DIR\sync-models.ps1" $Command
}
else {
    Write-Host "Error: Unknown command '$Command'" -ForegroundColor Red
    Write-Host ""
    Write-Host "Usage:"
    Write-Host "  .\scripts\sync-artifacts.ps1 upload    # Upload local changes to GCS"
    Write-Host "  .\scripts\sync-artifacts.ps1 download  # Download changes from GCS"
    Write-Host "  .\scripts\sync-artifacts.ps1 both      # Sync both directions"
    exit 1
}

Write-Host ""
Write-Host "=========================================="
Write-Host "Sync complete!"
Write-Host "=========================================="
