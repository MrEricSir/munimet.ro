# Sync trained models with Google Cloud Storage (Windows PowerShell version)
#
# Usage:
#   .\scripts\sync-models.ps1 upload    # Upload local changes to GCS
#   .\scripts\sync-models.ps1 download  # Download changes from GCS
#   .\scripts\sync-models.ps1 both      # Sync both directions (default)

param(
    [Parameter(Position=0)]
    [string]$Command = "both"
)

$ErrorActionPreference = "Stop"

$BUCKET = "gs://munimetro-annex"

# Exclusions based on .gitignore (regex patterns for gsutil -x flag)
$EXCLUDE_PATTERNS = @(
    ".*\.DS_Store$",        # macOS metadata
    ".*__MACOSX.*",         # macOS archive metadata
    ".*__pycache__.*",      # Python cache
    ".*\.pyc$",             # Python bytecode
    ".*\.pyo$",             # Python optimized bytecode
    ".*\.pyd$",             # Python DLL
    ".*\.log$",             # Log files
    ".*\.ipynb_checkpoints.*",  # Jupyter checkpoints
    ".*\.swp$",             # Vim swap files
    ".*\.swo$",             # Vim swap files
    ".*~$"                  # Backup files
)

# Build exclusion flags
$EXCLUDE_FLAGS = @()
foreach ($pattern in $EXCLUDE_PATTERNS) {
    $EXCLUDE_FLAGS += "-x"
    $EXCLUDE_FLAGS += $pattern
}

Write-Host "=========================================="
Write-Host "Sync Models with GCS"
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
if ($cmd -eq "upload") {
    Write-Host "Uploading models to GCS (~856MB)..."
    Write-Host ""
    $gsutilArgs = @("-m", "rsync", "-r") + $EXCLUDE_FLAGS + @("artifacts/models", "$BUCKET/models")
    & gsutil $gsutilArgs
    Write-Host ""
    Write-Host "OK Models uploaded to $BUCKET/models" -ForegroundColor Green
}
elseif ($cmd -eq "download") {
    Write-Host "Downloading models from GCS (~856MB)..."
    Write-Host ""
    New-Item -ItemType Directory -Force -Path "artifacts/models" | Out-Null
    $gsutilArgs = @("-m", "rsync", "-r") + $EXCLUDE_FLAGS + @("$BUCKET/models", "artifacts/models")
    & gsutil $gsutilArgs
    Write-Host ""
    Write-Host "OK Models downloaded to artifacts/models" -ForegroundColor Green
}
elseif ($cmd -eq "both") {
    Write-Host "Syncing models bidirectionally..."
    Write-Host ""
    $uploadArgs = @("-m", "rsync", "-r") + $EXCLUDE_FLAGS + @("artifacts/models", "$BUCKET/models")
    & gsutil $uploadArgs
    $downloadArgs = @("-m", "rsync", "-r") + $EXCLUDE_FLAGS + @("$BUCKET/models", "artifacts/models")
    & gsutil $downloadArgs
    Write-Host ""
    Write-Host "OK Models synced" -ForegroundColor Green
}
else {
    Write-Host "Error: Unknown command '$Command'" -ForegroundColor Red
    Write-Host ""
    Write-Host "Usage:"
    Write-Host "  .\scripts\sync-models.ps1 upload    # Upload local changes to GCS"
    Write-Host "  .\scripts\sync-models.ps1 download  # Download changes from GCS"
    Write-Host "  .\scripts\sync-models.ps1 both      # Sync both directions"
    exit 1
}

Write-Host ""
Write-Host "=========================================="
Write-Host "Complete!"
Write-Host "=========================================="
