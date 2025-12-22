# Sync training data with Google Cloud Storage (Windows PowerShell version)
#
# Usage:
#   .\scripts\sync-training-data.ps1 upload            # Upload local changes to GCS
#   .\scripts\sync-training-data.ps1 download          # Download changes from GCS
#   .\scripts\sync-training-data.ps1 both              # Sync both directions (default)
#   .\scripts\sync-training-data.ps1 delete <path>     # Delete file/directory from GCS

param(
    [Parameter(Position=0)]
    [string]$Command = "both",

    [Parameter(Position=1)]
    [string]$DeletePath = ""
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
Write-Host "Sync Training Data with GCS"
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
    Write-Host "Uploading training data to GCS..."
    Write-Host ""
    $gsutilArgs = @("-m", "rsync", "-r") + $EXCLUDE_FLAGS + @("artifacts/training_data", "$BUCKET/training_data")
    & gsutil $gsutilArgs
    Write-Host ""
    Write-Host "OK Training data uploaded to $BUCKET/training_data" -ForegroundColor Green
}
elseif ($cmd -eq "download") {
    Write-Host "Downloading training data from GCS (~270MB)..."
    Write-Host ""
    New-Item -ItemType Directory -Force -Path "artifacts/training_data" | Out-Null
    $gsutilArgs = @("-m", "rsync", "-r") + $EXCLUDE_FLAGS + @("$BUCKET/training_data", "artifacts/training_data")
    & gsutil $gsutilArgs
    Write-Host ""
    Write-Host "OK Training data downloaded to artifacts/training_data" -ForegroundColor Green
}
elseif ($cmd -eq "both") {
    Write-Host "Syncing training data bidirectionally..."
    Write-Host ""
    $uploadArgs = @("-m", "rsync", "-r") + $EXCLUDE_FLAGS + @("artifacts/training_data", "$BUCKET/training_data")
    & gsutil $uploadArgs
    $downloadArgs = @("-m", "rsync", "-r") + $EXCLUDE_FLAGS + @("$BUCKET/training_data", "artifacts/training_data")
    & gsutil $downloadArgs
    Write-Host ""
    Write-Host "OK Training data synced" -ForegroundColor Green
}
elseif ($cmd -eq "delete") {
    if ([string]::IsNullOrWhiteSpace($DeletePath)) {
        Write-Host "Error: delete command requires a path" -ForegroundColor Red
        Write-Host ""
        Write-Host "Usage: .\scripts\sync-training-data.ps1 delete <path>"
        Write-Host "Example: .\scripts\sync-training-data.ps1 delete training_data/2024-01/.DS_Store"
        exit 1
    }
    Write-Host "Deleting $BUCKET/$DeletePath from GCS..."
    Write-Host ""
    & gsutil -m rm -r "$BUCKET/$DeletePath"
    Write-Host ""
    Write-Host "OK Deleted $BUCKET/$DeletePath" -ForegroundColor Green
}
else {
    Write-Host "Error: Unknown command '$Command'" -ForegroundColor Red
    Write-Host ""
    Write-Host "Usage:"
    Write-Host "  .\scripts\sync-training-data.ps1 upload            # Upload local changes to GCS"
    Write-Host "  .\scripts\sync-training-data.ps1 download          # Download changes from GCS"
    Write-Host "  .\scripts\sync-training-data.ps1 both              # Sync both directions"
    Write-Host "  .\scripts\sync-training-data.ps1 delete <path>     # Delete file/directory from GCS"
    exit 1
}

Write-Host ""
Write-Host "=========================================="
Write-Host "Complete!"
Write-Host "=========================================="
