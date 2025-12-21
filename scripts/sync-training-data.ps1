# Sync training data with Google Cloud Storage (Windows PowerShell version)
#
# Usage:
#   .\scripts\sync-training-data.ps1 upload    # Upload local changes to GCS
#   .\scripts\sync-training-data.ps1 download  # Download changes from GCS
#   .\scripts\sync-training-data.ps1 both      # Sync both directions (default)

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

switch ($Command.ToLower()) {
    "upload" {
        Write-Host "Uploading training data to GCS..."
        Write-Host ""
        $args = @("-m", "rsync", "-r") + $EXCLUDE_FLAGS + @("artifacts/training_data", "$BUCKET/training_data")
        & gsutil $args
        Write-Host ""
        Write-Host "✓ Training data uploaded to $BUCKET/training_data" -ForegroundColor Green
    }

    "download" {
        Write-Host "Downloading training data from GCS (~270MB)..."
        Write-Host ""
        New-Item -ItemType Directory -Force -Path "artifacts/training_data" | Out-Null
        $args = @("-m", "rsync", "-r") + $EXCLUDE_FLAGS + @("$BUCKET/training_data", "artifacts/training_data")
        & gsutil $args
        Write-Host ""
        Write-Host "✓ Training data downloaded to artifacts/training_data" -ForegroundColor Green
    }

    "both" {
        Write-Host "Syncing training data bidirectionally..."
        Write-Host ""
        $args1 = @("-m", "rsync", "-r") + $EXCLUDE_FLAGS + @("artifacts/training_data", "$BUCKET/training_data")
        & gsutil $args1
        $args2 = @("-m", "rsync", "-r") + $EXCLUDE_FLAGS + @("$BUCKET/training_data", "artifacts/training_data")
        & gsutil $args2
        Write-Host ""
        Write-Host "✓ Training data synced" -ForegroundColor Green
    }

    default {
        Write-Host "Error: Unknown command '$Command'" -ForegroundColor Red
        Write-Host ""
        Write-Host "Usage:"
        Write-Host "  .\scripts\sync-training-data.ps1 upload    # Upload local changes to GCS"
        Write-Host "  .\scripts\sync-training-data.ps1 download  # Download changes from GCS"
        Write-Host "  .\scripts\sync-training-data.ps1 both      # Sync both directions"
        exit 1
    }
}

Write-Host ""
Write-Host "=========================================="
Write-Host "Complete!"
Write-Host "=========================================="
