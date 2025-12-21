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

switch ($Command.ToLower()) {
    "upload" {
        Write-Host "Uploading artifacts to GCS..."
        Write-Host ""

        Write-Host "[1/2] Uploading training data..."
        $args1 = @("-m", "rsync", "-r") + $EXCLUDE_FLAGS + @("artifacts/training_data", "$BUCKET/training_data")
        & gsutil $args1
        Write-Host "✓ Training data uploaded" -ForegroundColor Green
        Write-Host ""

        Write-Host "[2/2] Uploading models..."
        $args2 = @("-m", "rsync", "-r") + $EXCLUDE_FLAGS + @("artifacts/models", "$BUCKET/models")
        & gsutil $args2
        Write-Host "✓ Models uploaded" -ForegroundColor Green
    }

    "download" {
        Write-Host "Downloading artifacts from GCS..."
        Write-Host ""

        Write-Host "[1/2] Downloading training data (~270MB)..."
        New-Item -ItemType Directory -Force -Path "artifacts/training_data" | Out-Null
        $args1 = @("-m", "rsync", "-r") + $EXCLUDE_FLAGS + @("$BUCKET/training_data", "artifacts/training_data")
        & gsutil $args1
        Write-Host "✓ Training data downloaded" -ForegroundColor Green
        Write-Host ""

        Write-Host "[2/2] Downloading models (~856MB)..."
        New-Item -ItemType Directory -Force -Path "artifacts/models" | Out-Null
        $args2 = @("-m", "rsync", "-r") + $EXCLUDE_FLAGS + @("$BUCKET/models", "artifacts/models")
        & gsutil $args2
        Write-Host "✓ Models downloaded" -ForegroundColor Green
    }

    "both" {
        Write-Host "Syncing artifacts bidirectionally..."
        Write-Host ""

        Write-Host "[1/2] Syncing training data..."
        $args1 = @("-m", "rsync", "-r") + $EXCLUDE_FLAGS + @("artifacts/training_data", "$BUCKET/training_data")
        & gsutil $args1
        $args2 = @("-m", "rsync", "-r") + $EXCLUDE_FLAGS + @("$BUCKET/training_data", "artifacts/training_data")
        & gsutil $args2
        Write-Host "✓ Training data synced" -ForegroundColor Green
        Write-Host ""

        Write-Host "[2/2] Syncing models..."
        $args3 = @("-m", "rsync", "-r") + $EXCLUDE_FLAGS + @("artifacts/models", "$BUCKET/models")
        & gsutil $args3
        $args4 = @("-m", "rsync", "-r") + $EXCLUDE_FLAGS + @("$BUCKET/models", "artifacts/models")
        & gsutil $args4
        Write-Host "✓ Models synced" -ForegroundColor Green
    }

    default {
        Write-Host "Error: Unknown command '$Command'" -ForegroundColor Red
        Write-Host ""
        Write-Host "Usage:"
        Write-Host "  .\scripts\sync-artifacts.ps1 upload    # Upload local changes to GCS"
        Write-Host "  .\scripts\sync-artifacts.ps1 download  # Download changes from GCS"
        Write-Host "  .\scripts\sync-artifacts.ps1 both      # Sync both directions"
        exit 1
    }
}

Write-Host ""
Write-Host "=========================================="
Write-Host "Sync complete!"
Write-Host "=========================================="
