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
        gsutil -m rsync -r artifacts/training_data "$BUCKET/training_data"
        Write-Host "✓ Training data uploaded" -ForegroundColor Green
        Write-Host ""

        Write-Host "[2/2] Uploading models..."
        gsutil -m rsync -r artifacts/models "$BUCKET/models"
        Write-Host "✓ Models uploaded" -ForegroundColor Green
    }

    "download" {
        Write-Host "Downloading artifacts from GCS..."
        Write-Host ""

        Write-Host "[1/2] Downloading training data (~270MB)..."
        New-Item -ItemType Directory -Force -Path "artifacts/training_data" | Out-Null
        gsutil -m rsync -r "$BUCKET/training_data" artifacts/training_data
        Write-Host "✓ Training data downloaded" -ForegroundColor Green
        Write-Host ""

        Write-Host "[2/2] Downloading models (~856MB)..."
        New-Item -ItemType Directory -Force -Path "artifacts/models" | Out-Null
        gsutil -m rsync -r "$BUCKET/models" artifacts/models
        Write-Host "✓ Models downloaded" -ForegroundColor Green
    }

    "both" {
        Write-Host "Syncing artifacts bidirectionally..."
        Write-Host ""

        Write-Host "[1/2] Syncing training data..."
        gsutil -m rsync -r artifacts/training_data "$BUCKET/training_data"
        gsutil -m rsync -r "$BUCKET/training_data" artifacts/training_data
        Write-Host "✓ Training data synced" -ForegroundColor Green
        Write-Host ""

        Write-Host "[2/2] Syncing models..."
        gsutil -m rsync -r artifacts/models "$BUCKET/models"
        gsutil -m rsync -r "$BUCKET/models" artifacts/models
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
