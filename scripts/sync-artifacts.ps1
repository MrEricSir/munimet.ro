# Sync artifacts (training images) with Google Cloud Storage (Windows PowerShell version)
#
# This is an alias for sync-training-data.ps1 for convenience.
# ML models are no longer used (replaced with OpenCV-based detection).
#
# Usage:
#   .\scripts\sync-artifacts.ps1 upload    # Upload local changes to GCS
#   .\scripts\sync-artifacts.ps1 download  # Download changes from GCS
#   .\scripts\sync-artifacts.ps1 both      # Sync both directions
#   .\scripts\sync-artifacts.ps1 delete <path>  # Delete from GCS

param(
    [Parameter(Position=0)]
    [string]$Command = "both",

    [Parameter(Position=1)]
    [string]$Path = ""
)

$SCRIPT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path

# Forward to sync-training-data.ps1
if ($Path) {
    & "$SCRIPT_DIR\sync-training-data.ps1" $Command $Path
} else {
    & "$SCRIPT_DIR\sync-training-data.ps1" $Command
}
