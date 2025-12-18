# Helper script to set GCS credentials for git-annex
# Source this script to set environment variables for the current session:
#   . .\scripts\setup\gcs-env.ps1

$awsCredsFile = "$env:USERPROFILE\.aws\credentials"

if (-not (Test-Path $awsCredsFile)) {
    Write-Host "ERROR: Credentials file not found: $awsCredsFile" -ForegroundColor Red
    Write-Host "Run setup-gcs-credentials.ps1 first" -ForegroundColor Yellow
    return
}

$credsContent = Get-Content $awsCredsFile -Raw
$accessKeyMatch = [regex]::Match($credsContent, '(?<=\[munimetro\][\s\S]*?aws_access_key_id\s*=\s*)\S+')
$secretKeyMatch = [regex]::Match($credsContent, '(?<=\[munimetro\][\s\S]*?aws_secret_access_key\s*=\s*)\S+')

if ($accessKeyMatch.Success -and $secretKeyMatch.Success) {
    $env:AWS_ACCESS_KEY_ID = $accessKeyMatch.Value
    $env:AWS_SECRET_ACCESS_KEY = $secretKeyMatch.Value
    Write-Host "GCS credentials loaded for git-annex" -ForegroundColor Green
} else {
    Write-Host "ERROR: Could not find [munimetro] credentials in $awsCredsFile" -ForegroundColor Red
    Write-Host "Run setup-gcs-credentials.ps1 first" -ForegroundColor Yellow
}
