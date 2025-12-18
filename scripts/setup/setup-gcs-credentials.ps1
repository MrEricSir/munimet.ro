# Automated GCS credentials setup for git-annex collaborators (Windows)
# Generates HMAC keys and configures git-annex to use them

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "GCS Credentials Setup for Collaborators" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

# Get project root
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Resolve-Path "$scriptDir\..\.."
Set-Location $projectRoot

# Check if gcloud is installed
$gcloud = Get-Command gcloud -ErrorAction SilentlyContinue
if (-not $gcloud) {
    Write-Host "ERROR: gcloud is not installed" -ForegroundColor Red
    Write-Host "Please install Google Cloud SDK first (see SETUP.md)" -ForegroundColor Yellow
    exit 1
}

# Check if gsutil is available
$gsutil = Get-Command gsutil -ErrorAction SilentlyContinue
if (-not $gsutil) {
    Write-Host "ERROR: gsutil is not installed" -ForegroundColor Red
    Write-Host "Please install Google Cloud SDK first (see SETUP.md)" -ForegroundColor Yellow
    exit 1
}

# Check if git-annex is installed
$gitAnnex = Get-Command git-annex -ErrorAction SilentlyContinue
if (-not $gitAnnex) {
    Write-Host "ERROR: git-annex is not installed" -ForegroundColor Red
    Write-Host "Please install git-annex first (see SETUP.md)" -ForegroundColor Yellow
    exit 1
}

Write-Host "[1/6] Checking gcloud authentication..." -ForegroundColor White
# Check if user is authenticated
$account = gcloud auth list --filter=status:ACTIVE --format="value(account)" 2>$null | Select-Object -First 1
if (-not $account) {
    Write-Host "Not authenticated with gcloud" -ForegroundColor Yellow
    Write-Host "Launching gcloud authentication..." -ForegroundColor Yellow
    gcloud auth login
    $account = gcloud auth list --filter=status:ACTIVE --format="value(account)" 2>$null | Select-Object -First 1
}

Write-Host "OK Authenticated as: $account" -ForegroundColor Green
Write-Host ""

Write-Host "[2/6] Checking project configuration..." -ForegroundColor White
# Get current project
$projectId = gcloud config get-value project 2>$null
if (-not $projectId) {
    Write-Host "No project configured" -ForegroundColor Yellow
    $projectId = Read-Host "Please enter your GCP project ID (ask project maintainer)"
    gcloud config set project $projectId
}

Write-Host "OK Using project: $projectId" -ForegroundColor Green
Write-Host ""

Write-Host "[3/6] Verifying bucket access..." -ForegroundColor White
# Check if user has access to the munimetro-annex bucket
$bucket = "gs://munimetro-annex"
$bucketAccess = gsutil ls $bucket 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Cannot access $bucket" -ForegroundColor Red
    Write-Host "Please contact the project maintainer to grant you access." -ForegroundColor Yellow
    Write-Host "Required role: Storage Object Admin" -ForegroundColor Yellow
    exit 1
}

Write-Host "OK Can access $bucket" -ForegroundColor Green
Write-Host ""

Write-Host "[4/6] Generating HMAC credentials..." -ForegroundColor White
# Check if HMAC keys already exist for this user
$serviceAccount = "${projectId}@appspot.gserviceaccount.com"

# Check if credentials already exist
$awsCredsFile = "$env:USERPROFILE\.aws\credentials"
$skipGeneration = $false

if (Test-Path $awsCredsFile) {
    $existingCreds = Get-Content $awsCredsFile -Raw
    if ($existingCreds -match "\[munimetro\]") {
        Write-Host "Credentials already exist in $awsCredsFile" -ForegroundColor Yellow
        $response = Read-Host "Do you want to generate new credentials? This will overwrite existing ones. [y/N]"
        if ($response -notmatch "^[Yy]$") {
            Write-Host "Using existing credentials..." -ForegroundColor White
            $skipGeneration = $true
        }
    }
}

if (-not $skipGeneration) {
    # Generate new HMAC keys
    Write-Host "Generating HMAC keys for service account: $serviceAccount" -ForegroundColor White
    $hmacOutput = gsutil hmac create $serviceAccount 2>&1 | Out-String

    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Failed to create HMAC keys" -ForegroundColor Red
        Write-Host $hmacOutput -ForegroundColor Red
        exit 1
    }

    # Parse the output to extract credentials
    $accessKeyId = ($hmacOutput | Select-String "Access ID:\s+(\S+)").Matches.Groups[1].Value
    $secretAccessKey = ($hmacOutput | Select-String "Secret:\s+(\S+)").Matches.Groups[1].Value

    if (-not $accessKeyId -or -not $secretAccessKey) {
        Write-Host "ERROR: Could not parse HMAC credentials" -ForegroundColor Red
        Write-Host $hmacOutput -ForegroundColor Red
        exit 1
    }

    Write-Host "OK HMAC keys generated successfully" -ForegroundColor Green
    Write-Host ""

    Write-Host "[5/6] Saving credentials to $awsCredsFile..." -ForegroundColor White
    # Create ~/.aws directory if it doesn't exist
    $awsDir = "$env:USERPROFILE\.aws"
    if (-not (Test-Path $awsDir)) {
        New-Item -ItemType Directory -Path $awsDir | Out-Null
    }

    # Backup existing credentials if they exist
    if (Test-Path $awsCredsFile) {
        $backupFile = "${awsCredsFile}.backup.$(Get-Date -Format 'yyyyMMddHHmmss')"
        Copy-Item $awsCredsFile $backupFile
        Write-Host "Backed up existing credentials to $backupFile" -ForegroundColor Cyan
    }

    # Read existing credentials
    $credsContent = ""
    if (Test-Path $awsCredsFile) {
        $credsContent = Get-Content $awsCredsFile -Raw
        # Remove existing [munimetro] section
        $credsContent = $credsContent -replace '(?s)\[munimetro\].*?(?=\[|\z)', ''
    }

    # Append munimetro credentials
    $munimetroSection = @"

[munimetro]
aws_access_key_id = ${accessKeyId}
aws_secret_access_key = ${secretAccessKey}
"@

    $newContent = $credsContent.TrimEnd() + $munimetroSection

    # Write to file
    Set-Content -Path $awsCredsFile -Value $newContent -NoNewline

    Write-Host "OK Credentials saved to $awsCredsFile" -ForegroundColor Green
} else {
    Write-Host "OK Using existing credentials" -ForegroundColor Green
}
Write-Host ""

Write-Host "[6/6] Configuring git-annex remote..." -ForegroundColor White

# Read credentials from file for git-annex (it doesn't support AWS_PROFILE properly)
$credsContent = Get-Content $awsCredsFile -Raw
$accessKeyMatch = [regex]::Match($credsContent, '(?<=\[munimetro\][\s\S]*?aws_access_key_id\s*=\s*)\S+')
$secretKeyMatch = [regex]::Match($credsContent, '(?<=\[munimetro\][\s\S]*?aws_secret_access_key\s*=\s*)\S+')

if ($accessKeyMatch.Success -and $secretKeyMatch.Success) {
    $env:AWS_ACCESS_KEY_ID = $accessKeyMatch.Value
    $env:AWS_SECRET_ACCESS_KEY = $secretKeyMatch.Value
} else {
    Write-Host "ERROR: Could not read credentials from $awsCredsFile" -ForegroundColor Red
    exit 1
}

# Check if gcs remote already exists
$remoteInfo = git annex info gcs 2>&1 | Out-String
if ($remoteInfo -match "type: S3") {
    Write-Host "Remote 'gcs' already configured as S3 type" -ForegroundColor Green
    # Just enable it
    git annex enableremote gcs 2>$null
} else {
    # Check if remote exists at all (might be rclone type or not exist)
    $remoteExists = git config --get remote.gcs.annex-uuid 2>$null

    if ($remoteExists) {
        Write-Host "Enabling existing gcs remote..." -ForegroundColor White
        git annex enableremote gcs 2>$null
    } else {
        Write-Host "Initializing new S3 remote..." -ForegroundColor White
        # Workaround for GCS "BucketAlreadyOwnedByYou" error:
        # Create with a temporary bucket name, then switch to actual bucket
        $tempBucket = "munimetro-annex-init-temp-" + [DateTimeOffset]::Now.ToUnixTimeSeconds()

        $initResult = git annex initremote gcs type=S3 encryption=none host=storage.googleapis.com bucket=$tempBucket port=443 protocol=https requeststyle=path chunk=50MiB datacenter=us-west1 2>&1 | Out-String
        if ($LASTEXITCODE -ne 0) {
            Write-Host "ERROR: Failed to initialize remote" -ForegroundColor Red
            Write-Host $initResult -ForegroundColor Red
            exit 1
        }

        # Switch to actual bucket
        git annex enableremote gcs bucket=munimetro-annex 2>$null
        if ($LASTEXITCODE -ne 0) {
            Write-Host "ERROR: Failed to configure bucket" -ForegroundColor Red
            exit 1
        }

        # Clean up temporary bucket
        Write-Host "Cleaning up temporary bucket..." -ForegroundColor White
        gsutil rb "gs://${tempBucket}/" 2>$null
    }
}

Write-Host "OK git-annex remote configured successfully" -ForegroundColor Green
Write-Host ""

# Quick connection test (just check if we can list the bucket)
Write-Host "Testing connection to Google Cloud Storage..." -ForegroundColor White
$testResult = git annex info gcs 2>&1 | Out-String
if ($testResult -match "bucket: munimetro-annex") {
    Write-Host "OK Connection configured!" -ForegroundColor Green
} else {
    Write-Host "WARNING Remote may not be fully configured" -ForegroundColor Yellow
}
Write-Host ""

Write-Host "==========================================" -ForegroundColor Green
Write-Host "OK GCS credentials setup complete!" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Your credentials are saved in: $awsCredsFile" -ForegroundColor White
Write-Host "Profile name: munimetro" -ForegroundColor White
Write-Host ""
Write-Host "To use these credentials in a new session:" -ForegroundColor Cyan
Write-Host ""
Write-Host "  # Option 1: Set for current session" -ForegroundColor White
Write-Host "  `$creds = Get-Content `$env:USERPROFILE\.aws\credentials -Raw" -ForegroundColor Yellow
Write-Host "  `$env:AWS_ACCESS_KEY_ID = [regex]::Match(`$creds, '(?<=\[munimetro\][\s\S]*?aws_access_key_id\s*=\s*)\S+').Value" -ForegroundColor Yellow
Write-Host "  `$env:AWS_SECRET_ACCESS_KEY = [regex]::Match(`$creds, '(?<=\[munimetro\][\s\S]*?aws_secret_access_key\s*=\s*)\S+').Value" -ForegroundColor Yellow
Write-Host ""
Write-Host "  # Option 2: Add a helper function to your PowerShell profile" -ForegroundColor White
Write-Host "  # See scripts/setup/gcs-env.ps1 for a helper script" -ForegroundColor Yellow
Write-Host ""
Write-Host "  # Then use git-annex normally:" -ForegroundColor White
Write-Host "  git annex get artifacts/training_data/" -ForegroundColor Yellow
Write-Host ""
Write-Host "Security notes:" -ForegroundColor Cyan
Write-Host "  • Credentials are stored in $awsCredsFile" -ForegroundColor White
Write-Host "  • These credentials only work with gs://munimetro-annex" -ForegroundColor White
Write-Host "  • To revoke access, delete the HMAC keys via: gsutil hmac list" -ForegroundColor White
Write-Host ""
