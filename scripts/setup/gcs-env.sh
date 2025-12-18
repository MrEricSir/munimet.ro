#!/bin/bash
# Helper script to set GCS credentials for git-annex
# Source this script to set environment variables for the current session:
#   source scripts/setup/gcs-env.sh

AWS_CREDS_FILE="${HOME}/.aws/credentials"

if [ ! -f "$AWS_CREDS_FILE" ]; then
    echo "ERROR: Credentials file not found: $AWS_CREDS_FILE"
    echo "Run setup-gcs-credentials.sh first"
    return 1 2>/dev/null || exit 1
fi

export AWS_ACCESS_KEY_ID=$(grep -A2 '\[munimetro\]' "$AWS_CREDS_FILE" | grep aws_access_key_id | cut -d'=' -f2 | tr -d ' ')
export AWS_SECRET_ACCESS_KEY=$(grep -A2 '\[munimetro\]' "$AWS_CREDS_FILE" | grep aws_secret_access_key | cut -d'=' -f2 | tr -d ' ')

if [ -z "$AWS_ACCESS_KEY_ID" ] || [ -z "$AWS_SECRET_ACCESS_KEY" ]; then
    echo "ERROR: Could not find [munimetro] credentials in $AWS_CREDS_FILE"
    echo "Run setup-gcs-credentials.sh first"
    return 1 2>/dev/null || exit 1
fi

echo "GCS credentials loaded for git-annex"
