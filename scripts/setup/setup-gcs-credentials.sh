#!/bin/bash
# Automated GCS credentials setup for git-annex collaborators
# Generates HMAC keys and configures git-annex to use them

set -e

echo "=========================================="
echo "GCS Credentials Setup for Collaborators"
echo "=========================================="
echo ""

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Get the project root directory (2 levels up from this script)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT" || exit 1

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo -e "${RED}ERROR: gcloud is not installed${NC}"
    echo "Please install Google Cloud SDK first (see SETUP.md)"
    exit 1
fi

# Check if gsutil is available
if ! command -v gsutil &> /dev/null; then
    echo -e "${RED}ERROR: gsutil is not installed${NC}"
    echo "Please install Google Cloud SDK first (see SETUP.md)"
    exit 1
fi

# Check if git-annex is installed
if ! command -v git-annex &> /dev/null; then
    echo -e "${RED}ERROR: git-annex is not installed${NC}"
    echo "Please install git-annex first (see SETUP.md)"
    exit 1
fi

echo "[1/6] Checking gcloud authentication..."
# Check if user is authenticated
ACCOUNT=$(gcloud auth list --filter=status:ACTIVE --format="value(account)" 2>/dev/null | head -n1)
if [ -z "$ACCOUNT" ]; then
    echo -e "${YELLOW}Not authenticated with gcloud${NC}"
    echo "Launching gcloud authentication..."
    gcloud auth login
    ACCOUNT=$(gcloud auth list --filter=status:ACTIVE --format="value(account)" 2>/dev/null | head -n1)
fi

echo -e "${GREEN}✓ Authenticated as: $ACCOUNT${NC}"
echo ""

echo "[2/6] Checking project configuration..."
# Get current project
PROJECT_ID=$(gcloud config get-value project 2>/dev/null)
if [ -z "$PROJECT_ID" ]; then
    echo -e "${YELLOW}No project configured${NC}"
    echo "Please enter your GCP project ID (ask project maintainer):"
    read -r PROJECT_ID
    gcloud config set project "$PROJECT_ID"
fi

echo -e "${GREEN}✓ Using project: $PROJECT_ID${NC}"
echo ""

echo "[3/6] Verifying bucket access..."
# Check if user has access to the munimetro-annex bucket
BUCKET="gs://munimetro-annex"
if ! gsutil ls "$BUCKET" &>/dev/null; then
    echo -e "${RED}ERROR: Cannot access $BUCKET${NC}"
    echo "Please contact the project maintainer to grant you access."
    echo "Required role: Storage Object Admin"
    exit 1
fi

echo -e "${GREEN}✓ Can access $BUCKET${NC}"
echo ""

echo "[4/6] Generating HMAC credentials..."
# Check if HMAC keys already exist for this user
SERVICE_ACCOUNT="${PROJECT_ID}@appspot.gserviceaccount.com"

# Check if credentials already exist in ~/.aws/credentials
AWS_CREDS_FILE="${HOME}/.aws/credentials"
if [ -f "$AWS_CREDS_FILE" ] && grep -q "\[munimetro\]" "$AWS_CREDS_FILE"; then
    echo -e "${YELLOW}Credentials already exist in ${AWS_CREDS_FILE}${NC}"
    echo "Do you want to generate new credentials? This will overwrite existing ones. [y/N]"
    read -r RESPONSE
    if [[ ! $RESPONSE =~ ^[Yy]$ ]]; then
        echo "Using existing credentials..."
        SKIP_GENERATION=true
    fi
fi

if [ -z "$SKIP_GENERATION" ]; then
    # Generate new HMAC keys
    echo "Generating HMAC keys for service account: $SERVICE_ACCOUNT"
    HMAC_OUTPUT=$(gsutil hmac create "$SERVICE_ACCOUNT" 2>&1)

    if [ $? -ne 0 ]; then
        echo -e "${RED}ERROR: Failed to create HMAC keys${NC}"
        echo "$HMAC_OUTPUT"
        exit 1
    fi

    # Parse the output to extract credentials
    ACCESS_KEY_ID=$(echo "$HMAC_OUTPUT" | grep "Access ID:" | awk '{print $3}')
    SECRET_ACCESS_KEY=$(echo "$HMAC_OUTPUT" | grep "Secret:" | awk '{print $2}')

    if [ -z "$ACCESS_KEY_ID" ] || [ -z "$SECRET_ACCESS_KEY" ]; then
        echo -e "${RED}ERROR: Could not parse HMAC credentials${NC}"
        echo "$HMAC_OUTPUT"
        exit 1
    fi

    echo -e "${GREEN}✓ HMAC keys generated successfully${NC}"
    echo ""

    echo "[5/6] Saving credentials to ${AWS_CREDS_FILE}..."
    # Create ~/.aws directory if it doesn't exist
    mkdir -p "${HOME}/.aws"

    # Backup existing credentials if they exist
    if [ -f "$AWS_CREDS_FILE" ]; then
        cp "$AWS_CREDS_FILE" "${AWS_CREDS_FILE}.backup.$(date +%s)"
        echo -e "${CYAN}Backed up existing credentials${NC}"
    fi

    # Remove existing [munimetro] section if present
    if grep -q "\[munimetro\]" "$AWS_CREDS_FILE" 2>/dev/null; then
        # Remove the munimetro section (from [munimetro] to next section or EOF)
        sed -i.tmp '/\[munimetro\]/,/^\[.*\]/{/^\[munimetro\]/d; /^\[.*\]/!d;}' "$AWS_CREDS_FILE" 2>/dev/null || true
    fi

    # Append munimetro credentials
    cat >> "$AWS_CREDS_FILE" << EOF

[munimetro]
aws_access_key_id = ${ACCESS_KEY_ID}
aws_secret_access_key = ${SECRET_ACCESS_KEY}
EOF

    chmod 600 "$AWS_CREDS_FILE"
    echo -e "${GREEN}✓ Credentials saved to ${AWS_CREDS_FILE}${NC}"
else
    echo -e "${GREEN}✓ Using existing credentials${NC}"
fi
echo ""

echo "[6/6] Configuring git-annex to use munimetro profile..."
# Set environment variable for git-annex to use the munimetro profile
export AWS_PROFILE=munimetro

# Enable the google-cloud remote
if git annex enableremote google-cloud 2>/dev/null; then
    echo -e "${GREEN}✓ git-annex remote enabled successfully${NC}"
else
    echo -e "${YELLOW}⚠ Could not enable remote (may not exist yet)${NC}"
    echo "This is normal if the repository hasn't been fully set up yet"
fi
echo ""

# Test the connection
echo "Testing connection to Google Cloud Storage..."
if AWS_PROFILE=munimetro git annex testremote google-cloud --fast 2>/dev/null; then
    echo -e "${GREEN}✓ Connection test successful!${NC}"
else
    echo -e "${YELLOW}⚠ Connection test failed or remote not fully configured${NC}"
    echo "You may need to complete git-annex setup first"
fi
echo ""

echo "=========================================="
echo -e "${GREEN}✓ GCS credentials setup complete!${NC}"
echo "=========================================="
echo ""
echo "Your credentials are saved in: ${AWS_CREDS_FILE}"
echo "Profile name: munimetro"
echo ""
echo -e "${CYAN}To use these credentials:${NC}"
echo ""
echo "  # Option 1: Set for current session"
echo "  export AWS_PROFILE=munimetro"
echo ""
echo "  # Option 2: Set permanently (add to ~/.bashrc or ~/.zshrc)"
echo "  echo 'export AWS_PROFILE=munimetro' >> ~/.bashrc"
echo ""
echo "  # Then use git-annex normally:"
echo "  git annex get artifacts/training_data/"
echo ""
echo -e "${CYAN}Security notes:${NC}"
echo "  • Credentials are stored in ${AWS_CREDS_FILE} (mode 600)"
echo "  • These credentials only work with gs://munimetro-annex"
echo "  • To revoke access, delete the HMAC keys via: gsutil hmac list"
echo ""
