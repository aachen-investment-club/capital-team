#!/usr/bin/env bash
# Run once on a fresh Lightsail Ubuntu 22.04+ instance (2GB/2vCPU bundle or larger).
# Lightsail has no IAM-instance-profile support, so this box authenticates to AWS
# with a static key pair — use a purpose-scoped IAM user (S3 read on config/ and
# history/portfolio/*, S3 read/write/delete on backup/* only, DynamoDB read on
# fund-baskets only), never a broad personal/admin key.
# Usage: bash server-setup.sh <git-repo-url>
set -euo pipefail

REPO_URL=${1:?Usage: $0 <git-repo-url>}
APP_DIR=/opt/capital-dashboard
DOMAIN=portfolio.aachen-investment-club.de
EMAIL=mathis.makarski@aic.rwth-aachen.de

# --- system packages ---
sudo apt update && sudo apt upgrade -y
sudo apt install -y nginx certbot python3-certbot-nginx unattended-upgrades git curl

# auto security patches
echo 'Dpkg::Options { "--force-confdef"; "--force-confold"; };' | sudo tee /etc/apt/apt.conf.d/local
sudo dpkg-reconfigure --priority=low unattended-upgrades

# uv (installs python + manages the venv)
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# swap safety net for the 2 GB instance
if ! swapon --show | grep -q swapfile; then
  sudo fallocate -l 2G /swapfile && sudo chmod 600 /swapfile
  sudo mkswap /swapfile && sudo swapon /swapfile
  echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
fi

# --- app ---
sudo git clone "$REPO_URL" "$APP_DIR"
sudo chown -R ubuntu:ubuntu "$APP_DIR"
cd "$APP_DIR"
uv sync --frozen --extra ingest

# data + cache dirs
mkdir -p "$APP_DIR/data" "$APP_DIR/data/cache"

# env file — fill in the secrets before starting the services
cat > "$APP_DIR/.env" <<'EOF'
S3_BUCKET=aic-fund-public-data
AWS_REGION=eu-central-1
DDB_TABLE=fund-baskets
CAPITAL_DB=/opt/capital-dashboard/data/market.duckdb
CAPITAL_CACHE=/opt/capital-dashboard/data/cache
# LSEG (nightly EOD ingest)
LSEG_APP_KEY=REPLACE_ME
LSEG_USERNAME=REPLACE_ME
LSEG_PASSWORD=REPLACE_ME
# optional: free key from fred.stlouisfed.org — without it FRED history caps at ~3y
FRED_API_KEY=
# optional: healthchecks.io ping URL for the nightly ingest
HEALTHCHECK_URL=
# Static key for the scoped "capital-dashboard" IAM user (Lightsail has no
# instance-profile mechanism) — never a broad personal/admin key here.
AWS_ACCESS_KEY_ID=REPLACE_ME
AWS_SECRET_ACCESS_KEY=REPLACE_ME
EOF
chmod 600 "$APP_DIR/.env"

# --- seed the local store from the nightly S3 backup ---
echo "==> Seed the DuckDB store from the latest backup:"
echo "    aws s3 cp s3://aic-fund-public-data/backup/market.duckdb $APP_DIR/data/market.duckdb"
echo "    (or rebuild from LSEG: capital-ingest eod --start <inception> && capital-ingest fund market fred derived)"

# --- systemd: dashboard + nightly ingest ---
sudo cp deploy/capital-dashboard.service /etc/systemd/system/
sudo cp deploy/capital-ingest.service /etc/systemd/system/
sudo cp deploy/capital-ingest.timer /etc/systemd/system/
sudo cp deploy/capital-alert.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now capital-dashboard
sudo systemctl enable --now capital-ingest.timer

# --- nginx ---
sudo cp deploy/nginx-capital-dashboard.conf /etc/nginx/sites-available/capital-dashboard
sudo ln -sf /etc/nginx/sites-available/capital-dashboard /etc/nginx/sites-enabled/capital-dashboard
sudo nginx -t
sudo systemctl reload nginx

PUBLIC_IP=$(curl -s https://checkip.amazonaws.com)
HOSTED_ZONE_ID=Z02404541MO0M8NEH757G

echo ""
echo "==> Server setup done. Run this LOCALLY to create the DNS A record in Route 53:"
cat <<EOF

aws route53 change-resource-record-sets --hosted-zone-id $HOSTED_ZONE_ID --change-batch '{
  "Changes": [{
    "Action": "CREATE",
    "ResourceRecordSet": {
      "Name": "$DOMAIN",
      "Type": "A",
      "TTL": 300,
      "ResourceRecords": [{"Value": "$PUBLIC_IP"}]
    }
  }]
}'

EOF
echo "==> Then wait ~1 min for DNS to propagate and run on this server:"
echo "    sudo certbot --nginx -d $DOMAIN --email $EMAIL --agree-tos --non-interactive"
