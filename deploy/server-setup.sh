#!/usr/bin/env bash
# Run once on a fresh Lightsail Ubuntu 22.04 instance.
# Usage: bash server-setup.sh <git-repo-url>
set -euo pipefail

REPO_URL=${1:?Usage: $0 <git-repo-url>}
APP_DIR=/opt/capital-dashboard
DOMAIN=portfolio.aachen-investment-club.de
EMAIL=mathis.makarski@aic.rwth-aachen.de

# --- system packages ---
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3-pip python3-venv nginx certbot python3-certbot-nginx unattended-upgrades git

# auto security patches
echo 'Dpkg::Options { "--force-confdef"; "--force-confold"; };' | sudo tee /etc/apt/apt.conf.d/local
sudo dpkg-reconfigure --priority=low unattended-upgrades

# --- app ---
sudo git clone "$REPO_URL" "$APP_DIR"
sudo chown -R ubuntu:ubuntu "$APP_DIR"
cd "$APP_DIR"
python3 -m venv .venv
.venv/bin/pip install --quiet -e .

# env file — fill in AWS keys before running this script
cat > "$APP_DIR/.env" <<'EOF'
S3_BUCKET=aic-fund-public-data
AWS_REGION=eu-central-1
DDB_TABLE=fund-baskets
AWS_ACCESS_KEY_ID=REPLACE_ME
AWS_SECRET_ACCESS_KEY=REPLACE_ME
EOF
chmod 600 "$APP_DIR/.env"

# --- systemd ---
sudo cp deploy/capital-dashboard.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable capital-dashboard
sudo systemctl start capital-dashboard

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
