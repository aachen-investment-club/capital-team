#!/usr/bin/env bash
# Deploy latest code to the running server.
# Run from /opt/capital-dashboard on the instance.
set -euo pipefail

git pull

# reinstall only if deps changed
if git diff HEAD@{1} --name-only | grep -qE "pyproject.toml|uv.lock"; then
  uv sync --frozen --extra ingest
fi

sudo systemctl restart capital-dashboard
echo "Deployed. Status:"
sudo systemctl status capital-dashboard --no-pager
