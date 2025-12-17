#!/bin/bash
# Setup script for running SWE-bench experiments on a VPS
# Run from your LOCAL machine: bash scripts/setup_vps.sh

set -e

# Load .env if present
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
if [ -f "$SCRIPT_DIR/../.env" ]; then
    set -a
    source "$SCRIPT_DIR/../.env"
    set +a
fi

VPS_IP="${VPS_IP:?Set VPS_IP in .env or as env var}"
VPS_USER="${VPS_USER:-root}"
REPO_URL="https://github.com/Recursive-Safeguarding/the-complexity-trap.git"

echo "=== Deploying to $VPS_USER@$VPS_IP ==="

ssh "$VPS_USER@$VPS_IP" 'bash -s' << 'REMOTE_SCRIPT'
set -e
echo "=== Installing system dependencies ==="
apt-get update
apt-get install -y git curl tmux docker.io build-essential libssl-dev zlib1g-dev \
    libbz2-dev libreadline-dev libsqlite3-dev libncursesw5-dev xz-utils \
    tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev

echo "=== Starting Docker ==="
systemctl enable docker
systemctl start docker

echo "=== Installing uv ==="
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

echo "=== Cloning repository ==="
cd ~
rm -rf the-complexity-trap
git clone https://github.com/Recursive-Safeguarding/the-complexity-trap.git
cd the-complexity-trap

echo "=== Setting up Python environment ==="
uv venv .venv --python 3.12 --seed
source .venv/bin/activate
uv sync --extra dev  # Includes wandb, weave for sweeps

echo "=== Setup complete! ==="
REMOTE_SCRIPT

echo "=== Copying local .env to VPS ==="
scp .env "$VPS_USER@$VPS_IP:~/the-complexity-trap/.env"

echo ""
echo "=== Done! ==="
echo "SSH in and run:"
echo "  ssh $VPS_USER@$VPS_IP"
echo "  cd ~/the-complexity-trap && source .venv/bin/activate"
echo "  python scripts/run_sweep.py --model bedrock-qwen3-32b --strategy observation_masking --instances-slice :1 --wandb"
