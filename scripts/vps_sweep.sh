#!/bin/bash
# Launch a WandB sweep on Contabo VPS
# Usage: ./scripts/vps_sweep.sh [sweep_yaml] [--start]

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

if [ -f "$PROJECT_ROOT/.env" ]; then
    set -a
    source "$PROJECT_ROOT/.env"
    set +a
fi

VPS_IP="${VPS_IP:?Set VPS_IP in .env}"
SWEEP_YAML="${1:-sweeps/smart_search.yaml}"
START_AGENT="${2:-}"

echo "Syncing configs..."
rsync -az "$PROJECT_ROOT/sweeps/" "root@$VPS_IP:~/the-complexity-trap/sweeps/"
scp "$PROJECT_ROOT/.env" "root@$VPS_IP:~/the-complexity-trap/.env"

echo "Creating sweep..."
SWEEP_OUTPUT=$(ssh root@$VPS_IP "cd ~/the-complexity-trap && source .venv/bin/activate && set -a && source .env && set +a && wandb sweep $SWEEP_YAML 2>&1")
echo "$SWEEP_OUTPUT"

# Extract sweep ID (portable, no -P flag)
SWEEP_ID=$(echo "$SWEEP_OUTPUT" | sed -n 's/.*wandb agent \([^ ]*\).*/\1/p' | head -1)
if [ -z "$SWEEP_ID" ]; then
    echo "ERROR: Could not extract sweep ID"
    exit 1
fi

if [ "$START_AGENT" = "--start" ]; then
    # Check for existing session
    if ssh root@$VPS_IP "tmux has-session -t sweep 2>/dev/null"; then
        echo "ERROR: tmux session 'sweep' already exists"
        echo ""
        echo "Current status:"
        ssh root@$VPS_IP "tmux capture-pane -t sweep -p | tail -20"
        echo ""
        echo "  Attach: ssh root@\$VPS_IP 'tmux attach -t sweep'"
        echo "  Kill:   ssh root@\$VPS_IP 'tmux kill-session -t sweep'"
        exit 1
    fi

    echo "Starting agent in tmux..."
    ssh root@$VPS_IP "tmux new-session -d -s sweep 'cd ~/the-complexity-trap && source .venv/bin/activate && set -a && source .env && set +a && wandb agent $SWEEP_ID'"
    echo ""
    echo "Agent started."
    echo "  Monitor: ssh root@$VPS_IP 'tmux attach -t sweep'"
    echo "  Stop:    ssh root@$VPS_IP 'tmux kill-session -t sweep'"
else
    echo ""
    echo "Sweep created: $SWEEP_ID"
    echo ""
    echo "To start:"
    echo "  ssh root@$VPS_IP"
    echo "  tmux new -s sweep 'cd ~/the-complexity-trap && source .venv/bin/activate && set -a && source .env && set +a && wandb agent $SWEEP_ID'"
fi
