#!/usr/bin/env bash
#
# Bootstrap a fresh GPU VM for this assignment from a local checkout.
#
# Usage:
#   scripts/bootstrap_vm.sh khab1973@89.169.120.204
#
# Optional environment overrides:
#   REPO_URL=git@github.com:khab40/mlops-hw3.git
#   BRANCH=main
#   REMOTE_DIR=/home/khab1973/mlops-hw3
#   COPY_ENV=1              # copy local .env to VM
#   LOAD_DATA=1             # run scripts/load_data.py when data/bird is missing
#   START_TUNNEL=0          # set 1 to keep an SSH -L tunnel open after bootstrap
#   START_VLLM=1
#   START_AGENT=1
#   START_OBSERVABILITY=1

set -euo pipefail

usage() {
    sed -n '2,21p' "$0" >&2
}

if [[ $# -lt 1 || "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 1
fi

REMOTE="$1"
REPO_URL="${REPO_URL:-$(git remote get-url origin)}"
BRANCH="${BRANCH:-main}"
REMOTE_USER="${REMOTE%@*}"
if [[ "$REMOTE_USER" == "$REMOTE" ]]; then
    REMOTE_USER="${USER:-ubuntu}"
fi
REMOTE_DIR="${REMOTE_DIR:-/home/$REMOTE_USER/mlops-hw3}"
COPY_ENV="${COPY_ENV:-1}"
LOAD_DATA="${LOAD_DATA:-1}"
START_TUNNEL="${START_TUNNEL:-0}"
START_VLLM="${START_VLLM:-1}"
START_AGENT="${START_AGENT:-1}"
START_OBSERVABILITY="${START_OBSERVABILITY:-1}"
SSH_OPTS=(-o StrictHostKeyChecking=accept-new)

ssh_cmd() {
    ssh "${SSH_OPTS[@]}" "$REMOTE" "$@"
}

echo "Remote:      $REMOTE"
echo "Repo URL:    $REPO_URL"
echo "Branch:      $BRANCH"
echo "Remote dir:  $REMOTE_DIR"
echo
echo "Port-forward command:"
echo "ssh -L 3000:localhost:3000 -L 9090:localhost:9090 -L 3001:localhost:3001 -L 8000:localhost:8000 -L 8001:localhost:8001 $REMOTE"
echo

printf "Checking SSH access... "
ssh_cmd "true"
echo "ok"

echo "Installing VM system dependencies, Docker, uv, and repo..."
ssh_cmd "REPO_URL='$REPO_URL' BRANCH='$BRANCH' REMOTE_DIR='$REMOTE_DIR' LOAD_DATA='$LOAD_DATA' START_OBSERVABILITY='$START_OBSERVABILITY' bash -s" <<'REMOTE_BOOTSTRAP'
set -euo pipefail

log() {
    printf '\n[%s] %s\n' "$(date +%H:%M:%S)" "$*"
}

if command -v apt-get >/dev/null 2>&1; then
    log "Installing apt packages"
    sudo apt-get update
    sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
        build-essential \
        ca-certificates \
        curl \
        git \
        git-lfs \
        python3 \
        python3-dev \
        python3-pip \
        python3-venv \
        unzip \
        jq \
        docker.io \
        docker-compose-plugin
    sudo systemctl enable --now docker || true
else
    log "apt-get not found; assuming base packages are already installed"
fi

if ! command -v uv >/dev/null 2>&1; then
    log "Installing uv"
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"

if ! command -v docker >/dev/null 2>&1; then
    echo "docker is not installed or not on PATH" >&2
    exit 1
fi

if command -v nvidia-smi >/dev/null 2>&1; then
    log "GPU check"
    nvidia-smi
else
    log "nvidia-smi not found; vLLM may fail unless GPU drivers are installed"
fi

if [[ ! -d "$REMOTE_DIR/.git" ]]; then
    log "Cloning repository"
    mkdir -p "$(dirname "$REMOTE_DIR")"
    git clone --branch "$BRANCH" "$REPO_URL" "$REMOTE_DIR"
else
    log "Updating repository"
    git -C "$REMOTE_DIR" fetch origin "$BRANCH"
    git -C "$REMOTE_DIR" checkout "$BRANCH"
    git -C "$REMOTE_DIR" pull --ff-only origin "$BRANCH"
fi

cd "$REMOTE_DIR"

if [[ ! -f .env ]]; then
    log "Creating .env from .env.example"
    cp .env.example .env
fi

log "Installing Python dependencies"
uv sync

if [[ "$LOAD_DATA" == "1" && ! -d data/bird ]]; then
    log "Loading BIRD data"
    uv run python scripts/load_data.py
else
    log "Skipping BIRD load; data/bird already exists or LOAD_DATA=0"
fi

mkdir -p logs results

if [[ "$START_OBSERVABILITY" == "1" ]]; then
    log "Starting Prometheus, Grafana, and Langfuse"
    if docker compose version >/dev/null 2>&1; then
        docker compose up -d
    else
        sudo docker compose up -d
    fi
fi

log "Bootstrap dependency phase complete"
REMOTE_BOOTSTRAP

if [[ "$COPY_ENV" == "1" && -f .env ]]; then
    echo "Copying local .env to VM..."
    scp "${SSH_OPTS[@]}" .env "$REMOTE:$REMOTE_DIR/.env"
elif [[ "$COPY_ENV" == "1" ]]; then
    echo "COPY_ENV=1 but local .env was not found; VM keeps .env.example-derived file"
fi

if [[ "$START_VLLM" == "1" ]]; then
    echo "Starting vLLM..."
    ssh_cmd "REMOTE_DIR='$REMOTE_DIR' bash -s" <<'REMOTE_VLLM'
set -euo pipefail
cd "$REMOTE_DIR"
mkdir -p logs
if pgrep -af "vllm.entrypoints.openai.api_server" >/dev/null; then
    echo "vLLM already appears to be running"
else
    nohup scripts/start_vllm.sh > logs/vllm.log 2>&1 &
    echo $! > logs/vllm.pid
fi
for _ in $(seq 1 120); do
    if curl -fsS http://localhost:8000/v1/models >/dev/null 2>&1; then
        echo "vLLM is ready on :8000"
        exit 0
    fi
    sleep 5
done
echo "vLLM did not become ready within 10 minutes; check logs/vllm.log" >&2
exit 1
REMOTE_VLLM
fi

if [[ "$START_AGENT" == "1" ]]; then
    echo "Starting agent..."
    ssh_cmd "REMOTE_DIR='$REMOTE_DIR' bash -s" <<'REMOTE_AGENT'
set -euo pipefail
cd "$REMOTE_DIR"
mkdir -p logs
if pgrep -af "uvicorn agent.server:app" >/dev/null; then
    echo "Agent already appears to be running"
else
    nohup bash -lc 'set -a; source .env; set +a; uv run uvicorn agent.server:app --host 0.0.0.0 --port 8001' > logs/agent.log 2>&1 &
    echo $! > logs/agent.pid
fi
for _ in $(seq 1 60); do
    if curl -fsS http://localhost:8001/health >/dev/null 2>&1; then
        echo "Agent is ready on :8001"
        exit 0
    fi
    sleep 2
done
echo "Agent did not become ready; check logs/agent.log" >&2
exit 1
REMOTE_AGENT
fi

echo
echo "Bootstrap complete."
echo "Local URLs with SSH forwarding:"
echo "  Grafana:    http://localhost:3000    admin/admin"
echo "  Prometheus: http://localhost:9090"
echo "  Langfuse:   http://localhost:3001"
echo "  vLLM:       http://localhost:8000/v1/models"
echo "  Agent:      http://localhost:8001/health"

if [[ "$START_TUNNEL" == "1" ]]; then
    echo "Opening SSH tunnel; keep this process running."
    exec ssh "${SSH_OPTS[@]}" \
        -L 3000:localhost:3000 \
        -L 9090:localhost:9090 \
        -L 3001:localhost:3001 \
        -L 8000:localhost:8000 \
        -L 8001:localhost:8001 \
        "$REMOTE"
fi
