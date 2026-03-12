#!/usr/bin/env bash
# MusicVision launcher — run from WSL
# Caps the 5090 to 450W (via Windows UAC prompt), starts backend + frontend,
# restores power limit on exit.

set -euo pipefail

GPU_INDEX="${1:-1}"
POWER_LIMIT="${2:-450}"
PROJECT_DIR="${3:-}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

# --- Power limit helpers (call nvidia-smi on the Windows host via RunAs) ---

set_power_limit() {
    local gpu="$1" watts="$2"
    powershell.exe -Command \
        "Start-Process nvidia-smi -ArgumentList '-i $gpu -pl $watts' -Verb RunAs -Wait" \
        2>/dev/null
}

query_power_limit() {
    nvidia-smi -i "$1" --query-gpu=power.limit --format=csv,noheader,nounits 2>/dev/null | tr -d '[:space:]'
}

query_default_power_limit() {
    nvidia-smi -i "$1" --query-gpu=power.default_limit --format=csv,noheader,nounits 2>/dev/null | tr -d '[:space:]'
}

# --- Read default and apply cap ---

DEFAULT_PL="$(query_default_power_limit "$GPU_INDEX")"
echo "GPU $GPU_INDEX default power limit: ${DEFAULT_PL}W"

set_power_limit "$GPU_INDEX" "$POWER_LIMIT"
CURRENT="$(query_power_limit "$GPU_INDEX")"
echo "GPU $GPU_INDEX power limit set to ${CURRENT}W"

# --- Cleanup on exit ---

cleanup() {
    echo ""
    echo "Shutting down..."

    # Kill background jobs (backend + frontend)
    kill $BACKEND_PID $FRONTEND_PID 2>/dev/null || true
    wait $BACKEND_PID $FRONTEND_PID 2>/dev/null || true

    # Restore power limit
    set_power_limit "$GPU_INDEX" "${DEFAULT_PL%.*}"
    echo "GPU $GPU_INDEX power limit restored to ${DEFAULT_PL}W"
}
trap cleanup EXIT INT TERM

# --- Start backend ---

SERVE_CMD="cd $REPO_DIR && uv run musicvision serve"
if [ -n "$PROJECT_DIR" ]; then
    SERVE_CMD="$SERVE_CMD $PROJECT_DIR"
fi
bash -c "$SERVE_CMD" &
BACKEND_PID=$!

# --- Start frontend ---

cd "$REPO_DIR/frontend" && npm run dev &
FRONTEND_PID=$!

echo ""
echo "Backend:  http://localhost:8000"
echo "Frontend: http://localhost:5173"
echo "Press Ctrl+C to stop both and restore power limit."
echo ""

# Wait for either to exit
wait -n $BACKEND_PID $FRONTEND_PID
