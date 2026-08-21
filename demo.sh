#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"
SRC_DIR="$SCRIPT_DIR/src"
PIDS=()

cleanup() {
    echo ""
    echo "Shutting down..."
    for pid in "${PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    wait 2>/dev/null || true
    echo "Done."
}
trap cleanup EXIT INT TERM

# ── Python venv ──────────────────────────────────────────────────────
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
pip install -q -r "$SRC_DIR/requirements.txt"

# ── Ollama (optional) ───────────────────────────────────────────────
MODEL_NAME="qwen2.5:1.5b"
USE_OLLAMA=false

if command -v ollama &>/dev/null; then
    echo "Ollama found — checking if $MODEL_NAME is available..."
    if ollama list 2>/dev/null | grep -q "$MODEL_NAME"; then
        echo "Model $MODEL_NAME ready."
        USE_OLLAMA=true
    else
        echo "Pulling $MODEL_NAME (this may take a minute)..."
        if ollama pull "$MODEL_NAME"; then
            USE_OLLAMA=true
        else
            echo "Pull failed — falling back to demo mode."
        fi
    fi
else
    echo "Ollama not found — running in demo mode (simulated agent responses)."
fi

# ── Environment ──────────────────────────────────────────────────────
if $USE_OLLAMA; then
    export MODEL_ENDPOINT="http://localhost:11434/v1"
    export MODEL_NAME
    export DEMO_MODE="false"
    echo "Starting agents in LIVE mode (Ollama)..."
else
    export DEMO_MODE="true"
    export MODEL_ENDPOINT=""
    export MODEL_NAME
    echo "Starting agents in DEMO mode..."
fi

cd "$SRC_DIR"

# ── Start 3 A2A agents ──────────────────────────────────────────────
AGENT_NAME=triage AGENT_SKILLS=classify,prioritize AGENT_PORT=8001 \
    python3 -m uvicorn agent:app --host 127.0.0.1 --port 8001 &
PIDS+=($!)

AGENT_NAME=clinical AGENT_SKILLS=diagnose,recommend AGENT_PORT=8002 \
    python3 -m uvicorn agent:app --host 127.0.0.1 --port 8002 &
PIDS+=($!)

AGENT_NAME=scheduling AGENT_SKILLS=schedule,notify AGENT_PORT=8003 \
    python3 -m uvicorn agent:app --host 127.0.0.1 --port 8003 &
PIDS+=($!)

# Wait for all 3 agents
echo -n "Waiting for agents..."
for port in 8001 8002 8003; do
    for _ in $(seq 1 30); do
        if curl -sf "http://127.0.0.1:$port/health" >/dev/null 2>&1; then
            break
        fi
        sleep 1
        echo -n "."
    done
done
echo " ready."

# ── Start orchestrator (on :8000) ────────────────────────────────────
export AGENT_URLS="http://127.0.0.1:8001,http://127.0.0.1:8002,http://127.0.0.1:8003"
python3 -m uvicorn orchestrator:app --host 127.0.0.1 --port 8000 &
PIDS+=($!)

echo -n "Waiting for orchestrator..."
for _ in $(seq 1 30); do
    if curl -sf http://127.0.0.1:8000/health >/dev/null 2>&1; then
        echo " ready."
        break
    fi
    sleep 1
    echo -n "."
done

# ── Start Gradio UI (on :7860) ───────────────────────────────────────
export ORCHESTRATOR_URL="http://127.0.0.1:8000"
python3 ui.py &
PIDS+=($!)

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  Multi-Agent Health Assistant — running"
echo ""
echo "  Triage Agent:     http://127.0.0.1:8001"
echo "  Clinical Agent:   http://127.0.0.1:8002"
echo "  Scheduling Agent: http://127.0.0.1:8003"
echo "  Orchestrator:     http://127.0.0.1:8000"
echo "  Gradio UI:        http://127.0.0.1:7860"
echo ""
if $USE_OLLAMA; then
    echo "  Mode: LIVE (Ollama + $MODEL_NAME)"
else
    echo "  Mode: DEMO (simulated agent responses)"
fi
echo "════════════════════════════════════════════════════════════"
echo "Press Ctrl+C to stop."
echo ""

wait
