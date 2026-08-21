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
MODEL_NAME="${MODEL_NAME:-qwen2.5:1.5b}"
USE_OLLAMA=false
USE_EXTERNAL_MODEL=false
FORCE_DEMO=false

case "${DEMO_MODE:-}" in
    true|TRUE|1|yes|YES) FORCE_DEMO=true ;;
esac

if $FORCE_DEMO; then
    echo "Demo mode explicitly requested; no model endpoint will be used."
elif [ -n "${MODEL_ENDPOINT:-}" ]; then
    echo "Using configured OpenAI-compatible model endpoint."
    USE_EXTERNAL_MODEL=true
elif command -v ollama &>/dev/null; then
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
if $USE_EXTERNAL_MODEL; then
    export MODEL_ENDPOINT
    export MODEL_NAME
    export DEMO_MODE="false"
    echo "Starting agents in LIVE mode (configured model endpoint)..."
elif $USE_OLLAMA; then
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

wait_for_ready() {
    local service_name="$1"
    local ready_url="$2"
    local service_pid="$3"

    echo -n "Waiting for $service_name..."
    for _ in $(seq 1 30); do
        if curl -sf "$ready_url" >/dev/null 2>&1; then
            echo " ready."
            return 0
        fi
        if ! kill -0 "$service_pid" 2>/dev/null; then
            echo " failed."
            echo "$service_name exited before becoming ready. Review the error above."
            return 1
        fi
        sleep 1
        echo -n "."
    done
    echo " timed out."
    return 1
}

# ── Start 3 A2A agents ──────────────────────────────────────────────
AGENT_NAME=triage AGENT_SKILLS=classify,prioritize AGENT_PORT=8001 \
    AGENT_PUBLIC_URL=http://127.0.0.1:8001/a2a \
    python3 -m uvicorn agent:app --host 127.0.0.1 --port 8001 &
TRIAGE_PID=$!
PIDS+=("$TRIAGE_PID")

AGENT_NAME=clinical AGENT_SKILLS=diagnose,recommend AGENT_PORT=8002 \
    AGENT_PUBLIC_URL=http://127.0.0.1:8002/a2a \
    python3 -m uvicorn agent:app --host 127.0.0.1 --port 8002 &
CLINICAL_PID=$!
PIDS+=("$CLINICAL_PID")

AGENT_NAME=scheduling AGENT_SKILLS=schedule,notify AGENT_PORT=8003 \
    AGENT_PUBLIC_URL=http://127.0.0.1:8003/a2a \
    python3 -m uvicorn agent:app --host 127.0.0.1 --port 8003 &
SCHEDULING_PID=$!
PIDS+=("$SCHEDULING_PID")

# Wait for all 3 agents
wait_for_ready "triage agent" "http://127.0.0.1:8001/ready" "$TRIAGE_PID"
wait_for_ready "clinical agent" "http://127.0.0.1:8002/ready" "$CLINICAL_PID"
wait_for_ready "scheduling agent" "http://127.0.0.1:8003/ready" "$SCHEDULING_PID"

# ── Start orchestrator (on :8000) ────────────────────────────────────
export AGENT_URLS="http://127.0.0.1:8001,http://127.0.0.1:8002,http://127.0.0.1:8003"
python3 -m uvicorn orchestrator:app --host 127.0.0.1 --port 8000 &
ORCHESTRATOR_PID=$!
PIDS+=("$ORCHESTRATOR_PID")

wait_for_ready "orchestrator" "http://127.0.0.1:8000/ready" "$ORCHESTRATOR_PID"

# ── Start Gradio UI (on :7860) ───────────────────────────────────────
export ORCHESTRATOR_URL="http://127.0.0.1:8000"
python3 ui.py &
PIDS+=("$!")

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
if $USE_EXTERNAL_MODEL; then
    echo "  Mode: LIVE (configured model endpoint + $MODEL_NAME)"
elif $USE_OLLAMA; then
    echo "  Mode: LIVE (Ollama + $MODEL_NAME)"
else
    echo "  Mode: DEMO (simulated agent responses)"
fi
echo "════════════════════════════════════════════════════════════"
echo "Press Ctrl+C to stop."
echo ""

wait
