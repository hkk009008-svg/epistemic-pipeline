#!/bin/bash
# ============================================================
# Epistemic Verification Pipeline — Startup Script
# Launches both the FastAPI pipeline and n8n orchestrator
#
# Environment variables (all optional, defaults shown):
#   PORT=8000                    FastAPI port
#   N8N_PORT=5678                n8n port
#   OPENAI_API_KEY=              Pre-set API key (skip UI config)
#   OPENAI_MODEL=gpt-4o-mini    Default model
#   PIPELINE_URL=http://localhost:8000   URL n8n uses to reach API
#   N8N_DEFAULT_USER_EMAIL=admin@epistemic.local
#   N8N_DEFAULT_USER_PASSWORD=EpistemicPipeline2024!
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PATH="$HOME/node/bin:$HOME/bin:$PATH"

# Defaults (override via env)
export PORT="${PORT:-8000}"
export N8N_PORT="${N8N_PORT:-5678}"
export PIPELINE_URL="${PIPELINE_URL:-http://localhost:$PORT}"
export N8N_DEFAULT_USER_EMAIL="${N8N_DEFAULT_USER_EMAIL:-admin@epistemic.local}"
export N8N_DEFAULT_USER_PASSWORD="${N8N_DEFAULT_USER_PASSWORD:-EpistemicPipeline2024!}"

echo "============================================"
echo "  Epistemic Verification Pipeline"
echo "============================================"
echo ""

# Check dependencies
if ! command -v python3 &>/dev/null; then
  echo "ERROR: python3 not found"
  exit 1
fi

if ! command -v node &>/dev/null; then
  echo "ERROR: Node.js not found. Expected at ~/node/bin/node"
  exit 1
fi

# Kill any existing instances
echo "[1/4] Stopping existing services..."
pkill -f "uvicorn.*app:app" 2>/dev/null || true
pkill -f "uvicorn.*portfolio_api" 2>/dev/null || true
pkill -f "n8n start" 2>/dev/null || true
sleep 2

# Start FastAPI pipeline
echo "[2/4] Starting FastAPI pipeline on :$PORT..."
cd "$SCRIPT_DIR"
python3 -m uvicorn app:app --host 0.0.0.0 --port "$PORT" &
FASTAPI_PID=$!
echo "       PID: $FASTAPI_PID"

# Wait for FastAPI to be ready
echo "       Waiting for FastAPI..."
for i in $(seq 1 30); do
  if curl -s "http://localhost:$PORT/health" > /dev/null 2>&1; then
    echo "       FastAPI ready!"
    break
  fi
  sleep 1
done

# Start n8n
echo "[3/4] Starting n8n orchestrator on :$N8N_PORT..."
N8N_PORT="$N8N_PORT" \
  PIPELINE_URL="$PIPELINE_URL" \
  n8n start &>/dev/null &
N8N_PID=$!
echo "       PID: $N8N_PID"

# Wait for n8n to be ready
echo "       Waiting for n8n..."
for i in $(seq 1 30); do
  if curl -s "http://localhost:$N8N_PORT/healthz" > /dev/null 2>&1; then
    echo "       n8n ready!"
    break
  fi
  sleep 1
done

echo ""
echo "[4/4] All services running!"
echo ""
echo "============================================"
echo "  ENDPOINTS"
echo "============================================"
echo ""
echo "  Pipeline UI:      http://localhost:$PORT"
echo "  Pipeline API:     http://localhost:$PORT/api/pipeline"
echo "  Health Check:     http://localhost:$PORT/health"
echo "  Stress Test API:  http://localhost:$PORT/api/stress"
echo ""
echo "  n8n Dashboard:    http://localhost:$N8N_PORT"
echo "    Login:          $N8N_DEFAULT_USER_EMAIL"
echo ""
echo "  n8n Webhooks:"
echo "    Verify:   POST http://localhost:$N8N_PORT/webhook/epistemic-verify"
echo "              Body: {\"prompt\": \"your question\"}"
echo ""
echo "    Stress:   POST http://localhost:$N8N_PORT/webhook/epistemic-stress"
echo "              Body: {\"api_key\": \"sk-...\", \"category\": \"...\", \"count\": 5}"
echo ""
echo "    Health:   Automatic every 6 hours"
echo ""
echo "============================================"
echo "  Press Ctrl+C to stop all services"
echo "============================================"

# Wait for either to exit
trap "echo 'Shutting down...'; kill $FASTAPI_PID $N8N_PID 2>/dev/null; exit 0" INT TERM
wait
