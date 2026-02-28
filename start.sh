#!/bin/bash
# ============================================================
# Epistemic Verification Pipeline — Startup Script
# Launches both the FastAPI pipeline and n8n orchestrator
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PATH="$HOME/node/bin:$HOME/bin:$PATH"

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
pkill -f "uvicorn.*portfolio_api" 2>/dev/null || true
pkill -f "n8n start" 2>/dev/null || true
sleep 2

# Start FastAPI pipeline (port 8000)
echo "[2/4] Starting FastAPI pipeline on :8000..."
cd "$SCRIPT_DIR"
python3 -m uvicorn portfolio_api:app --host 0.0.0.0 --port 8000 &
FASTAPI_PID=$!
echo "       PID: $FASTAPI_PID"

# Wait for FastAPI to be ready
echo "       Waiting for FastAPI..."
for i in $(seq 1 30); do
  if curl -s http://localhost:8000/ > /dev/null 2>&1; then
    echo "       FastAPI ready!"
    break
  fi
  sleep 1
done

# Start n8n (port 5678)
echo "[3/4] Starting n8n orchestrator on :5678..."
N8N_PORT=5678 n8n start &>/dev/null &
N8N_PID=$!
echo "       PID: $N8N_PID"

# Wait for n8n to be ready
echo "       Waiting for n8n..."
for i in $(seq 1 30); do
  if curl -s http://localhost:5678/healthz > /dev/null 2>&1; then
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
echo "  Pipeline UI:      http://localhost:8000"
echo "  Pipeline API:     http://localhost:8000/api/pipeline"
echo "  Stress Test API:  http://localhost:8000/api/stress"
echo ""
echo "  n8n Dashboard:    http://localhost:5678"
echo "    Login:          \${N8N_BASIC_AUTH_USER:-admin@epistemic.local}"
echo "    Password:       (set via N8N_BASIC_AUTH_PASSWORD env var)"
echo ""
echo "  n8n Webhooks:"
echo "    Verify:   POST http://localhost:5678/webhook/epistemic-verify"
echo "              Body: {\"prompt\": \"your question\"}"
echo ""
echo "    Stress:   POST http://localhost:5678/webhook/epistemic-stress"
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
