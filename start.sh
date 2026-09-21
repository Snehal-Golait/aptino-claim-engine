#!/usr/bin/env bash
set -e

echo "=== Aptino Multi-Agent Claim Decision Engine ==="
echo "Starting FastAPI backend on port 8000..."
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 &
FASTAPI_PID=$!

echo "Waiting for FastAPI backend to initialize..."
until curl -s http://localhost:8000/health | grep -q '"status":"healthy"'; do
    sleep 1
done
echo "FastAPI backend is healthy!"

echo "Starting Streamlit frontend on port 8501..."
streamlit run src/frontend/app.py \
    --server.port 8501 \
    --server.address 0.0.0.0 \
    --server.headless true \
    --browser.gatherUsageStats false &
STREAMLIT_PID=$!

echo "Both services running:"
echo "  - FastAPI:   http://localhost:8000 (docs at /docs)"
echo "  - Streamlit: http://localhost:8501"

# Keep container alive and trap termination
trap "kill $FASTAPI_PID $STREAMLIT_PID; exit 0" SIGINT SIGTERM
wait
