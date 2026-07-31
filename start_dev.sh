#!/usr/bin/env bash
set -e

echo "========================================"
echo "  StoMar - Full Stack Startup"
echo "========================================"
echo

cleanup() {
    echo
    echo "Shutting down..."
    kill $API_PID $WEB_PID 2>/dev/null
    wait $API_PID $WEB_PID 2>/dev/null
    exit 0
}
trap cleanup SIGINT SIGTERM

DIR="$(cd "$(dirname "$0")" && pwd)"

echo "[1/3] Starting FastAPI backend on :8000..."
cd "$DIR"
python -m uvicorn api.main:app --reload --port 8000 &
API_PID=$!

echo "[2/3] Waiting for API to start..."
sleep 3

echo "[3/3] Starting React frontend on :5173..."
cd "$DIR/web"
npm run dev &
WEB_PID=$!

echo
echo "========================================"
echo "  Both servers running!"
echo "  API:    http://localhost:8000"
echo "  UI:     http://localhost:5173"
echo "  Docs:   http://localhost:8000/docs"
echo "========================================"
echo
echo "Press Ctrl+C to stop both servers."

wait
