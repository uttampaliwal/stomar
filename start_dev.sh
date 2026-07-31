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

# --- Ensure uv is installed ---
if ! command -v uv >/dev/null 2>&1; then
    echo "[setup] 'uv' not found. Installing uv (Python package manager)..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
    if ! command -v uv >/dev/null 2>&1; then
        echo "[error] uv installation failed. Install it manually: https://docs.astral.sh/uv/getting-started/installation/"
        exit 1
    fi
fi

# --- Sync Python env (creates .venv with pinned Python, installs deps from uv.lock) ---
echo "[setup] Syncing Python dependencies with uv..."
cd "$DIR"
uv sync

# --- Install Node deps if vite missing (npm ci = reproducible from lockfile) ---
if [ ! -f "$DIR/web/node_modules/.bin/vite" ]; then
    if ! command -v node >/dev/null 2>&1; then
        echo "[error] Node.js not found. Install Node 22+ (https://nodejs.org) or use nvm (see web/.nvmrc)."
        exit 1
    fi
    echo "[setup] Installing Node dependencies..."
    cd "$DIR/web"
    npm ci
fi

echo "[1/3] Starting FastAPI backend on :8000..."
cd "$DIR"
uv run uvicorn api.main:app --reload --port 8000 &
API_PID=$!

echo "[2/3] Waiting for API to start..."
sleep 3

echo "[3/3] Starting React frontend on :5173..."
cd "$DIR/web"
npx vite --host &
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
