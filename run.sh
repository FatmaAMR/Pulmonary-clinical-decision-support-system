#!/usr/bin/env bash
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

echo "=================================================="
echo "Starting CHESTWISE - Complete Clinical System"
echo "=================================================="

# 1. Activate Virtual Environment
if [ -d "$PROJECT_ROOT/.venv" ]; then
    source "$PROJECT_ROOT/.venv/bin/activate"
    echo "✓ Activated virtual environment from .venv"
elif [ -d "$PROJECT_ROOT/rag_engine/.venv" ]; then
    source "$PROJECT_ROOT/rag_engine/.venv/bin/activate"
    echo "✓ Activated virtual environment from rag_engine/.venv"
elif [ -d "$HOME/Documents/rag_engine/.venv" ]; then
    source "$HOME/Documents/rag_engine/.venv/bin/activate"
    echo "✓ Activated virtual environment from ~/Documents/rag_engine/.venv"
fi

# 2. Export Environment Variables from .env if present
if [ -f "$PROJECT_ROOT/.env" ]; then
    set -a
    source "$PROJECT_ROOT/.env"
    set +a
    echo "✓ Loaded environment variables from .env"
elif [ -f "$PROJECT_ROOT/rag_engine/.env" ]; then
    set -a
    source "$PROJECT_ROOT/rag_engine/.env"
    set +a
    echo "✓ Loaded environment variables from rag_engine/.env"
fi

export PYTHONPATH="$PROJECT_ROOT/rag_engine:$PYTHONPATH"

# Function to stop all background background jobs on exit (Ctrl + C)
cleanup() {
    echo -e "\nShutting down all CHESTWISE services..."
    kill $(jobs -p) 2>/dev/null || true
    wait 2>/dev/null || true
    echo "✓ All servers stopped cleanly."
}
trap cleanup EXIT INT TERM

# 3. Start Pneumonia Classifier Server on port 8001
echo "⚙️  Starting Classifier on http://127.0.0.1:8001 ..."
(
    cd "$PROJECT_ROOT/classifiers/poneumonia"
    uvicorn main:app --reload --port 8001
) &

# 4. Start Local Web Server for Client UI on port 5500
CLIENT_DIR="$PROJECT_ROOT/client"
if [ -d "$CLIENT_DIR" ]; then
    echo "🖥️  Starting Client UI on http://127.0.0.1:5500 ..."
    (
        cd "$CLIENT_DIR"
        python3 -m http.server 5500 --bind 127.0.0.1 >/dev/null 2>&1
    ) &
fi

# Give background processes a second to spin up
sleep 1.5

# 5. Start Main RAG Engine on port 8000 in foreground
echo "Starting RAG Core API on http://127.0.0.1:8000 ..."
echo "--------------------------------------------------"
echo "Client UI:     http://localhost:5500"
echo "RAG API Docs:  http://localhost:8000/docs"
echo "Classifier:    http://localhost:8001/docs"
echo "--------------------------------------------------"
echo "Press CTRL+C anytime to stop all servers."

uvicorn api.main:app --reload --port 8000
