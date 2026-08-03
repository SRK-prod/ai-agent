#!/usr/bin/env bash
# One-command dev-mode startup: backing services, backend, then the overlay UI.
# See `make run`.
set -euo pipefail
cd "$(dirname "$0")/.."

docker compose up -d qdrant redis

if [ -d .venv ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

cleanup() {
  echo "Shutting down backend..."
  kill "${BACKEND_PID:-0}" 2>/dev/null || true
}
trap cleanup EXIT

uvicorn meeting_copilot.server.main:app --loop uvloop \
  --host "${MEETING_COPILOT_HOST:-127.0.0.1}" \
  --port "${MEETING_COPILOT_PORT:-8765}" &
BACKEND_PID=$!

sleep 2  # let the backend finish booting before the UI tries to connect

python -m meeting_copilot.desktop.app
