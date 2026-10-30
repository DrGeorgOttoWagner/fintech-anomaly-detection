#!/usr/bin/env bash
# Development mode: FastAPI with auto-reload on :8790 and Vite with HMR on :5173.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/backend"
"$ROOT/backend/.venv/bin/python" -m uvicorn ftad.app:create_app --factory --host 127.0.0.1 --port 8790 --reload &
API=$!
trap 'kill $API 2>/dev/null' EXIT
cd "$ROOT/frontend" && npm run dev
