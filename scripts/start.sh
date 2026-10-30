#!/usr/bin/env bash
# Start FTAD (API + built UI) on http://127.0.0.1:8790 — loopback only.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
if [ ! -f "$ROOT/frontend/dist/index.html" ]; then
  echo "› Building frontend…"; (cd "$ROOT/frontend" && npm run build)
fi
cd "$ROOT/backend"
echo "› FTAD running at http://127.0.0.1:${FTAD_PORT:-8790}"
exec "$ROOT/backend/.venv/bin/python" -m ftad.app
