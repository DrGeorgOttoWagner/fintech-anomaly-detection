#!/usr/bin/env bash
# One-time setup: Python virtualenv + backend deps, frontend deps and production build.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PYTHON:-python3.12}"
command -v "$PY" >/dev/null || PY=python3
echo "› Python: $($PY --version)"
"$PY" -m venv "$ROOT/backend/.venv"
"$ROOT/backend/.venv/bin/pip" install --upgrade pip >/dev/null
"$ROOT/backend/.venv/bin/pip" install -r "$ROOT/backend/requirements.lock"
"$ROOT/backend/.venv/bin/pip" install -e "$ROOT/backend" --no-deps >/dev/null
if [[ "$(uname)" == "Darwin" ]] && ! [ -e /opt/homebrew/opt/libomp/lib/libomp.dylib ] && ! [ -e /usr/local/opt/libomp/lib/libomp.dylib ]; then
  echo "! XGBoost/LightGBM need the OpenMP runtime on macOS: brew install libomp"
fi
(cd "$ROOT/frontend" && npm ci && npm run build)
[ -f "$ROOT/.env" ] || cp "$ROOT/.env.example" "$ROOT/.env"
echo "✓ Setup complete. Start FTAD with: ./scripts/start.sh"
