#!/usr/bin/env bash
# Run backend unit/integration tests and the frontend type-check/build.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
"$ROOT/backend/.venv/bin/python" -m pytest "$@"
(cd "$ROOT/frontend" && npx tsc -b && npx oxlint src)
