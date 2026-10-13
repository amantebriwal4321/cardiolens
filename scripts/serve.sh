#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
export BACKEND_URL="${BACKEND_URL:-http://127.0.0.1:8091}"
uvicorn backend.app.main:app --host 127.0.0.1 --port 8091 &
api_pid=$!
trap 'kill "$api_pid"' EXIT
cd web
exec npx next start -p 3921 -H 0.0.0.0
