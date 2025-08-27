#!/usr/bin/env bash
# Entrypoint for serving the ExoLife HTTP API.  This script
# launches a FastAPI application defined in `api/server.py` using
# uvicorn.  Adjust the module path or port as needed.  The
# service listens on all interfaces at port 8080 by default.

set -euo pipefail

APP_MODULE="api.server:app"
HOST="0.0.0.0"
PORT="8080"

echo "🌐 Starting ExoLife API on http://${HOST}:${PORT}" 
exec uvicorn "$APP_MODULE" --host "$HOST" --port "$PORT" --workers 2