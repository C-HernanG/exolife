#!/usr/bin/env bash
# Entrypoint for the development container.  Installs git hooks
# via pre-commit if available and starts JupyterLab without a
# password or token.  This script is invoked by `compose.yaml` for
# the `exolife-dev` service.

set -euo pipefail

# Install pre‑commit hooks when present.  Failure should not abort
# container startup.
if command -v pre-commit > /dev/null 2>&1; then
  pre-commit install --install-hooks || true
fi

echo "🚀 Starting JupyterLab..."
exec jupyter lab \
  --ip=0.0.0.0 \
  --port=8888 \
  --no-browser \
  --NotebookApp.token='' \
  --NotebookApp.password=''