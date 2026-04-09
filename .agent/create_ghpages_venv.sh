#!/usr/bin/env bash
# Create a top-level .venv for MkDocs builds
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
VENV_DIR="$REPO_ROOT/.venv"

cd "$REPO_ROOT"

python3.14 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

python3.14 -m pip install --upgrade pip
python3.14 -m pip install mkdocs-material

echo ""
echo "Done. Activate with:"
echo "  source $VENV_DIR/bin/activate"
