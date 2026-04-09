#!/usr/bin/env bash
# Rebuild and deploy MkDocs site to GitHub Pages
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
VENV_DIR="$REPO_ROOT/.venv"

cd "$REPO_ROOT"

# Activate venv
if [[ ! -d "$VENV_DIR" ]]; then
  echo "ERROR: venv not found at $VENV_DIR" >&2
  echo "Run: python3.14 -m venv $VENV_DIR && pip install mkdocs-material" >&2
  exit 1
fi
source "$VENV_DIR/bin/activate"

# Build and deploy
mkdocs gh-deploy --force
