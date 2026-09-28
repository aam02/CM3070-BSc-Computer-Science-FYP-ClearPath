#!/usr/bin/env bash
# macOS: put libomp on DYLD path for LightGBM.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/.venv/bin/activate"
if command -v brew >/dev/null 2>&1; then
  OMP_PREFIX="$(brew --prefix libomp 2>/dev/null || true)"
  if [[ -n "${OMP_PREFIX}" && -d "${OMP_PREFIX}/lib" ]]; then
    export DYLD_LIBRARY_PATH="${OMP_PREFIX}/lib:${DYLD_LIBRARY_PATH:-}"
  fi
fi
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
exec "$@"
