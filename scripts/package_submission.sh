#!/usr/bin/env bash
# Build submission zip (code + demo artefacts).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${1:-$ROOT/ClearPath_submission.zip}"
cd "$ROOT"
rm -f "$OUT"

zip -r -X "$OUT" \
  clearpath \
  scripts/run.sh \
  scripts/download_oulad.py \
  tests \
  requirements.txt \
  README.md \
  artifacts/models \
  artifacts/metrics \
  artifacts/figures \
  artifacts/shap/faithfulness.json \
  artifacts/shap/stability.json \
  artifacts/shap/shap_demo_v4_meta.json \
  artifacts/shap/shap_demo_v4.parquet \
  data/processed/rankings_demo_v4.parquet \
  -x "**/__pycache__/*" \
     "**/*.pyc" \
     "**/.DS_Store" \
     "**/__MACOSX/*" \
     "**/*.log" \
     "clearpath/scripts/*"

echo "Wrote $OUT ($(du -h "$OUT" | awk '{print $1}'))"
echo "Contents check (should have no Rebuild/Guide/plan md, no videos):"
unzip -l "$OUT" | grep -Ei 'rebuild|guide|plan|\.mp4|\.md' || true
unzip -l "$OUT" | tail -5
