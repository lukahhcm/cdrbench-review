#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RELEASE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
export PYTHONPATH="$RELEASE_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
if [[ $# -eq 0 ]]; then
  shopt -s nullglob
  set -- "${BENCHMARK_ROOT:-$RELEASE_ROOT/data/benchmark_v3}"/tracks/*.jsonl
  if [[ $# -eq 0 ]]; then
    echo "No benchmark tracks found. Download the benchmark first or pass JSONL paths." >&2
    exit 1
  fi
fi
python -m cdrbench_v3.validate_benchmark "$@"
