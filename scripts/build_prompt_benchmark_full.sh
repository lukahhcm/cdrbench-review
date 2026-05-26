#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  build_prompt_benchmark_full.sh [options]

Build a prompt-aware benchmark-full directly from benchmark instances plus an
existing prompt library. This does not regenerate prompts.

Options:
  --benchmark-dir <path>                  Default: data/processed/benchmark_instances
  --prompt-library <path>                 Default: data/processed/prompt_library
  --output-dir <path>                     Default: data/benchmark_full_prompt
  --tracks <csv>                          Default: atomic_ops,main,order_sensitivity
  --min-prompt-variants-per-sample <int>  Default: 1
  --drop-unmatched-rows                   Drop rows with missing/insufficient prompt pools
  --python-bin <path>                     Default: .venv-ops/bin/python or python3
  -h, --help                              Show this help
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

BENCHMARK_DIR="data/processed/benchmark_instances"
PROMPT_LIBRARY="data/processed/prompt_library"
OUTPUT_DIR="data/benchmark_full_prompt"
TRACKS_CSV="atomic_ops,main,order_sensitivity"
MIN_PROMPT_VARIANTS_PER_SAMPLE=1
DROP_UNMATCHED_ROWS="false"

PYTHON_BIN="$REPO_ROOT/.venv-ops/bin/python"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="python3"
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --benchmark-dir)
      BENCHMARK_DIR="$2"
      shift 2
      ;;
    --prompt-library)
      PROMPT_LIBRARY="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --tracks)
      TRACKS_CSV="$2"
      shift 2
      ;;
    --min-prompt-variants-per-sample)
      MIN_PROMPT_VARIANTS_PER_SAMPLE="$2"
      shift 2
      ;;
    --drop-unmatched-rows)
      DROP_UNMATCHED_ROWS="true"
      shift 1
      ;;
    --python-bin)
      PYTHON_BIN="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

IFS=',' read -r -a TRACKS <<< "$TRACKS_CSV"

CMD=(
  "$PYTHON_BIN" -m cdrbench.prompting.build_prompt_benchmark_full
  --benchmark-dir "$BENCHMARK_DIR"
  --prompt-library "$PROMPT_LIBRARY"
  --output-dir "$OUTPUT_DIR"
  --min-prompt-variants-per-sample "$MIN_PROMPT_VARIANTS_PER_SAMPLE"
  --tracks "${TRACKS[@]}"
)

if [[ "$DROP_UNMATCHED_ROWS" == "true" ]]; then
  CMD+=(--drop-unmatched-rows)
fi

printf 'Running:'
for token in "${CMD[@]}"; do
  printf ' %q' "$token"
done
printf '\n'

"${CMD[@]}"
