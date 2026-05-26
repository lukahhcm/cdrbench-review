#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  score_benchmark_tracks.sh [options]

Score previously saved inference outputs for one or more CDR-Bench tracks.
By default this script targets all v2 benchmark tracks:

  1. atomic_m
  2. atomic_f
  3. agnostic_m
  4. order_m
  5. order_f

This step only reads predictions, computes metrics, and writes reports under each
track's `score/` subdirectory. It also writes slice CSVs such as `by_domain.csv`.

Options:
  --predictions-root <path>           Inference root. Default: data/evaluation_v2
  --model-dirname <name>              Model subdirectory name. Required
  --predictions-filename <name>       Predictions filename per track. Default: predictions.jsonl
  --score-dirname <name>              Score subdirectory per track. Default: score
  --prompt-variant-sample-size <int>  Deterministically sample this many prompt styles at score time. Default: 3
  --prompt-variant-sampling-seed <int>  Sampling seed used with --prompt-variant-sample-size. Default: 0
  --tracks <csv>                      Comma-separated tracks. Default: atomic_m,atomic_f,agnostic_m,order_m,order_f
  --progress-every <int>              Default: 20
  --resume                            Resume scoring from existing report files in the same directory
  --no-resume                         Disable resume even if a wrapper defaults it on
  -h, --help                          Show this help

Examples:
  ./scripts/score_benchmark_tracks.sh \
    --predictions-root data/evaluation_v2 \
    --model-dirname gpt54

  ./scripts/score_benchmark_tracks.sh \
    --tracks atomic_m,atomic_f,agnostic_m,order_m,order_f \
    --predictions-root data/evaluation_v2 \
    --model-dirname local_model
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="$REPO_ROOT/.venv-ops/bin/python"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="python3"
fi

PREDICTIONS_ROOT="data/evaluation_v2"
MODEL_DIRNAME=""
PREDICTIONS_FILENAME="predictions.jsonl"
SCORE_DIRNAME="score"
PROMPT_VARIANT_SAMPLE_SIZE="3"
PROMPT_VARIANT_SAMPLING_SEED="0"
TRACKS_CSV="atomic_m,atomic_f,agnostic_m,order_m,order_f"
PROGRESS_EVERY="20"
RESUME="false"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --predictions-root)
      PREDICTIONS_ROOT="$2"
      shift 2
      ;;
    --model-dirname)
      MODEL_DIRNAME="$2"
      shift 2
      ;;
    --predictions-filename)
      PREDICTIONS_FILENAME="$2"
      shift 2
      ;;
    --score-dirname)
      SCORE_DIRNAME="$2"
      shift 2
      ;;
    --prompt-variant-sample-size)
      PROMPT_VARIANT_SAMPLE_SIZE="$2"
      shift 2
      ;;
    --prompt-variant-sampling-seed)
      PROMPT_VARIANT_SAMPLING_SEED="$2"
      shift 2
      ;;
    --tracks)
      TRACKS_CSV="$2"
      shift 2
      ;;
    --progress-every)
      PROGRESS_EVERY="$2"
      shift 2
      ;;
    --resume)
      RESUME="true"
      shift 1
      ;;
    --no-resume)
      RESUME="false"
      shift 1
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

if [[ -z "$MODEL_DIRNAME" ]]; then
  echo "--model-dirname is required." >&2
  exit 1
fi

for track in "${TRACKS[@]}"; do
  predictions_path="$PREDICTIONS_ROOT/$track/$MODEL_DIRNAME/$PREDICTIONS_FILENAME"
  output_dir="$PREDICTIONS_ROOT/$track/$MODEL_DIRNAME/$SCORE_DIRNAME"
  mkdir -p "$output_dir"

  if [[ ! -f "$predictions_path" ]]; then
    echo "Missing predictions file for track=$track: $predictions_path" >&2
    exit 1
  fi

  cmd=(
    "$PYTHON_BIN" -m cdrbench.eval.run_benchmark_score
    --predictions-path "$predictions_path"
    --output-dir "$output_dir"
    --prompt-variant-sample-size "$PROMPT_VARIANT_SAMPLE_SIZE"
    --prompt-variant-sampling-seed "$PROMPT_VARIANT_SAMPLING_SEED"
    --progress-every "$PROGRESS_EVERY"
    --write-csv-slices
  )
  if [[ "$RESUME" == "true" ]]; then
    cmd+=(--resume)
  fi

  echo "[run] track=$track step=score output_dir=$output_dir"
  "${cmd[@]}"
  echo "[done] track=$track scored=$output_dir"
done

echo "[complete] scoring finished for tracks: ${TRACKS[*]}"
echo "[complete] reports written under each track's $SCORE_DIRNAME/ subdirectory in: $PREDICTIONS_ROOT"
