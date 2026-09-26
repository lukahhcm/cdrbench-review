#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  score_suite.sh --model-dirname NAME [options]

Options:
  --track-family <main|core_rule|semantic|semantic_extension|all>  Default: all
  --tracks <csv>                                     Override track list
  --predictions-root <path>                          Default: data/results
  --model-dirname <name>                             Required
  --predictions-filename <name>                      Default: predictions_direct_k3_seed0.jsonl
  --score-dirname <name>                             Default: score_direct_k3_seed0
  --rs-at-k <int>                                    Default: 3
  --prompt-variant-sampling-seed <int>               Default: 0
  --progress-every <int>                             Accepted for compatibility
  --write-csv                                        Write CSV reports
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RELEASE_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

TRACK_FAMILY="all"
TRACKS_CSV=""
PREDICTIONS_ROOT="data/results"
MODEL_DIRNAME=""
PREDICTIONS_FILENAME=""
SCORE_DIRNAME=""
RS_AT_K="3"
PROMPT_VARIANT_SAMPLING_SEED="0"
PROGRESS_EVERY="20"
WRITE_CSV=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --track-family) TRACK_FAMILY="$2"; shift 2 ;;
    --tracks) TRACKS_CSV="$2"; shift 2 ;;
    --predictions-root) PREDICTIONS_ROOT="$2"; shift 2 ;;
    --model-dirname) MODEL_DIRNAME="$2"; shift 2 ;;
    --predictions-filename) PREDICTIONS_FILENAME="$2"; shift 2 ;;
    --score-dirname) SCORE_DIRNAME="$2"; shift 2 ;;
    --rs-at-k) RS_AT_K="$2"; shift 2 ;;
    --prompt-variant-sampling-seed) PROMPT_VARIANT_SAMPLING_SEED="$2"; shift 2 ;;
    --progress-every) PROGRESS_EVERY="$2"; shift 2 ;;
    --write-csv) WRITE_CSV=(--write-csv); shift 1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ -z "$MODEL_DIRNAME" ]]; then
  usage >&2
  exit 1
fi

if [[ -z "$PREDICTIONS_FILENAME" ]]; then
  PREDICTIONS_FILENAME="predictions_direct_k${RS_AT_K}_seed${PROMPT_VARIANT_SAMPLING_SEED}.jsonl"
fi
if [[ -z "$SCORE_DIRNAME" ]]; then
  SCORE_DIRNAME="score_direct_k${RS_AT_K}_seed${PROMPT_VARIANT_SAMPLING_SEED}"
fi

if [[ -n "$TRACKS_CSV" ]]; then
  IFS=',' read -r -a TRACKS <<< "$TRACKS_CSV"
else
  case "$TRACK_FAMILY" in
    main|core_rule)
      TRACKS=(atomic_m atomic_f agnostic_m order_m order_f)
      ;;
    semantic|semantic_extension)
      TRACKS=(semantic_pii_atomic semantic_pii_compositional semantic_hallu_atomic semantic_hallu_compositional semantic_rubric_atomic semantic_rubric_compositional semantic_safety_atomic semantic_safety_compositional)
      ;;
    all)
      TRACKS=(atomic_m atomic_f agnostic_m order_m order_f semantic_pii_atomic semantic_pii_compositional semantic_hallu_atomic semantic_hallu_compositional semantic_rubric_atomic semantic_rubric_compositional semantic_safety_atomic semantic_safety_compositional)
      ;;
    *)
      echo "Invalid --track-family: $TRACK_FAMILY" >&2
      exit 1
      ;;
  esac
fi

for track in "${TRACKS[@]}"; do
  predictions_path="$RELEASE_ROOT/$PREDICTIONS_ROOT/$track/$MODEL_DIRNAME/$PREDICTIONS_FILENAME"
  output_dir="$RELEASE_ROOT/$PREDICTIONS_ROOT/$track/$MODEL_DIRNAME/$SCORE_DIRNAME"
  if [[ ! -f "$predictions_path" ]]; then
    echo "Missing predictions for track=$track: $predictions_path" >&2
    exit 1
  fi
  echo "[run] score track=$track predictions=$predictions_path"
  "${RELEASE_ROOT}/scripts/score/score_predictions.sh" \
    --predictions-path "$predictions_path" \
    --output-dir "$output_dir" \
    --rs-at-k "$RS_AT_K" \
    --prompt-variant-sampling-seed "$PROMPT_VARIANT_SAMPLING_SEED" \
    ${WRITE_CSV[@]+"${WRITE_CSV[@]}"}
done
