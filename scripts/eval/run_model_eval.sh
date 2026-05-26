#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  run_model_eval.sh [infer|score|all]
  run_model_eval.sh --mode <infer|score|all>
  run_model_eval.sh [infer|score|all] --mode <direct|few_shot|plan_first|state_aware>
  run_model_eval.sh --prompt-mode <direct|few_shot|plan_first|state_aware>

Modes:
  infer   Run inference only, resuming from existing outputs and rerunning anything that did not complete successfully.
  score   Recompute scores only, replacing each track's entire score/ directory.
  all     Default. Run infer + score sequentially per track.

Prompt modes:
  direct
  few_shot
  plan_first
  state_aware

Configuration is provided by environment variables from thin model wrappers.
  Preferred wrapper inputs:
  EVALUATION_ROOT   Final root for per-track evaluation outputs. Default: data/evaluation_v2
  MODEL_SLUG        Directory name under each track, for example local_model or gpt_4o_mini
  RESUME_ONLY_EXISTING_ROWS  When true, resume only over instance_ids already present in predictions
  ENABLE_THINKING  true/false override for supported models. Default leaves inference behavior unchanged.

Compatibility note:
  OUTPUT_ROOT is still accepted for older wrappers. Legacy values such as
  data/evaluation/infer/<model_slug> are normalized to the final layout above.

For API models, leave BASE_URL unset to auto-resolve the correct endpoint from the model config.
If PROMPT_API_KEY=true and API_KEY is empty, the script will prompt for a key before inference.
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${REPO_ROOT}/.venv-ops/bin/python"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="python3"
fi

MODE="all"
PROMPT_MODE_OVERRIDE=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    infer|score|all)
      MODE="$1"
      shift 1
      ;;
    --mode)
      if [[ $# -lt 2 ]]; then
        echo "--mode requires a value." >&2
        usage
        exit 1
      fi
      case "$2" in
        infer|score|all)
          MODE="$2"
          ;;
        direct|few_shot|plan_first|state_aware)
          PROMPT_MODE_OVERRIDE="$2"
          ;;
        *)
          echo "Unsupported --mode value: $2" >&2
          usage
          exit 1
          ;;
      esac
      shift 2
      ;;
    --prompt-mode)
      if [[ $# -lt 2 ]]; then
        echo "--prompt-mode requires a value." >&2
        usage
        exit 1
      fi
      PROMPT_MODE_OVERRIDE="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift 1
      ;;
  esac
done

case "${MODE}" in
  infer|score|all)
    ;;
  *)
    echo "Unsupported mode: ${MODE}" >&2
    usage
    exit 1
    ;;
esac

TRACKS="${TRACKS:-atomic_m,atomic_f,agnostic_m,order_m,order_f}"
EVAL_ROOT="${EVAL_ROOT:-data/benchmark_v2}"
MODEL="${MODEL:-}"
BASE_URL="${BASE_URL:-}"
API_KEY="${API_KEY:-}"
PROMPT_API_KEY="${PROMPT_API_KEY:-false}"
EVALUATION_ROOT="${EVALUATION_ROOT:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${EVALUATION_ROOT:-data/evaluation_v2}}"
MODEL_DIRNAME="${MODEL_DIRNAME:-${MODEL_SLUG:-}}"
PROMPT_VARIANT_INDICES="${PROMPT_VARIANT_INDICES:-all}"
PROMPT_VARIANT_SAMPLE_SIZE="${PROMPT_VARIANT_SAMPLE_SIZE:-3}"
PROMPT_VARIANT_SAMPLING_SEED="${PROMPT_VARIANT_SAMPLING_SEED:-0}"
PROMPT_MODE="${PROMPT_MODE:-direct}"
FEW_SHOT_SOURCE_ROOT="${FEW_SHOT_SOURCE_ROOT:-data/benchmark_v2}"
PREDICTIONS_FILENAME="${PREDICTIONS_FILENAME:-}"
SCORE_DIRNAME="${SCORE_DIRNAME:-}"
SCORE_PROMPT_VARIANT_SAMPLE_SIZE="${SCORE_PROMPT_VARIANT_SAMPLE_SIZE:-3}"
SCORE_PROMPT_VARIANT_SAMPLING_SEED="${SCORE_PROMPT_VARIANT_SAMPLING_SEED:-0}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
MAX_INPUT_CHARS="${MAX_INPUT_CHARS:-0}"
MAX_TOKENS="${MAX_TOKENS:-0}"
CONCURRENCY="${CONCURRENCY:-4}"
PROGRESS_EVERY="${PROGRESS_EVERY:-20}"
RESUME="${RESUME:-true}"
RESUME_ONLY_EXISTING_ROWS="${RESUME_ONLY_EXISTING_ROWS:-false}"
ENABLE_THINKING="${ENABLE_THINKING:-}"

sanitize_model_dirname() {
  local value="$1"
  value="$(printf '%s' "$value" | tr -cs '[:alnum:]._-' '_')"
  value="${value##_}"
  value="${value%%_}"
  printf '%s' "${value:-model}"
}

derive_model_dirname() {
  if [[ -n "${MODEL_DIRNAME}" ]]; then
    printf '%s' "${MODEL_DIRNAME}"
    return
  fi
  if [[ -n "${MODEL_SLUG:-}" ]]; then
    printf '%s' "${MODEL_SLUG}"
    return
  fi
  if [[ -n "${OUTPUT_ROOT}" ]]; then
    local base
    base="$(basename "${OUTPUT_ROOT}")"
    if [[ "${base}" != "evaluation" && "${base}" != "infer" && "${base}" != "." ]]; then
      printf '%s' "${base}"
      return
    fi
  fi
  printf '%s' "$(sanitize_model_dirname "${MODEL}")"
}

derive_evaluation_root() {
  if [[ -n "${EVALUATION_ROOT:-}" ]]; then
    printf '%s' "${EVALUATION_ROOT}"
    return
  fi
  if [[ "${OUTPUT_ROOT}" == */infer/* ]]; then
    printf '%s' "$(dirname "$(dirname "${OUTPUT_ROOT}")")"
    return
  fi
  printf '%s' "${OUTPUT_ROOT}"
}

if [[ -n "${PROMPT_MODE_OVERRIDE}" ]]; then
  PROMPT_MODE="${PROMPT_MODE_OVERRIDE}"
fi

case "${PROMPT_MODE}" in
  direct|few_shot|plan_first|state_aware)
    ;;
  *)
    echo "Unsupported prompt mode: ${PROMPT_MODE}" >&2
    usage
    exit 1
    ;;
esac

MODEL_DIRNAME="$(derive_model_dirname)"
OUTPUT_ROOT="$(derive_evaluation_root)"
PREDICTIONS_ROOT="${PREDICTIONS_ROOT:-${OUTPUT_ROOT}}"

infer_sampling_suffix=""
if [[ "${PROMPT_VARIANT_SAMPLE_SIZE}" =~ ^[0-9]+$ ]] && [[ "${PROMPT_VARIANT_SAMPLE_SIZE}" -gt 0 ]]; then
  infer_sampling_suffix="_k${PROMPT_VARIANT_SAMPLE_SIZE}_seed${PROMPT_VARIANT_SAMPLING_SEED}"
fi

score_sampling_suffix=""
if [[ "${SCORE_PROMPT_VARIANT_SAMPLE_SIZE}" =~ ^[0-9]+$ ]] && [[ "${SCORE_PROMPT_VARIANT_SAMPLE_SIZE}" -gt 0 ]]; then
  score_sampling_suffix="_k${SCORE_PROMPT_VARIANT_SAMPLE_SIZE}_seed${SCORE_PROMPT_VARIANT_SAMPLING_SEED}"
fi

if [[ -z "${PREDICTIONS_FILENAME}" ]]; then
  PREDICTIONS_FILENAME="predictions_${PROMPT_MODE}${infer_sampling_suffix}.jsonl"
fi

if [[ -z "${SCORE_DIRNAME}" ]]; then
  SCORE_DIRNAME="score_${PROMPT_MODE}${score_sampling_suffix}"
fi

prompt_for_api_key_if_needed() {
  if [[ "${PROMPT_API_KEY}" != "true" || -n "${API_KEY}" ]]; then
    return
  fi
  IFS= read -rsp "API key for ${MODEL}: " API_KEY
  echo
  if [[ -z "${API_KEY}" ]]; then
    echo "API key cannot be empty." >&2
    exit 1
  fi
}

run_infer() {
  if [[ -z "${MODEL}" ]]; then
    cat >&2 <<'EOF'
MODEL is required for infer mode.

Use a per-model wrapper for the default infer+score flow, for example:
  bash scripts/eval/api/eval_gpt_5_4.sh
  bash scripts/eval/api/eval_gpt_5_4.sh infer
  bash scripts/eval/api/eval_gpt_5_4.sh score

Or call the shared driver directly with MODEL set:
  MODEL=gpt-4o-mini PROMPT_API_KEY=true bash scripts/eval/run_model_eval.sh
EOF
    exit 1
  fi

  prompt_for_api_key_if_needed

  cmd=(
    "${REPO_ROOT}/scripts/infer_benchmark_tracks.sh"
    --tracks "${TRACKS}"
    --eval-root "${EVAL_ROOT}"
    --model "${MODEL}"
    --output-root "${OUTPUT_ROOT}"
    --model-dirname "${MODEL_DIRNAME}"
    --predictions-filename "${PREDICTIONS_FILENAME}"
    --prompt-variant-indices "${PROMPT_VARIANT_INDICES}"
    --prompt-variant-sample-size "${PROMPT_VARIANT_SAMPLE_SIZE}"
    --prompt-variant-sampling-seed "${PROMPT_VARIANT_SAMPLING_SEED}"
    --prompt-mode "${PROMPT_MODE}"
    --few-shot-source-root "${FEW_SHOT_SOURCE_ROOT}"
    --max-samples "${MAX_SAMPLES}"
    --max-input-chars "${MAX_INPUT_CHARS}"
    --max-tokens "${MAX_TOKENS}"
    --concurrency "${CONCURRENCY}"
    --progress-every "${PROGRESS_EVERY}"
  )
  if [[ -n "${BASE_URL}" ]]; then
    cmd+=(--base-url "${BASE_URL}")
  fi
  if [[ -n "${API_KEY}" ]]; then
    cmd+=(--api-key "${API_KEY}")
  fi
  if [[ "${RESUME}" == "true" ]]; then
    cmd+=(--resume)
  fi
  if [[ "${RESUME_ONLY_EXISTING_ROWS}" == "true" ]]; then
    cmd+=(--resume-only-existing-rows)
  fi
  case "${ENABLE_THINKING}" in
    true|TRUE|1|yes|YES|on|ON)
      cmd+=(--enable-thinking)
      ;;
    false|FALSE|0|no|NO|off|OFF)
      cmd+=(--disable-thinking)
      ;;
    "")
      ;;
    *)
      echo "Unsupported ENABLE_THINKING value: ${ENABLE_THINKING}" >&2
      echo "Use true or false." >&2
      exit 1
      ;;
  esac
  if [[ $# -gt 0 ]]; then
    cmd+=("$@")
  fi

  "${cmd[@]}"
}

remove_existing_scores() {
  IFS=',' read -r -a TRACK_LIST <<< "${TRACKS}"
  for track in "${TRACK_LIST[@]}"; do
    score_dir="${PREDICTIONS_ROOT}/${track}/${MODEL_DIRNAME}/${SCORE_DIRNAME}"
    rm -rf "${score_dir}"
  done
}

run_score() {
  remove_existing_scores

  cmd=(
    "${REPO_ROOT}/scripts/score_benchmark_tracks.sh"
    --tracks "${TRACKS}"
    --predictions-root "${PREDICTIONS_ROOT}"
    --model-dirname "${MODEL_DIRNAME}"
    --predictions-filename "${PREDICTIONS_FILENAME}"
    --score-dirname "${SCORE_DIRNAME}"
    --prompt-variant-sample-size "${SCORE_PROMPT_VARIANT_SAMPLE_SIZE}"
    --prompt-variant-sampling-seed "${SCORE_PROMPT_VARIANT_SAMPLING_SEED}"
    --progress-every "${PROGRESS_EVERY}"
  )
  if [[ $# -gt 0 ]]; then
    cmd+=("$@")
  fi

  "${cmd[@]}"
}

run_all_per_track() {
  IFS=',' read -r -a TRACK_LIST <<< "${TRACKS}"
  for track in "${TRACK_LIST[@]}"; do
    TRACKS="${track}" run_infer
    TRACKS="${track}" run_score
  done
}

case "${MODE}" in
  infer)
    run_infer "${EXTRA_ARGS[@]}"
    ;;
  score)
    run_score "${EXTRA_ARGS[@]}"
    ;;
  all)
    run_all_per_track
    ;;
esac
