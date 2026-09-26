#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  run_model_eval.sh [infer|score|all]
  run_model_eval.sh --mode <infer|score|all>
  run_model_eval.sh [infer|score|all] --mode <direct|few_shot|plan_first|state_aware>
  run_model_eval.sh --prompt-mode <direct|few_shot|plan_first|state_aware>

Environment variables, usually set by thin wrappers:
  EVAL_SUITE                     main or semantic. Default: main
  TRACKS                         Optional override. main defaults to the five core tracks;
                                 semantic defaults to implemented atomic/compositional extension tracks
  BENCHMARK_ROOT                 Default: data/benchmark_v3
  EVAL_ROOT                      Alias for BENCHMARK_ROOT
  BENCHMARK_TRACKS_SUBDIR        Default: tracks. Use tracks_all_prompts for custom prompt-seed sweeps
  EVALUATION_ROOT                Default: data/evaluation
  OUTPUT_ROOT                    Alias for EVALUATION_ROOT
  MODEL                          Required for infer/all
  MODEL_SLUG                     Directory name under each track
  BACKEND                        api or vllm. Default: api
  BASE_URL                       Optional; API models auto-resolve if unset
  API_KEY                        Optional; vLLM defaults to EMPTY
  PROMPT_VARIANT_INDICES         Default: all
  PROMPT_STYLE_IDS               Semantic default: direct,imperative_checklist,application_context
  PROMPT_VARIANT_SAMPLE_SIZE     Default: 3
  PROMPT_VARIANT_SAMPLING_SEED   Default: 0
  PROMPT_MODE                    Default: direct
  MAX_SAMPLES                    Default: 0
  MAX_INPUT_CHARS                Default: 0
  MAX_TOKENS                     Default: 0
  TEMPERATURE                    Default: 0
  MAX_RETRIES                    Default: 1
  RETRY_SLEEP_SECONDS            Default: 2.0
  CONCURRENCY                    Default: 4 for api wrappers, 128 for vllm wrappers
  RESUME                         Default: true
  RESUME_ONLY_EXISTING_ROWS      Default: false
  ENABLE_THINKING                true/false. Default: false in model wrappers and lower-level inference.
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RELEASE_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${RELEASE_ROOT}"

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
  infer|score|all) ;;
  *) echo "Unsupported mode: ${MODE}" >&2; usage; exit 1 ;;
esac

EVAL_SUITE="${EVAL_SUITE:-main}"
MAIN_TRACKS="atomic_m,atomic_f,agnostic_m,order_m,order_f"
SEMANTIC_IMPLEMENTED_TRACKS="semantic_pii_atomic,semantic_pii_compositional,semantic_hallu_atomic,semantic_hallu_compositional,semantic_rubric_atomic,semantic_rubric_compositional,semantic_safety_atomic,semantic_safety_compositional"
SEMANTIC_EXTRA_TRACKS="${SEMANTIC_EXTRA_TRACKS:-}"
case "${EVAL_SUITE}" in
  main)
    DEFAULT_TRACKS="${MAIN_TRACKS}"
    DEFAULT_PROMPT_STYLE_IDS=""
    ;;
  semantic)
    DEFAULT_TRACKS="${SEMANTIC_IMPLEMENTED_TRACKS}${SEMANTIC_EXTRA_TRACKS:+,${SEMANTIC_EXTRA_TRACKS}}"
    DEFAULT_PROMPT_STYLE_IDS="direct,imperative_checklist,application_context"
    ;;
  *)
    echo "Unsupported EVAL_SUITE: ${EVAL_SUITE}. Use main or semantic." >&2
    exit 1
    ;;
esac

TRACKS="${TRACKS:-${DEFAULT_TRACKS}}"
BENCHMARK_ROOT="${BENCHMARK_ROOT:-${EVAL_ROOT:-data/benchmark_v3}}"
BENCHMARK_TRACKS_SUBDIR="${BENCHMARK_TRACKS_SUBDIR:-tracks}"
EVALUATION_ROOT="${EVALUATION_ROOT:-${OUTPUT_ROOT:-data/evaluation}}"
PREDICTIONS_ROOT="${PREDICTIONS_ROOT:-${EVALUATION_ROOT}}"
MODEL="${MODEL:-}"
MODEL_SLUG="${MODEL_SLUG:-${MODEL_DIRNAME:-}}"
BACKEND="${BACKEND:-api}"
BASE_URL="${BASE_URL:-}"
API_KEY="${API_KEY:-}"
PROMPT_VARIANT_INDICES="${PROMPT_VARIANT_INDICES:-all}"
PROMPT_STYLE_IDS="${PROMPT_STYLE_IDS:-${DEFAULT_PROMPT_STYLE_IDS}}"
PROMPT_VARIANT_SAMPLE_SIZE="${PROMPT_VARIANT_SAMPLE_SIZE:-3}"
PROMPT_VARIANT_SAMPLING_SEED="${PROMPT_VARIANT_SAMPLING_SEED:-0}"
PROMPT_MODE="${PROMPT_MODE:-direct}"
if [[ -n "${PROMPT_MODE_OVERRIDE}" ]]; then
  PROMPT_MODE="${PROMPT_MODE_OVERRIDE}"
fi
case "${PROMPT_MODE}" in
  direct|few_shot|plan_first|state_aware) ;;
  *)
    echo "Unsupported prompt mode in release_v3: ${PROMPT_MODE}" >&2
    exit 1
    ;;
esac
MAX_SAMPLES="${MAX_SAMPLES:-0}"
MAX_INPUT_CHARS="${MAX_INPUT_CHARS:-0}"
MAX_TOKENS="${MAX_TOKENS:-0}"
if [[ "${MAX_TOKENS}" != "0" ]]; then
  echo "MAX_TOKENS must be 0 for benchmark evaluation; got ${MAX_TOKENS}." >&2
  echo "Use the model/server context length instead of truncating generation at the runner level." >&2
  exit 1
fi
TEMPERATURE="${TEMPERATURE:-0}"
MAX_RETRIES="${MAX_RETRIES:-1}"
RETRY_SLEEP_SECONDS="${RETRY_SLEEP_SECONDS:-${RETRY_DELAY:-2.0}}"
CONCURRENCY="${CONCURRENCY:-4}"
PROGRESS_EVERY="${PROGRESS_EVERY:-20}"
RESUME="${RESUME:-true}"
RESUME_ONLY_EXISTING_ROWS="${RESUME_ONLY_EXISTING_ROWS:-false}"
ENABLE_THINKING="${ENABLE_THINKING:-}"
SCORE_PROMPT_VARIANT_SAMPLE_SIZE="${SCORE_PROMPT_VARIANT_SAMPLE_SIZE:-${RS_AT_K:-3}}"
SCORE_PROMPT_VARIANT_SAMPLING_SEED="${SCORE_PROMPT_VARIANT_SAMPLING_SEED:-${PROMPT_VARIANT_SAMPLING_SEED}}"

sanitize_model_dirname() {
  local value="$1"
  value="$(printf '%s' "$value" | tr -cs '[:alnum:]._-' '_')"
  value="${value##_}"
  value="${value%%_}"
  printf '%s' "${value:-model}"
}

if [[ -z "${MODEL_SLUG}" ]]; then
  MODEL_SLUG="$(sanitize_model_dirname "${MODEL}")"
fi

if [[ "${EVAL_SUITE}" == "semantic" ]]; then
  PREDICTIONS_FILENAME="${PREDICTIONS_FILENAME:-predictions_semantic_styles3.jsonl}"
  SCORE_DIRNAME="${SCORE_DIRNAME:-score_semantic_styles3}"
else
  infer_sampling_suffix=""
  if [[ "${PROMPT_VARIANT_SAMPLE_SIZE}" =~ ^[0-9]+$ ]] && [[ "${PROMPT_VARIANT_SAMPLE_SIZE}" -gt 0 ]]; then
    infer_sampling_suffix="_k${PROMPT_VARIANT_SAMPLE_SIZE}_seed${PROMPT_VARIANT_SAMPLING_SEED}"
  fi
  score_sampling_suffix=""
  if [[ "${SCORE_PROMPT_VARIANT_SAMPLE_SIZE}" =~ ^[0-9]+$ ]] && [[ "${SCORE_PROMPT_VARIANT_SAMPLE_SIZE}" -gt 0 ]]; then
    score_sampling_suffix="_k${SCORE_PROMPT_VARIANT_SAMPLE_SIZE}_seed${SCORE_PROMPT_VARIANT_SAMPLING_SEED}"
  fi
  PREDICTIONS_FILENAME="${PREDICTIONS_FILENAME:-predictions_${PROMPT_MODE}${infer_sampling_suffix}.jsonl}"
  SCORE_DIRNAME="${SCORE_DIRNAME:-score_${PROMPT_MODE}${score_sampling_suffix}}"
fi

run_infer() {
  if [[ -z "${MODEL}" ]]; then
    echo "MODEL is required for infer/all mode." >&2
    exit 1
  fi
  cmd=(
    bash "${RELEASE_ROOT}/scripts/infer/run_inference_suite.sh"
    --tracks "${TRACKS}"
    --benchmark-root "${BENCHMARK_ROOT}"
    --benchmark-tracks-subdir "${BENCHMARK_TRACKS_SUBDIR}"
    --output-root "${EVALUATION_ROOT}"
    --model "${MODEL}"
    --model-dirname "${MODEL_SLUG}"
    --backend "${BACKEND}"
    --predictions-filename "${PREDICTIONS_FILENAME}"
    --prompt-variant-indices "${PROMPT_VARIANT_INDICES}"
    --prompt-style-ids "${PROMPT_STYLE_IDS}"
    --prompt-variant-sample-size "${PROMPT_VARIANT_SAMPLE_SIZE}"
    --prompt-variant-sampling-seed "${PROMPT_VARIANT_SAMPLING_SEED}"
    --prompt-mode "${PROMPT_MODE}"
    --max-samples "${MAX_SAMPLES}"
    --max-input-chars "${MAX_INPUT_CHARS}"
    --max-tokens "${MAX_TOKENS}"
    --temperature "${TEMPERATURE}"
    --max-retries "${MAX_RETRIES}"
    --retry-sleep-seconds "${RETRY_SLEEP_SECONDS}"
    --concurrency "${CONCURRENCY}"
    --progress-every "${PROGRESS_EVERY}"
  )
  if [[ -n "${BASE_URL}" ]]; then cmd+=(--base-url "${BASE_URL}"); fi
  if [[ -n "${API_KEY}" ]]; then export API_KEY; fi
  if [[ "${RESUME}" == "true" ]]; then cmd+=(--resume); fi
  if [[ "${RESUME_ONLY_EXISTING_ROWS}" == "true" ]]; then cmd+=(--resume-only-existing-rows); fi
  case "${ENABLE_THINKING}" in
    true|TRUE|1|yes|YES|on|ON) cmd+=(--enable-thinking) ;;
    false|FALSE|0|no|NO|off|OFF) cmd+=(--disable-thinking) ;;
    "") ;;
    *) echo "Unsupported ENABLE_THINKING value: ${ENABLE_THINKING}" >&2; exit 1 ;;
  esac
  "${cmd[@]}" ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
}

run_score() {
  IFS=',' read -r -a TRACK_LIST <<< "${TRACKS}"
  for track in "${TRACK_LIST[@]}"; do
    rm -rf "${RELEASE_ROOT}/${PREDICTIONS_ROOT}/${track}/${MODEL_SLUG}/${SCORE_DIRNAME}"
  done

  bash "${RELEASE_ROOT}/scripts/score/score_suite.sh" \
    --tracks "${TRACKS}" \
    --predictions-root "${PREDICTIONS_ROOT}" \
    --model-dirname "${MODEL_SLUG}" \
    --predictions-filename "${PREDICTIONS_FILENAME}" \
    --score-dirname "${SCORE_DIRNAME}" \
    --rs-at-k "${SCORE_PROMPT_VARIANT_SAMPLE_SIZE}" \
    --prompt-variant-sampling-seed "${SCORE_PROMPT_VARIANT_SAMPLING_SEED}" \
    --progress-every "${PROGRESS_EVERY}" \
    --write-csv
}

case "${MODE}" in
  infer) run_infer ;;
  score) run_score ;;
  all)
    IFS=',' read -r -a TRACK_LIST <<< "${TRACKS}"
    for track in "${TRACK_LIST[@]}"; do
      TRACKS="${track}" run_infer
      TRACKS="${track}" run_score
    done
    ;;
esac
