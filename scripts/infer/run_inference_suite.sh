#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  run_inference_suite.sh --model MODEL [options]

Options:
  --track-family <main|core_rule|semantic|semantic_extension|all>  Default: all
  --benchmark-root <path>                            Default: data/benchmark_v3
  --benchmark-tracks-subdir <name>                   Default: tracks
  --output-root <path>                               Default: data/results
  --tracks <csv>                                     Override track list
  --model <name>                                     Required
  --model-dirname <name>                             Default: sanitized model name
  --backend <api|vllm>                               Default: api
  --base-url <url>                                   Optional
  --api-key <key>                                    Optional
  --predictions-filename <name>                      Default: predictions_direct_k3_seed0.jsonl
  --prompt-variant-indices <csv|all>                 Default: all
  --prompt-style-ids <csv>                           Optional style-id filter
  --prompt-variant-sample-size <int>                 Default: 3
  --prompt-variant-sampling-seed <int>               Default: 0
  --prompt-mode <direct|few_shot|plan_first|state_aware>
                                                    Default: direct
  --max-samples <int>                                Default: 0
  --max-input-chars <int>                            Default: 0
  --concurrency <int>                                Default: 5
  --max-tokens <int>                                 Default: 0
  --temperature <float>                              Default: 0
  --max-retries <int>                                Default: 1
  --retry-sleep-seconds <float>                      Default: 2.0
  --retry-delay <float>                              Alias for --retry-sleep-seconds
  --progress-every <int>                             Default: 20
  --enable-thinking                                  Enable thinking mode when supported
  --disable-thinking                                 Disable thinking mode
  --resume                                           Resume existing outputs
  --resume-only-existing-rows                        Only fill rows already present in existing output
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RELEASE_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

TRACK_FAMILY="all"
TRACKS_CSV=""
BENCHMARK_ROOT="data/benchmark_v3"
BENCHMARK_TRACKS_SUBDIR="tracks"
OUTPUT_ROOT="data/results"
MODEL=""
MODEL_DIRNAME=""
BACKEND="api"
BASE_URL=""
API_KEY=""
PREDICTIONS_FILENAME=""
PROMPT_VARIANT_INDICES="all"
PROMPT_STYLE_IDS=""
PROMPT_VARIANT_SAMPLE_SIZE="3"
PROMPT_VARIANT_SAMPLING_SEED="0"
PROMPT_MODE="direct"
MAX_SAMPLES="0"
MAX_INPUT_CHARS="0"
CONCURRENCY="5"
MAX_TOKENS="0"
TEMPERATURE="0"
MAX_RETRIES="1"
RETRY_DELAY="2.0"
PROGRESS_EVERY="20"
RESUME_ARGS=()
RESUME_ONLY_EXISTING_ARGS=()
THINKING_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --track-family) TRACK_FAMILY="$2"; shift 2 ;;
    --tracks) TRACKS_CSV="$2"; shift 2 ;;
    --benchmark-root) BENCHMARK_ROOT="$2"; shift 2 ;;
    --benchmark-tracks-subdir) BENCHMARK_TRACKS_SUBDIR="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --model-dirname) MODEL_DIRNAME="$2"; shift 2 ;;
    --backend) BACKEND="$2"; shift 2 ;;
    --base-url) BASE_URL="$2"; shift 2 ;;
    --api-key) API_KEY="$2"; shift 2 ;;
    --predictions-filename) PREDICTIONS_FILENAME="$2"; shift 2 ;;
    --prompt-variant-indices) PROMPT_VARIANT_INDICES="$2"; shift 2 ;;
    --prompt-style-ids) PROMPT_STYLE_IDS="$2"; shift 2 ;;
    --prompt-variant-sample-size) PROMPT_VARIANT_SAMPLE_SIZE="$2"; shift 2 ;;
    --prompt-variant-sampling-seed) PROMPT_VARIANT_SAMPLING_SEED="$2"; shift 2 ;;
    --prompt-mode) PROMPT_MODE="$2"; shift 2 ;;
    --max-samples) MAX_SAMPLES="$2"; shift 2 ;;
    --max-input-chars) MAX_INPUT_CHARS="$2"; shift 2 ;;
    --concurrency) CONCURRENCY="$2"; shift 2 ;;
    --max-tokens) MAX_TOKENS="$2"; shift 2 ;;
    --temperature) TEMPERATURE="$2"; shift 2 ;;
    --max-retries) MAX_RETRIES="$2"; shift 2 ;;
    --retry-sleep-seconds) RETRY_DELAY="$2"; shift 2 ;;
    --retry-delay) RETRY_DELAY="$2"; shift 2 ;;
    --progress-every) PROGRESS_EVERY="$2"; shift 2 ;;
    --enable-thinking) THINKING_ARGS=(--enable-thinking); shift 1 ;;
    --disable-thinking) THINKING_ARGS=(--disable-thinking); shift 1 ;;
    --resume) RESUME_ARGS=(--resume); shift 1 ;;
    --resume-only-existing-rows) RESUME_ONLY_EXISTING_ARGS=(--resume-only-existing-rows); shift 1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ -z "$MODEL" ]]; then
  usage >&2
  exit 1
fi

if [[ -z "$MODEL_DIRNAME" ]]; then
  MODEL_DIRNAME="$(echo "$MODEL" | tr '/:.' '___')"
fi

if [[ -z "$PREDICTIONS_FILENAME" ]]; then
  PREDICTIONS_FILENAME="predictions_${PROMPT_MODE}_k${PROMPT_VARIANT_SAMPLE_SIZE}_seed${PROMPT_VARIANT_SAMPLING_SEED}.jsonl"
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
  benchmark_path="$RELEASE_ROOT/$BENCHMARK_ROOT/$BENCHMARK_TRACKS_SUBDIR/$track.jsonl"
  output_path="$RELEASE_ROOT/$OUTPUT_ROOT/$track/$MODEL_DIRNAME/$PREDICTIONS_FILENAME"
  cmd=(
    "${RELEASE_ROOT}/scripts/infer/run_inference.sh"
    --benchmark-path "$benchmark_path"
    --output-path "$output_path"
    --model "$MODEL"
    --backend "$BACKEND"
    --prompt-variant-indices "$PROMPT_VARIANT_INDICES"
    --prompt-style-ids "$PROMPT_STYLE_IDS"
    --prompt-variant-sample-size "$PROMPT_VARIANT_SAMPLE_SIZE"
    --prompt-variant-sampling-seed "$PROMPT_VARIANT_SAMPLING_SEED"
    --prompt-mode "$PROMPT_MODE"
    --max-samples "$MAX_SAMPLES"
    --max-input-chars "$MAX_INPUT_CHARS"
    --concurrency "$CONCURRENCY"
    --max-tokens "$MAX_TOKENS"
    --temperature "$TEMPERATURE"
    --max-retries "$MAX_RETRIES"
    --retry-delay "$RETRY_DELAY"
    --progress-every "$PROGRESS_EVERY"
    ${RESUME_ARGS[@]+"${RESUME_ARGS[@]}"}
    ${RESUME_ONLY_EXISTING_ARGS[@]+"${RESUME_ONLY_EXISTING_ARGS[@]}"}
    ${THINKING_ARGS[@]+"${THINKING_ARGS[@]}"}
  )
  if [[ -n "$BASE_URL" ]]; then cmd+=(--base-url "$BASE_URL"); fi
  if [[ -n "$API_KEY" ]]; then cmd+=(--api-key "$API_KEY"); fi
  echo "[run] inference track=$track output=$output_path"
  "${cmd[@]}"
done
