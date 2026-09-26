#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

MODEL_PATH="${1:-${MODEL_PATH:?'MODEL_PATH required'}}"
MODEL_NAME="${2:-${MODEL_NAME:?'MODEL_NAME required'}}"
PORT="${3:-${PORT:-8901}}"
GPU_IDS="${4:-${GPU_IDS:-0}}"
TP_SIZE="${5:-${TP_SIZE:-1}}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-32768}"

export VLLM_USE_MODELSCOPE="${VLLM_USE_MODELSCOPE:-False}"

if ! python - <<'PY' >/dev/null 2>&1
import vllm  # noqa: F401
PY
then
  cat >&2 <<'EOF'
[start_vllm] ERROR: vLLM is not installed in the current Python environment.

The CDR-Bench requirements.txt intentionally installs only the lightweight
evaluation client dependencies. To use scripts/start_vllm.sh, activate a
separate Linux/CUDA serving environment and install vLLM following the official
vLLM documentation for your hardware and PyTorch/CUDA stack.
EOF
  exit 1
fi

echo "========================================================"
echo "[start_vllm] MODEL_PATH = ${MODEL_PATH}"
echo "[start_vllm] MODEL_NAME = ${MODEL_NAME}"
echo "[start_vllm] PORT       = ${PORT}"
echo "[start_vllm] GPU_IDS    = ${GPU_IDS}"
echo "[start_vllm] TP_SIZE    = ${TP_SIZE}"
echo "[start_vllm] MAX_MODEL_LEN = ${MAX_MODEL_LEN}"
echo "[start_vllm] MAX_NUM_BATCHED_TOKENS = ${MAX_NUM_BATCHED_TOKENS}"
echo "========================================================"

if command -v setsid >/dev/null 2>&1; then
  setsid env CUDA_VISIBLE_DEVICES="${GPU_IDS}" python -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_PATH}" \
    --served-model-name "${MODEL_NAME}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
    --trust-remote-code \
    --tensor-parallel-size "${TP_SIZE}" \
    --port "${PORT}" \
    --disable-log-stats &
else
  CUDA_VISIBLE_DEVICES="${GPU_IDS}" python -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_PATH}" \
    --served-model-name "${MODEL_NAME}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
    --trust-remote-code \
    --tensor-parallel-size "${TP_SIZE}" \
    --port "${PORT}" \
    --disable-log-stats &
fi

VLLM_PID=$!
echo "${VLLM_PID}" > "/tmp/vllm_${PORT}.pid"
echo "${PORT}" > "/tmp/vllm_${PORT}.port"
echo "[start_vllm] vLLM started (PID=${VLLM_PID}), waiting for ready..."

MAX_WAIT="${VLLM_START_MAX_WAIT:-1200}"
WAITED=0
until curl --noproxy '*' -sf "http://127.0.0.1:${PORT}/ping" > /dev/null 2>&1; do
  if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
    echo "[start_vllm] ERROR: vLLM process exited before health check passed (PID=${VLLM_PID})"
    rm -f "/tmp/vllm_${PORT}.pid" "/tmp/vllm_${PORT}.port"
    exit 1
  fi
  sleep 5
  WAITED=$((WAITED + 5))
  if [[ "${WAITED}" -ge "${MAX_WAIT}" ]]; then
    echo "[start_vllm] ERROR: server did not start within ${MAX_WAIT}s"
    kill "${VLLM_PID}" 2>/dev/null || true
    exit 1
  fi
done

echo "[start_vllm] Server ready at http://127.0.0.1:${PORT}"
