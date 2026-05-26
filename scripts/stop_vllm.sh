#!/usr/bin/env bash
set -euo pipefail

PORT="${1:-8901}"
PID_FILE="/tmp/vllm_${PORT}.pid"
GRACE_SECONDS="${GRACE_SECONDS:-30}"

find_port_pids() {
  local port="$1"
  if command -v lsof >/dev/null 2>&1; then
    lsof -tiTCP:"${port}" -sTCP:LISTEN 2>/dev/null || true
    return
  fi
  if command -v fuser >/dev/null 2>&1; then
    fuser "${port}/tcp" 2>/dev/null || true
    return
  fi
  if command -v ss >/dev/null 2>&1; then
    ss -ltnp "sport = :${port}" 2>/dev/null \
      | sed -n 's/.*pid=\([0-9][0-9]*\).*/\1/p' \
      | sort -u || true
  fi
}

child_pids_recursive() {
  local parent="$1"
  local children child
  children="$(pgrep -P "${parent}" 2>/dev/null || true)"
  for child in ${children}; do
    child_pids_recursive "${child}"
    printf '%s\n' "${child}"
  done
}

is_running() {
  local pid="$1"
  [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null
}

terminate_pid_tree() {
  local pid="$1"
  local pgid
  if ! is_running "${pid}"; then
    return
  fi

  pgid="$(ps -o pgid= -p "${pid}" 2>/dev/null | tr -d '[:space:]' || true)"
  if [[ "${pgid}" == "${pid}" ]]; then
    echo "[stop_vllm] SIGTERM process group -${pid}"
    kill -TERM "-${pid}" 2>/dev/null || true
  else
    echo "[stop_vllm] SIGTERM PID tree rooted at ${pid}"
    child_pids_recursive "${pid}" | sort -rn | xargs -r kill -TERM 2>/dev/null || true
    kill -TERM "${pid}" 2>/dev/null || true
  fi
}

kill_pid_tree() {
  local pid="$1"
  local pgid
  if ! is_running "${pid}"; then
    return
  fi

  pgid="$(ps -o pgid= -p "${pid}" 2>/dev/null | tr -d '[:space:]' || true)"
  if [[ "${pgid}" == "${pid}" ]]; then
    echo "[stop_vllm] SIGKILL process group -${pid}"
    kill -KILL "-${pid}" 2>/dev/null || true
  else
    echo "[stop_vllm] SIGKILL PID tree rooted at ${pid}"
    child_pids_recursive "${pid}" | sort -rn | xargs -r kill -KILL 2>/dev/null || true
    kill -KILL "${pid}" 2>/dev/null || true
  fi
}

pids=()
if [[ -f "${PID_FILE}" ]]; then
  pid="$(cat "${PID_FILE}")"
  if [[ -n "${pid}" ]]; then
    pids+=("${pid}")
  fi
else
  echo "[stop_vllm] No PID file found for port ${PORT}; probing listening process."
fi

while IFS= read -r pid; do
  [[ -n "${pid}" ]] && pids+=("${pid}")
done < <(find_port_pids "${PORT}")

if [[ "${#pids[@]}" -eq 0 ]]; then
  echo "[stop_vllm] No vLLM process found for port ${PORT}."
  rm -f "${PID_FILE}" "/tmp/vllm_${PORT}.port"
  exit 0
fi

mapfile -t unique_pids < <(printf '%s\n' "${pids[@]}" | sort -u)
echo "[stop_vllm] Stopping vLLM on port ${PORT}: PIDs=${unique_pids[*]}"

for pid in "${unique_pids[@]}"; do
  terminate_pid_tree "${pid}"
done

deadline=$((SECONDS + GRACE_SECONDS))
while [[ "${SECONDS}" -lt "${deadline}" ]]; do
  still_running=()
  for pid in "${unique_pids[@]}"; do
    if is_running "${pid}"; then
      still_running+=("${pid}")
    fi
  done
  if [[ "${#still_running[@]}" -eq 0 ]]; then
    break
  fi
  sleep 1
done

for pid in "${unique_pids[@]}"; do
  if is_running "${pid}"; then
    kill_pid_tree "${pid}"
  fi
done

rm -f "${PID_FILE}" "/tmp/vllm_${PORT}.port"
echo "[stop_vllm] Done."
