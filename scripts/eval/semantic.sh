#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_SUITE="semantic" exec bash "${SCRIPT_DIR}/run_model_eval.sh" "$@"
