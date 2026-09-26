#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RELEASE_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

export PYTHONPATH="$RELEASE_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
python -m cdrbench_v3.run_inference "$@"
