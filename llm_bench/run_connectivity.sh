#!/usr/bin/env bash
# Usage: ./run_connectivity.sh [target] [bench.py options]
# Example: ./run_connectivity.sh ark
set -euo pipefail
cd "$(dirname "$0")"

# shellcheck source=target_runner.sh
source ./target_runner.sh
parse_target_cli "$@"
[ "${#TARGET_NAMES[@]}" -eq 1 ] ||
  target_die "连通性自检一次只接受一个目标"
prepare_first_target

PY=".venv/bin/python"
[ -x "$PY" ] || PY="python3"

echo "[connectivity] 目标=${FIRST_TARGET_NAME} model=${FIRST_TARGET_MODEL}"
exec "$PY" bench.py connectivity \
  "${FIRST_TARGET_ARGS[@]}" \
  ${BENCH_ARGS[@]+"${BENCH_ARGS[@]}"}
