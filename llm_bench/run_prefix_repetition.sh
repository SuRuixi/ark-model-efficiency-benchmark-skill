#!/usr/bin/env bash
# Usage: ./run_prefix_repetition.sh [target ...] [bench.py options]
# Example: ./run_prefix_repetition.sh ark tencent --max-concurrency 20
set -euo pipefail
cd "$(dirname "$0")"

# shellcheck source=target_runner.sh
source ./target_runner.sh
parse_target_cli "$@"
prepare_first_target
prepare_compare_targets

PY=".venv/bin/python"
[ -x "$PY" ] || PY="python3"
EXTRA_ARGS=()
[ -n "${OUTPUT:-}" ] && EXTRA_ARGS+=(--output "$OUTPUT")
[ -n "${SEED:-}" ] && EXTRA_ARGS+=(--seed "$SEED")

exec "$PY" bench.py prefix-repetition \
  "${FIRST_TARGET_ARGS[@]}" \
  --label "$FIRST_TARGET_LABEL" \
  --prefix-len "${PREFIX_LEN:-12000}" \
  --suffix-len "${SUFFIX_LEN:-2000}" \
  --num-prefixes "${NUM_PREFIXES:-10}" \
  --num-requests "${NUM_REQUESTS:-200}" \
  --max-concurrency "${MAX_CONCURRENCY:-5}" \
  ${COMPARE_ARGS[@]+"${COMPARE_ARGS[@]}"} \
  ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} \
  ${BENCH_ARGS[@]+"${BENCH_ARGS[@]}"}
