#!/usr/bin/env bash
# Usage: ./run_multi_turn.sh [target ...] [bench.py options]
# Example: ./run_multi_turn.sh ark tencent --num-sessions 50
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

exec "$PY" bench.py multi-turn \
  "${FIRST_TARGET_ARGS[@]}" \
  --label "$FIRST_TARGET_LABEL" \
  --initial-len "${INITIAL_LEN:-3000}" \
  --question-len "${QUESTION_LEN:-256}" \
  --num-sessions "${NUM_SESSIONS:-10}" \
  --max-turns "${MAX_TURNS:-20}" \
  --max-concurrency "${MAX_CONCURRENCY:-5}" \
  ${COMPARE_ARGS[@]+"${COMPARE_ARGS[@]}"} \
  ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} \
  ${BENCH_ARGS[@]+"${BENCH_ARGS[@]}"}
