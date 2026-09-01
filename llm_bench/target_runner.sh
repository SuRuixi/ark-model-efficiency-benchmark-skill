#!/usr/bin/env bash
# Shared target-profile loader for run_*.sh.
#
# New interface:
#   ./run_prefix_repetition.sh ark
#   ./run_prefix_repetition.sh ark tencent --max-concurrency 20

TARGETS_DIR="${TARGETS_DIR:-targets}"
TARGET_NAMES=()
BENCH_ARGS=()
INHERITED_TARGET_MODEL_SET=0
if [ -n "${TARGET_MODEL+x}" ]; then
  INHERITED_TARGET_MODEL_SET=1
fi
unset TARGET_MODEL

target_die() {
  echo "[target] $*" >&2
  exit 1
}

parse_target_cli() {
  TARGET_NAMES=()
  BENCH_ARGS=()

  while [ "$#" -gt 0 ]; do
    case "$1" in
      --)
        shift
        BENCH_ARGS=("$@")
        break
        ;;
      -*)
        BENCH_ARGS=("$@")
        break
        ;;
      *)
        TARGET_NAMES+=("$1")
        shift
        ;;
    esac
  done

  if [ "${#TARGET_NAMES[@]}" -eq 0 ]; then
    TARGET_NAMES=("ark")
  fi

  local arg
  for arg in ${BENCH_ARGS[@]+"${BENCH_ARGS[@]}"}; do
    case "$arg" in
      --model|--model=*|--base-url|--base-url=*|--api-key|--api-key=*|\
      --label|--label=*|--compare|--compare=*|--output-param|--output-param=*|\
      --thinking|--thinking=*|--reasoning-effort|--reasoning-effort=*|\
      --target-max-completion-tokens|--target-max-completion-tokens=*)
        target_die "${arg%%=*} 是目标专属参数，请在对应的 targets/<目标>.env 中配置"
        ;;
    esac
  done

  if [ "$INHERITED_TARGET_MODEL_SET" = "1" ]; then
    echo "[target] 忽略终端中的 TARGET_MODEL；请在各 targets/<目标>.env 中分别配置模型" >&2
  fi
}

target_network_skipped() {
  local arg
  for arg in ${BENCH_ARGS[@]+"${BENCH_ARGS[@]}"}; do
    case "$arg" in
      -h|--help|--dump-data|--dump-data=*) return 0 ;;
    esac
  done
  return 1
}

load_target() {
  local name="$1"
  local file

  case "$name" in
    ""|"."|".."|*/*) target_die "非法目标名: ${name}" ;;
  esac
  file="${TARGETS_DIR}/${name}.env"
  [ -f "$file" ] || target_die "未找到 ${file}（可用目标见 ${TARGETS_DIR}/ 目录）"

  # TARGET_MODEL belongs to this profile only; never reuse a value from the
  # parent shell or a previously loaded comparison target.
  unset TARGET_LABEL TARGET_MODEL TARGET_BASE_URL TARGET_API_KEY_ENV TARGET_API_KEY
  unset TARGET_OUTPUT_PARAM TARGET_THINKING TARGET_REASONING_EFFORT
  unset TARGET_MAX_COMPLETION_TOKENS

  # shellcheck disable=SC1090
  source "$file"

  case "${TARGET_LABEL:-}${TARGET_MODEL:-}${TARGET_BASE_URL:-}${TARGET_API_KEY_ENV:-}" in
    *"<"*) target_die "${file} 仍有 <...> 占位符未替换" ;;
  esac
  [ -n "${TARGET_MODEL:-}" ] || target_die "${file} 未设置 TARGET_MODEL"
  [ -n "${TARGET_BASE_URL:-}" ] || target_die "${file} 未设置 TARGET_BASE_URL"
  [ -z "${TARGET_API_KEY:-}" ] ||
    target_die "${file} 禁止直接设置 TARGET_API_KEY；请使用 TARGET_API_KEY_ENV 引用外部环境变量"
  [ -n "${TARGET_API_KEY_ENV:-}" ] ||
    target_die "${file} 未设置 TARGET_API_KEY_ENV"

  if [[ ! "$TARGET_API_KEY_ENV" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]]; then
    target_die "${file} 的 TARGET_API_KEY_ENV 不是合法环境变量名"
  fi

  LOADED_NAME="$name"
  LOADED_LABEL="${TARGET_LABEL:-$name}"
  LOADED_MODEL="$TARGET_MODEL"
  LOADED_BASE_URL="$TARGET_BASE_URL"
  LOADED_API_KEY_ENV="$TARGET_API_KEY_ENV"
  LOADED_OUTPUT_PARAM="${TARGET_OUTPUT_PARAM:-}"
  LOADED_THINKING="${TARGET_THINKING:-}"
  LOADED_MAX_COMPLETION_TOKENS="${TARGET_MAX_COMPLETION_TOKENS:-}"
  LOADED_REASONING_EFFORT="${TARGET_REASONING_EFFORT:-}"
  if [ -n "${TARGET_REASONING_EFFORT+x}" ]; then
    LOADED_REASONING_EFFORT_SET=1
  else
    LOADED_REASONING_EFFORT_SET=0
  fi
}

resolve_loaded_api_key() {
  RESOLVED_API_KEY="${!LOADED_API_KEY_ENV:-}"
  if [ -z "$RESOLVED_API_KEY" ] && ! target_network_skipped; then
    target_die "${TARGETS_DIR}/${LOADED_NAME}.env 需要环境变量 ${LOADED_API_KEY_ENV}"
  fi
}

prepare_first_target() {
  load_target "${TARGET_NAMES[0]}"

  FIRST_TARGET_NAME="$LOADED_NAME"
  FIRST_TARGET_LABEL="$LOADED_LABEL"
  # Snapshot the model before another target profile is loaded.
  FIRST_TARGET_MODEL="$LOADED_MODEL"

  resolve_loaded_api_key
  FIRST_TARGET_API_KEY="$RESOLVED_API_KEY"
  FIRST_TARGET_BASE_URL="$LOADED_BASE_URL"
  export LLM_BENCH_API_KEY="$FIRST_TARGET_API_KEY"
  export LLM_BENCH_BASE_URL="$FIRST_TARGET_BASE_URL"

  FIRST_TARGET_ARGS=(--model "$FIRST_TARGET_MODEL")
  [ -n "$LOADED_OUTPUT_PARAM" ] &&
    FIRST_TARGET_ARGS+=(--output-param "$LOADED_OUTPUT_PARAM")
  [ -n "$LOADED_THINKING" ] &&
    FIRST_TARGET_ARGS+=(--thinking "$LOADED_THINKING")
  if [ "$LOADED_REASONING_EFFORT_SET" = "1" ]; then
    FIRST_TARGET_ARGS+=(--reasoning-effort "$LOADED_REASONING_EFFORT")
  fi
  [ -n "$LOADED_MAX_COMPLETION_TOKENS" ] &&
    FIRST_TARGET_ARGS+=(--target-max-completion-tokens "$LOADED_MAX_COMPLETION_TOKENS")
  return 0
}

prepare_compare_targets() {
  COMPARE_ARGS=()
  local i name model spec key_env

  i=1
  while [ "$i" -lt "${#TARGET_NAMES[@]}" ]; do
    name="${TARGET_NAMES[$i]}"
    load_target "$name"
    # Keep this profile's model in its own --compare specification.
    model="$LOADED_MODEL"

    resolve_loaded_api_key
    key_env="LLM_BENCH_COMPARE_API_KEY_${i}"
    printf -v "$key_env" "%s" "$RESOLVED_API_KEY"
    export "$key_env"
    spec="label=${LOADED_LABEL};model=${model};base_url=${LOADED_BASE_URL}"
    spec="${spec};api_key_env=${key_env}"
    [ -n "$LOADED_OUTPUT_PARAM" ] &&
      spec="${spec};output_param=${LOADED_OUTPUT_PARAM}"
    [ -n "$LOADED_THINKING" ] &&
      spec="${spec};thinking=${LOADED_THINKING}"
    if [ "$LOADED_REASONING_EFFORT_SET" = "1" ]; then
      spec="${spec};reasoning_effort=${LOADED_REASONING_EFFORT}"
    fi
    [ -n "$LOADED_MAX_COMPLETION_TOKENS" ] &&
      spec="${spec};max_completion_tokens=${LOADED_MAX_COMPLETION_TOKENS}"
    COMPARE_ARGS+=(--compare "$spec")
    i=$((i + 1))
  done

  # Keep the first target in neutral internal variables. Provider-specific
  # variables remain untouched and cannot contaminate another target.
  export LLM_BENCH_API_KEY="$FIRST_TARGET_API_KEY"
  export LLM_BENCH_BASE_URL="$FIRST_TARGET_BASE_URL"
}
