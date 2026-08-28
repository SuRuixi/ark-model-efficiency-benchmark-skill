#!/usr/bin/env bash
# 模式 A：prefix（对标 vLLM prefix_repetition）
#
# 核心概念：
#   命中来源       = 横向：不同请求复用同一个固定 prefix（模拟多用户共享系统提示词）
#   三层相同性     = 不同 prefix 之间内容不同；同一 prefix 的多个请求前缀逐字相同（命中来源）；
#                    每个请求的 suffix 各不相同（保证不额外命中、避免服务端去重）
#   命中率上限     ≈ prefix_len×(每前缀请求数−1) / [(prefix_len+suffix_len)×每前缀请求数]
#                    NUM_PREFIXES 越小命中率越高；每前缀首个请求为冷启动未命中，属预期
#
# 可用环境变量覆盖（均为用户可调）：
#   MODEL             被测模型 ID
#   PREFIX_LEN        前缀长度（token，主变量）
#   SUFFIX_LEN        后缀长度（token，主变量）
#   NUM_PREFIXES      前缀池个数（决定命中率上限）
#   NUM_REQUESTS      总请求数
#   MAX_CONCURRENCY   并发上限（请求间无依赖，可全并发）
#   OUTPUT            结果 JSON 路径（默认 reports/prefix_<时间戳>/result_prefix.json）
#   PEER              可选：友商对比配置名，加载 peers/<名字>.env；不设即单边压测
#   COMPARE           同步对比友商模型（同一份样本数据、同一时间窗并发发起）：
#                     格式 'label=显示名;model=模型ID;base_url=地址;api_key_env=环境变量名'
#                     多个对比目标可给 COMPARE_1 / COMPARE_2 / ...
#                     示例：COMPARE='label=友商A;model=qwen-max;base_url=https://dashscope.aliyun.com/compatible-mode/v1;api_key_env=DASHSCOPE_API_KEY'
set -euo pipefail
cd "$(dirname "$0")"

export ARK_BASE_URL="${ARK_BASE_URL:-https://ark.cn-beijing.volces.com/api/v3}"
# export ARK_API_KEY="..."      # 必须提前 export（或 OPENAI_API_KEY）
# 友商对比（可选）：PEER=<名字> 加载 peers/<名字>.env，未设 PEER 即单边压测；
# PEER 设了但配置不存在时直接报错退出（避免静默跑成单边）
if [ -n "${PEER:-}" ]; then
  if [ -f "peers/${PEER}.env" ]; then
    source "peers/${PEER}.env"
    echo "[peer] 已加载友商配置 peers/${PEER}.env（${PEER_LABEL:-?} / ${PEER_MODEL:-?}）"
    # 填空残留自检：占位符没替换 / cp 别家配置漏改（key 变量名对不上）时拦下，
    # 避免静默用错 key 或带着 <...> 占位符去请求
    case "${PEER_LABEL:-}${PEER_MODEL:-}${PEER_BASE_URL:-}" in
      *"<"*) echo "[peer] PEER_* 仍有 <...> 占位符未替换，请按 peers/_template.env 注释填空" >&2; exit 1 ;;
    esac
    # PEER 设了但没填 PEER_MODEL 会静默跑成单边压测，必须拦下
    if [ -z "${PEER_MODEL:-}" ]; then
      echo "[peer] peers/${PEER}.env 未设置 PEER_MODEL（必填），请按 peers/_template.env 注释填空" >&2
      exit 1
    fi
    if [ -n "${PEER_API_KEY_ENV:-}" ] && [ -z "$(eval echo "\$${PEER_API_KEY_ENV}")" ]; then
      echo "[peer] PEER_API_KEY_ENV=${PEER_API_KEY_ENV} 未定义--多半是 cp 了别家配置没改 key 变量名，请对照 peers/_template.env 检查" >&2
      exit 1
    fi
  else
    echo "[peer] 未找到 peers/${PEER}.env（可用配置见 peers/ 目录）" >&2
    exit 1
  fi
fi

PY=".venv/bin/python"; [ -x "$PY" ] || PY="python3"
# 结果 JSON / 报告 / 图像默认落进本次运行独立目录 reports/prefix_<时间戳>/；
# 显式指定 OUTPUT 则用用户给的路径
EXTRA_ARGS=()
[ -n "${OUTPUT:-}" ] && EXTRA_ARGS+=(--output "$OUTPUT")
# 常用形态：peers 配置里设了 PEER_MODEL 就自动拼 --compare，无需手写 spec
if [ -n "${PEER_MODEL:-}" ]; then
  spec="model=${PEER_MODEL}"
  [ -n "${PEER_LABEL:-}" ] && spec="label=${PEER_LABEL};${spec}"
  [ -n "${PEER_BASE_URL:-}" ] && spec="${spec};base_url=${PEER_BASE_URL}"
  if [ -n "${PEER_API_KEY_ENV:-}" ]; then
    spec="${spec};api_key_env=${PEER_API_KEY_ENV}"
  elif [ -n "${PEER_API_KEY:-}" ]; then
    spec="${spec};api_key=${PEER_API_KEY}"
  fi
  [ -n "${PEER_OUTPUT_PARAM:-}" ] && spec="${spec};output_param=${PEER_OUTPUT_PARAM}"
  [ -n "${PEER_THINKING:-}" ] && spec="${spec};thinking=${PEER_THINKING}"
  # PEER_REASONING_EFFORT：覆盖思考深度；置空字符串表示对该厂商不发送（如阿里
  # 只认 low/medium/high/xhigh/max、拒绝 none）
  [ -n "${PEER_REASONING_EFFORT+x}" ] && spec="${spec};reasoning_effort=${PEER_REASONING_EFFORT}"
  [ -n "${PEER_ENABLE_THINKING:-}" ] && spec="${spec};enable_thinking=${PEER_ENABLE_THINKING}"
  EXTRA_ARGS+=(--compare "$spec")
fi
# 通用形态：多友商时用 COMPARE / COMPARE_1 / ... 手写完整 spec（与 PEER_* 可叠加）
for v in "${COMPARE:-}" "${COMPARE_1:-}" "${COMPARE_2:-}" "${COMPARE_3:-}"; do
  [ -n "$v" ] && EXTRA_ARGS+=(--compare "$v")
done
exec "$PY" bench.py prefix \
  --model "${MODEL:-deepseek-v4-flash-ga-260731}" \
  --prefix-len "${PREFIX_LEN:-12000}" \
  --suffix-len "${SUFFIX_LEN:-2000}" \
  --num-prefixes "${NUM_PREFIXES:-10}" \
  --num-requests "${NUM_REQUESTS:-200}" \
  --max-concurrency "${MAX_CONCURRENCY:-5}" \
  ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} "$@"
