# llm-bench

通过 OpenAI 兼容的 Chat Completions 接口测试一个或多个 LLM 服务的 **TTFT、TPOT、E2E 和 Cache 命中率**。

> 在 `ark-model-benchmark` Skill 中，请优先运行
> `bash ../scripts/run.sh --model "<模型名称>" --scenario "<模式>"`。该包装器会从
> Ark CLI 自动注入后付费 Profile、Base URL 和 API Key，无需修改 `targets/ark.env`。

## 安装

```bash
cd llm-bench
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 使用方式

只需要修改和选择两样东西：

- **目标**：`targets/` 中的一份服务配置，例如 `ark.env`、`mytarget.env`。
- **场景**：固定前缀池使用 `run_prefix_repetition.sh`，多轮会话测缓存命中使用
  `run_multi_turn.sh`。

### 1. 配置目标

每个服务商对应一个 `targets/*.env`。新增目标时先复制模板：

```bash
cp targets/_template.env targets/mytarget.env
```

运行前打开目标文件，按文件内注释填写 API Key、模型 ID 和 Base URL。例如
`targets/ark.env`：

```bash
export ARK_API_KEY="..."
export TARGET_MODEL="your-model-id"
```

每个目标的模型独立配置，不要在终端全局设置 `TARGET_MODEL`。API Key 可按目标文件内的注释从终端继承。

### 2. 检查连通性

每个参与压测的目标都先检查一次：

```bash
./run_connectivity.sh ark
./run_connectivity.sh mytarget
```

确认模型 ID、Token 统计和思考状态符合预期后再开始压测。

### 3. 运行压测

一个目标表示单压，多个目标表示同步对比：

```bash
# 单目标
./run_prefix_repetition.sh ark
./run_multi_turn.sh ark

# 两个目标对比
./run_prefix_repetition.sh ark mytarget
./run_multi_turn.sh ark mytarget

# 多目标对比
./run_prefix_repetition.sh ark mytarget another-target
```

命令格式为“脚本 + 一个或多个目标 + 可选参数”。多个目标表示同步对比，目标顺序只决定报告中的展示顺序；所有目标使用各自的配置和同一份输入，并分别统计耗时。

## 目标配置

项目已在 `targets/` 中预置若干常见服务配置，均不包含凭据，可用目标
以目录内容为准。目标名就是 `targets/<目标名>.env` 的文件名，也可复制 `_template.env` 新增目标。

更换模型时，直接修改对应文件中的 `TARGET_MODEL`。也可以在项目根目录使用命令修改对应配置文件并确认：

```bash
perl -pi -e 's/^export TARGET_MODEL=.*/export TARGET_MODEL="your-model-id"/' targets/mytarget.env
grep '^export TARGET_MODEL=' targets/mytarget.env
./run_connectivity.sh mytarget
```

命令中的模型 ID 和配置文件路径应按目标替换。

不同厂商可能用不同 ID 表示同一模型版本。跨厂比较前应确认版本映射，并保持思考设置和输出上限一致。

默认发送 `thinking.type=disabled`。若连通性检查发现模型不允许关闭思考，会根据服务端错误提示在对应 target 中设置 `TARGET_THINKING="enabled"`；模型要求推理档位时再设置 `TARGET_REASONING_EFFORT`。

## 压测模式

### Prefix Repetition

N 个不同前缀组成固定前缀池。每个前缀由多个请求重复使用，每个请求使用不同后缀。该场景对齐 vLLM 的 `prefix_repetition` workload。

```bash
./run_prefix_repetition.sh ark mytarget \
  --prefix-len 12000 \
  --suffix-len 2000 \
  --num-prefixes 10 \
  --num-requests 400 \
  --max-concurrency 5
```

每个前缀的第一次请求是冷启动。高并发下，前几个请求可能在 cache 写入完成前到达，因此命中率可能低于稳态值。

### Multi-turn

每个 session 是一条串行对话链，session 之间并发。Turn 1 发送长输入，Turn 2+ 增加短追问，历史回答会进入后续上下文。

```bash
./run_multi_turn.sh ark mytarget \
  --initial-len 3000 \
  --question-len 256 \
  --num-sessions 20 \
  --max-turns 20 \
  --max-concurrency 10
```

查看全部参数：

```bash
./run_prefix_repetition.sh --help
./run_multi_turn.sh --help
```

常用参数：

- **Prefix Repetition**：`--prefix-len` 为每个固定前缀的 token 数，
  `--suffix-len` 为每个请求独立后缀的 token 数，`--num-prefixes` 为前缀池大小，
  `--num-requests` 为请求总数。
- **Multi-turn**：`--initial-len` 为首轮长输入的 token 数，`--question-len` 为
  后续每轮新增问题的 token 数，`--num-sessions` 为会话总数，`--max-turns` 为
  每个会话的最大轮数。
- **通用**：`--max-concurrency` 是每个目标独立的并发上限，在 Prefix
  Repetition 中表示并发请求数，在 Multi-turn 中表示并发 session 数；多个目标
  同时压测时，整体最大在途请求数约为目标数乘以该值；`--seed` 用于重放同一批输入；
  `--output` 指定逐请求 JSON；`--max-completion-tokens` 统一覆盖所有目标的
  单次输出上限。仅需修改某个目标时，在对应配置文件中设置
  `TARGET_MAX_COMPLETION_TOKENS`。

## 报告

每次运行生成独立目录，目录名包含模式、服务商、模型和时间：

```text
reports/<mode>_<服务商>-<模型>[_vs_...]_<时间戳>/
report_<mode>.md   Markdown 报告
report_<mode>*.png 图表
result_<mode>.json 逐请求原始数据
```

逐请求结果会保存服务商响应中的 `request_id`，并在可用时保存响应头中的
`provider_log_id`，便于定位单次请求；服务商未返回时对应字段为空。

配置中的长度是本地语料净长度；实际输入 token 数来自服务端
`usage.prompt_tokens`，包含模板、当前输入和对话历史。

Cache 命中率只统计成功且返回 cache 字段的请求：

```text
命中率 = Σ cached_tokens / Σ prompt_tokens
```

multi-turn 每轮的平均输入 tokens 只统计成功到达该轮的 session。session 在某轮
失败后立即退出，后续轮次不补零；图底部会显示每轮成功样本数并标出终止轮次。

P99 建议至少采集 1000 个成功样本。跨批次比较时应保持模型版本、思考设置、
输出上限、并发和 seed 一致。

## 运行规则

- 语料来自本地 `data/sharegpt_pool.txt`，不需要在线下载。
- 默认发送 `thinking.type=disabled` 关闭思考，不做 warmup；
  `reasoning_effort` 仅在对应 target 明确配置时发送。
- 失败请求不进入时延统计。
- multi-turn 对 429 和连接错误最多重试 3 次，其他失败会终止当前 session。
- 连续出现配置性 4xx 时会 fail-fast，避免整轮请求持续失败。
