# Ark Model Efficiency Benchmark Skill

基于 Ark CLI 凭据体系与内置 `llm-bench` 的一键大语言模型性能评测 Skill。用户只需用自然语言指定 Ark 模型或 Endpoint，即可通过后付费 API 的预置推理接入点执行标准化压测，并获得 Markdown、JSON 与 PNG 报告，全程无需手写 API Key。

[English](#english)

## 中文

### 核心能力

- 自动检查 Ark CLI 登录状态。
- 自动选择 Ark CLI 的 `platform` 后付费 Profile。
- 将模型名称解析为 Model ID，直接调用 Ark 预置推理接入点；无需创建自定义 Endpoint。
- 通过 Ark CLI 的机器接口读取平台凭据，密钥仅用于请求头，不写入报告或日志。
- 执行连通性、Prefix 前缀复用、多轮长上下文三类测试。
- 统计 TTFT、TPOT、E2E、缓存命中率，以及 Mean、P50、P99。
- 使用随 Skill 分发的 ShareGPT 本地语料，无需联网下载数据集。
- 输出请求级 JSON、Markdown 摘要、运行日志和 PNG 趋势图。

### 前置条件

- macOS 或 Linux
- Python 3.9+
- Node.js 与 npm
- 已安装并登录 Ark CLI

安装和登录 Ark CLI：

```bash
npm config set registry https://registry.npmjs.org/
npm install -g @volcengine/ark-cli@latest
arkcli auth login volc-sso
arkcli auth status
```

登录完成后，Ark CLI 会管理 Profile 与 API Key。评测过程不要求用户复制密钥。

内部用户已经安装 Ark CLI、但尚未完成 SSO 登录时：

1. 打开[内部方舟账号页面](https://babi.bytedance.net/finance/basic/volcManage/?fullscreen=true&volc_account_category=1&tab=my&status=1)，登录内部账号。
2. 执行 `arkcli auth login volc-sso`，完成浏览器 SSO 授权。
3. 执行 `arkcli auth status --format json`，确认 `logged_in=true`。
4. 重新运行原评测命令，无需填写 API Key。

### 安装 Skill

在目标项目根目录执行：

```bash
mkdir -p .trae/skills
git clone https://github.com/SuRuixi/ark-model-efficiency-benchmark-skill.git \
  .trae/skills/ark-model-benchmark
```

重新打开 Agent 会话，使项目级 Skill 被加载。

### 自然语言使用

可以直接输入：

```text
帮我测试一下 Ark 上豆包 Seed 2.0 Mini 模型的性能
```

也支持：

```text
压测 Ark 上的 doubao-seed-evolving
测一下 ep-xxxxxxxx 的 TTFT、TPOT 和缓存命中率
比较 Ark 上两个模型的延迟
```

Skill 会自动解析模型、选择后付费 `platform` Profile、运行评测并汇总报告。模型名称存在歧义时，会返回候选列表供用户选择。Agent Plan 与 Coding Plan 不参与评测调用。

### 选择测试模式

当用户只提出「测试模型性能」而未指定场景时，Skill 会先询问：

```text
请选择测试模式：连通性检查（1 个请求）、Prefix 前缀复用（200 个请求）、
多轮长上下文（200 个请求），或完整测试（401 个请求）？
```

| 模式 | 适用目标 |
|---|---|
| `connectivity` | 验证模型、鉴权和基础流式指标 |
| `prefix` | 测试固定前缀复用和缓存命中 |
| `multiturn` | 测试对话历史增长后的延迟与缓存 |
| `all` | 顺序执行以上三种模式 |

用户已经明确说明模式时，Skill 会直接执行。多模型对比是 `prefix` 或 `multiturn` 的附加能力，不属于独立模式。

### 命令行使用

完整标准评测：

```bash
bash scripts/run.sh --model "doubao-seed-2-0-mini" --scenario all --preset standard
```

低成本快速验证：

```bash
bash scripts/run.sh --model "doubao-seed-2-0-mini" --scenario all --preset quick
```

仅运行指定场景：

```bash
bash scripts/run.sh --model "doubao-seed-2-0-mini" --scenario connectivity
bash scripts/run.sh --model "doubao-seed-2-0-mini" --scenario prefix --preset standard
bash scripts/run.sh --model "doubao-seed-2-0-mini" --scenario multiturn --preset standard
```

固定 Ark CLI Profile：

```bash
bash scripts/run.sh \
  --model "doubao-seed-evolving" \
  --profile "<platform-profile-name>" \
  --preset quick
```

显式指定的 Profile 必须为 `type=platform`。自然语言模型名会通过 Ark 模型目录解析为 Model ID，并由平台自动路由至预置推理接入点。

首次运行会在 Skill 目录创建隔离的 `.venv`，并自动安装 `aiohttp`、`numpy`、`tiktoken` 与 `matplotlib`。

### 评测口径

| 场景 | `standard` 参数 |
|---|---|
| Prefix 前缀复用 | 12,000 前缀 Token + 2,000 后缀 Token，10 个前缀，200 个请求，并发 5 |
| 多轮长上下文 | 3,000 初始 Token + 每轮 256 Token，10 个会话，20 轮，并发 5 |
| 输出设置 | Prefix 最多 512 Token，多轮最多 1,024 Token，默认关闭推理 |

`standard` 完整流程包含 1 个连通性请求、200 个 Prefix 请求和 200 个多轮生成，共 401 个请求，可能产生较高 Token 用量。`quick` 用于功能验证与低成本预检。

指标定义：

- TTFT：请求开始至首个输出 Token 到达的时间。
- E2E：请求开始至最终响应到达的时间。
- TPOT：`(E2E - TTFT) / (输出 Token 数 - 1)`。
- 缓存命中率：成功请求的缓存输入 Token 总数除以输入 Token 总数。

跨模型对比应保持机器、网络、输入输出长度、并发度、推理设置和测试时间窗口一致。

### 参数覆盖

```text
--prefix-len
--suffix-len
--num-prefixes
--num-requests
--initial-len
--question-len
--num-sessions
--max-turns
--max-concurrency
--max-output-tokens
--reasoning-effort
--thinking
--seed
--profile
--output-dir
```

查看完整帮助：

```bash
bash scripts/run.sh --help
```

### 报告结构

默认输出到当前工作目录的 `reports/`：

```text
reports/
├── connectivity_<timestamp>/
│   ├── report.md
│   └── connectivity.log
├── prefix_<timestamp>/
│   ├── report_prefix-repetition.md
│   ├── result_prefix-repetition.json
│   ├── report_prefix-repetition_reuse.png
│   └── run.log
└── multiturn_<timestamp>/
    ├── report_multi-turn.md
    ├── result_multi-turn.json
    ├── report_multi-turn_turns.png
    └── run.log
```

Skill 对外继续使用 `prefix` 与 `multiturn`，执行器会分别映射到新版
`llm-bench` 的 `prefix-repetition` 与 `multi-turn` 子命令。凭据仅通过
`LLM_BENCH_BASE_URL` 与 `LLM_BENCH_API_KEY` 注入子进程。

### 安全机制

- 凭据优先通过 `arkcli profile apikey get --plain` 读取；旧版本不支持该参数时，自动回退到 `--format json` 并解析 `api_key` 字段。
- API Key 不通过命令参数传入，不写入报告、日志或 JSON。
- HTTP 错误写入报告前会执行密钥脱敏。
- 请勿在调试时启用包含环境变量的 Shell 跟踪。

### 测试

```bash
.venv/bin/python -m unittest discover -s tests -v
```

---

## English

### Overview

Ark Model Efficiency Benchmark is a one-command LLM performance evaluation Skill built on Ark CLI authentication and the bundled `llm-bench` engine. It invokes Ark preset inference endpoints through the postpaid platform API. A user can specify an Ark model or endpoint in natural language and receive Markdown, JSON, and PNG reports without manually entering an API key.

### Features

- Verifies Ark CLI authentication.
- Selects an Ark CLI `platform` profile and excludes Agent Plan and Coding Plan profiles.
- Resolves model aliases to Model IDs and invokes Ark preset inference endpoints without creating a custom endpoint.
- Retrieves the platform credential through Ark CLI without persisting it.
- Runs connectivity, prefix-reuse, and multi-turn long-context workloads.
- Measures TTFT, TPOT, E2E latency, cache hit rate, Mean, P50, and P99.
- Uses a bundled local ShareGPT corpus without downloading a dataset.
- Produces request-level JSON, Markdown summaries, redacted logs, and PNG charts.

### Requirements

- macOS or Linux
- Python 3.9+
- Node.js and npm
- An authenticated Ark CLI installation

Install and authenticate Ark CLI:

```bash
npm config set registry https://registry.npmjs.org/
npm install -g @volcengine/ark-cli@latest
arkcli auth login volc-sso
arkcli auth status
```

For internal users who installed Ark CLI but have not completed SSO:

1. Sign in through the [internal Ark account page](https://babi.bytedance.net/finance/basic/volcManage/?fullscreen=true&volc_account_category=1&tab=my&status=1).
2. Run `arkcli auth login volc-sso` and complete browser authorization.
3. Verify `logged_in=true` with `arkcli auth status --format json`.
4. Rerun the original benchmark command without entering an API key.

### Install the Skill

Run from the target project root:

```bash
mkdir -p .trae/skills
git clone https://github.com/SuRuixi/ark-model-efficiency-benchmark-skill.git \
  .trae/skills/ark-model-benchmark
```

Restart the Agent session so the project-level Skill is loaded.

### Natural-Language Usage

Example:

```text
Benchmark the performance of Doubao Seed 2.0 Mini on Ark.
```

The Skill resolves the model to a postpaid Model ID, selects a `platform` profile, runs the benchmark, and summarizes the generated reports. Ambiguous model names produce a ranked candidate list.

### Select a Benchmark Mode

For a generic performance-test request, the Skill asks the user to select one
mode before execution:

| Mode | Purpose |
|---|---|
| `connectivity` | Validate model access and basic streaming metrics with one request |
| `prefix` | Measure fixed-prefix reuse and cache behavior |
| `multiturn` | Measure growing conversation context |
| `all` | Run all three modes sequentially |

An explicitly stated mode runs immediately. Multi-model comparison is an option
within `prefix` or `multiturn`, not a separate mode.

### CLI Usage

Run the complete standard benchmark:

```bash
bash scripts/run.sh --model "doubao-seed-2-0-mini" --scenario all --preset standard
```

Run a low-cost validation:

```bash
bash scripts/run.sh --model "doubao-seed-2-0-mini" --scenario all --preset quick
```

Run one workload:

```bash
bash scripts/run.sh --model "doubao-seed-2-0-mini" --scenario connectivity
bash scripts/run.sh --model "doubao-seed-2-0-mini" --scenario prefix --preset standard
bash scripts/run.sh --model "doubao-seed-2-0-mini" --scenario multiturn --preset standard
```

The first run creates an isolated `.venv` and installs `aiohttp`, `numpy`, `tiktoken`, and `matplotlib`.

The wrapper maps `prefix` to the bundled `prefix-repetition` subcommand and
`multiturn` to `multi-turn`. Ark credentials are passed only through
`LLM_BENCH_BASE_URL` and `LLM_BENCH_API_KEY`.

### Standard Workload

| Scenario | `standard` parameters |
|---|---|
| Prefix reuse | 12,000 prefix tokens + 2,000 suffix tokens, 10 prefixes, 200 requests, concurrency 5 |
| Multi-turn context | 3,000 initial tokens + 256 tokens per follow-up, 10 sessions, 20 turns, concurrency 5 |
| Generation | Up to 512 prefix output tokens and 1,024 multi-turn output tokens; reasoning disabled |

The complete `standard` run sends 401 requests and may consume a substantial number of tokens. Use `quick` for functional validation.

### Metrics

- TTFT: time from request start to the first output token.
- E2E: time from request start to the final response event.
- TPOT: `(E2E - TTFT) / (output tokens - 1)`.
- Cache hit rate: total cached input tokens divided by total input tokens across successful requests.

Cross-model comparisons require the same machine, network, token lengths, concurrency, reasoning settings, and test window.

### Security

- Credentials are retrieved through `arkcli profile apikey get --plain`, with `--format json` as a compatibility fallback for older CLI versions.
- API keys are never passed as command-line arguments or stored in generated artifacts.
- HTTP errors are redacted before they are recorded.

### Tests

```bash
.venv/bin/python -m unittest discover -s tests -v
```
