# Ark Model Efficiency Benchmark Skill

基于 Ark CLI 凭据体系的一键大语言模型性能评测 Skill。用户只需用自然语言指定 Ark 模型或 Endpoint，即可通过后付费 API 的预置推理接入点执行标准化压测，并获得 Markdown、JSON、CSV 与 PNG 报告，全程无需手写 API Key。

[English](#english)

## 中文

### 核心能力

- 自动检查 Ark CLI 登录状态。
- 自动选择 Ark CLI 的 `platform` 后付费 Profile。
- 将模型名称解析为 Model ID，直接调用 Ark 预置推理接入点；无需创建自定义 Endpoint。
- 通过 Ark CLI 的机器接口读取平台凭据，密钥仅用于请求头，不写入报告或日志。
- 执行连通性、Prefix 前缀复用、多轮长上下文三类测试。
- 统计 TTFT、TPOT、E2E、缓存命中率，以及 Mean、P50、P99。
- 输出请求级 CSV、结构化 JSON、Markdown 摘要和 PNG 趋势图。

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

### 命令行使用

完整标准评测：

```bash
bash scripts/run.sh --model "doubao-seed-2-0-mini" --preset standard
```

低成本快速验证：

```bash
bash scripts/run.sh --model "doubao-seed-2-0-mini" --preset quick
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

首次运行会在 Skill 目录创建隔离的 `.venv`，并自动安装 `aiohttp`、`tiktoken` 与 `matplotlib`。

### 评测口径

| 场景 | `standard` 参数 |
|---|---|
| Prefix 前缀复用 | 12,000 前缀 Token + 2,000 后缀 Token，10 个前缀，200 个请求，并发 5 |
| 多轮长上下文 | 7,000 初始 Token + 每轮 256 Token，5 个会话，30 轮，并发 5 |
| 输出设置 | 每请求最多 512 Token，默认关闭推理 |

`standard` 完整流程包含 1 个连通性请求、200 个 Prefix 请求和 150 个多轮请求，共 351 个请求，可能产生较高 Token 用量。`quick` 用于功能验证与低成本预检。

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
│   ├── result.json
│   └── requests.csv
├── prefix_<timestamp>/
│   ├── report.md
│   ├── result.json
│   ├── requests.csv
│   └── report_prefix.png
└── multiturn_<timestamp>/
    ├── report.md
    ├── result.json
    ├── requests.csv
    └── report_multiturn.png
```

### 安全机制

- 凭据通过 `arkcli profile apikey get --plain` 在运行时读取。
- API Key 不通过命令参数传入，不写入报告、CSV 或 JSON。
- HTTP 错误写入报告前会执行密钥脱敏。
- 请勿在调试时启用包含环境变量的 Shell 跟踪。

### 测试

```bash
.venv/bin/python -m unittest discover -s tests -v
```

---

## English

### Overview

Ark Model Efficiency Benchmark is a one-command LLM performance evaluation Skill built on Ark CLI authentication. It invokes Ark preset inference endpoints through the postpaid platform API. A user can specify an Ark model or endpoint in natural language and receive Markdown, JSON, CSV, and PNG reports without manually entering an API key.

### Features

- Verifies Ark CLI authentication.
- Selects an Ark CLI `platform` profile and excludes Agent Plan and Coding Plan profiles.
- Resolves model aliases to Model IDs and invokes Ark preset inference endpoints without creating a custom endpoint.
- Retrieves the platform credential through Ark CLI without persisting it.
- Runs connectivity, prefix-reuse, and multi-turn long-context workloads.
- Measures TTFT, TPOT, E2E latency, cache hit rate, Mean, P50, and P99.
- Produces request-level CSV, structured JSON, Markdown summaries, and PNG charts.

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

### CLI Usage

Run the complete standard benchmark:

```bash
bash scripts/run.sh --model "doubao-seed-2-0-mini" --preset standard
```

Run a low-cost validation:

```bash
bash scripts/run.sh --model "doubao-seed-2-0-mini" --preset quick
```

Run one workload:

```bash
bash scripts/run.sh --model "doubao-seed-2-0-mini" --scenario connectivity
bash scripts/run.sh --model "doubao-seed-2-0-mini" --scenario prefix --preset standard
bash scripts/run.sh --model "doubao-seed-2-0-mini" --scenario multiturn --preset standard
```

The first run creates an isolated `.venv` and installs `aiohttp`, `tiktoken`, and `matplotlib`.

### Standard Workload

| Scenario | `standard` parameters |
|---|---|
| Prefix reuse | 12,000 prefix tokens + 2,000 suffix tokens, 10 prefixes, 200 requests, concurrency 5 |
| Multi-turn context | 7,000 initial tokens + 256 tokens per follow-up, 5 sessions, 30 turns, concurrency 5 |
| Generation | Up to 512 output tokens per request; reasoning disabled by default |

The complete `standard` run sends 351 requests and may consume a substantial number of tokens. Use `quick` for functional validation.

### Metrics

- TTFT: time from request start to the first output token.
- E2E: time from request start to the final response event.
- TPOT: `(E2E - TTFT) / (output tokens - 1)`.
- Cache hit rate: total cached input tokens divided by total input tokens across successful requests.

Cross-model comparisons require the same machine, network, token lengths, concurrency, reasoning settings, and test window.

### Security

- Credentials are retrieved at runtime through `arkcli profile apikey get --plain`.
- API keys are never passed as command-line arguments or stored in generated artifacts.
- HTTP errors are redacted before they are recorded.

### Tests

```bash
.venv/bin/python -m unittest discover -s tests -v
```
