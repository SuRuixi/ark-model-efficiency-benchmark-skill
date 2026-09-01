---
name: "ark-model-benchmark"
description: "Benchmarks Ark LLM latency, throughput, and cache performance without manual API keys. Invoke when users ask to test, benchmark, or compare an Ark model."
---

# Ark Model Benchmark

Run a standardized performance evaluation through Ark's postpaid API and preset
inference endpoints. The runner uses a `platform` Ark CLI profile, obtains its
credential automatically, and delegates measurement to the bundled `llm_bench`
engine. It never asks the user to paste an API key.

## Trigger

Invoke for requests such as:

- 「帮我测试一下 Ark 上某某模型的性能」
- 「压测豆包模型」
- 「测一下某个 Ark Endpoint 的 TTFT、TPOT」
- 「比较 Ark 上两个模型的延迟」

## Mode Selection

Before running a benchmark, determine whether the user explicitly selected a
mode. Supported choices are:

| Mode | Purpose | Standard request count |
|---|---|---:|
| `connectivity` | Send one request to validate model access and inspect basic TTFT, E2E, usage, cache fields, and reasoning state | 1 |
| `prefix` | Measure fixed-prefix reuse, TTFT, TPOT, E2E, and cache hit behavior across independent requests | 200 |
| `multiturn` | Measure latency and cache behavior as conversation history grows across turns | 200 |
| `all` | Run connectivity, prefix, and multi-turn sequentially | 401 |

If the user has not explicitly selected a mode, ask this question before any
Ark CLI or benchmark command:

```text
请选择测试模式：连通性检查（1 个请求）、Prefix 前缀复用（200 个请求）、
多轮长上下文（200 个请求），或完整测试（401 个请求）？
```

Do not infer `all` from a generic request such as 「测试模型性能」. If the user
already names a mode, execute it directly without asking again. Treat
「前缀缓存」 as `prefix`, 「长上下文」 or 「多轮」 as `multiturn`,
「连通性」 as `connectivity`, and 「完整」「全部」 as `all`.

Multi-model comparison is an option within `prefix` or `multiturn`; it is not a
separate benchmark mode.

## Fast Invocation

After the mode is selected, run the benchmark wrapper immediately:

```bash
bash <skill-dir>/scripts/run.sh \
  --model "<user model wording>" \
  --scenario "<selected-mode>" \
  --preset standard
```

Do not perform separate model searches, profile listing, API-key lookup, or
connectivity probes before this command. The wrapper performs these operations
once and reuses their results. Redundant Ark CLI discovery adds several seconds
to startup.

If the wrapper reports that Ark CLI is missing, install it:

```bash
npm install -g @volcengine/ark-cli@latest
```

If the wrapper reports that authentication is required, run:

1. Give the user this internal Ark account URL and ask them to sign in:

```text
https://babi.bytedance.net/finance/basic/volcManage/?fullscreen=true&volc_account_category=1&tab=my&status=1
```

2. After the user confirms that the internal account login is complete, initiate
   Ark CLI SSO authentication:

```bash
arkcli auth login volc-sso
```

3. Wait for browser authentication to finish and verify:

```bash
arkcli auth status --format json
```

4. When `logged_in=true`, rerun the original wrapper command with the previously
   selected model and mode. Do not ask the user to select them again.

Never request an API key from the user. Do not start the benchmark before both
the internal account login and Ark CLI SSO authentication are complete.

## Ark CLI Call Sequence

The runner uses the following fixed sequence. These commands are documented here
for auditability; the Agent should call the wrapper instead of repeating them.

1. Read authentication state and the available Profile summaries:

```bash
arkcli auth status --format json
```

2. Select the default Profile whose `type` is `platform`. Ignore
   `agent-plan`, `coding-plan`, and other subscription Profiles.

3. For a natural-language model name, search the Ark text-model catalog:

```bash
arkcli models search "<normalized-model-name>" \
  --profile "<profile-name>" \
  --modality text \
  --size 30 \
  --format json
```

Resolve the user's wording against `name`, `display_name`, and
`primary_version`, then construct the callable Model ID. A complete Model ID or
`ep-...` Endpoint ID bypasses catalog search.

4. Fetch the platform API base URL and credential concurrently:

```bash
arkcli profile show "<profile-name>" --format json
arkcli profile apikey get --profile "<profile-name>" --plain
```

Ark CLI `1.0.22` supports `--plain`. For versions where that flag is unavailable,
the runner automatically retries with:

```bash
arkcli profile apikey get --profile "<profile-name>" --format json
```

It then reads the top-level `api_key` field. Capture either output directly in
process memory. Do not print it. Use it only as the `Authorization` header for
benchmark requests.

5. Export the resolved values only to the child-process environment:

```bash
LLM_BENCH_BASE_URL="<base-url>"
LLM_BENCH_API_KEY="<in-memory-key>"
```

6. Invoke `llm_bench/bench.py`, which calls
   `<base_url>/chat/completions` with streaming enabled and
   `stream_options.include_usage=true`. Ark automatically routes a Model ID to
   its preset inference endpoint and charges by consumed tokens through the
   postpaid platform account. Do not call `arkcli +deploy`, do not create a
   custom Endpoint, and do not use an Agent Plan credential.

The wrapper maps user-facing modes to the bundled engine:

- `prefix` -> `prefix-repetition`;
- `multiturn` -> `multi-turn`;
- `connectivity` -> `connectivity`.

The engine defaults to `thinking.type=disabled` and omits
`reasoning_effort`. Use `--thinking enabled` and an explicit
`--reasoning-effort` only for models that require thinking.

## Model Resolution

Pass the user's model wording directly to `--model`. The runner resolves aliases
and partial names from the Ark model catalog. Normal requests use the resolved
Model ID and Ark's preset inference endpoint.

If resolution is ambiguous, the command exits with code 2 and prints ranked
candidates. Ask the user to choose one candidate, then rerun with that exact ID.

## Run

Defaults:

- The Skill must always pass the user-selected `--scenario`; the CLI fallback is
  `all` for direct terminal use.
- `--preset standard`: document-aligned parameters.
- `--output-dir ./reports`: reports are written under the current workspace.

For an explicit quick check, use:

```bash
bash <skill-dir>/scripts/run.sh --model "<model>" --preset quick
```

For one scenario:

```bash
bash <skill-dir>/scripts/run.sh --model "<model>" --scenario connectivity
bash <skill-dir>/scripts/run.sh --model "<model>" --scenario prefix --preset standard
bash <skill-dir>/scripts/run.sh --model "<model>" --scenario multiturn --preset standard
```

Supported overrides include `--prefix-len`, `--suffix-len`, `--num-prefixes`,
`--num-requests`, `--initial-len`, `--question-len`, `--num-sessions`,
`--max-turns`, `--max-concurrency`, `--max-output-tokens`,
`--reasoning-effort`, `--thinking`, `--seed`, and `--profile`. An explicitly
supplied Profile must have `type=platform`.

## Standard Parameters

| Scenario | Parameters |
|---|---|
| Prefix reuse | 12,000 prefix tokens, 2,000 suffix tokens, 10 prefixes, 200 requests, concurrency 5 |
| Multi-turn | 3,000 initial tokens, 256-token follow-ups, 10 sessions, 20 turns, concurrency 5 |
| Generation | Prefix up to 512 output tokens; multi-turn up to 1,024; reasoning disabled |

These settings can incur substantial model usage. The user explicitly requesting
a performance test is sufficient authorization to run it. State the selected
Model ID, postpaid platform Profile, scenario, and request count before starting;
do not ask for an additional confirmation.

The complete `standard` run sends 401 requests: one connectivity request, 200
prefix requests, and 200 multi-turn generations.

## Results

The runner prints a final JSON object containing report paths. Read the generated
`report.md` and summarize:

- successful and failed requests;
- TTFT, TPOT, and E2E mean/P50/P99;
- weighted cache hit rate;
- observed input/output tokens;
- anomalies or failed-request patterns;
- the absolute report directory.

Reports are produced by the bundled `llm_bench` implementation:

- `report_prefix-repetition.md` / `report_multi-turn.md`;
- `result_prefix-repetition.json` / `result_multi-turn.json`;
- latency and cache PNG charts;
- `run.log` with the redacted console transcript.

Do not claim model quality, correctness, or cost efficiency from these latency
measurements. Note that cross-model comparisons are valid only under the same
machine, network, parameters, and time window.

## Security

- Retrieve credentials only through
  `arkcli profile apikey get --profile <name> --plain`, with
  `--format json` as the compatibility fallback.
- Never print, persist, summarize, or send API keys to another process except as
  the HTTP `Authorization` header.
- Generated artifacts must not contain credentials.
- Do not use shell tracing (`set -x`) around credential retrieval.

## Benchmark Methodology

Measure all candidates on the same machine and network, with identical input
lengths, output limits, concurrency, reasoning settings, and test windows.
Network round trips and proxies are part of TTFT and E2E, so environment drift
can invalidate a comparison.

Use two workload shapes:

- Prefix reuse: select a fixed prefix from a bounded pool, append a changing
  suffix, and repeat requests to measure cache reuse and its effect on TTFT.
- Multi-turn context: begin with one long input, append a fixed-size follow-up
  and the preceding assistant response on each turn, and measure how latency and
  cache behavior change as the conversation grows.

Compute metrics as follows:

- TTFT = first output token arrival time - request start time.
- E2E = final response arrival time - request start time.
- TPOT = `(E2E - TTFT) / (output tokens - 1)`.
- Cache hit rate = `sum(cached input tokens) / sum(input tokens)`.
- Aggregate TTFT, TPOT, and E2E with Mean, P50, and P99.
- Calculate cache hit rate as a token-weighted ratio, not an average of
  per-request percentages.

Record the exact model or endpoint, profile, API base URL, reasoning effort,
maximum output tokens, requested and observed token lengths, concurrency,
request count, random seed, test time, and duration. Preserve request-level
metrics and failures in JSON, then provide a Markdown summary and scenario
charts.
