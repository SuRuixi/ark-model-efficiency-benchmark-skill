---
name: "ark-model-benchmark"
description: "Benchmarks Ark LLM latency, throughput, and cache performance without manual API keys. Invoke when users ask to test, benchmark, or compare an Ark model."
---

# Ark Model Benchmark

Run a standardized performance evaluation for an Ark text model or endpoint. The
runner obtains credentials from the authenticated Ark CLI profile and never asks
the user to paste an API key.

## Trigger

Invoke for requests such as:

- 「帮我测试一下 Ark 上某某模型的性能」
- 「压测豆包模型」
- 「测一下某个 Ark Endpoint 的 TTFT、TPOT」
- 「比较 Ark 上两个模型的延迟」

## Preconditions

1. Confirm `arkcli` exists with `command -v arkcli`.
2. Run `arkcli auth status --format json`.
3. If `logged_in` is false, run `arkcli auth login volc-sso` and wait for the
   user to finish browser authentication. Do not request or display an API key.

## Model Resolution

Pass the user's model wording directly to the runner. It searches the text
resources available to every Ark CLI profile and resolves aliases and partial
names. Exact model IDs and `ep-...` endpoint IDs are accepted.

If resolution is ambiguous, the command exits with code 2 and prints ranked
candidates. Ask the user to choose one candidate, then rerun with that exact ID.

## Run

Resolve this skill's directory and execute:

```bash
bash <skill-dir>/scripts/run.sh --model "<user model wording>" --preset standard
```

Defaults:

- `--scenario all`: connectivity, prefix reuse, and multi-turn context.
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
`--reasoning-effort`, and `--profile`.

## Standard Parameters

| Scenario | Parameters |
|---|---|
| Prefix reuse | 12,000 prefix tokens, 2,000 suffix tokens, 10 prefixes, 200 requests, concurrency 5 |
| Multi-turn | 7,000 initial tokens, 256-token follow-ups, 5 sessions, 30 turns, concurrency 5 |
| Generation | Up to 512 output tokens per request; reasoning disabled by default |

These settings can incur substantial model usage. The user explicitly requesting
a performance test is sufficient authorization to run it. State the selected
model, profile, scenario, and request count before starting; do not ask for an
additional confirmation.

## Results

The runner prints a final JSON object containing report paths. Read the generated
`report.md` and summarize:

- successful and failed requests;
- TTFT, TPOT, and E2E mean/P50/P99;
- weighted cache hit rate;
- observed input/output tokens;
- anomalies or failed-request patterns;
- the absolute report directory.

Do not claim model quality, correctness, or cost efficiency from these latency
measurements. Note that cross-model comparisons are valid only under the same
machine, network, parameters, and time window.

## Security

- Retrieve credentials only through
  `arkcli profile apikey get --profile <name> --plain`.
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
metrics and failures in CSV and JSON, then provide a Markdown summary and
scenario chart.
