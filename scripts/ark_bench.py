#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import csv
import difflib
import json
import math
import random
import re
import statistics
import subprocess
import sys
import time
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

warnings.filterwarnings(
    "ignore",
    message="urllib3 v2 only supports OpenSSL 1.1.1+",
)

import aiohttp
import matplotlib
import tiktoken

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
CORPUS = (
    "在分布式系统中，服务需要处理并发请求、状态同步、故障恢复和容量规划。"
    "可靠的性能评估应固定输入输出长度、并发度、网络环境和采样参数，并记录长尾延迟。"
    "缓存能够减少重复前缀的计算开销，但命中效果受模型、内容长度、请求顺序和平台策略影响。"
    "工程团队需要基于可复现的数据分析瓶颈，并区分网络耗时、排队耗时和模型生成耗时。"
)


class BenchmarkError(RuntimeError):
    pass


@dataclass
class Target:
    model: str
    profile: str
    profile_type: str
    base_url: str


@dataclass
class RequestMetric:
    scenario: str
    request_id: str
    success: bool
    ttft_ms: float | None = None
    e2e_ms: float | None = None
    tpot_ms: float | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    prefix_index: int | None = None
    reuse_index: int | None = None
    session_index: int | None = None
    turn_index: int | None = None
    output_text: str = ""
    error: str | None = None


def run_json(argv: list[str]) -> dict[str, Any]:
    proc = subprocess.run(argv, text=True, capture_output=True, check=False)
    if proc.returncode:
        message = proc.stderr.strip() or proc.stdout.strip()
        raise BenchmarkError(f"Command failed: {' '.join(argv[:3])}: {message}")
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise BenchmarkError(f"Invalid JSON from {' '.join(argv[:3])}") from exc


def check_auth() -> None:
    status = run_json(["arkcli", "auth", "status", "--format", "json"])
    if not status.get("logged_in"):
        raise BenchmarkError(
            "Ark CLI is not authenticated. Run `arkcli auth login volc-sso` first."
        )


def normalize_model_name(value: str) -> str:
    replacements = {
        "豆包": "doubao",
        "深度求索": "deepseek",
        "迷你": "mini",
        "轻量": "lite",
        "专业": "pro",
        "极速": "flash",
        "正式版": "ga",
        "模型": "",
        "端点": "",
        "接入点": "",
    }
    value = value.lower()
    for source, target in replacements.items():
        value = value.replace(source, target)
    return re.sub(r"[^a-z0-9]+", "", value)


def candidate_score(query: str, candidate: str) -> float:
    q = normalize_model_name(query)
    c = normalize_model_name(candidate)
    if q == c:
        return 1.0
    if q and (q in c or c in q):
        return 0.94 - min(abs(len(q) - len(c)), 20) / 200
    return difflib.SequenceMatcher(None, q, c).ratio()


def discover_candidates(profile_filter: str | None = None) -> list[dict[str, str]]:
    profile_data = run_json(["arkcli", "profile", "list", "--format", "json"])
    profiles = profile_data.get("profiles", [])
    if profile_filter:
        profiles = [item for item in profiles if item.get("name") == profile_filter]
        if not profiles:
            raise BenchmarkError(f"Ark CLI profile not found: {profile_filter}")

    candidates: list[dict[str, str]] = []
    for profile in profiles:
        name = profile["name"]
        try:
            resources = run_json(
                [
                    "arkcli",
                    "resources",
                    "list",
                    "--profile",
                    name,
                    "--modality",
                    "text",
                    "--format",
                    "json",
                ]
            )
        except BenchmarkError:
            continue
        for item in resources.get("items", []):
            resource_id = item.get("id")
            if not resource_id or resource_id == "auto":
                continue
            candidates.append(
                {
                    "id": resource_id,
                    "profile": name,
                    "profile_type": profile.get("type", "unknown"),
                }
            )
    return candidates


def resolve_target(query: str, profile_filter: str | None = None) -> Target:
    candidates = discover_candidates(profile_filter)
    if not candidates:
        raise BenchmarkError(
            "No callable text resources were found in the selected Ark CLI profile(s)."
        )

    ranked = sorted(
        (
            {**candidate, "score": candidate_score(query, candidate["id"])}
            for candidate in candidates
        ),
        key=lambda item: item["score"],
        reverse=True,
    )
    best = ranked[0]
    if best["score"] < 0.45:
        shown = ", ".join(item["id"] for item in ranked[:5])
        raise BenchmarkError(f"Could not resolve model {query!r}. Candidates: {shown}")
    if (
        len(ranked) > 1
        and best["score"] < 0.93
        and best["score"] - ranked[1]["score"] < 0.06
    ):
        payload = {
            "error": "ambiguous_model",
            "query": query,
            "candidates": [
                {
                    "model": item["id"],
                    "profile": item["profile"],
                    "score": round(item["score"], 3),
                }
                for item in ranked[:5]
            ],
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2), file=sys.stderr)
        raise SystemExit(2)

    profile = run_json(
        ["arkcli", "profile", "show", best["profile"], "--format", "json"]
    )
    return Target(
        model=best["id"],
        profile=best["profile"],
        profile_type=best["profile_type"],
        base_url=profile.get("base_url") or DEFAULT_BASE_URL,
    )


def get_api_key(profile: str) -> str:
    proc = subprocess.run(
        [
            "arkcli",
            "profile",
            "apikey",
            "get",
            "--profile",
            profile,
            "--plain",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    key = proc.stdout.strip()
    if proc.returncode or not key:
        raise BenchmarkError(
            f"Ark CLI could not provide a key for profile {profile}. "
            "Refresh authentication with `arkcli auth login volc-sso`."
        )
    return key


def token_text(encoding: Any, token_count: int, salt: str) -> str:
    if token_count <= 0:
        return ""
    source = f"{salt}。{CORPUS}" * (token_count // 40 + 8)
    tokens = encoding.encode(source)
    while len(tokens) < token_count:
        source += CORPUS
        tokens = encoding.encode(source)
    text = encoding.decode(tokens[:token_count])
    while len(encoding.encode(text)) > token_count:
        text = text[:-1]
    return text


def extract_usage(response: dict[str, Any]) -> tuple[int, int, int]:
    usage = response.get("usage") or {}
    input_tokens = int(usage.get("input_tokens") or usage.get("prompt_tokens") or 0)
    output_tokens = int(
        usage.get("output_tokens") or usage.get("completion_tokens") or 0
    )
    details = usage.get("input_tokens_details") or usage.get(
        "prompt_tokens_details"
    ) or {}
    cached_tokens = int(details.get("cached_tokens") or 0)
    return input_tokens, output_tokens, cached_tokens


def extract_output_text(response: dict[str, Any]) -> str:
    parts: list[str] = []
    for item in response.get("output") or []:
        if item.get("type") != "message":
            continue
        for content in item.get("content") or []:
            text = content.get("text")
            if text:
                parts.append(text)
    return "".join(parts)


def is_token_delta(event_type: str, delta: Any) -> bool:
    return (
        event_type.endswith((".delta", "_delta"))
        and isinstance(delta, str)
        and bool(delta)
    )


def is_success_response(response: dict[str, Any]) -> bool:
    status = response.get("status")
    if status in {"completed", None}:
        return True
    details = response.get("incomplete_details") or {}
    return status == "incomplete" and details.get("reason") == "length"


async def stream_request(
    session: aiohttp.ClientSession,
    url: str,
    api_key: str,
    payload: dict[str, Any],
    scenario: str,
    request_id: str,
    **dimensions: int,
) -> RequestMetric:
    first_token_at: float | None = None
    final_response: dict[str, Any] = {}
    output_parts: list[str] = []
    retry_payload = dict(payload)

    for attempt in range(4):
        start = time.perf_counter()
        first_token_at = None
        final_response = {}
        output_parts = []
        try:
            async with session.post(
                url,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=retry_payload,
            ) as response:
                if response.status >= 400:
                    body = await response.text()
                    if (
                        response.status == 400
                        and "thinking" in retry_payload
                        and ("thinking" in body.lower() or "unsupported" in body.lower())
                    ):
                        retry_payload.pop("thinking", None)
                        continue
                    if (
                        response.status == 400
                        and "caching" in retry_payload
                        and "caching" in body.lower()
                    ):
                        retry_payload.pop("caching", None)
                        continue
                    if response.status == 429 or response.status >= 500:
                        if attempt < 3:
                            await asyncio.sleep(2**attempt + random.random())
                            continue
                    raise BenchmarkError(
                        f"HTTP {response.status}: {body[:500].replace(api_key, '<redacted>')}"
                    )

                event_type = ""
                data_lines: list[str] = []
                async for raw_line in response.content:
                    line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
                    if line:
                        if line.startswith("event:"):
                            event_type = line[6:].strip()
                        elif line.startswith("data:"):
                            data_lines.append(line[5:].strip())
                        continue
                    if not data_lines:
                        continue
                    raw_data = "\n".join(data_lines)
                    data_lines = []
                    if raw_data == "[DONE]":
                        continue
                    try:
                        event = json.loads(raw_data)
                    except json.JSONDecodeError:
                        continue
                    current_type = event.get("type") or event_type
                    delta = event.get("delta")
                    if (
                        first_token_at is None
                        and current_type
                        and is_token_delta(current_type, delta)
                    ):
                        first_token_at = time.perf_counter()
                    if current_type == "response.output_text.delta" and isinstance(
                        delta, str
                    ):
                        output_parts.append(delta)
                    if current_type in {
                        "response.completed",
                        "response.incomplete",
                        "response.failed",
                    }:
                        final_response = event.get("response") or {}

                end = time.perf_counter()
                if not final_response:
                    raise BenchmarkError("Streaming response ended without a final event")
                if not is_success_response(final_response):
                    reason = final_response.get("incomplete_details") or final_response.get(
                        "error"
                    )
                    raise BenchmarkError(f"Response status is not completed: {reason}")
                input_tokens, output_tokens, cached_tokens = extract_usage(final_response)
                text = "".join(output_parts) or extract_output_text(final_response)
                if first_token_at is None:
                    first_token_at = end
                ttft_ms = (first_token_at - start) * 1000
                e2e_ms = (end - start) * 1000
                tpot_ms = (
                    (e2e_ms - ttft_ms) / (output_tokens - 1)
                    if output_tokens > 1
                    else None
                )
                return RequestMetric(
                    scenario=scenario,
                    request_id=request_id,
                    success=True,
                    ttft_ms=ttft_ms,
                    e2e_ms=e2e_ms,
                    tpot_ms=tpot_ms,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cached_tokens=cached_tokens,
                    output_text=text,
                    **dimensions,
                )
        except (aiohttp.ClientError, asyncio.TimeoutError, BenchmarkError) as exc:
            if attempt < 3 and not isinstance(exc, BenchmarkError):
                await asyncio.sleep(2**attempt + random.random())
                continue
            return RequestMetric(
                scenario=scenario,
                request_id=request_id,
                success=False,
                error=str(exc).replace(api_key, "<redacted>"),
                **dimensions,
            )

    return RequestMetric(
        scenario=scenario,
        request_id=request_id,
        success=False,
        error="Retry limit exceeded",
        **dimensions,
    )


def make_payload(
    model: str,
    input_data: str | list[dict[str, str]],
    max_output_tokens: int,
    reasoning_effort: str,
    enable_cache: bool = False,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "input": input_data,
        "max_output_tokens": max_output_tokens,
        "stream": True,
        "store": False,
    }
    if reasoning_effort == "none":
        payload["thinking"] = {"type": "disabled"}
    else:
        payload["reasoning"] = {"effort": reasoning_effort}
    if enable_cache:
        payload["caching"] = {"type": "enabled", "prefix": True}
    return payload


async def run_connectivity(
    session: aiohttp.ClientSession,
    target: Target,
    api_key: str,
    max_output_tokens: int,
    reasoning_effort: str,
) -> list[RequestMetric]:
    payload = make_payload(
        target.model,
        "请用一段简短文字说明性能测试中的首字延迟概念。",
        min(max_output_tokens, 64),
        reasoning_effort,
    )
    return [
        await stream_request(
            session,
            f"{target.base_url.rstrip('/')}/responses",
            api_key,
            payload,
            "connectivity",
            "connectivity-1",
        )
    ]


async def run_prefix(
    session: aiohttp.ClientSession,
    target: Target,
    api_key: str,
    args: argparse.Namespace,
    encoding: Any,
) -> list[RequestMetric]:
    prefixes = [
        token_text(encoding, args.prefix_len, f"公共前缀编号 {index + 1}")
        for index in range(args.num_prefixes)
    ]
    semaphore = asyncio.Semaphore(args.max_concurrency)

    async def one(index: int) -> RequestMetric:
        prefix_index = index % args.num_prefixes
        reuse_index = index // args.num_prefixes + 1
        suffix = token_text(
            encoding, args.suffix_len, f"变化后缀编号 {index + 1}"
        )
        prompt = (
            prefixes[prefix_index]
            + suffix
            + "\n请概括以上内容，并给出三项可执行的性能优化建议。"
        )
        payload = make_payload(
            target.model,
            prompt,
            args.max_output_tokens,
            args.reasoning_effort,
            enable_cache=True,
        )
        async with semaphore:
            return await stream_request(
                session,
                f"{target.base_url.rstrip('/')}/responses",
                api_key,
                payload,
                "prefix",
                f"prefix-{index + 1}",
                prefix_index=prefix_index + 1,
                reuse_index=reuse_index,
            )

    return list(await asyncio.gather(*(one(i) for i in range(args.num_requests))))


async def run_multiturn(
    session: aiohttp.ClientSession,
    target: Target,
    api_key: str,
    args: argparse.Namespace,
    encoding: Any,
) -> list[RequestMetric]:
    semaphore = asyncio.Semaphore(args.max_concurrency)

    async def session_run(session_index: int) -> list[RequestMetric]:
        history: list[dict[str, str]] = [
            {
                "role": "user",
                "content": token_text(
                    encoding,
                    args.initial_len,
                    f"长上下文会话编号 {session_index + 1}",
                ),
            }
        ]
        metrics: list[RequestMetric] = []
        for turn_index in range(args.max_turns):
            if turn_index:
                history.append(
                    {
                        "role": "user",
                        "content": token_text(
                            encoding,
                            args.question_len,
                            f"会话 {session_index + 1} 第 {turn_index + 1} 轮追问",
                        )
                        + "\n请结合此前上下文继续分析。",
                    }
                )
            payload = make_payload(
                target.model,
                history,
                args.max_output_tokens,
                args.reasoning_effort,
                enable_cache=True,
            )
            async with semaphore:
                metric = await stream_request(
                    session,
                    f"{target.base_url.rstrip('/')}/responses",
                    api_key,
                    payload,
                    "multiturn",
                    f"session-{session_index + 1}-turn-{turn_index + 1}",
                    session_index=session_index + 1,
                    turn_index=turn_index + 1,
                )
            metrics.append(metric)
            if not metric.success:
                break
            history.append(
                {
                    "role": "assistant",
                    "content": metric.output_text
                    or "已完成本轮分析，并保留前文信息供后续追问。",
                }
            )
        return metrics

    nested = await asyncio.gather(
        *(session_run(i) for i in range(args.num_sessions))
    )
    return [metric for group in nested for metric in group]


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def summarize(metrics: list[RequestMetric]) -> dict[str, Any]:
    successful = [metric for metric in metrics if metric.success]

    def metric_stats(field: str) -> dict[str, float | None]:
        values = [
            float(value)
            for metric in successful
            if (value := getattr(metric, field)) is not None
        ]
        return {
            "mean": statistics.fmean(values) if values else None,
            "p50": percentile(values, 0.50),
            "p99": percentile(values, 0.99),
        }

    total_input = sum(metric.input_tokens for metric in successful)
    total_cached = sum(metric.cached_tokens for metric in successful)
    return {
        "requests": len(metrics),
        "successful_requests": len(successful),
        "failed_requests": len(metrics) - len(successful),
        "input_tokens": total_input,
        "output_tokens": sum(metric.output_tokens for metric in successful),
        "cached_tokens": total_cached,
        "cache_hit_rate": total_cached / total_input if total_input else None,
        "ttft_ms": metric_stats("ttft_ms"),
        "tpot_ms": metric_stats("tpot_ms"),
        "e2e_ms": metric_stats("e2e_ms"),
    }


def fmt_number(value: float | None, digits: int = 2) -> str:
    return "N/A" if value is None else f"{value:.{digits}f}"


def write_csv(path: Path, metrics: list[RequestMetric]) -> None:
    fields = list(asdict(metrics[0]).keys()) if metrics else list(
        RequestMetric.__dataclass_fields__.keys()
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for metric in metrics:
            row = asdict(metric)
            row["output_text"] = ""
            writer.writerow(row)


def write_chart(path: Path, scenario: str, metrics: list[RequestMetric]) -> None:
    successful = [metric for metric in metrics if metric.success]
    if not successful:
        return
    plt.rcParams.update({"font.size": 9, "axes.unicode_minus": False})

    if scenario == "prefix":
        grouped: dict[int, list[RequestMetric]] = {}
        for metric in successful:
            grouped.setdefault(metric.reuse_index or 0, []).append(metric)
        x = sorted(grouped)
        ttft = [statistics.fmean(m.ttft_ms or 0 for m in grouped[i]) for i in x]
        tpot = [
            statistics.fmean(
                m.tpot_ms for m in grouped[i] if m.tpot_ms is not None
            )
            if any(m.tpot_ms is not None for m in grouped[i])
            else 0
            for i in x
        ]
        cache = [
            sum(m.cached_tokens for m in grouped[i])
            / max(sum(m.input_tokens for m in grouped[i]), 1)
            * 100
            for i in x
        ]
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), dpi=180)
        for axis, values, title, ylabel in [
            (axes[0], ttft, "TTFT by prefix reuse", "ms"),
            (axes[1], tpot, "TPOT by prefix reuse", "ms/token"),
            (axes[2], cache, "Cache hit rate", "%"),
        ]:
            axis.plot(x, values, marker="o", color="#2563eb", linewidth=1.5)
            axis.set_title(title)
            axis.set_xlabel("Reuse sequence")
            axis.set_ylabel(ylabel)
            axis.grid(alpha=0.25)
    else:
        grouped = {}
        for metric in successful:
            grouped.setdefault(metric.turn_index or 0, []).append(metric)
        x = sorted(grouped)
        ttft = [statistics.fmean(m.ttft_ms or 0 for m in grouped[i]) for i in x]
        e2e = [statistics.fmean(m.e2e_ms or 0 for m in grouped[i]) for i in x]
        cache = [
            sum(m.cached_tokens for m in grouped[i])
            / max(sum(m.input_tokens for m in grouped[i]), 1)
            * 100
            for i in x
        ]
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), dpi=180)
        for axis, values, title, ylabel in [
            (axes[0], ttft, "TTFT by turn", "ms"),
            (axes[1], e2e, "E2E by turn", "ms"),
            (axes[2], cache, "Cache hit rate by turn", "%"),
        ]:
            axis.plot(x, values, marker="o", color="#0f766e", linewidth=1.5)
            axis.set_title(title)
            axis.set_xlabel("Turn")
            axis.set_ylabel(ylabel)
            axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def write_report(
    directory: Path,
    scenario: str,
    target: Target,
    args: argparse.Namespace,
    metrics: list[RequestMetric],
    started_at: datetime,
    duration: float,
) -> dict[str, Any]:
    directory.mkdir(parents=True, exist_ok=True)
    summary = summarize(metrics)
    metadata = {
        "scenario": scenario,
        "model": target.model,
        "profile": target.profile,
        "profile_type": target.profile_type,
        "base_url": target.base_url,
        "started_at": started_at.astimezone().isoformat(),
        "duration_seconds": duration,
        "preset": args.preset,
        "parameters": {
            key: value
            for key, value in vars(args).items()
            if key
            not in {
                "model",
                "output_dir",
                "profile",
                "scenario",
            }
        },
    }
    result = {
        "metadata": metadata,
        "summary": summary,
        "requests": [
            {key: value for key, value in asdict(metric).items() if key != "output_text"}
            for metric in metrics
        ],
    }
    (directory / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_csv(directory / "requests.csv", metrics)
    if scenario in {"prefix", "multiturn"}:
        write_chart(directory / f"report_{scenario}.png", scenario, metrics)

    lines = [
        f"# Ark LLM 性能评测报告：{scenario}",
        "",
        f"- 测试时间：{metadata['started_at']}",
        f"- 模型：`{target.model}`",
        f"- Ark CLI Profile：`{target.profile}`（{target.profile_type}）",
        f"- API 地址：`{target.base_url}`",
        f"- 预设：`{args.preset}`",
        f"- 耗时：{duration:.2f} 秒",
        "",
        "## 总体结果",
        "",
        "| 指标 | 结果 |",
        "|---|---:|",
        f"| 成功请求 | {summary['successful_requests']} |",
        f"| 失败请求 | {summary['failed_requests']} |",
        f"| 输入 Token | {summary['input_tokens']} |",
        f"| 输出 Token | {summary['output_tokens']} |",
        f"| 缓存命中 Token | {summary['cached_tokens']} |",
        f"| 缓存命中率 | {fmt_number((summary['cache_hit_rate'] or 0) * 100)}% |",
        "",
        "## 延迟分布",
        "",
        "| 指标 | Mean | P50 | P99 |",
        "|---|---:|---:|---:|",
        (
            f"| TTFT (ms) | {fmt_number(summary['ttft_ms']['mean'])} | "
            f"{fmt_number(summary['ttft_ms']['p50'])} | "
            f"{fmt_number(summary['ttft_ms']['p99'])} |"
        ),
        (
            f"| TPOT (ms/token) | {fmt_number(summary['tpot_ms']['mean'])} | "
            f"{fmt_number(summary['tpot_ms']['p50'])} | "
            f"{fmt_number(summary['tpot_ms']['p99'])} |"
        ),
        (
            f"| E2E (ms) | {fmt_number(summary['e2e_ms']['mean'])} | "
            f"{fmt_number(summary['e2e_ms']['p50'])} | "
            f"{fmt_number(summary['e2e_ms']['p99'])} |"
        ),
        "",
        "## 口径说明",
        "",
        "- TTFT：请求发出至首个流式输出 Token 到达的时间。",
        "- TPOT：`(E2E - TTFT) / (输出 Token 数 - 1)`。",
        "- 缓存命中率：所有成功请求的缓存 Token 总数除以输入 Token 总数。",
        "- 结果包含本机网络往返时间；跨模型比较应保持机器、网络、参数和时间窗口一致。",
    ]
    if scenario in {"prefix", "multiturn"}:
        lines.extend(["", "## 图表", "", f"![性能图表](report_{scenario}.png)"])
    errors = [metric.error for metric in metrics if metric.error]
    if errors:
        counts: dict[str, int] = {}
        for error in errors:
            counts[error or "unknown"] = counts.get(error or "unknown", 0) + 1
        lines.extend(["", "## 失败摘要", ""])
        lines.extend(f"- {count} 次：{error}" for error, count in counts.items())
    (directory / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def apply_preset(args: argparse.Namespace) -> None:
    presets = {
        "quick": {
            "prefix_len": 2000,
            "suffix_len": 256,
            "num_prefixes": 2,
            "num_requests": 6,
            "initial_len": 1000,
            "question_len": 128,
            "num_sessions": 2,
            "max_turns": 3,
            "max_concurrency": 2,
            "max_output_tokens": 64,
        },
        "standard": {
            "prefix_len": 12000,
            "suffix_len": 2000,
            "num_prefixes": 10,
            "num_requests": 200,
            "initial_len": 7000,
            "question_len": 256,
            "num_sessions": 5,
            "max_turns": 30,
            "max_concurrency": 5,
            "max_output_tokens": 512,
        },
    }
    for key, value in presets[args.preset].items():
        if getattr(args, key) is None:
            setattr(args, key, value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark an Ark LLM using credentials managed by Ark CLI."
    )
    parser.add_argument("--model", required=True, help="Model name, alias, or endpoint ID")
    parser.add_argument("--profile", help="Restrict resolution to one Ark CLI profile")
    parser.add_argument(
        "--scenario",
        choices=["all", "connectivity", "prefix", "multiturn"],
        default="all",
    )
    parser.add_argument("--preset", choices=["quick", "standard"], default="standard")
    parser.add_argument("--output-dir", default="reports")
    parser.add_argument("--prefix-len", type=int)
    parser.add_argument("--suffix-len", type=int)
    parser.add_argument("--num-prefixes", type=int)
    parser.add_argument("--num-requests", type=int)
    parser.add_argument("--initial-len", type=int)
    parser.add_argument("--question-len", type=int)
    parser.add_argument("--num-sessions", type=int)
    parser.add_argument("--max-turns", type=int)
    parser.add_argument("--max-concurrency", type=int)
    parser.add_argument("--max-output-tokens", type=int)
    parser.add_argument(
        "--reasoning-effort",
        choices=["none", "minimal", "low", "medium", "high"],
        default="none",
    )
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--seed", type=int, default=20260827)
    return parser


async def async_main(args: argparse.Namespace) -> dict[str, Any]:
    random.seed(args.seed)
    check_auth()
    target = resolve_target(args.model, args.profile)
    api_key = get_api_key(target.profile)
    encoding = tiktoken.get_encoding("cl100k_base")
    output_root = Path(args.output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    scenarios = (
        ["connectivity", "prefix", "multiturn"]
        if args.scenario == "all"
        else [args.scenario]
    )
    expected = {
        "connectivity": 1,
        "prefix": args.num_requests,
        "multiturn": args.num_sessions * args.max_turns,
    }
    print(
        json.dumps(
            {
                "event": "benchmark_start",
                "model": target.model,
                "profile": target.profile,
                "profile_type": target.profile_type,
                "scenarios": scenarios,
                "expected_requests": sum(expected[item] for item in scenarios),
                "preset": args.preset,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    timeout = aiohttp.ClientTimeout(total=args.timeout)
    connector = aiohttp.TCPConnector(limit=max(args.max_concurrency * 2, 10))
    report_paths: list[str] = []
    summaries: dict[str, Any] = {}
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        for scenario in scenarios:
            started_at = datetime.now().astimezone()
            started = time.perf_counter()
            if scenario == "connectivity":
                metrics = await run_connectivity(
                    session,
                    target,
                    api_key,
                    args.max_output_tokens,
                    args.reasoning_effort,
                )
            elif scenario == "prefix":
                metrics = await run_prefix(
                    session, target, api_key, args, encoding
                )
            else:
                metrics = await run_multiturn(
                    session, target, api_key, args, encoding
                )
            duration = time.perf_counter() - started
            report_dir = output_root / f"{scenario}_{timestamp}"
            result = write_report(
                report_dir,
                scenario,
                target,
                args,
                metrics,
                started_at,
                duration,
            )
            report_paths.append(str(report_dir / "report.md"))
            summaries[scenario] = result["summary"]
            print(
                json.dumps(
                    {
                        "event": "scenario_complete",
                        "scenario": scenario,
                        "successful": result["summary"]["successful_requests"],
                        "failed": result["summary"]["failed_requests"],
                        "report": str(report_dir / "report.md"),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            if scenario == "connectivity" and not metrics[0].success:
                raise BenchmarkError(
                    f"Connectivity check failed: {metrics[0].error}"
                )

    return {
        "ok": True,
        "model": target.model,
        "profile": target.profile,
        "reports": report_paths,
        "summaries": summaries,
    }


def main() -> int:
    args = build_parser().parse_args()
    apply_preset(args)
    try:
        result = asyncio.run(async_main(args))
    except SystemExit:
        raise
    except (BenchmarkError, KeyboardInterrupt) as exc:
        print(
            json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False),
            file=sys.stderr,
        )
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
