"""异步请求引擎：OpenAI 兼容 /chat/completions 流式调用 + SSE 解析 + 指标采集。"""
from __future__ import annotations

import asyncio
import json
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, List, Optional

import aiohttp

from metrics import cache_usage_error

MAX_COMPLETION_TOKENS = 512  # 含回答+思维链的总输出上限（prefix-repetition 模式缺省值）
OUTPUT_CAP_FIELDS = ("max_completion_tokens", "max_tokens")
# reasoning_effort 不是通用字段：仅由对应 target 显式配置，缺省不发送。
DEFAULT_REASONING_EFFORT = ""


def requires_enabled_thinking(error: str) -> bool:
    """服务端错误是否明确表示 thinking 不能关闭。"""
    text = error.casefold()
    return (
        ("thinking" in text or "思考" in text)
        and (
            any(phrase in text for phrase in (
                "must be enabled", "must enable", "only supports enabled",
                "only support enabled", "cannot be disabled",
                "does not support disabled", "not support disabled",
                "不支持关闭", "无法关闭", "必须开启", "仅支持开启",
                "只支持 enabled", "仅支持 enabled", "必须为 enabled",
            ))
            or (
                "disabled" in text
                and any(word in text for word in (
                    "invalid", "unsupported", "not allowed", "not supported",
                    "only", "must", "不支持", "不允许", "无效",
                ))
            )
        )
    )


def invalid_reasoning_effort(error: str) -> bool:
    """服务端错误是否明确拒绝 reasoning_effort 的值或字段。"""
    text = error.casefold()
    return (
        "reasoning_effort" in text
        and any(word in text for word in (
            "invalid", "unsupported", "not allowed", "not supported",
            "only", "must be one of", "仅支持", "不支持", "可选值",
        ))
    )


def thinking_error_hint(error: str) -> str:
    """根据服务端错误体生成强制思考或 effort 档位修复提示。"""
    thinking_required = requires_enabled_thinking(error)
    effort_invalid = invalid_reasoning_effort(error)
    hints = []
    if thinking_required:
        hints.append(
            "服务端表明该模型不支持关闭思考：请在对应 targets/*.env 设置 "
            'TARGET_THINKING="enabled"。')
    if effort_invalid:
        hints.append(
            "服务端拒绝 reasoning_effort：请按错误返回的允许档位设置 "
            "TARGET_REASONING_EFFORT；GLM-5.3/GLM-5.3-FLASH 仅支持 low/high/max。")
    elif thinking_required:
        hints.append(
            "若模型要求推理档位，请同时设置 TARGET_REASONING_EFFORT；"
            "GLM-5.3/GLM-5.3-FLASH 可使用 low/high/max。")
    return "\n  ".join(hints)


@dataclass
class RequestResult:
    ok: bool = False
    error: str = ""
    request_id: str = ""          # 服务商响应体中的 request_id/id；缺失时回退到响应头
    provider_log_id: str = ""     # x-request-id / x-tt-logid 等服务商链路标识
    ttft: float = 0.0            # 秒
    e2e: float = 0.0             # 秒
    output_tokens: int = 0       # usage.completion_tokens；缺失时请求失败，不做估算
    prompt_tokens: int = 0
    usage_received: bool = False  # 是否收到非 null 的 usage 对象
    usage_error: str = ""         # usage 缺失/字段非法；该请求不进入任何指标
    done_received: bool = False   # 是否收到 OpenAI SSE 的 [DONE] 结束标记
    stream_error: str = ""        # finish_reason/[DONE] 缺失；响应可能被截断
    cached_tokens: int = 0
    cache_miss_tokens: Optional[int] = None  # DeepSeek 可用于校验 hit + miss = prompt
    has_cache_field: bool = False  # 读到任一方言的缓存命中字段（见 send_chat 解析）
    cache_error: str = ""          # 非法 cache usage；该样本不进入命中率统计
    reasoning_tokens: int = 0      # DeepSeek: completion_tokens_details.reasoning_tokens
    turn: int = 0                # multi-turn 用，prefix-repetition 恒 0
    session_id: int = -1
    completion_text: str = ""    # 聚合的 assistant 全文（multi-turn 追加历史用）
    finish_reason: str = ""      # stop / length / ...；length 说明被 max_completion_tokens 截断
    prefix_id: int = -1
    reuse_n: int = -1           # prefix-repetition 用：第 N 次提交，失败也保持原序号
    throttled: bool = False    # HTTP 429：限流拦截（配额相关），与真实失败区分
    # 流中出现过 reasoning_content（思考开着的铁证）：基线口径是思考关闭，
    # 出现说明该厂商的关闭参数被静默忽略，统计已混入思考时延，必须告警排查
    had_reasoning: bool = False
    # 思考开启时的分相指标（思考关闭时与 ttft/e2e 相等）：
    # ttft_content = 首个 content token（跳过整段 reasoning_content），
    # e2e_content = 最后一个 content token，think_time = ttft_content - ttft
    ttft_content: float = 0.0     # 秒；思考关闭时等于 ttft
    e2e_content: float = 0.0      # 秒；思考关闭时等于 e2e
    content_chunks: int = 0       # content chunk 数（不含 reasoning），TPOT 分母口径


def build_payload(model: str, messages: List[Dict[str, str]],
                  reasoning_effort: str = DEFAULT_REASONING_EFFORT,
                  max_completion_tokens: int = MAX_COMPLETION_TOKENS,
                  output_cap_field: str = "max_completion_tokens",
                  thinking: Optional[str] = "disabled") -> Dict[str, Any]:
    if output_cap_field not in OUTPUT_CAP_FIELDS:
        raise ValueError(
            f"output_cap_field must be one of {OUTPUT_CAP_FIELDS}, "
            f"got {output_cap_field!r}")
    payload = {
        "model": model,
        "messages": messages,
        # max_completion_tokens 限制模型输出总长（回答+思维链），reasoning 模型下
        # 比 max_tokens 更准确；无思维链时两者等价。
        # 部分厂商（如 DeepSeek）不识别 max_completion_tokens 而只认 max_tokens：
        # 不报错但静默忽略，导致输出不被封顶、对比失真——由 output_cap_field 切换
        output_cap_field: max_completion_tokens,
        "stream": True,
        # 命门约束：流式默认不返回 usage，必须显式开启
        "stream_options": {"include_usage": True},
        # 不带 seed：对齐 vLLM benchmark_serve，请求采样用服务端默认随机性，
        # 输出分布贴近真实流量；seed 只用于本地语料采样（见 bench.py）
    }
    if reasoning_effort:
        # 关闭思维链，保证 TTFT/TPOT 口径不含思考阶段
        payload["reasoning_effort"] = reasoning_effort
    if thinking:
        # 当前 targets 已验证兼容的统一思考开关。
        payload["thinking"] = {"type": thinking}
    return payload


def _response_request_id(payload: object) -> str:
    """Extract a provider request/response ID from a JSON object."""
    if not isinstance(payload, dict):
        return ""
    candidates = [payload.get("request_id"), payload.get("id")]
    error = payload.get("error")
    if isinstance(error, dict):
        candidates.extend([error.get("request_id"), error.get("id")])
    for value in candidates:
        if isinstance(value, (str, int)) and not isinstance(value, bool) and str(value):
            return str(value)
    return ""


async def send_chat(session: aiohttp.ClientSession, url: str, headers: Dict[str, str],
                    model: str, messages: List[Dict[str, str]],
                    timeout_sec: float = 600.0,
                    reasoning_effort: str = DEFAULT_REASONING_EFFORT,
                    max_completion_tokens: int = MAX_COMPLETION_TOKENS,
                    output_cap_field: str = "max_completion_tokens",
                    thinking: Optional[str] = "disabled") -> RequestResult:
    """发一次流式 chat 请求并解析 SSE，采集 TTFT / E2E / usage。不重试。"""
    res = RequestResult()
    payload = build_payload(model, messages, reasoning_effort, max_completion_tokens,
                            output_cap_field, thinking)
    t0 = time.perf_counter()
    try:
        async with session.post(url, json=payload, headers=headers,
                                timeout=aiohttp.ClientTimeout(total=timeout_sec)) as resp:
            for header_name in (
                    "x-request-id", "x-tt-logid", "x-requestid",
                    "request-id", "x-trace-id", "trace-id"):
                header_value = resp.headers.get(header_name)
                if header_value:
                    res.provider_log_id = header_value
                    res.request_id = header_value
                    break
            if resp.status >= 400:
                # errors="replace"：错误体不一定是合法 UTF-8，严格解码会把压测炸掉
                raw_body = await resp.text(errors="replace")
                try:
                    body_request_id = _response_request_id(json.loads(raw_body))
                except json.JSONDecodeError:
                    body_request_id = ""
                if body_request_id:
                    res.request_id = body_request_id
                body = raw_body[:1000]
                res.error = f"HTTP {resp.status}: {body}"
                if resp.status == 429:
                    res.throttled = True  # 限流拦截：配额问题，不是服务故障
                return res
            ttft: Optional[float] = None
            ttft_content: Optional[float] = None
            t_last = t0
            t_last_content = t0
            delta_chunks = 0
            content_chunks = 0
            prompt_tokens_seen = False
            completion_tokens_seen = False
            prompt_tokens_error = ""
            completion_tokens_error = ""
            text_parts: List[str] = []
            async for raw_line in resp.content:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line or not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    res.done_received = True
                    # 不更新 t_last：E2E/TPOT 口径到「最后一个输出 token」为止，
                    # 不含最后一个 token 之后到 [DONE] 的服务端收尾时间（对齐 vLLM）
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                response_id = _response_request_id(chunk)
                if response_id:
                    res.request_id = response_id
                t_now = time.perf_counter()
                usage = chunk.get("usage")
                if usage is not None:
                    res.usage_received = True
                    if not isinstance(usage, dict):
                        res.usage_error = (
                            f"invalid usage: expected object, got "
                            f"{type(usage).__name__}")
                        continue
                    # 无论 usage 是独立的收尾 chunk（OpenAI 风格）还是挂在最后一个
                    # 带 choices 的 chunk 上（部分兼容网关），都解析，保证 token 数准确
                    if "prompt_tokens" in usage:
                        prompt_tokens_seen = True
                        value = usage["prompt_tokens"]
                        if (isinstance(value, int) and not isinstance(value, bool)
                                and value >= 0):
                            res.prompt_tokens = value
                            prompt_tokens_error = ""
                        else:
                            prompt_tokens_error = (
                                "usage.prompt_tokens must be a non-negative "
                                f"integer, got {value!r}")
                    if "completion_tokens" in usage:
                        completion_tokens_seen = True
                        value = usage["completion_tokens"]
                        if (isinstance(value, int) and not isinstance(value, bool)
                                and value > 0):
                            res.output_tokens = value
                            completion_tokens_error = ""
                        else:
                            completion_tokens_error = (
                                "usage.completion_tokens must be a positive "
                                f"integer for a non-empty stream, got {value!r}")
                    # 缓存命中字段四种方言：
                    #   OpenAI/GLM: usage.prompt_tokens_details.cached_tokens
                    #   DeepSeek:   usage.prompt_cache_hit_tokens（扁平字段，
                    #               恒等式 prompt_tokens = hit + miss，miss 字段可校验）
                    #   Kimi:      usage.cached_tokens（扁平字段，与 prompt_tokens 平级）
                    #   TokenHub:  usage.cache_read_tokens（扁平字段）
                    details = usage.get("prompt_tokens_details") or {}
                    if "cached_tokens" in details:
                        res.has_cache_field = True
                        res.cached_tokens = details.get("cached_tokens", 0) or 0
                    elif "prompt_cache_hit_tokens" in usage:
                        res.has_cache_field = True
                        res.cached_tokens = usage.get("prompt_cache_hit_tokens", 0) or 0
                        if "prompt_cache_miss_tokens" in usage:
                            res.cache_miss_tokens = usage.get(
                                "prompt_cache_miss_tokens")
                    elif "cached_tokens" in usage:
                        res.has_cache_field = True
                        res.cached_tokens = usage.get("cached_tokens", 0) or 0
                    elif "cache_read_tokens" in usage:
                        res.has_cache_field = True
                        res.cached_tokens = usage.get("cache_read_tokens", 0) or 0
                    # DeepSeek 的思维链 token 数（GLM 文档未保证返回，仅作参考值）
                    cd = usage.get("completion_tokens_details") or {}
                    res.reasoning_tokens = cd.get("reasoning_tokens", 0) or 0
                choices = chunk.get("choices") or []
                if choices:
                    fr = choices[0].get("finish_reason")
                    if fr:
                        res.finish_reason = fr
                    delta = (choices[0].get("delta") or {})
                    # reasoning 模型先流 reasoning_content（content 为空串）；
                    # TTFT 口径 = 第一个输出 token（content 或 reasoning_content 均算）
                    if delta.get("reasoning_content"):
                        res.had_reasoning = True
                    content = delta.get("content") or delta.get("reasoning_content") or ""
                    if content:
                        if ttft is None:
                            ttft = t_now  # 第一个带内容的 chunk，跳过空 role chunk
                        delta_chunks += 1
                        # 历史只回填 content：reasoning_content 按协议不回传
                        #（DeepSeek 明确 400），且思考链不构成下一轮上下文
                        if delta.get("content"):
                            text_parts.append(delta["content"])
                            content_chunks += 1
                            if ttft_content is None:
                                ttft_content = t_now  # 首个 content token（思考段之后）
                            t_last_content = t_now
                        t_last = t_now
            res.completion_text = "".join(text_parts)
            if delta_chunks == 0:
                # 无内容 chunk 的流（如只有 usage）：不是有效时延样本，计为失败，
                # 否则 ttft/e2e=0 会污染统计
                res.error = "empty stream (no content delta)"
                return res
            res.e2e = t_last - t0
            res.ttft = (ttft - t0) if ttft is not None else res.e2e
            # 分相口径：思考关闭时 ttft_content==ttft、e2e_content==e2e（同一 chunk）；
            # 开思考时 content 可能为空（被 max_completion_tokens 截断在思考段），
            # 此时退回 ttft/e2e，避免 0 污染
            if content_chunks > 0:
                res.ttft_content = (ttft_content - t0) if ttft_content is not None else res.ttft
                res.e2e_content = (t_last_content - t0)
            else:
                res.ttft_content = res.ttft
                res.e2e_content = res.e2e
            res.content_chunks = content_chunks

            stream_errors = []
            if not res.finish_reason:
                stream_errors.append("missing finish_reason")
            if not res.done_received:
                stream_errors.append("missing [DONE] terminator")
            if stream_errors:
                res.stream_error = (
                    "incomplete stream: " + "; ".join(stream_errors))

            usage_errors = []
            if not res.usage_received:
                usage_errors.append(
                    "missing usage: stream ended without a usage object "
                    "(stream_options.include_usage=true was requested)")
            else:
                if res.usage_error:
                    usage_errors.append(res.usage_error)
                if not prompt_tokens_seen:
                    usage_errors.append("missing usage.prompt_tokens")
                elif prompt_tokens_error:
                    usage_errors.append(prompt_tokens_error)
                if not completion_tokens_seen:
                    usage_errors.append("missing usage.completion_tokens")
                elif completion_tokens_error:
                    usage_errors.append(completion_tokens_error)
            errors = []
            if res.stream_error:
                errors.append(res.stream_error)
            if usage_errors:
                res.usage_error = "; ".join(usage_errors)
                errors.append(res.usage_error)
            if errors:
                res.error = "; ".join(errors)
                return res
            if res.has_cache_field:
                res.cache_error = cache_usage_error(
                    res.prompt_tokens, res.cached_tokens,
                    res.cache_miss_tokens)
            res.ok = True
            return res
    except asyncio.TimeoutError:
        res.error = "timeout"
        return res
    except (aiohttp.ClientError, OSError) as e:
        res.error = f"connection error: {e}"
        return res


class EarlyAbortError(Exception):
    """fail-fast 早停：前若干请求全部因非 429 的 HTTP 4xx 失败且零成功（参数/
    配置错误），继续跑只会让整轮请求逐个撞同一堵墙。message 已是可直接打印
    的中文提示（含定位与修改建议）。"""


class FailFast:
    """非 429 的 HTTP 4xx 零成功监测：完成的请求数达阈值且无任何成功即触发。

    定向场景：强制思考模型（GLM-5.3/5.3-FLASH 等）拒绝 thinking=disabled、
    key 无效、模型 ID 写错、方言参数不被接受--这类配置性错误逐请求重放没有
    意义（失败不重试是既定口径），跑完 200 个请求只浪费配额和时间。
    429（配额限流）与连接错误不触发：它们是运行环境问题，不是配置问题。"""

    THRESHOLD = 3  # 前 3 个完成的请求全是 4xx（且零成功）即认定配置性失败

    def __init__(self, label: str, engine: "Engine") -> None:
        self.label = label
        self.engine = engine
        self.fatal: List[RequestResult] = []   # 非 429 的 4xx 失败
        self.succeeded = 0

    def record(self, r: RequestResult) -> None:
        if r.ok:
            self.succeeded += 1
        elif not r.throttled and r.error.startswith("HTTP 4"):
            self.fatal.append(r)

    @property
    def triggered(self) -> bool:
        return self.succeeded == 0 and len(self.fatal) >= self.THRESHOLD

    def error(self) -> EarlyAbortError:
        """构造带修改指引的早停异常（仅在 triggered 时调用）。"""
        lines = [f"[fail-fast] 目标「{self.label}」前 {len(self.fatal)} 个完成的请求"
                 f"全部为 4xx 失败且零成功，提前终止本次压测（不再空跑剩余请求）",
                 f"  首个错误: {self.fatal[0].error[:200]}"]
        hint = thinking_error_hint(self.fatal[0].error)
        if hint:
            lines.append(f"  参数提示: {hint}")
        lines.append(
            "  其他常见原因: key / 模型 ID / 方言参数不被该厂商接受。先跑 "
            "python bench.py connectivity 自检逐项排查后再压测")
        return EarlyAbortError("\n".join(lines))


class Engine:
    """带并发信号量的请求引擎，prefix-repetition / multi-turn 共用。"""

    def __init__(self, base_url: str, api_key: str, timeout_sec: float = 600.0,
                 reasoning_effort: str = DEFAULT_REASONING_EFFORT,
                 max_completion_tokens: int = MAX_COMPLETION_TOKENS,
                 output_cap_field: str = "max_completion_tokens",
                 thinking: Optional[str] = "disabled") -> None:
        # 兼容不带/带 /chat/completions 的 base_url
        base = base_url.rstrip("/")
        self.chat_url = base + ("/chat/completions" if not base.endswith("/chat/completions") else "")
        self.headers = {"Authorization": f"Bearer {api_key}"}
        self.timeout_sec = timeout_sec
        self.reasoning_effort = reasoning_effort
        self.max_completion_tokens = max_completion_tokens
        # 输出封顶参数字段名：OpenAI/Ark 用 max_completion_tokens，
        # DeepSeek 等只认 max_tokens 的厂商需切换，否则输出不被封顶、对比失真
        if output_cap_field not in OUTPUT_CAP_FIELDS:
            raise ValueError(
                f"output_cap_field must be one of {OUTPUT_CAP_FIELDS}, "
                f"got {output_cap_field!r}")
        self.output_cap_field = output_cap_field
        # DeepSeek 风格思考开关（"enabled"/"disabled"，None=不发送）；
        # 与 reasoning_effort 是不同厂商的等价物
        self.thinking = thinking
        self._inflight = 0           # 当前在途请求数（实测峰值并发用）
        self.peak_concurrency = 0

    def client(self, max_concurrency: int = 1) -> aiohttp.ClientSession:
        """keep-alive 连接池：连接复用，避免每个请求重复 TCP/TLS 握手
        （握手耗时会计入 TTFT/E2E，系统性抬高时延）。"""
        return aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(limit=max_concurrency))

    @asynccontextmanager
    async def inflight(self) -> AsyncIterator[None]:
        """登记一个在途请求，用于实测峰值并发。"""
        self._inflight += 1
        self.peak_concurrency = max(self.peak_concurrency, self._inflight)
        try:
            yield
        finally:
            self._inflight -= 1

    async def run_requests(self, model: str,
                           message_lists: List[List[Dict[str, str]]],
                           max_concurrency: int,
                           tags: Optional[List[Dict[str, int]]] = None,
                           label: Optional[str] = None) -> List[RequestResult]:
        """并发跑一批独立请求（prefix-repetition 模式；multi-turn 单轮也复用）。

        全批次共享一个 keep-alive 会话：第一波并发照常建连（保持冷启动口径），
        后续请求复用连接，不再重复握手。超配额吃到的 429 会被标记 throttled，
        由上层计为失败（对齐 vLLM 口径），时延统计仅含成功请求。
        fail-fast：前 3 个完成的请求全是非 429 的 4xx 且零成功时抛
        EarlyAbortError（配置性失败，不再空跑），并取消未发的请求。"""
        sem = asyncio.Semaphore(max_concurrency)
        ff = FailFast(label or model, self)

        async def one(i: int, messages: List[Dict[str, str]], sess: aiohttp.ClientSession) -> RequestResult:
            if ff.triggered:  # 已判定配置性失败：排队中的请求不再发出
                raise ff.error()
            async with sem:
                async with self.inflight():
                    r = await send_chat(sess, self.chat_url, self.headers, model,
                                        messages, self.timeout_sec,
                                        self.reasoning_effort,
                                        self.max_completion_tokens,
                                        self.output_cap_field,
                                        self.thinking)
            if tags:
                r.prefix_id = tags[i].get("prefix_id", -1)
                r.reuse_n = tags[i].get("reuse_n", -1)
                r.session_id = tags[i].get("session_id", -1)
                r.turn = tags[i].get("turn", 0)
            ff.record(r)
            if ff.triggered:
                raise ff.error()
            return r

        async with self.client(max_concurrency) as sess:
            tasks = [asyncio.create_task(one(i, m, sess))
                     for i, m in enumerate(message_lists)]
            try:
                return list(await asyncio.gather(*tasks))
            except EarlyAbortError:
                for t in tasks:
                    t.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                raise
