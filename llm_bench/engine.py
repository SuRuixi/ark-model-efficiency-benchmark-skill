"""异步请求引擎：OpenAI 兼容 /chat/completions 流式调用 + SSE 解析 + 指标采集。"""
from __future__ import annotations

import asyncio
import json
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, List, Optional

import aiohttp

MAX_COMPLETION_TOKENS = 512  # 含回答+思维链的总输出上限（prefix 模式缺省值，--max-completion-tokens 可覆盖）
# Ark 等服务商的思考深度参数；none = 关闭思维链（压测时延口径必须无思考）
DEFAULT_REASONING_EFFORT = "none"


@dataclass
class RequestResult:
    ok: bool = False
    error: str = ""
    ttft: float = 0.0            # 秒
    e2e: float = 0.0             # 秒
    output_tokens: int = 0       # 优先 usage.completion_tokens，否则 delta chunk 数
    prompt_tokens: int = 0
    cached_tokens: int = 0
    has_cache_field: bool = False  # usage.prompt_tokens_details.cached_tokens 是否存在
    turn: int = 0                # multiturn 用，prefix 恒 0
    session_id: int = -1
    completion_text: str = ""    # 聚合的 assistant 全文（multiturn 追加历史用）
    finish_reason: str = ""      # stop / length / ...；length 说明被 max_completion_tokens 截断
    prefix_id: int = -1
    throttled: bool = False    # HTTP 429：限流拦截（配额相关），与真实失败区分
    # 流中出现过 reasoning_content（思考开着的铁证）：基线口径是思考关闭，
    # 出现说明该厂商的关闭参数被静默忽略，统计已混入思考时延，必须告警排查
    had_reasoning: bool = False


def build_payload(model: str, messages: List[Dict[str, str]],
                  reasoning_effort: str = DEFAULT_REASONING_EFFORT,
                  max_completion_tokens: int = MAX_COMPLETION_TOKENS,
                  output_cap_field: str = "max_completion_tokens",
                  thinking: Optional[str] = None,
                  enable_thinking: Optional[bool] = None) -> Dict[str, Any]:
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
        # DeepSeek 等厂商的思考开关：{"thinking": {"type": "enabled/disabled"}}
        # 与 reasoning_effort 是不同厂商的等价物，按目标模型各自发送
        payload["thinking"] = {"type": thinking}
    if enable_thinking is not None:
        # 阿里 DashScope 风格思考开关：body 顶层布尔 enable_thinking（HTTP 直调
        # 与 model/messages 平级）。DeepSeek-V4 系在阿里侧默认开思考，对齐
        # reasoning 关闭口径须显式发 enable_thinking=false
        payload["enable_thinking"] = enable_thinking
    return payload


async def send_chat(session: aiohttp.ClientSession, url: str, headers: Dict[str, str],
                    model: str, messages: List[Dict[str, str]],
                    timeout_sec: float = 600.0,
                    reasoning_effort: str = DEFAULT_REASONING_EFFORT,
                    max_completion_tokens: int = MAX_COMPLETION_TOKENS,
                    output_cap_field: str = "max_completion_tokens",
                    thinking: Optional[str] = None,
                    enable_thinking: Optional[bool] = None) -> RequestResult:
    """发一次流式 chat 请求并解析 SSE，采集 TTFT / E2E / usage。不重试。"""
    res = RequestResult()
    payload = build_payload(model, messages, reasoning_effort, max_completion_tokens,
                            output_cap_field, thinking, enable_thinking)
    t0 = time.perf_counter()
    try:
        async with session.post(url, json=payload, headers=headers,
                                timeout=aiohttp.ClientTimeout(total=timeout_sec)) as resp:
            if resp.status >= 400:
                # errors="replace"：错误体不一定是合法 UTF-8，严格解码会把压测炸掉
                body = (await resp.text(errors="replace"))[:300]
                res.error = f"HTTP {resp.status}: {body}"
                if resp.status == 429:
                    res.throttled = True  # 限流拦截：配额问题，不是服务故障
                return res
            ttft: Optional[float] = None
            t_last = t0
            delta_chunks = 0
            text_parts: List[str] = []
            async for raw_line in resp.content:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line or not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    # 不更新 t_last：E2E/TPOT 口径到「最后一个输出 token」为止，
                    # 不含最后一个 token 之后到 [DONE] 的服务端收尾时间（对齐 vLLM）
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                t_now = time.perf_counter()
                usage = chunk.get("usage")
                if usage:
                    # 无论 usage 是独立的收尾 chunk（OpenAI 风格）还是挂在最后一个
                    # 带 choices 的 chunk 上（部分兼容网关），都解析，保证 token 数准确
                    res.prompt_tokens = usage.get("prompt_tokens", 0) or 0
                    res.output_tokens = usage.get("completion_tokens", 0) or res.output_tokens
                    details = usage.get("prompt_tokens_details") or {}
                    if "cached_tokens" in details:
                        res.has_cache_field = True
                        res.cached_tokens = details.get("cached_tokens", 0) or 0
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
                        t_last = t_now
            res.completion_text = "".join(text_parts)
            if delta_chunks == 0:
                # 无内容 chunk 的流（如只有 usage）：不是有效时延样本，计为失败，
                # 否则 ttft/e2e=0 会污染统计
                res.error = "empty stream (no content delta)"
                return res
            res.e2e = t_last - t0
            res.ttft = (ttft - t0) if ttft is not None else res.e2e
            if res.output_tokens == 0:
                res.output_tokens = delta_chunks  # 降级：用 delta chunk 数估算
            res.ok = True
            return res
    except asyncio.TimeoutError:
        res.error = "timeout"
        return res
    except (aiohttp.ClientError, OSError) as e:
        res.error = f"connection error: {e}"
        return res


class Engine:
    """带并发信号量的请求引擎，prefix / multiturn 共用。"""

    def __init__(self, base_url: str, api_key: str, timeout_sec: float = 600.0,
                 reasoning_effort: str = DEFAULT_REASONING_EFFORT,
                 max_completion_tokens: int = MAX_COMPLETION_TOKENS,
                 output_cap_field: str = "max_completion_tokens",
                 thinking: Optional[str] = None,
                 enable_thinking: Optional[bool] = None) -> None:
        # 兼容不带/带 /chat/completions 的 base_url
        base = base_url.rstrip("/")
        self.chat_url = base + ("/chat/completions" if not base.endswith("/chat/completions") else "")
        self.headers = {"Authorization": f"Bearer {api_key}"}
        self.timeout_sec = timeout_sec
        self.reasoning_effort = reasoning_effort
        self.max_completion_tokens = max_completion_tokens
        # 输出封顶参数字段名：OpenAI/Ark 用 max_completion_tokens，
        # DeepSeek 等只认 max_tokens 的厂商需切换，否则输出不被封顶、对比失真
        self.output_cap_field = output_cap_field
        # DeepSeek 风格思考开关（"enabled"/"disabled"，None=不发送）；
        # 与 reasoning_effort 是不同厂商的等价物
        self.thinking = thinking
        # 阿里 DashScope 风格思考开关（True/False，None=不发送）；
        # 与 reasoning_effort / thinking 是不同厂商的等价物
        self.enable_thinking = enable_thinking
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
                           tags: Optional[List[Dict[str, int]]] = None) -> List[RequestResult]:
        """并发跑一批独立请求（prefix 模式；multiturn 单轮也复用）。

        全批次共享一个 keep-alive 会话：第一波并发照常建连（保持冷启动口径），
        后续请求复用连接，不再重复握手。超配额吃到的 429 会被标记 throttled，
        由上层计为失败（对齐 vLLM 口径），时延统计仅含成功请求。"""
        sem = asyncio.Semaphore(max_concurrency)

        async def one(i: int, messages: List[Dict[str, str]], sess: aiohttp.ClientSession) -> RequestResult:
            async with sem:
                async with self.inflight():
                    r = await send_chat(sess, self.chat_url, self.headers, model,
                                        messages, self.timeout_sec,
                                        self.reasoning_effort,
                                        self.max_completion_tokens,
                                        self.output_cap_field,
                                        self.thinking,
                                        self.enable_thinking)
            if tags:
                r.prefix_id = tags[i].get("prefix_id", -1)
                r.session_id = tags[i].get("session_id", -1)
                r.turn = tags[i].get("turn", 0)
            return r

        async with self.client(max_concurrency) as sess:
            return list(await asyncio.gather(
                *(one(i, m, sess) for i, m in enumerate(message_lists))))
