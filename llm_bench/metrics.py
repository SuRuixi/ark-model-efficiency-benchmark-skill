"""TTFT / TPOT / E2E / Cache 命中率统计与分位数。"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

QUANTILES = (("P50", 50), ("P99", 99))


def cache_usage_error(prompt_tokens: object, cached_tokens: object,
                      cache_miss_tokens: object = None) -> str:
    """Return a reason when cache token counters cannot form a valid ratio."""
    for name, value in (
            ("prompt_tokens", prompt_tokens),
            ("cached_tokens", cached_tokens)):
        if isinstance(value, bool) or not isinstance(value, int):
            return f"{name} must be an integer, got {value!r}"
        if value < 0:
            return f"{name} must be >= 0, got {value}"
    if prompt_tokens == 0:
        return "prompt_tokens must be > 0 when a cache field is present"
    if cached_tokens > prompt_tokens:
        return (f"cached_tokens ({cached_tokens}) exceeds "
                f"prompt_tokens ({prompt_tokens})")
    if cache_miss_tokens is not None:
        if (isinstance(cache_miss_tokens, bool)
                or not isinstance(cache_miss_tokens, int)):
            return (f"cache_miss_tokens must be an integer, got "
                    f"{cache_miss_tokens!r}")
        if cache_miss_tokens < 0:
            return f"cache_miss_tokens must be >= 0, got {cache_miss_tokens}"
        if cached_tokens + cache_miss_tokens != prompt_tokens:
            return (
                f"cached_tokens + cache_miss_tokens "
                f"({cached_tokens + cache_miss_tokens}) does not equal "
                f"prompt_tokens ({prompt_tokens})"
            )
    return ""


def percentile_stats(values: Sequence[float]) -> Dict[str, float]:
    """返回 AVG / P50 / P99；空序列返回全 NaN。"""
    if not values:
        return {"AVG": float("nan"), **{name: float("nan") for name, _ in QUANTILES}}
    arr = np.asarray(list(values), dtype=float)
    out = {"AVG": float(arr.mean())}
    for name, q in QUANTILES:
        out[name] = float(np.percentile(arr, q))
    return out


def fmt_ms(v: float) -> str:
    if v != v:  # NaN
        return "-"
    return f"{v * 1000:.1f}"

def fmt_ratio(v: float) -> str:
    if v != v:
        return "-"
    return f"{v * 100:.2f}%"


class RequestMetrics:
    """聚合单个模式所有成功请求的指标。"""

    def __init__(self) -> None:
        self.ttft: List[float] = []
        self.tpot: List[float] = []
        self.e2e: List[float] = []
        # 思考开启时的分相统计（关闭时 think_time 全为 0，与 ttft_content==ttft
        # 一致）；截断在思考段的请求不进这两组（见 add_success）
        self.ttft_content: List[float] = []
        self.think_time: List[float] = []
        self.cached_sum: int = 0          # 加权口径分子分母（仅统计读到字段的请求）
        self.prompt_cache_sum: int = 0
        self.output_tokens: List[int] = []
        self.prompt_tokens: List[int] = []
        self.total = 0
        self.failed = 0
        # 明确请求了 stream_options.include_usage 后仍未收到完整 token usage。
        # 这些请求按失败处理，不能用 SSE chunk 数伪造 Token/TPOT。
        self.usage_error_count = 0
        # 缺少 finish_reason 或 [DONE] 的响应可能被截断，不进入任何时延指标。
        self.stream_error_count = 0
        self.invalid_cache_count = 0
        # 返回了合法 Cache token 字段的成功请求数：
        # 加权命中率的分母只含这些请求，与「Total input tokens」（全部成功请求）
        # 口径不同；非法字段另计 invalid_cache_count，不进入分子/分母
        self.cache_field_count = 0
        # 输出被 max_completion_tokens 截断的请求数（finish_reason=length）：
        # 截断会让 TPOT/E2E 偏短，某家偏高即疑似隐藏输出上限
        self.truncated = 0

    def add_success(self, ttft: float, e2e: float, output_tokens: int,
                    prompt_tokens: int, cached_tokens: int, has_cache_field: bool,
                    ttft_content: float = 0.0, e2e_content: float = 0.0,
                    content_chunks: int = 0, had_reasoning: bool = False,
                    finish_reason: str = "", cache_error: str = "") -> None:
        self.total += 1
        self.ttft.append(ttft)
        self.e2e.append(e2e)
        self.output_tokens.append(output_tokens)
        self.prompt_tokens.append(prompt_tokens)
        # 分相统计（思考关闭时 think_time 全为 0，与 ttft_content==ttft 一致）。
        # 截断在思考段（content_chunks==0）的请求不含 content 相，
        # 回退值会低估 Think Time，直接不进分相统计
        if content_chunks > 0:
            self.ttft_content.append(ttft_content)
            self.think_time.append(max(ttft_content - ttft, 0.0))
        # TPOT 对齐 vLLM 口径：全部输出 token（含思维链）的平均解码间隔。
        # 思考/回答 token 解码机制相同、速度一致，无需分相；分相只对 TTFT 有意义
        if output_tokens > 1:
            self.tpot.append((e2e - ttft) / (output_tokens - 1))
        if finish_reason == "length":
            self.truncated += 1
        if has_cache_field and not cache_error:
            cache_error = cache_usage_error(prompt_tokens, cached_tokens)
        if cache_error:
            self.invalid_cache_count += 1
        elif has_cache_field and prompt_tokens > 0:
            self.cache_field_count += 1
            self.cached_sum += cached_tokens
            self.prompt_cache_sum += prompt_tokens

    def add_failure(self, usage_error: str = "",
                    stream_error: str = "") -> None:
        self.total += 1
        self.failed += 1
        if usage_error:
            self.usage_error_count += 1
        if stream_error:
            self.stream_error_count += 1

    @property
    def weighted_cache_hit(self) -> float:
        """加权命中率 = Σcached_tokens / Σprompt_tokens（token 口径，主口径）。"""
        if self.prompt_cache_sum == 0:
            return float("nan")
        return self.cached_sum / self.prompt_cache_sum
