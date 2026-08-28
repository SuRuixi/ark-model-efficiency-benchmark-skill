"""TTFT / TPOT / E2E / Cache 命中率统计与分位数。"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

QUANTILES = (("P50", 50), ("P99", 99))


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
        self.cached_sum: int = 0          # 加权口径分子分母（仅统计读到字段的请求）
        self.prompt_cache_sum: int = 0
        self.output_tokens: List[int] = []
        self.prompt_tokens: List[int] = []
        self.total = 0
        self.failed = 0
        # 返回了 usage.prompt_tokens_details.cached_tokens 字段的成功请求数：
        # 加权命中率的分母只含这些请求，与「Total input tokens」（全部成功请求）
        # 口径不同，报告需标注覆盖率供读者换算
        self.cache_field_count = 0

    def add_success(self, ttft: float, e2e: float, output_tokens: int,
                    prompt_tokens: int, cached_tokens: int, has_cache_field: bool) -> None:
        self.total += 1
        self.ttft.append(ttft)
        self.e2e.append(e2e)
        self.output_tokens.append(output_tokens)
        self.prompt_tokens.append(prompt_tokens)
        if output_tokens > 1:
            self.tpot.append((e2e - ttft) / (output_tokens - 1))
        if has_cache_field and prompt_tokens > 0:
            self.cache_field_count += 1
            self.cached_sum += cached_tokens
            self.prompt_cache_sum += prompt_tokens

    def add_failure(self) -> None:
        self.total += 1
        self.failed += 1

    @property
    def weighted_cache_hit(self) -> float:
        """加权命中率 = Σcached_tokens / Σprompt_tokens（token 口径，主口径）。"""
        if self.prompt_cache_sum == 0:
            return float("nan")
        return self.cached_sum / self.prompt_cache_sum
