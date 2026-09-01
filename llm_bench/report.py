"""压测报告：vLLM benchmark_serving 风格总览。

呈现层全部收口在本文件：bench.py 只负责发请求拿 RequestResult，
总览的构建由这里完成。只保留 Serving Benchmark Result 基准块
（TTFT / TPOT / 端到端时延）。
"""
from __future__ import annotations

import datetime
import hashlib
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from metrics import (RequestMetrics, cache_usage_error, fmt_ms, fmt_ratio,
                     percentile_stats)

try:
    import matplotlib
    matplotlib.use("Agg")  # 无显示环境也能落盘
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

# macOS / Windows / Linux 常见中文字体依次回退
_CJK_FONTS = ["PingFang SC", "Hiragino Sans GB", "Arial Unicode MS",
              "Microsoft YaHei", "Noto Sans CJK SC", "sans-serif"]


def _kv(label: str, value: str) -> str:
    return f"{label:<40}{value}"


def _stat_block(name: str, values_s: Sequence[float]) -> List[str]:
    """Mean / P50 / P99 三行（输入秒，输出毫秒）。"""
    st = percentile_stats(values_s)
    return [
        _kv(f"Mean {name} (ms):", f"{st['AVG'] * 1000:.2f}"),
        _kv(f"P50 {name} (ms):", f"{st['P50'] * 1000:.2f}"),
        _kv(f"P99 {name} (ms):", f"{st['P99'] * 1000:.2f}"),
    ]


_NOTICE_PARAM_KEYS = {"⚠️ 重试提示"}


def _mode_title(mode: str) -> str:
    return {
        "prefix-repetition": "Prefix Repetition",
        "multi-turn": "Multi-turn",
    }.get(mode, mode)


def _workload_summary(mode: str, params: Dict[str, str]) -> str:
    if mode == "prefix-repetition":
        prefixes = int(params.get("num_prefixes", "0") or 0)
        requests = int(params.get("num_requests", "0") or 0)
        if prefixes and requests % prefixes == 0:
            scale = f"{prefixes} 个前缀 × 每个 {requests // prefixes} 个请求"
        else:
            scale = f"{prefixes} 个前缀 · 共 {requests} 个请求"
        return (
            f"{scale} · 前缀 {params.get('prefix_len（语料净长度）', '-')} tokens · "
            f"后缀 {params.get('suffix_len（语料净长度）', '-')} tokens"
        )
    if mode == "multi-turn":
        return (
            f"{params.get('num_conversations', '-')} sessions × "
            f"{params.get('max_turns', '-')} turns · "
            f"首轮 {params.get('initial_len（语料净长度）', '-')} tokens · "
            f"每轮追问 {params.get('question_len（语料净长度）', '-')} tokens"
        )
    return "见完整配置"


def _execution_summary(mode: str, params: Dict[str, str],
                       output_limits: Sequence[int]) -> str:
    unit = "sessions" if mode == "multi-turn" else "requests"
    limits = sorted(set(v for v in output_limits if v > 0))
    output = (f"输出上限 {limits[0]} tokens" if len(limits) == 1
              else "输出上限按目标配置")
    return (f"并发 {params.get('max_concurrency', '-')} {unit} · {output}")


def _thinking_markdown(value: str) -> str:
    if not value or value == "未发送思考参数":
        return "未发送"
    return " · ".join(f"`{part.strip()}`" for part in value.split(";") if part.strip())


def _full_config_lines(items: Sequence[Tuple[str, str]]) -> List[str]:
    lines = [
        "<details>",
        "<summary>完整配置</summary>",
        "",
        "| 参数 | 值 |",
        "| --- | --- |",
    ]
    seen = set()
    for key, raw_value in items:
        if key in seen or key in _NOTICE_PARAM_KEYS:
            continue
        seen.add(key)
        value = str(raw_value).replace("|", "\\|").replace("\n", "<br>")
        lines.append(f"| `{key}` | {value} |")
    lines.extend(["", "</details>", ""])
    return lines


def _notice_lines(params: Dict[str, str]) -> List[str]:
    lines = []
    for key in _NOTICE_PARAM_KEYS:
        if params.get(key):
            lines.extend([f"> **{key}** {params[key]}", ""])
    return lines


def _planned_requests(mode: str, params: Dict[str, str]) -> Optional[int]:
    """Return the configured logical request count when it is well-defined."""
    if mode != "multi-turn":
        return None
    try:
        sessions = int(params.get("num_conversations", "0") or 0)
        turns = int(params.get("max_turns", "0") or 0)
    except (TypeError, ValueError):
        return None
    if sessions <= 0 or turns <= 0:
        return None
    return sessions * turns


def serving_summary_lines(m: RequestMetrics, duration_sec: float,
                          peak_concurrency: int,
                          planned_requests: Optional[int] = None) -> List[str]:
    """vLLM benchmark_serving 风格的总体结果块。"""
    prompt_tokens = [v for v in m.prompt_tokens if v > 0]
    prompt_summary = (
        f"{sum(prompt_tokens) / len(prompt_tokens):.1f} / "
        f"{min(prompt_tokens)} / {max(prompt_tokens)}"
        if prompt_tokens else "-"
    )
    lines = ["============ Serving Benchmark Result ============"]
    if planned_requests is not None:
        lines.extend([
            _kv("Planned requests:", str(planned_requests)),
            _kv("Attempted requests:", str(m.total)),
            _kv("Skipped (session terminated):",
                str(max(planned_requests - m.total, 0))),
        ])
    lines.extend([
        _kv("Successful requests:", str(m.total - m.failed)),
        _kv("Failed requests:", str(m.failed)),
        _kv("Missing/invalid usage:", str(m.usage_error_count)),
        _kv("Incomplete streams:", str(m.stream_error_count)),
        _kv("Benchmark duration (s):", f"{duration_sec:.2f}"),
        _kv("Total input tokens:", str(sum(m.prompt_tokens))),
        # 服务端 usage 返回的最终 prompt token 数，包含固定指令、历史消息和协议模板；
        # 与 CLI 的语料净长度分开披露，避免把切片长度误认为最终请求长度。
        _kv("Input tokens/request (avg/min/max):", prompt_summary),
        _kv("Total generated tokens:", str(sum(m.output_tokens))),
        _kv("Truncated (finish=length):", str(m.truncated)),
        # 命中率口径披露：加权命中率分母仅含返回 cache 字段的请求，
        # 与 Total input tokens（全部成功请求）不同，部分缺失时不能直接互换算
        _kv("Valid cache samples:", f"{m.cache_field_count}/{m.total - m.failed}"
                                    f" requests (hit-rate denominator)"),
        _kv("Invalid cache samples:", str(m.invalid_cache_count)),
        _kv("Peak concurrent requests:", f"{peak_concurrency:d}"),
        "---------------Time to First Token----------------",
    ])
    lines += _stat_block("TTFT", m.ttft)
    # 思考开启时的分相口径：TTFT(content) = 首个 content token（思考段之后），
    # Think Time = TTFT(content) - TTFT（思维链解码耗时）。思考关闭时全为 0，不显示
    if m.think_time and max(m.think_time) > 0:
        lines.append("--------TTFT (first content token, post-thinking)--------")
        lines += _stat_block("TTFT (content)", m.ttft_content)
        lines.append("----------------Thinking Time---------------------")
        lines += _stat_block("Think Time", m.think_time)
    lines.append("-----Time per Output Token (excl. 1st token)------")
    lines += _stat_block("TPOT", m.tpot)
    lines.append("---------------End-to-End Latency-----------------")
    lines += _stat_block("E2E", m.e2e)
    lines.append("==================================================")
    return lines


class Report:
    """一次压测的完整报告数据，可渲染为控制台输出与 Markdown 文档。"""

    def __init__(self, mode: str, model: str, params: Dict[str, str],
                 start_time: Optional[datetime.datetime] = None,
                 label: str = "", thinking: str = "") -> None:
        self.mode = mode
        self.model = model
        self.label = label or model
        self.thinking = thinking
        self.params = params
        self.start_time = start_time or datetime.datetime.now()
        self.elapsed_sec: float = 0.0
        self.peak_concurrency: int = 0   # 实测峰值并发
        self.metrics: Optional[RequestMetrics] = None
        self.images: List[Tuple[str, str]] = []    # (图题, 相对 md 的图片路径)
        # 口径告警（如思考未关闭导致 E2E 混入思维链）：渲染进 Markdown，
        # 保证只看报告的人也能看到口径问题
        self.warnings: List[str] = []

    # ---------------------------------------------------------------- console

    def render_console(self) -> None:
        if self.metrics:
            print()
            for line in serving_summary_lines(self.metrics, self.elapsed_sec,
                                              self.peak_concurrency,
                                              _planned_requests(
                                                  self.mode, self.params)):
                print(line)
        for title, img in self.images:
            print(f"[chart] {title}: {img}")

    # ---------------------------------------------------------------- markdown

    def render_markdown(self) -> str:
        m = self.metrics
        warnings = list(self.warnings)
        if m and m.usage_error_count:
            warnings.append(
                f"{m.usage_error_count} 个请求未返回完整 usage，已按失败处理；"
                "未使用 SSE chunk 数估算 Token，且未计入 TTFT/TPOT/E2E。")
        if m and m.stream_error_count:
            warnings.append(
                f"{m.stream_error_count} 个请求缺少 finish_reason 或 [DONE]，"
                "已作为不完整流按失败处理，未计入 TTFT/TPOT/E2E。")
        lines: List[str] = []
        lines.append(f"# {_mode_title(self.mode)} 压测报告")
        lines.append("")
        lines.append(f"**目标**　{self.label} · `{self.model}`")
        lines.append("")
        lines.append(f"**思考参数**　{_thinking_markdown(self.thinking)}")
        lines.append("")
        lines.append(f"**工作负载**　{_workload_summary(self.mode, self.params)}")
        lines.append("")
        lines.append(
            f"**执行设置**　{_execution_summary(self.mode, self.params, [int(self.params.get('max_completion_tokens', 0) or 0)])}")
        lines.append("")
        lines.append(f"**时间**　{self.start_time.strftime('%Y-%m-%d %H:%M:%S')}"
                     f" · 耗时 {self.elapsed_sec:.1f}s")
        lines.append("")
        lines.extend(_notice_lines(self.params))
        if warnings:
            lines.append("## ⚠️ 口径告警")
            lines.append("")
            lines.append("```")
            lines.extend(warnings)
            lines.append("```")
            lines.append("")
        lines.append("## 总体结果")
        lines.append("")
        if m:
            lines.append("```")
            lines.extend(serving_summary_lines(m, self.elapsed_sec,
                                               self.peak_concurrency,
                                               _planned_requests(
                                                   self.mode, self.params)))
            lines.append("```")
            lines.append("")
        if self.images:
            for title, img in self.images:
                lines.append(f"## {title}")
                lines.append("")
                lines.append(f"![{title}]({img})")
                lines.append("")
        lines.extend(_full_config_lines([
            ("模式", self.mode),
            ("服务商", self.label),
            ("模型", self.model),
            ("思考参数", self.thinking or "未发送"),
            *self.params.items(),
        ]))
        return "\n".join(lines)

    def write_markdown(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.render_markdown())
        print(f"\n[report] Markdown 报告已写入 {path}")


# 报告与结果 JSON 统一输出目录（相对 llm-bench 运行目录）
REPORT_DIR = "reports"


class ComparisonReport:
    """多模型同步对比压测的报告：每模型一个 Serving Benchmark 块 + 汇总对比表。

    场景：客户在同一时间窗、同一份样本数据下并发压测我方与友商模型，
    报告可直接用于逐模型对照（时延/吞吐/命中率口径完全一致）。"""

    def __init__(self, mode: str, params: Dict[str, str],
                 start_time: Optional[datetime.datetime] = None) -> None:
        self.mode = mode
        self.params = params
        self.start_time = start_time or datetime.datetime.now()
        # label, model, metrics, elapsed, peak, thinking, base_url, output limit
        self.entries: List[
            Tuple[str, str, "RequestMetrics", float, int, str, str, int]
        ] = []
        self.images: List[Tuple[str, str]] = []
        # 口径告警（如思考未关闭导致 E2E 混入思维链）：渲染进 Markdown，
        # 保证只看报告的人也能看到口径问题
        self.warnings: List[str] = []

    def add(self, label: str, model: str, m: "RequestMetrics",
            elapsed: float, peak_concurrency: int, thinking: str = "",
            base_url: str = "", max_completion_tokens: int = 0) -> None:
        self.entries.append((label, model, m, elapsed, peak_concurrency,
                             thinking, base_url, max_completion_tokens))

    # ---------------------------------------------------------------- console

    def render_console(self) -> None:
        # 精简层级：总览表置顶 + 每模型一行关键指标，完整大块只进 Markdown，
        # 多模型对比时控制台不用翻几十行重复的 vLLM 块
        print("\n== 多模型对比总览 ==")
        for line in self.compare_table_lines():
            print(line)
        print()
        for label, model, m, elapsed, peak, thinking, base_url, output_limit in self.entries:
            ttft = percentile_stats(m.ttft)
            tpot = percentile_stats(m.tpot)
            e2e = percentile_stats(m.e2e)
            print(f"{label}: ok={m.total - m.failed} fail={m.failed}"
                  f" usage异常={m.usage_error_count}"
                  f" 流异常={m.stream_error_count}"
                  f"  TTFT {fmt_ms(ttft['AVG'])}/{fmt_ms(ttft['P99'])}ms"
                  f"  TPOT {fmt_ms(tpot['AVG'])}ms  E2E {fmt_ms(e2e['AVG'])}ms"
                  f"  hit {fmt_ratio(m.weighted_cache_hit)}")
        for title, img in self.images:
            print(f"[chart] {title}: {img}")
        print("\n（各模型完整指标块见 Markdown 报告）")

    # ---------------------------------------------------------------- markdown

    def compare_table_lines(self) -> List[str]:
        """关键指标对比表（Markdown 表格行）。"""
        head = ["服务商", "成功/失败", "Usage异常", "流异常",
                "TTFT avg (ms)", "TTFT P99 (ms)", "TPOT avg (ms)",
                "E2E avg (ms)", "加权Cache命中率"]
        rows = []
        for label, model, m, elapsed, peak, thinking, base_url, output_limit in self.entries:
            ttft = percentile_stats(m.ttft)
            tpot = percentile_stats(m.tpot)
            e2e = percentile_stats(m.e2e)
            rows.append([
                f"{label}", f"{m.total - m.failed}/{m.failed}",
                str(m.usage_error_count), str(m.stream_error_count),
                f"{ttft['AVG'] * 1000:.1f}", f"{ttft['P99'] * 1000:.1f}",
                f"{tpot['AVG'] * 1000:.1f}", f"{e2e['AVG'] * 1000:.1f}",
                fmt_ratio(m.weighted_cache_hit),
            ])
        lines = ["| " + " | ".join(head) + " |",
                 "|" + "---|" * len(head)]
        for r in rows:
            lines.append("| " + " | ".join(r) + " |")
        return lines

    def target_table_lines(self) -> List[str]:
        lines = [
            "| 服务商 | 模型 | 思考参数 |",
            "| --- | --- | --- |",
        ]
        for label, model, m, elapsed, peak, thinking, base_url, output_limit in self.entries:
            lines.append(
                f"| {label} | `{model}` | {_thinking_markdown(thinking)} |")
        return lines

    def render_markdown(self) -> str:
        warnings = list(self.warnings)
        warnings.extend(
            f"{label}: {m.usage_error_count} 个请求未返回完整 usage，"
            "已按失败处理；未使用 SSE chunk 数估算 Token，且未计入 "
            "TTFT/TPOT/E2E。"
            for label, model, m, elapsed, peak, thinking, base_url, output_limit
            in self.entries
            if m.usage_error_count
        )
        warnings.extend(
            f"{label}: {m.stream_error_count} 个请求缺少 finish_reason 或 "
            "[DONE]，已作为不完整流按失败处理，未计入 TTFT/TPOT/E2E。"
            for label, model, m, elapsed, peak, thinking, base_url, output_limit
            in self.entries
            if m.stream_error_count
        )
        lines: List[str] = []
        lines.append(f"# {_mode_title(self.mode)} 多服务商对比报告")
        lines.append("")
        lines.extend(self.target_table_lines())
        lines.append("")
        lines.append(f"**工作负载**　{_workload_summary(self.mode, self.params)}")
        lines.append("")
        lines.append(f"**执行设置**　{_execution_summary(self.mode, self.params, [entry[7] for entry in self.entries])}")
        lines.append("")
        lines.append(f"**时间**　{self.start_time.strftime('%Y-%m-%d %H:%M:%S')}"
                     " · 同步发起，各自独立计时")
        lines.append("")
        lines.extend(_notice_lines(self.params))
        if warnings:
            lines.append("## ⚠️ 口径告警")
            lines.append("")
            lines.append("```")
            lines.extend(warnings)
            lines.append("```")
            lines.append("")
        lines.append("## 多模型对比总览")
        lines.append("")
        lines.extend(self.compare_table_lines())
        lines.append("")
        for label, model, m, elapsed, peak, thinking, base_url, output_limit in self.entries:
            lines.append(f"## {label}")
            lines.append("")
            lines.append(f"<sub>模型 ID: `{model}`</sub>")
            lines.append("")
            lines.append("```")
            lines.extend(serving_summary_lines(
                m, elapsed, peak, _planned_requests(self.mode, self.params)))
            lines.append("```")
            lines.append("")
        if self.images:
            for title, img in self.images:
                lines.append(f"## {title}")
                lines.append("")
                lines.append(f"![{title}]({img})")
                lines.append("")
        config: List[Tuple[str, str]] = [("模式", self.mode)]
        for label, model, m, elapsed, peak, thinking, base_url, output_limit in self.entries:
            config.extend([
                (f"{label}.模型", model),
                (f"{label}.API 地址", base_url),
                (f"{label}.思考参数", thinking or "未发送"),
                (f"{label}.max_completion_tokens", str(output_limit)),
            ])
        config.extend(
            (key, value) for key, value in self.params.items()
            if key not in {"API 地址", "max_completion_tokens", "各目标思考口径"})
        lines.extend(_full_config_lines(config))
        return "\n".join(lines)

    def write_markdown(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.render_markdown())
        print(f"\n[report] Markdown 对比报告已写入 {path}")


def _path_slug(value: str, max_chars: int) -> str:
    """生成短且可读的路径片段；超长值保留前缀并附哈希避免重名。"""
    chars = []
    previous_dash = False
    for char in value.strip():
        if char.isalnum() or char in "._-":
            chars.append(char)
            previous_dash = False
        elif not previous_dash:
            chars.append("-")
            previous_dash = True
    slug = "".join(chars).strip("._-") or "unknown"
    if len(slug) <= max_chars:
        return slug
    digest = hashlib.sha1(slug.encode("utf-8")).hexdigest()[:6]
    return f"{slug[:max_chars - 7]}-{digest}"


def _target_path_tag(targets: Sequence[Tuple[str, str]]) -> str:
    """服务商+模型的紧凑标识；多目标以 vs 连接并限制目录组件长度。"""
    parts = []
    for label, model in targets:
        label_slug = _path_slug(label, 16)
        model_slug = _path_slug(model, 32)
        # kimi + kimi-k2.6 这类模型名已包含服务商时避免重复。
        if (model_slug.lower() == label_slug.lower()
                or model_slug.lower().startswith(label_slug.lower() + "-")):
            parts.append(model_slug)
        else:
            parts.append(f"{label_slug}-{model_slug}")
    tag = "_vs_".join(parts)
    if len(tag.encode("utf-8")) <= 160:
        return tag
    digest = hashlib.sha1(tag.encode("utf-8")).hexdigest()[:8]
    prefix = tag.encode("utf-8")[:147]
    while True:
        try:
            return f"{prefix.decode('utf-8')}-{digest}"
        except UnicodeDecodeError:
            prefix = prefix[:-1]


def default_report_path(
        mode: str, start_time: datetime.datetime,
        targets: Optional[Sequence[Tuple[str, str]]] = None) -> str:
    """默认报告路径包含模式、目标和时间；同秒并发运行自动加序号消歧。"""
    parts = [mode]
    if targets:
        parts.append(_target_path_tag(targets))
    parts.append(start_time.strftime("%Y%m%d_%H%M%S"))
    base = os.path.join(REPORT_DIR, "_".join(parts))
    suffix = 0
    while True:
        run_dir = base if suffix == 0 else f"{base}_{suffix}"
        try:
            # 排他创建：即使多个进程同一时刻启动，也不会复用同一目录。
            os.makedirs(run_dir)
            return os.path.join(run_dir, f"report_{mode}.md")
        except FileExistsError:
            suffix += 1


def default_output_path(report_path: str, mode: str) -> str:
    """结果 JSON 默认与报告同目录（即本次运行的独立目录）。"""
    return os.path.join(os.path.dirname(report_path) or ".", f"result_{mode}.json")


# ---------------------------------------------------------------- 图像生成


def _ensure_mpl() -> bool:
    """matplotlib 可用则初始化字体并返回 True。"""
    if not HAS_MPL:
        print("[report] ⚠️ 未安装 matplotlib，跳过图像生成（pip install matplotlib）")
        return False
    matplotlib.rcParams["font.sans-serif"] = _CJK_FONTS
    matplotlib.rcParams["axes.unicode_minus"] = False
    return True


def multi_turn_charts(results: Sequence, report_path: str,
                      label: str = "", model: str = "") -> List[Tuple[str, str]]:
    """multi-turn 图像（单模型）：2×2 面板（TTFT / 归一化 TTFT / TPOT / Cache 命中率），
    每面板 AVG 实线 + P5–P95 阴影带。PNG 与 md 同目录同名前缀。"""
    if not _ensure_mpl():
        return []
    base = _report_base(report_path)
    by_turn = _group_by_turn(results)
    if not by_turn:
        return []
    turns = sorted(by_turn)
    p = f"{base}_turns.png"
    _plot_turn_fig([(label or model, "tab:green", turns, by_turn,
                     _failure_turns(results))], p, [(label, model)])
    return [("按轮次延迟与命中率（TTFT / 归一化 TTFT / TPOT / Cache 命中率）",
             os.path.basename(p))]


def prefix_charts(results: Sequence, report_path: str,
                  label: str = "", model: str = "") -> List[Tuple[str, str]]:
    """prefix-repetition 图像（单模型）：1×3 面板（TTFT / 归一化 TTFT / TPOT），
    横轴为同一前缀的第 N 次请求（第 1 次=首次尝试），
    每面板 AVG 实线 + P5–P95 阴影带。PNG 与 md 同目录同名前缀。"""
    if not _ensure_mpl():
        return []
    base = _report_base(report_path)
    grouped = _group_by_reuse(results)
    if not grouped or not any(grouped[1].values()):
        return []
    reuse_n, by_reuse, attempts = grouped
    p = f"{base}_reuse.png"
    _plot_reuse_fig(
        [(label or model, "tab:green", reuse_n, by_reuse, attempts)],
        p, [(label, model)])
    return [("按同一前缀请求次数的延迟（TTFT / 归一化 TTFT / TPOT）",
             os.path.basename(p))]


# ---------------------------------------------------------------- 多模型对比

# 对比图的模型配色（循环使用）；首选色与单模型图一致
SERIES_COLORS = ["tab:green", "tab:blue", "tab:orange", "tab:red",
                 "tab:purple", "tab:brown", "tab:pink", "tab:gray"]


def _report_base(report_path: str) -> str:
    base = report_path[:-len(".md")] if report_path.endswith(".md") else report_path
    os.makedirs(os.path.dirname(base) or ".", exist_ok=True)
    return base


def _group_by_turn(results: Sequence) -> Dict[int, List]:
    by_turn: Dict[int, List] = {}
    for r in results:
        if r.ok:
            by_turn.setdefault(r.turn, []).append(r)
    return by_turn


def _failure_turns(results: Sequence) -> Dict[int, int]:
    """每轮失败请求数（turn -> 次数），用于图上标出会话终止位置。

    失败会话从下一轮起退出统计，均值曲线可能出现台阶；在图上标出失败轮，
    让台阶回落可自解释，避免误读为「上下文缩短」。"""
    fail: Dict[int, int] = {}
    for r in results:
        if not r.ok:
            fail[r.turn] = fail.get(r.turn, 0) + 1
    return fail


def _group_by_reuse(
        results: Sequence
) -> Optional[Tuple[List[int], Dict[int, List], Dict[int, int]]]:
    """按固定 reuse_n 分组，失败请求占据原序号但不进入时延统计。

    返回 (reuse_n 列表, reuse_n -> 成功请求, reuse_n -> 尝试请求数)。
    兼容没有 reuse_n 的旧结果，降级为按提交顺序为各 prefix 编号。
    """
    by_reuse: Dict[int, List] = {}
    attempts: Dict[int, int] = {}
    fallback_seen: Dict[int, int] = {}
    for r in results:
        if r.prefix_id < 0:
            continue
        fallback_seen[r.prefix_id] = fallback_seen.get(r.prefix_id, 0) + 1
        n = r.reuse_n if getattr(r, "reuse_n", -1) > 0 else fallback_seen[r.prefix_id]
        attempts[n] = attempts.get(n, 0) + 1
        by_reuse.setdefault(n, [])
        if r.ok:
            by_reuse[n].append(r)
    if not attempts:
        return None
    return sorted(attempts), by_reuse, attempts


def prefix_charts_compare(series: Sequence[Tuple[str, str, Sequence]],
                          report_path: str) -> List[Tuple[str, str]]:
    """prefix-repetition 多模型对比：一张叠加对比图（所有模型同面板同色区分）。
    series: List[(label, model, results)]。"""
    if not _ensure_mpl():
        return []
    base = _report_base(report_path)
    prepared = []
    title_targets = []
    for i, (label, model, results) in enumerate(series):
        grouped = _group_by_reuse(results)
        if grouped and any(grouped[1].values()):
            prepared.append((label, SERIES_COLORS[i % len(SERIES_COLORS)],
                             *grouped))
            title_targets.append((label, model))
    images: List[Tuple[str, str]] = []
    if prepared:
        p = f"{base}_compare_reuse.png"
        _plot_reuse_fig(prepared, p, title_targets)
        images.append(("多模型对比：按同一前缀请求次数的延迟"
                       "（TTFT / 归一化 TTFT / TPOT）",
                       os.path.basename(p)))
    return images


def multi_turn_charts_compare(series: Sequence[Tuple[str, str, Sequence]],
                              report_path: str) -> List[Tuple[str, str]]:
    """multi-turn 多模型对比：一张叠加对比图。"""
    if not _ensure_mpl():
        return []
    base = _report_base(report_path)
    prepared = []
    title_targets = []
    for i, (label, model, results) in enumerate(series):
        by_turn = _group_by_turn(results)
        if by_turn:
            prepared.append((label, SERIES_COLORS[i % len(SERIES_COLORS)],
                             sorted(by_turn), by_turn,
                             _failure_turns(results)))
            title_targets.append((label, model))
    images: List[Tuple[str, str]] = []
    if prepared:
        p = f"{base}_compare_turns.png"
        _plot_turn_fig(prepared, p, title_targets)
        images.append(("多模型对比：按轮次延迟与命中率（TTFT / 归一化 TTFT / TPOT / Cache 命中率）",
                       os.path.basename(p)))
    return images


def weighted_hit(rs: Sequence) -> float:
    """Σcached / Σprompt；与总体汇总一致地排除所有非法 Cache 样本。"""
    with_field = [
        r for r in rs
        if r.has_cache_field
        and not getattr(r, "cache_error", "")
        and not cache_usage_error(
            r.prompt_tokens,
            r.cached_tokens,
            getattr(r, "cache_miss_tokens", None),
        )
    ]
    if not with_field:
        return float("nan")
    return (sum(r.cached_tokens for r in with_field) /
            sum(r.prompt_tokens for r in with_field))


def _turn_band(per_turn_values: Sequence[Sequence[float]]
               ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """逐轮 (AVG, P5, P95) 三条曲线；无样本的轮次为 NaN（图中断开）。"""
    avg = np.full(len(per_turn_values), np.nan)
    p5 = np.full(len(per_turn_values), np.nan)
    p95 = np.full(len(per_turn_values), np.nan)
    for i, vals in enumerate(per_turn_values):
        if vals:
            arr = np.asarray(list(vals), dtype=float)
            avg[i] = arr.mean()
            p5[i], p95[i] = np.percentile(arr, [5, 95])
    return avg, p5, p95


def _int_ticks(ax) -> None:
    """横轴为轮次/序号，强制整数刻度。"""
    from matplotlib.ticker import MaxNLocator
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))


def _plot_band(ax, turns: Sequence[int], avg: np.ndarray, p5: np.ndarray,
               p95: np.ndarray, color: str, label: str) -> None:
    """AVG 实线 + P5–P95 阴影带。"""
    ax.fill_between(turns, p5, p95, color=color, alpha=0.2, linewidth=0)
    ax.plot(turns, avg, "-", marker="o", markersize=3, color=color,
            linewidth=1.5, label=label)


def _avg_box(ax, text: str) -> None:
    """右上角总体均值标注框。"""
    ax.text(0.98, 0.97, text, transform=ax.transAxes, ha="right", va="top",
            fontsize=9, bbox=dict(boxstyle="round,pad=0.35", fc="white",
                                  ec="0.6", alpha=0.9))


def _target_title_lines(targets: Sequence[Tuple[str, str]],
                        max_chars: int = 92) -> List[str]:
    """将“服务商 · 模型”按目标边界换行，避免长 ID 挤压图表标题。"""
    items = []
    for label, model in targets:
        if label and model and label != model:
            items.append(f"{label} · {model}")
        else:
            items.append(label or model or "未知目标")
    lines: List[str] = []
    current = ""
    for item in items:
        candidate = f"{current}    |    {item}" if current else item
        if current and len(candidate) > max_chars:
            lines.append(current)
            current = item
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def _chart_header(fig, title: str, targets: Sequence[Tuple[str, str]],
                  note: str) -> float:
    """绘制三层图头并返回 tight_layout 可用的顶部边界。"""
    target_lines = _target_title_lines(targets)
    fig.suptitle(title, fontsize=14, fontweight="semibold", y=0.985)
    fig.text(0.5, 0.945, "\n".join(target_lines), ha="center", va="top",
             fontsize=10.5, color="#303030", linespacing=1.25)
    note_y = 0.945 - 0.035 * len(target_lines)
    fig.text(0.5, note_y, note, ha="center", va="top",
             fontsize=9, color="#666666")
    return note_y - 0.035


# 每序列按面板 key 取值：输入分组结果，输出逐横轴取值的嵌套列表
def _reuse_values(key: str, reuse_n: Sequence[int],
                  by_reuse: Dict[int, List]) -> List[List[float]]:
    if key == "ttft":
        return [[r.ttft * 1000 for r in by_reuse[n]] for n in reuse_n]
    if key == "ttft_norm":
        return [[r.ttft * 1000 / (r.prompt_tokens / 1000)
                 for r in by_reuse[n] if r.prompt_tokens > 0] for n in reuse_n]
    # TPOT 对齐 vLLM 口径：全部输出 token（含思维链）的平均解码间隔
    return [[(r.e2e - r.ttft) * 1000 / (r.output_tokens - 1)
             for r in by_reuse[n] if r.output_tokens > 1] for n in reuse_n]


def _turn_values(key: str, turns: Sequence[int],
                 by_turn: Dict[int, List]) -> List[List[float]]:
    if key == "ttft":
        return [[r.ttft * 1000 for r in by_turn[t]] for t in turns]
    if key == "ttft_norm":
        return [[r.ttft * 1000 / (r.prompt_tokens / 1000)
                 for r in by_turn[t] if r.prompt_tokens > 0] for t in turns]
    return [[(r.e2e - r.ttft) * 1000 / (r.output_tokens - 1)
             for r in by_turn[t] if r.output_tokens > 1] for t in turns]


def _overall(per_x: Sequence[Sequence[float]]) -> float:
    """跨全体横轴点的总均值（NaN 安全）。"""
    vals = [v for v in per_x if v]
    if not vals:
        return float("nan")
    return float(np.nanmean(np.concatenate(
        [np.asarray(v, dtype=float) for v in vals])))


def _plot_reuse_fig(
        series: Sequence[
            Tuple[str, str, Sequence[int], Dict[int, List], Dict[int, int]]
        ],
        path: str, title_targets: Sequence[Tuple[str, str]]) -> None:
    """prefix-repetition 1×3 面板：(a) TTFT (b) 归一化 TTFT/千Token (c) TPOT。

    series: [(label, color, reuse_n, by_reuse, attempts)]。失败请求保留原
    reuse_n，但不进入时延曲线；单序列时总体均值放面板标题。"""
    multi = len(series) > 1
    fig, axes = plt.subplots(1, 3, figsize=(13, 5.0))
    header_top = _chart_header(
        fig, "按同一前缀请求次数的延迟", title_targets,
        "横轴=同一前缀第 N 次请求（N=1 为首次尝试）  |  "
        "实线=avg，阴影=P5-P95 区间")

    panels = [
        ("(a) TTFT - 首字延迟", "TTFT (ms)", "ttft", "avg={:.0f}ms"),
        ("(b) 归一化 TTFT - 每千输入Token", "TTFT (ms/千Token)", "ttft_norm",
         "avg={:.1f}ms/千Tok"),
        ("(c) TPOT - 每输出Token延迟", "TPOT (ms/token)", "tpot",
         "avg={:.1f}ms/tok"),
    ]
    for ax, (name, ylabel, key, fmt) in zip(axes.flat, panels):
        for label, color, reuse_n, by_reuse, _attempts in series:
            per_n = _reuse_values(key, reuse_n, by_reuse)
            avg, p5, p95 = _turn_band(per_n)
            line_label = (f"{label} ({fmt.format(_overall(per_n))})"
                          if multi else "avg")
            _plot_band(ax, reuse_n, avg, p5, p95, color, line_label)
            if not multi:
                # 总体均值放在面板标题左侧（不进图区，避免与右上角图例重叠）
                ax.set_title(f"[{fmt.format(_overall(per_n))}]   {name}", fontsize=11)
        if multi:
            ax.set_title(name, fontsize=11)
        ax.set_xlabel("同一前缀第 N 次请求（N=1 为首次尝试）")
        _int_ticks(ax)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=9)

    failure_notes = []
    for label, _color, reuse_n, by_reuse, attempts in series:
        failed = [
            f"N={n} {attempts[n] - len(by_reuse.get(n, []))}/{attempts[n]}"
            for n in reuse_n if attempts[n] > len(by_reuse.get(n, []))
        ]
        if failed:
            failure_notes.append(f"{label or '当前目标'}: " + ", ".join(failed))
    if failure_notes:
        fig.text(0.5, 0.01, "失败请求（失败/尝试，不参与曲线且不重编号）: "
                 + "；".join(failure_notes),
                 ha="center", va="bottom", fontsize=8, wrap=True)
    fig.tight_layout(rect=(0, 0.07 if failure_notes else 0, 1, header_top))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_turn_fig(series: Sequence[Tuple[str, str, Sequence[int], Dict[int, List],
                                     Dict[int, int]]],
                   path: str,
                   title_targets: Sequence[Tuple[str, str]]) -> None:
    """multi-turn 2×2 面板 + 底部样本数条带：(a) TTFT+上下文长度 (b) 归一化 TTFT/千Token
    (c) TPOT (d) Cache 命中率；条带=每轮存活请求数（柱）+ 失败轮标记（×）。

    series: [(label, color, turns, by_turn, fail_turns)]。(a) 每模型在右轴叠加每轮平均
    prompt tokens 虚线，直观呈现「输入逐轮膨胀、TTFT 是否持平」；(d) 命中率
    仅保留标准口径 token 加权 Σcached/Σprompt（与 OpenAI/Anthropic/vLLM
    一致）实线；右轴叠加每轮平均 prompt tokens 点线（同色），呈现
    「上下文膨胀 vs 命中率」对照。
    底部条带：柱高=该轮成功请求数；红 ×=该轮有会话失败的轮位（该会话自
    下一轮起退出统计），用于解释上方曲线因样本构成变化出现的台阶回落。"""
    multi = len(series) > 1
    fig = plt.figure(figsize=(11, 9.5))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.3])
    axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(2)]
    ax_n = fig.add_subplot(gs[2, :])
    header_top = _chart_header(
        fig, "按轮次延迟与命中率", title_targets,
        "实线=avg，阴影=P5-P95 区间")

    panels = [
        ("(a) TTFT - 首字延迟 vs 上下文长度", "TTFT (ms)", "ttft", "avg={:.0f}ms"),
        ("(b) 归一化 TTFT - 每千输入Token", "TTFT (ms/千Token)", "ttft_norm",
         "avg={:.1f}ms/千Tok"),
        ("(c) TPOT - 每输出Token延迟", "TPOT (ms/token)", "tpot",
         "avg={:.1f}ms/tok"),
    ]
    for ax, (name, ylabel, key, fmt) in zip(axes, panels):
        ax2 = None
        for label, color, turns, by_turn, _fails in series:
            per_turn = _turn_values(key, turns, by_turn)
            avg, p5, p95 = _turn_band(per_turn)
            line_label = (f"{label} ({fmt.format(_overall(per_turn))})"
                          if multi else "avg")
            _plot_band(ax, turns, avg, p5, p95, color, line_label)
            if not multi:
                _avg_box(ax, fmt.format(_overall(per_turn)))
            # (a) 右轴叠加每轮平均 prompt tokens：体现输入膨胀 vs TTFT 走势
            if key == "ttft":
                ctx = [np.mean([r.prompt_tokens for r in by_turn[t]
                                if r.prompt_tokens > 0])
                       if by_turn.get(t) else float("nan") for t in turns]
                if ax2 is None:
                    ax2 = ax.twinx()
                ax2.plot(turns, ctx, "--", color=color, linewidth=1.5,
                         label=(f"{label} 上下文" if multi else "平均输入tokens(右轴)"))
        if ax2 is not None:
            ax2.set_ylabel("平均输入 tokens", fontsize=9)
            ax2.tick_params(axis="y", labelsize=8)
            ax2.grid(False)
            # 合并左右轴图例，避免两框遮挡
            h1, l1 = ax.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8)
        else:
            ax.legend(loc="upper left", fontsize=9)
        ax.set_title(name, fontsize=11)
        ax.set_xlabel("对话轮次")
        _int_ticks(ax)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

    # (d) Cache 命中率：token 加权 Σcached/Σprompt（标准口径）实线；
    # 右轴叠加每轮平均 prompt tokens 点线：命中率走势与上下文膨胀同图对照
    ax = axes[3]
    ax2 = ax.twinx()
    for label, color, turns, by_turn, _fails in series:
        # 仅保留标准口径：token 加权 Σcached/Σprompt（OpenAI/Anthropic/vLLM 一致）
        weighted = np.array([weighted_hit(by_turn[t]) * 100 for t in turns])
        all_requests = [r for t in turns for r in by_turn[t]]
        overall = weighted_hit(all_requests) * 100
        overall_text = "-" if overall != overall else f"{overall:.1f}%"
        ax.plot(turns, weighted, "-", color=color, linewidth=1.5,
                label=(f"{label} token 加权(全局={overall_text})" if multi
                       else "每轮 token 加权（Σcached/Σprompt）"))
        if not multi:
            _avg_box(ax, f"全局token加权={overall_text}")
        ctx = [np.mean([r.prompt_tokens for r in by_turn[t]
                        if r.prompt_tokens > 0])
               if by_turn.get(t) else float("nan") for t in turns]
        ax2.plot(turns, ctx, ":", color=color, linewidth=1.5,
                 label=(f"{label} 上下文" if multi else "平均输入tokens(右轴)"))
    ax2.set_ylabel("平均输入 tokens", fontsize=9)
    ax2.tick_params(axis="y", labelsize=8)
    ax2.grid(False)
    # 合并左右轴图例
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="center right", fontsize=9)
    ax.set_title("(d) Cache 命中率 - 随轮次变化", fontsize=11)
    ax.set_xlabel("对话轮次")
    _int_ticks(ax)
    ax.set_ylabel("命中率 (%)")
    ax.set_ylim(-3, 103)
    ax.grid(True, alpha=0.3)

    # 底部条带：每轮成功请求数（柱）+ 失败轮标记（红 ×）。失败会话自下一轮起
    # 退出统计，均值/命中率曲线可能因此出现台阶回落；× 标出位置供读者归因。
    n_series = len(series)
    width = 0.8 / n_series
    fail_x: List[float] = []
    fail_label: List[str] = []
    for i, (label, color, turns, by_turn, fails) in enumerate(series):
        counts = [len(by_turn.get(t, [])) for t in turns]
        xs = [t + (i - (n_series - 1) / 2) * width for t in turns]
        ax_n.bar(xs, counts, width=width, color=color, alpha=0.55,
                 label=(f"{label} 成功请求数" if multi else "成功请求数"))
        if fails:
            turn_set = set(turns) | set(fails)
            fx = [t for t in sorted(turn_set) if fails.get(t)]
            fy = [len(by_turn.get(t, [])) + max(fails.get(t, 0), 0) for t in fx]
            ax_n.scatter(fx, fy, marker="x", color="red", s=70, zorder=5,
                         linewidths=2.0,
                         label=(f"{label} 会话终止" if multi else "会话终止(该轮失败)"))
            fail_x.extend(fx)
            fail_label.extend(f"{t}轮×{fails[t]}次" for t in fx if fails.get(t))
    if fail_x:
        # 轮位注释放在条带顶部，避免与柱体重叠；轮数多时字号已收小，
        # 仍重叠时由 tight_layout 兜底不裁切
        ax_n.set_title("× 会话终止轮位: " + ", ".join(fail_label)
                       + "（该会话后续轮退出统计）", fontsize=8.5, color="dimgray")
    ax_n.set_xlabel("对话轮次")
    _int_ticks(ax_n)
    ax_n.set_ylabel("成功请求数", fontsize=9)
    ax_n.grid(True, axis="y", alpha=0.3)
    ax_n.legend(loc="upper right", fontsize=8)

    fig.tight_layout(rect=(0, 0, 1, header_top))
    fig.savefig(path, dpi=150)
    plt.close(fig)
