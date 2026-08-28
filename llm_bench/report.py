"""压测报告：vLLM benchmark_serving 风格总览。

呈现层全部收口在本文件：bench.py 只负责发请求拿 RequestResult，
总览的构建由这里完成。只保留 Serving Benchmark Result 基准块
（TTFT / TPOT / 端到端时延）。
"""
from __future__ import annotations

import datetime
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from metrics import (RequestMetrics, fmt_ratio, percentile_stats)

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


def serving_summary_lines(m: RequestMetrics, duration_sec: float,
                          peak_concurrency: int) -> List[str]:
    """vLLM benchmark_serving 风格的总体结果块。"""
    lines = [
        "============ Serving Benchmark Result ============",
        _kv("Successful requests:", str(m.total - m.failed)),
        _kv("Failed requests:", str(m.failed)),
        _kv("Benchmark duration (s):", f"{duration_sec:.2f}"),
        _kv("Total input tokens:", str(sum(m.prompt_tokens))),
        _kv("Total generated tokens:", str(sum(m.output_tokens))),
        # 命中率口径披露：加权命中率分母仅含返回 cache 字段的请求，
        # 与 Total input tokens（全部成功请求）不同，部分缺失时不能直接互换算
        _kv("Cache field coverage:", f"{m.cache_field_count}/{m.total - m.failed}"
                                     f" requests (hit-rate denominator)"),
        _kv("Peak concurrent requests:", f"{peak_concurrency:d}"),
        "---------------Time to First Token----------------",
    ]
    lines += _stat_block("TTFT", m.ttft)
    lines.append("-----Time per Output Token (excl. 1st token)------")
    lines += _stat_block("TPOT", m.tpot)
    lines.append("---------------End-to-End Latency-----------------")
    lines += _stat_block("E2E", m.e2e)
    lines.append("==================================================")
    return lines


class Report:
    """一次压测的完整报告数据，可渲染为控制台输出与 Markdown 文档。"""

    def __init__(self, mode: str, model: str, params: Dict[str, str],
                 start_time: Optional[datetime.datetime] = None) -> None:
        self.mode = mode
        self.model = model
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
                                              self.peak_concurrency):
                print(line)
        for title, img in self.images:
            print(f"[chart] {title}: {img}")

    # ---------------------------------------------------------------- markdown

    def render_markdown(self) -> str:
        m = self.metrics
        lines: List[str] = []
        lines.append(f"# LLM Benchmark Report - {self.mode}")
        lines.append("")
        lines.append(f"- **压测时间**: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}"
                     f"（耗时 {self.elapsed_sec:.1f}s）")
        lines.append(f"- **模式**: {self.mode}    **模型**: `{self.model}`")
        for k, v in self.params.items():
            lines.append(f"- **{k}**: {v}")
        lines.append("")
        if self.warnings:
            lines.append("## ⚠️ 口径告警")
            lines.append("")
            lines.append("```")
            lines.extend(self.warnings)
            lines.append("```")
            lines.append("")
        lines.append("## 总体结果")
        lines.append("")
        if m:
            lines.append("```")
            lines.extend(serving_summary_lines(m, self.elapsed_sec,
                                               self.peak_concurrency))
            lines.append("```")
            lines.append("")
        if self.images:
            lines.append("## 图像")
            lines.append("")
            for title, img in self.images:
                lines.append(f"### {title}")
                lines.append("")
                lines.append(f"![{title}]({img})")
                lines.append("")
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
        # (label, model, metrics, elapsed_sec, peak_concurrency)
        self.entries: List[Tuple[str, str, "RequestMetrics", float, int]] = []
        self.images: List[Tuple[str, str]] = []
        # 口径告警（如思考未关闭导致 E2E 混入思维链）：渲染进 Markdown，
        # 保证只看报告的人也能看到口径问题
        self.warnings: List[str] = []

    def add(self, label: str, model: str, m: "RequestMetrics",
            elapsed: float, peak_concurrency: int) -> None:
        self.entries.append((label, model, m, elapsed, peak_concurrency))

    # ---------------------------------------------------------------- console

    def render_console(self) -> None:
        for label, model, m, elapsed, peak in self.entries:
            print(f"\n>>>>>> {label} <<<<<<")
            for line in serving_summary_lines(m, elapsed, peak):
                print(line)
        print("\n== 多模型对比总览 ==")
        for line in self.compare_table_lines():
            print(line)
        for title, img in self.images:
            print(f"[chart] {title}: {img}")

    # ---------------------------------------------------------------- markdown

    def compare_table_lines(self) -> List[str]:
        """关键指标对比表（Markdown 表格行）。"""
        head = ["模型", "成功/失败", "TTFT avg (ms)", "TTFT P99 (ms)",
                "TPOT avg (ms)", "E2E avg (ms)", "加权Cache命中率"]
        rows = []
        for label, model, m, elapsed, peak in self.entries:
            ttft = percentile_stats(m.ttft)
            tpot = percentile_stats(m.tpot)
            e2e = percentile_stats(m.e2e)
            rows.append([
                f"{label}", f"{m.total - m.failed}/{m.failed}",
                f"{ttft['AVG'] * 1000:.1f}", f"{ttft['P99'] * 1000:.1f}",
                f"{tpot['AVG'] * 1000:.1f}", f"{e2e['AVG'] * 1000:.1f}",
                fmt_ratio(m.weighted_cache_hit),
            ])
        lines = ["| " + " | ".join(head) + " |",
                 "|" + "---|" * len(head)]
        for r in rows:
            lines.append("| " + " | ".join(r) + " |")
        return lines

    def render_markdown(self) -> str:
        lines: List[str] = []
        lines.append(f"# LLM Benchmark Compare Report - {self.mode}")
        lines.append("")
        lines.append(f"- **压测时间**: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}"
                     f"（多模型同步并发，各自独立计时）")
        lines.append(f"- **模式**: {self.mode}")
        for k, v in self.params.items():
            lines.append(f"- **{k}**: {v}")
        lines.append("")
        if self.warnings:
            lines.append("## ⚠️ 口径告警")
            lines.append("")
            lines.append("```")
            lines.extend(self.warnings)
            lines.append("```")
            lines.append("")
        lines.append("## 多模型对比总览")
        lines.append("")
        lines.extend(self.compare_table_lines())
        lines.append("")
        for label, model, m, elapsed, peak in self.entries:
            lines.append(f"## {label}")
            lines.append("")
            lines.append("```")
            lines.extend(serving_summary_lines(m, elapsed, peak))
            lines.append("```")
            lines.append("")
        if self.images:
            lines.append("## 图像")
            lines.append("")
            for title, img in self.images:
                lines.append(f"### {title}")
                lines.append("")
                lines.append(f"![{title}]({img})")
                lines.append("")
        return "\n".join(lines)

    def write_markdown(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.render_markdown())
        print(f"\n[report] Markdown 对比报告已写入 {path}")


def default_report_path(mode: str, start_time: datetime.datetime) -> str:
    """默认报告路径：每次运行一个独立目录 reports/<mode>_<时间戳>/，
    Markdown 报告、结果 JSON 全部落在该目录内，多次运行互不混杂。"""
    run_dir = os.path.join(REPORT_DIR,
                           f"{mode}_{start_time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(run_dir, exist_ok=True)
    return os.path.join(run_dir, f"report_{mode}.md")


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


def multiturn_charts(results: Sequence, report_path: str,
                     model: str = "") -> List[Tuple[str, str]]:
    """multiturn 图像（单模型）：2×2 面板（TTFT / 归一化 TTFT / TPOT / Cache 命中率），
    每面板 AVG 实线 + P5–P95 阴影带。PNG 与 md 同目录同名前缀。"""
    if not _ensure_mpl():
        return []
    base = _report_base(report_path)
    by_turn = _group_by_turn(results)
    if not by_turn:
        return []
    turns = sorted(by_turn)
    p = f"{base}_turns.png"
    _plot_turn_fig([(model or "", "tab:green", turns, by_turn)], p)
    return [("按轮次延迟与命中率（TTFT / 归一化 TTFT / TPOT / Cache 命中率）",
             os.path.basename(p))]


def prefix_charts(results: Sequence, report_path: str,
                  model: str = "") -> List[Tuple[str, str]]:
    """prefix 图像（单模型）：1×3 面板（TTFT / 归一化 TTFT / TPOT），
    横轴为同一前缀的第 N 次请求（第 1 次=冷启动未命中），
    每面板 AVG 实线 + P5–P95 阴影带。PNG 与 md 同目录同名前缀。"""
    if not _ensure_mpl():
        return []
    base = _report_base(report_path)
    grouped = _group_by_reuse(results)
    if not grouped:
        return []
    reuse_n, by_reuse = grouped
    p = f"{base}_reuse.png"
    _plot_reuse_fig([(model or "", "tab:green", reuse_n, by_reuse)], p)
    return [("按前缀复用序号的延迟（TTFT / 归一化 TTFT / TPOT）",
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


def _group_by_reuse(results: Sequence) -> Optional[Tuple[List[int], Dict[int, List]]]:
    """results -> (reuse_n 列表, reuse_n -> 请求列表)。

    run_requests 用 asyncio.gather 返回，results 保持提交顺序；
    同一 prefix_id 内按出现顺序编号即「第 N 次复用该前缀」"""
    by_prefix: Dict[int, List] = {}
    for r in results:
        if r.ok and r.prefix_id >= 0:
            by_prefix.setdefault(r.prefix_id, []).append(r)
    if not by_prefix:
        return None
    # reuse_n（1 起）-> 该序号下所有前缀的请求
    by_reuse: Dict[int, List] = {}
    for rs in by_prefix.values():
        for n, r in enumerate(rs, start=1):
            by_reuse.setdefault(n, []).append(r)
    return sorted(by_reuse), by_reuse


def prefix_charts_compare(series: Sequence[Tuple[str, Sequence]],
                          report_path: str) -> List[Tuple[str, str]]:
    """prefix 多模型对比：一张叠加对比图（所有模型同面板同色区分）。
    series: List[(label, results)]。"""
    if not _ensure_mpl():
        return []
    base = _report_base(report_path)
    prepared = []
    for i, (label, results) in enumerate(series):
        grouped = _group_by_reuse(results)
        if grouped:
            prepared.append((label, SERIES_COLORS[i % len(SERIES_COLORS)],
                             *grouped))
    images: List[Tuple[str, str]] = []
    if prepared:
        p = f"{base}_compare_reuse.png"
        _plot_reuse_fig(prepared, p)
        images.append(("多模型对比：按前缀复用序号的延迟（TTFT / 归一化 TTFT / TPOT）",
                       os.path.basename(p)))
    return images


def multiturn_charts_compare(series: Sequence[Tuple[str, Sequence]],
                             report_path: str) -> List[Tuple[str, str]]:
    """multiturn 多模型对比：一张叠加对比图。"""
    if not _ensure_mpl():
        return []
    base = _report_base(report_path)
    prepared = []
    for i, (label, results) in enumerate(series):
        by_turn = _group_by_turn(results)
        if by_turn:
            prepared.append((label, SERIES_COLORS[i % len(SERIES_COLORS)],
                             sorted(by_turn), by_turn))
    images: List[Tuple[str, str]] = []
    if prepared:
        p = f"{base}_compare_turns.png"
        _plot_turn_fig(prepared, p)
        images.append(("多模型对比：按轮次延迟与命中率（TTFT / 归一化 TTFT / TPOT / Cache 命中率）",
                       os.path.basename(p)))
    return images


def weighted_hit(rs: Sequence) -> float:
    """Σcached / Σprompt（仅统计读到 cache 字段的请求）。"""
    with_field = [r for r in rs if r.has_cache_field and r.prompt_tokens]
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
    ax.plot(turns, avg, "-", color=color, linewidth=1.5, label=label)


def _avg_box(ax, text: str) -> None:
    """右上角总体均值标注框。"""
    ax.text(0.98, 0.97, text, transform=ax.transAxes, ha="right", va="top",
            fontsize=9, bbox=dict(boxstyle="round,pad=0.35", fc="white",
                                  ec="0.6", alpha=0.9))


# 每序列按面板 key 取值：输入分组结果，输出逐横轴取值的嵌套列表
def _reuse_values(key: str, reuse_n: Sequence[int],
                  by_reuse: Dict[int, List]) -> List[List[float]]:
    if key == "ttft":
        return [[r.ttft * 1000 for r in by_reuse[n]] for n in reuse_n]
    if key == "ttft_norm":
        return [[r.ttft * 1000 / (r.prompt_tokens / 1000)
                 for r in by_reuse[n] if r.prompt_tokens > 0] for n in reuse_n]
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


def _plot_reuse_fig(series: Sequence[Tuple[str, str, Sequence[int], Dict[int, List]]],
                    path: str) -> None:
    """prefix 1×3 面板：(a) TTFT (b) 归一化 TTFT/千Token (c) TPOT。

    series: [(label, color, reuse_n, by_reuse)]。单序列时总体均值放面板标题
    （与历史形态一致）；多序列时进图例，便于逐模型对照。"""
    multi = len(series) > 1
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    title = "按前缀复用序号的延迟"
    if not multi and series[0][0]:
        title += f" - {series[0][0]}"
    fig.suptitle(f"{title} | 横轴=同一前缀第 N 次请求(N=1 为冷启动) | "
                 f"实线=avg, 阴影=P5-P95区间")

    panels = [
        ("(a) TTFT - 首字延迟", "TTFT (ms)", "ttft", "avg={:.0f}ms"),
        ("(b) 归一化 TTFT - 每千输入Token", "TTFT (ms/千Token)", "ttft_norm",
         "avg={:.1f}ms/千Tok"),
        ("(c) TPOT - 每输出Token延迟", "TPOT (ms/token)", "tpot",
         "avg={:.1f}ms/tok"),
    ]
    for ax, (name, ylabel, key, fmt) in zip(axes.flat, panels):
        for label, color, reuse_n, by_reuse in series:
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
        ax.set_xlabel("同一前缀第 N 次请求（N=1 为冷启动）")
        _int_ticks(ax)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=9)

    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_turn_fig(series: Sequence[Tuple[str, str, Sequence[int], Dict[int, List]]],
                   path: str) -> None:
    """multiturn 2×2 面板：(a) TTFT+上下文长度 (b) 归一化 TTFT/千Token (c) TPOT (d) Cache 命中率。

    series: [(label, color, turns, by_turn)]。(a) 每模型在右轴叠加每轮平均
    prompt tokens 虚线，直观呈现「输入逐轮膨胀、TTFT 是否持平」；(d) 命中率
    仅保留标准口径 token 加权 Σcached/Σprompt（与 OpenAI/Anthropic/vLLM
    一致）实线；右轴叠加每轮平均 prompt tokens 点线（同色），呈现
    「上下文膨胀 vs 命中率」对照。"""
    multi = len(series) > 1
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    title = "按轮次延迟与命中率"
    if not multi and series[0][0]:
        title += f" - {series[0][0]}"
    fig.suptitle(f"{title} | 实线=avg, 阴影=P5-P95区间")

    panels = [
        ("(a) TTFT - 首字延迟 vs 上下文长度", "TTFT (ms)", "ttft", "avg={:.0f}ms"),
        ("(b) 归一化 TTFT - 每千输入Token", "TTFT (ms/千Token)", "ttft_norm",
         "avg={:.1f}ms/千Tok"),
        ("(c) TPOT - 每输出Token延迟", "TPOT (ms/token)", "tpot",
         "avg={:.1f}ms/tok"),
    ]
    for ax, (name, ylabel, key, fmt) in zip(axes.flat, panels):
        ax2 = None
        for label, color, turns, by_turn in series:
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
    ax = axes.flat[3]
    ax2 = ax.twinx()
    for label, color, turns, by_turn in series:
        # 仅保留标准口径：token 加权 Σcached/Σprompt（OpenAI/Anthropic/vLLM 一致）
        weighted = np.array([weighted_hit(by_turn[t]) * 100 for t in turns])
        ax.plot(turns, weighted, "-", color=color, linewidth=1.5,
                label=(f"{label} token 加权(标准)" if multi
                       else "token 加权（Σcached/Σprompt）"))
        if not multi:
            # 某轮全部请求都未返回 cache 字段时整体可能全 NaN，显示 "-" 避免 "nan%"
            overall = float(np.nanmean(weighted)) if np.isfinite(weighted).any() else float("nan")
            _avg_box(ax, "-" if overall != overall else f"加权avg={overall:.1f}%")
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

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=150)
    plt.close(fig)
