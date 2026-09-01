#!/usr/bin/env python3
"""轻量级 LLM Benchmark Serve 工具入口。

子命令：
  connectivity       快速连通性自检（发 1 个请求打印 TTFT/usage）
  prefix-repetition  固定前缀池重复 + 可变 suffix，全并发
  multi-turn         会话内逐轮累加历史，会话内串行 / 会话间并发
"""
from __future__ import annotations

import argparse
import asyncio
import copy
import datetime
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from typing import Awaitable, Dict, List, Optional, Tuple, TypeVar

import aiohttp

import datasets
import report as report_lib
from engine import (MAX_COMPLETION_TOKENS, OUTPUT_CAP_FIELDS, EarlyAbortError,
                    Engine, FailFast, RequestResult, requires_enabled_thinking,
                    send_chat, thinking_error_hint)
from metrics import RequestMetrics, fmt_ms, fmt_ratio
from report import ComparisonReport, Report, default_report_path

ENV_BASE_URL = "LLM_BENCH_BASE_URL"
ENV_API_KEY = "LLM_BENCH_API_KEY"
# run_*.sh 将第一个目标固化到这两个中立变量；厂商变量（如 ARK_API_KEY）
# 只属于对应 target，不能兼任内部传参。

# multi-turn 语料切片本身不是问句，裸发会让模型偶尔“不接话”（输出只有几个 token，
# 上下文不累积、命中率形态失真）。包一层指令把它变成真实提问形态。
# 指令文本所有 session 相同，但位于每轮新提问的开头（上下文末尾），不影响前缀 cache 口径。
MULTI_TURN_TURN1_TEMPLATE = "请仔细阅读以下材料，之后我会基于它连续提问：\n\n{}"
MULTI_TURN_FOLLOWUP_TEMPLATE = "基于前面的材料，继续分析下面这段内容并给出你的解读：\n\n{}"


def wrap_session_questions(sessions_questions: List[List[str]]) -> List[List[str]]:
    """把语料切片包装成真实提问形态（Turn1 阅读指令，Turn2+ 追问指令）。"""
    return [[(MULTI_TURN_TURN1_TEMPLATE if t == 1 else MULTI_TURN_FOLLOWUP_TEMPLATE).format(q)
             for t, q in enumerate(qs, start=1)]
            for qs in sessions_questions]


def _require_nonempty(value: str) -> str:
    """拦下空字符串：required=True 只拦「没传」，拦不住 --model "$EMPTY" 展开为空的情况，
    空值会一路发到服务端才报 400，提前在这里拦下。"""
    if not value or not value.strip():
        raise argparse.ArgumentTypeError("值不能为空（多半是 --model \"$MODEL\" 而 $MODEL 未定义）")
    return value


def _positive_float(value: str) -> float:
    """把需要参与长度换算的浮点参数限制为正数。"""
    try:
        parsed = float(value)
    except ValueError as e:
        raise argparse.ArgumentTypeError("必须是数字") from e
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("必须是大于 0 的有限数字")
    return parsed


def _positive_int(value: str) -> int:
    """把规模与 token 上限参数限制为正整数。"""
    try:
        parsed = int(value)
    except ValueError as e:
        raise argparse.ArgumentTypeError("必须是整数") from e
    if parsed < 1:
        raise argparse.ArgumentTypeError("必须为正整数")
    return parsed


def add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", required=True, type=_require_nonempty,
                   help="服务商模型 ID")
    p.add_argument("--base-url",
                   default=os.environ.get(ENV_BASE_URL, ""),
                   help=f"API 地址（默认取环境变量 {ENV_BASE_URL}）")
    p.add_argument("--api-key",
                   default=os.environ.get(ENV_API_KEY, ""),
                   help=f"鉴权密钥（默认取环境变量 {ENV_API_KEY}）")
    p.add_argument("--tokenizer", default=None,
                   help="可选：指定 transformers tokenizer 用于语料切片")
    p.add_argument("--chars-per-token", type=_positive_float,
                   default=datasets.DEFAULT_CHARS_PER_TOKEN,
                   help=f"无 tokenizer 时的字符/token 估算系数（默认 {datasets.DEFAULT_CHARS_PER_TOKEN}）")
    p.add_argument("--timeout", type=_positive_float, default=600.0,
                   help="单请求超时秒数")
    p.add_argument("--verbose", action="store_true",
                   help="逐请求进度日志（缺省每 20 轮打一行摘要；多模型对比时推荐用摘要）")
    p.add_argument("--max-completion-tokens", type=_positive_int, default=None,
                   help="所有目标的单次输出上限（回答+思维链）；"
                        "prefix-repetition 默认 512，multi-turn 默认 1024")
    p.add_argument("--target-max-completion-tokens", type=_positive_int, default=None,
                   help=argparse.SUPPRESS)
    p.add_argument("--reasoning-effort", default="",
                   help="可选思考深度；缺省不发送，仅在目标模型需要时显式配置")
    # 当前 targets 已验证兼容 thinking.type；缺省统一关闭思考。
    # 注意 GLM-5.3 只允许 enabled，传 disabled 会报错，需显式改传 enabled
    p.add_argument("--thinking", default="disabled", choices=["enabled", "disabled"],
                   help="统一思考开关，缺省发送 thinking.type=disabled；"
                        "测 GLM-5.3 等强制思考模型时"
                        "显式传 enabled")
    p.add_argument("--output-param", default=None, choices=OUTPUT_CAP_FIELDS,
                   help="输出封顶字段名（缺省 max_completion_tokens）；直连只认 max_tokens "
                        "的厂商（如 DeepSeek 官方）时设为 max_tokens，否则输出不被封顶、对比失真")
    p.add_argument("--output", default=None,
                   help="可选：结果 JSON 落盘路径（逐请求数据）；"
                        "缺省写入本次运行的 reports/<mode>_<目标>_<时间戳>/result_<mode>.json")
    p.add_argument("--report", default=None,
                   help="可选：Markdown 报告路径；"
                        "缺省写入 reports/<mode>_<目标>_<时间戳>/report_<mode>.md")
    p.add_argument("--dump-data", default=None,
                   help="可选：只生成测试数据预览 JSON（每请求/每轮的 token 数+首尾预览）"
                        "并退出，不发送任何请求，无需 API key；用于压测前检查数据形态")


def add_compare_args(p: argparse.ArgumentParser) -> None:
    """多模型同步对比参数（prefix-repetition / multi-turn 可用，connectivity 不适用）。"""
    p.add_argument("--label", default=None,
                   help="可选：主模型在报告/图例中的显示名（缺省用模型 ID）")
    p.add_argument("--compare", action="append", default=None, metavar="SPEC",
                   help="同步对比的友商模型（可重复传入多个）。SPEC 为分号分隔的 "
                        "key=value：label=显示名（缺省用 model）、model=模型 ID（必填）、"
                        "base_url=API 地址（缺省沿用主模型）、api_key=密钥 或 "
                        "api_key_env=环境变量名（二选一）、reasoning_effort / "
                        "max_completion_tokens（reasoning_effort 缺省不发送）、"
                        "output_param=输出封顶字段名（缺省 max_completion_tokens；"
                        "DeepSeek 等只认 max_tokens 的厂商须设为 max_tokens，否则输出不封顶）、"
                        "thinking=思考开关 disabled/enabled（缺省 disabled）。"
                        "示例：--compare 'label=友商A;model=qwen-max;"
                        "base_url=https://dashscope.example.com/v1;api_key_env=DASHSCOPE_KEY'")


# ---------------------------------------------------------------- 对比目标

@dataclass
class Target:
    """一个压测目标，各自独立持有 URL、密钥和请求参数。"""
    label: str      # 报告/图例显示名
    model: str
    engine: Engine


def parse_compare_spec(spec: str, args: argparse.Namespace,
                       default_mct: int) -> Target:
    fields: Dict[str, str] = {}
    for part in spec.split(";"):
        part = part.strip()
        if not part:
            continue
        k, sep, v = part.partition("=")
        if not sep or not k.strip():
            sys.exit(f"[compare] 无法解析对比目标片段 {part!r}（应为 key=value，"
                     f"分号分隔；支持 label/model/base_url/api_key/api_key_env/"
                     f"reasoning_effort/max_completion_tokens/output_param/thinking）")
        fields[k.strip()] = v.strip()
    allowed = {
        "label", "model", "base_url", "api_key", "api_key_env",
        "reasoning_effort", "max_completion_tokens", "output_param", "thinking",
    }
    unknown = sorted(set(fields) - allowed)
    if unknown:
        sys.exit(f"[compare] 不支持的字段: {', '.join(unknown)}")
    model = fields.get("model")
    if not model:
        sys.exit("[compare] 对比目标缺少 model=<友商模型 ID>")
    peer_key = fields.get("api_key") or ""
    env_name = fields.get("api_key_env")
    if env_name:
        peer_key = peer_key or os.environ.get(env_name, "")
    peer_url = fields.get("base_url") or args.base_url
    if not peer_key or not peer_url:
        sys.exit(f"[compare] 对比目标 {model} 缺少 api_key/api_key_env 或 base_url")
    # reasoning_effort 是目标专属参数，不能从其他目标继承；缺省不发送。
    reasoning = fields.get("reasoning_effort", "")
    try:
        if args.max_completion_tokens is not None:
            mct = args.max_completion_tokens
        else:
            mct = (int(fields["max_completion_tokens"])
                   if fields.get("max_completion_tokens") else default_mct)
    except ValueError:
        sys.exit(f"[compare] max_completion_tokens 需为整数，"
                 f"收到 {fields['max_completion_tokens']!r}（对比目标 {model}）")
    if mct < 1:
        sys.exit(f"[compare] max_completion_tokens 必须为正整数，"
                 f"收到 {mct!r}（对比目标 {model}）")
    # 输出封顶参数字段名：DeepSeek 等厂商只认 max_tokens（不认则静默忽略封顶，对比失真）
    cap_field = fields.get("output_param") or "max_completion_tokens"
    if cap_field not in OUTPUT_CAP_FIELDS:
        sys.exit(f"[compare] output_param 仅支持 "
                 f"{'/'.join(OUTPUT_CAP_FIELDS)}，收到 {cap_field!r}")
    # 所有目标缺省独立发送 disabled；不会继承其他目标的 enabled。
    thinking = fields.get("thinking") or "disabled"
    return Target(fields.get("label") or model, model,
                  Engine(peer_url, peer_key, timeout_sec=args.timeout,
                         reasoning_effort=reasoning,
                         max_completion_tokens=mct,
                         output_cap_field=cap_field,
                         thinking=thinking))


def build_targets(args: argparse.Namespace) -> List[Target]:
    """第一个模型 + 全部 --compare 对比目标。所有目标共用同一样本数据与并发上限，
    由调用方 asyncio.gather 同步并发发起。"""
    if not args.base_url or not args.api_key:
        sys.exit(f"请先 export {ENV_API_KEY} 和 "
                 f"{ENV_BASE_URL}（或用 --base-url/--api-key 传入）")
    default_mct = 1024 if args.cmd == "multi-turn" else MAX_COMPLETION_TOKENS
    mct = (args.max_completion_tokens if args.max_completion_tokens is not None
           else getattr(args, "target_max_completion_tokens", None) or default_mct)
    # 第一个目标和其余目标都由 targets/*.env 转换为现有 CLI 参数；bench.py 本身
    # 仍保持 provider 无关，只接收已经展开的模型、地址与方言设置。
    targets = [Target(args.label or args.model, args.model,
                      Engine(args.base_url, args.api_key, timeout_sec=args.timeout,
                             reasoning_effort=args.reasoning_effort,
                             max_completion_tokens=mct,
                             output_cap_field=args.output_param or "max_completion_tokens",
                             thinking=args.thinking))]
    for spec in args.compare or []:
        t = parse_compare_spec(spec, args, default_mct)
        targets.append(t)
    # label 用于图像文件名，重复会导致互相覆盖，自动加序号消歧
    seen: Dict[str, int] = {}
    for t in targets:
        seen[t.label] = seen.get(t.label, 0) + 1
        if seen[t.label] > 1:
            t.label = f"{t.label}_{seen[t.label]}"
    return targets


def print_targets(targets: List[Target]) -> None:
    if len(targets) > 1:
        print(f"[compare] 同步对比 {len(targets)} 个模型：")
        for t in targets:
            print(f"[compare]   - {t.label}: model={t.model} "
                  f"base_url={t.engine.chat_url} "
                  f"max_completion_tokens={t.engine.max_completion_tokens} "
                  f"thinking={_thinking_desc(t.engine)}")
        print("[compare] 所有模型共用同一份样本数据与并发上限，同一时间窗内并发发起")


def check_env(args: argparse.Namespace) -> Engine:
    if not args.base_url or not args.api_key:
        sys.exit(f"请先 export {ENV_API_KEY} 和 "
                 f"{ENV_BASE_URL}（或用 --base-url/--api-key 传入）")
    default_mct = 1024 if args.cmd == "multi-turn" else MAX_COMPLETION_TOKENS
    mct = (args.max_completion_tokens if args.max_completion_tokens is not None
           else getattr(args, "target_max_completion_tokens", None) or default_mct)
    return Engine(args.base_url, args.api_key, timeout_sec=args.timeout,
                  reasoning_effort=args.reasoning_effort, max_completion_tokens=mct,
                  output_cap_field=args.output_param or "max_completion_tokens",
                  thinking=args.thinking)


def warn_reasoning(label: str, results: List[RequestResult]) -> str:
    """基线口径是思考关闭；流里出现 reasoning_content 说明关闭参数被该厂商
    静默忽略。口径影响：TTFT 仍为首输出 token 时延（含首思维链 token，语义不变），
    但 E2E 含完整思维链解码，与思考关闭的目标不可比。
    返回告警文案（写入报告），无污染返回空串。"""
    bad = [r for r in results if r.had_reasoning and r.ok]
    if not bad:
        return ""
    good = [r for r in results if not r.had_reasoning and r.ok]
    lines = [f"[reasoning] ⚠️ {label}: {len(bad)}/{len(results)} 个请求流中出现 "
             f"reasoning_content（思考未关闭，关闭参数被该厂商忽略）"]
    if good:
        lines.append(f"  E2E avg: 思考未关 {fmt_ms(sum(r.e2e for r in bad) / len(bad))}ms"
                     f" vs 已关 {fmt_ms(sum(r.e2e for r in good) / len(good))}ms"
                     f"——该目标 E2E 混入思维链解码，与其他目标不可比")
    else:
        lines.append(f"  E2E avg: {fmt_ms(sum(r.e2e for r in bad) / len(bad))}ms"
                     f"（全部请求均混入思维链，与其他目标不可比）")
    lines.append("  TTFT 口径不受影响（仍为首输出 token 时延）；"
                 "TPOT 含思维链 token（对齐 vLLM：全部输出 token 平均解码间隔）")
    thinks = [r.ttft_content - r.ttft for r in bad if r.content_chunks > 0]
    if thinks:
        lines.append(f"  思考耗时 avg: {fmt_ms(sum(thinks) / len(thinks))}ms"
                     f"（= TTFT(content) - TTFT，报告另有分相统计）")
    lines.append("  建议：正式测算请以思考关闭的口径为准；思考开启的数据仅用于"
                 "评估开启思考时的体验（分相指标：TTFT(content) / Think Time）")
    msg = "\n".join(lines)
    print(msg, file=sys.stderr)
    return msg


def collect_metrics(results: List[RequestResult]) -> RequestMetrics:
    m = RequestMetrics()
    for r in results:
        if r.ok:
            m.add_success(r.ttft, r.e2e, r.output_tokens, r.prompt_tokens,
                          r.cached_tokens, r.has_cache_field,
                          ttft_content=r.ttft_content, e2e_content=r.e2e_content,
                          content_chunks=r.content_chunks,
                          had_reasoning=r.had_reasoning,
                          finish_reason=r.finish_reason,
                          cache_error=r.cache_error)
        elif r.throttled:
            # 429 计为失败（对齐 vLLM：影响成功率）；时延统计仅含成功请求
            m.add_failure(r.usage_error, r.stream_error)
        else:
            m.add_failure(r.usage_error, r.stream_error)
    return m


def build_report(mode: str, model: str, m: RequestMetrics,
                 params: Dict[str, str],
                 start_time, elapsed: float, peak_concurrency: int,
                 label: str = "", thinking: str = "") -> Report:
    """汇总指标，返回 Report（可渲染控制台/Markdown）。"""
    rep = Report(mode, model, params, start_time, label, thinking)
    rep.metrics = m
    rep.elapsed_sec = elapsed
    rep.peak_concurrency = peak_concurrency
    return rep


def emit_report(rep: Report, args: argparse.Namespace, path: Optional[str] = None) -> None:
    """控制台打印 + Markdown 落盘（path 可指定，缺省自动生成）。"""
    rep.render_console()
    path = path or args.report or default_report_path(rep.mode, rep.start_time)
    rep.write_markdown(path)


def new_dataset_rng(seed: Optional[int] = None) -> "tuple[random.Random, int]":
    """未指定时生成随机 seed；指定时复现对应语料采样。

    对齐 vLLM benchmark_serve 的随机采样思想：语料切片起点由 seed 随机化，
    多轮压测之间取材不同，避免上一轮的 KV cache 污染本轮冷启动测量。
    seed 打印并写进报告参数，指标异常时可用同 seed 复现当时的确切数据。
    """
    source = "命令行指定" if seed is not None else "随机生成"
    if seed is None:
        seed = random.randrange(1 << 32)
    print(f"[dataset] 本轮语料采样 seed={seed}（{source}，写入报告）")
    return random.Random(seed), seed


def _check_positive(args: argparse.Namespace, *names: str, cmd: str) -> None:
    """压测规模参数必须为正整数：0/负数会在深处炸出裸 traceback
    （如 num_prefixes=0 触发 ZeroDivisionError），提前拦下给出可读报错。"""
    for n in names:
        if getattr(args, n) < 1:
            sys.exit(f"[{cmd}] --{n.replace('_', '-')} 必须为正整数，收到 {getattr(args, n)}")


def _thinking_desc(e: Engine) -> str:
    """一个目标实际发送的思考控制参数（对比报告逐目标展示用）。"""
    parts = [f"thinking.type={e.thinking}"] if e.thinking else []
    if e.reasoning_effort:
        parts.append(f"reasoning_effort={e.reasoning_effort}")
    return "; ".join(parts) if parts else "未发送思考参数"


def common_params(args: argparse.Namespace, slicer: "datasets.TokenSlicer",
                  max_completion_tokens: int, dataset_seed: int) -> Dict[str, str]:
    return {
        "API 地址": args.base_url,
        "长度口径": f"{slicer.tokenizer_label}（构造口径）" if not slicer.approx
                   else f"估算 ~{args.chars_per_token} chars/token",
        "max_completion_tokens": str(max_completion_tokens),
        "语料采样 seed": str(dataset_seed),
    }


def _result_dict(r: RequestResult) -> Dict:
    return {
        "request_id": r.request_id, "provider_log_id": r.provider_log_id,
        "ok": r.ok, "error": r.error, "ttft_s": r.ttft, "e2e_s": r.e2e,
        "ttft_content_s": r.ttft_content, "e2e_content_s": r.e2e_content,
        "content_chunks": r.content_chunks, "had_reasoning": r.had_reasoning,
        "output_tokens": r.output_tokens, "prompt_tokens": r.prompt_tokens,
        "usage_received": r.usage_received, "usage_error": r.usage_error,
        "done_received": r.done_received, "stream_error": r.stream_error,
        "cached_tokens": r.cached_tokens, "has_cache_field": r.has_cache_field,
        "cache_miss_tokens": r.cache_miss_tokens,
        "cache_error": r.cache_error,
        "reasoning_tokens": r.reasoning_tokens,
        "finish_reason": r.finish_reason,
        "turn": r.turn, "session_id": r.session_id, "prefix_id": r.prefix_id,
        "reuse_n": r.reuse_n,
        "throttled": r.throttled,
    }


def _write_json(path: str, data: Dict) -> None:
    """写 JSON 前创建父目录，支持用户直接指定新的嵌套输出路径。"""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def dump_output(path: str, mode: str, results: List[RequestResult],
                targets: Optional[List[Tuple[str, str, List[RequestResult]]]] = None) -> None:
    """逐请求数据落盘。多模型对比时按 targets 分组（label/model 各自一组）；
    单模型保持原有 {mode, results} 结构不变。"""
    if targets:
        data = {
            "mode": mode,
            "targets": [{"label": label, "model": model,
                         "results": [_result_dict(r) for r in rs]}
                        for label, model, rs in targets],
        }
    else:
        data = {"mode": mode, "results": [_result_dict(r) for r in results]}
    _write_json(path, data)
    print(f"\n[output] 逐请求数据已写入 {path}")


def _preview(text: str, n: int = 120) -> Dict[str, str]:
    return {"head": text[:n], "tail": text[-n:] if len(text) > n else ""}


def dump_multi_turn_data(path: str, args: argparse.Namespace,
                         slicer: "datasets.TokenSlicer",
                         sessions_questions: List[List[str]],
                         dataset_seed: int) -> None:
    """multi-turn 数据预览：每 session 每轮的 token 数 / 字符数 / 首尾预览。"""
    sessions = []
    for sid, qs in enumerate(sessions_questions):
        sessions.append({
            "session_id": sid,
            "turns": [{"turn": t, "role": "user",
                       "tokens": slicer.count_tokens(q), "chars": len(q),
                       **_preview(q)}
                      for t, q in enumerate(qs, start=1)],
        })
    data = {
        "mode": "multi-turn",
        "params": {"initial_len": args.initial_len, "question_len": args.question_len,
                   "num_conversations": args.num_sessions, "max_turns": args.max_turns,
                   "长度口径": slicer.tokenizer_label, "语料采样 seed": dataset_seed},
        "sessions": sessions,
    }
    _write_json(path, data)
    print(f"[dump-data] multi-turn 数据预览已写入 {path}"
          f"（{args.num_sessions} conversations × {args.max_turns} turns，未发送任何请求）")


def dump_prefix_repetition_data(path: str, args: argparse.Namespace,
                                slicer: "datasets.TokenSlicer",
                                message_lists: List[List[Dict[str, str]]],
                                prefix_ids: List[int], dataset_seed: int) -> None:
    """prefix-repetition 数据预览：每请求的 prefix/suffix token 数与首尾预览。"""
    requests = []
    reuse_numbers = _prefix_reuse_numbers(prefix_ids)
    for i, (msgs, pid, reuse_n) in enumerate(
            zip(message_lists, prefix_ids, reuse_numbers)):
        content = msgs[0]["content"]
        requests.append({"request_id": i, "prefix_id": pid, "reuse_n": reuse_n,
                         "tokens": slicer.count_tokens(content), "chars": len(content),
                         **_preview(content)})
    data = {
        "mode": "prefix-repetition",
        "params": {"prefix_len": args.prefix_len, "suffix_len": args.suffix_len,
                   "num_prefixes": args.num_prefixes, "num_requests": args.num_requests,
                   "长度口径": slicer.tokenizer_label, "语料采样 seed": dataset_seed},
        "requests": requests,
    }
    _write_json(path, data)
    print(f"[dump-data] prefix-repetition 数据预览已写入 {path}"
          f"（{len(requests)} 请求，未发送任何请求）")


def _prefix_reuse_numbers(prefix_ids: List[int]) -> List[int]:
    """按提交顺序固定每个 prefix 的复用序号，失败后也不重新编号。"""
    seen: Dict[int, int] = {}
    reuse_numbers = []
    for pid in prefix_ids:
        seen[pid] = seen.get(pid, 0) + 1
        reuse_numbers.append(seen[pid])
    return reuse_numbers


async def _gather_cancel_on_error(coros):
    """gather 的取消增强版：任一子任务抛错（如对比目标触发 fail-fast 早停）时，
    先取消其余子任务再上抛--不让健康目标的请求跟着空跑到底。"""
    tasks = [asyncio.ensure_future(c) for c in coros]
    try:
        return await asyncio.gather(*tasks)
    except BaseException:
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise


T = TypeVar("T")


async def _timed(coro: Awaitable[T]) -> Tuple[T, float]:
    """独立记录单个压测目标的墙钟耗时。"""
    started = time.perf_counter()
    result = await coro
    return result, time.perf_counter() - started


# ---------------------------------------------------------------- connectivity

async def cmd_connectivity(args: argparse.Namespace) -> None:
    engine = check_env(args)
    payload_msgs = [{"role": "user", "content": "你好，请回复：连通性正常"}]

    async def _probe(e: Engine) -> RequestResult:
        async with aiohttp.ClientSession() as sess:
            # 用 engine 的实际配置发请求：连通性自检的 payload 形态须与正式压测一致，
            # 否则对严格校验字段的第三方厂商会误报失败
            return await send_chat(sess, e.chat_url, e.headers, args.model,
                                   payload_msgs, timeout_sec=args.timeout,
                                   reasoning_effort=e.reasoning_effort,
                                   max_completion_tokens=e.max_completion_tokens,
                                   output_cap_field=e.output_cap_field,
                                   thinking=e.thinking)

    r = await _probe(engine)
    if (not r.ok and not r.throttled and engine.thinking == "disabled"
            and requires_enabled_thinking(r.error)):
        # 强制思考模型兜底：GLM-5.3/5.3-FLASH 等不支持 thinking.type=disabled，
        # 请求会被 4xx 拒绝。自动改用 enabled + reasoning_effort=low（其支持的
        # 最低档）重探一次；成功则说明是强制思考模型，不是配置/密钥问题
        fb = copy.copy(engine)
        fb.thinking = "enabled"
        if engine.reasoning_effort in ("", "none"):
            fb.reasoning_effort = "low"  # 强制思考模型仅认 low/high/max
        r2 = await _probe(fb)
        if r2.ok:
            print("  ⚠️ 检测到强制思考模型：thinking.type=disabled 被拒绝（4xx），"
                  "用 enabled + reasoning_effort=low 重探成功")
            print("  正式压测请改传：--thinking enabled"
                  + (" --reasoning-effort low" if engine.reasoning_effort in ("", "none")
                     else ""))
            print("  （注意：正式压测不会自动套用这组兜底参数，须在命令行或 targets 配置里显式给出）")
            print("  口径提示：E2E 含思维链解码，其长度由模型行为决定，"
                  "与思考关闭的目标不可比；TTFT/TPOT 仍可比"
                  "（首 token 与逐 token 解码速度不受思考影响）；"
                  "思考耗时见分相指标 TTFT(content) / Think Time")
            r = r2
            engine = fb  # 后续指标打印用开思考口径
        else:
            hint = thinking_error_hint(r2.error)
            detail = f"\n  参数提示: {hint}" if hint else ""
            sys.exit(
                f"连通性失败: 检测到强制思考错误，改用 enabled + "
                f"{fb.reasoning_effort or '默认 effort'} 复探仍失败\n"
                f"  初始错误: {r.error}\n  复探错误: {r2.error}{detail}")
    elif not r.ok:
        hint = thinking_error_hint(r.error)
        detail = f"\n  参数提示: {hint}" if hint else ""
        sys.exit(f"连通性失败: {r.error}{detail}")
    print(f"连通性 OK  model={args.model}")
    print(f"  TTFT = {fmt_ms(r.ttft)} ms   E2E = {fmt_ms(r.e2e)} ms")
    print(f"  prompt_tokens = {r.prompt_tokens}  completion_tokens = {r.output_tokens}")
    # reasoning_effort 非 none 或 thinking=enabled 任一命中即视为思考开启。
    want_thinking = (engine.reasoning_effort not in ("", "none")
                     or engine.thinking == "enabled")
    if r.had_reasoning:
        if want_thinking:
            # 有意开思考：不算故障，打印分相指标后放行。TPOT 对齐 vLLM 口径
            # （含思维链 token），E2E 含思维链，与思考关闭的目标不可比
            print("  ℹ️ 思考已开启（按用户要求）：分相指标如下")
            print(f"    TTFT(首个思考token) = {fmt_ms(r.ttft)} ms")
            print(f"    TTFT(content)      = {fmt_ms(r.ttft_content)} ms"
                  f"   思考耗时 = {fmt_ms(r.ttft_content - r.ttft)} ms")
            print("    提示：正式对比测算建议以思考关闭口径为准，"
                  "开启思考的数据仅用于评估开启思考时的体验")
        else:
            # 思考关不掉会让 TTFT/E2E 混入思维链时延，口径直接失真
            print("  ⚠️ 检测到 reasoning_content：思考未关闭！该厂商忽略了当前关闭参数，"
                  "请确认 thinking.type=disabled 是否生效，或按模型要求配置 "
                  "--reasoning-effort 后重试；"
                  "强行压测口径不可比")
            sys.exit(1)
    elif want_thinking:
        # 思考开着但本条无思维链输出（如强制思考模型遇到极简 prompt 产出空链），
        # 不能打成“思考已关闭”，会误读口径
        print("  ✓ 未出现 reasoning_content（思考已开启，本条无思维链输出）")
    else:
        print("  ✓ 未出现 reasoning_content（思考已关闭）")
    if r.cache_error:
        print(f"  ⚠️ 服务端返回的 Cache token 数据无效，已排除该样本: {r.cache_error}")
    elif r.has_cache_field:
        hit = r.cached_tokens / r.prompt_tokens if r.prompt_tokens else 0.0
        print(f"  cached_tokens = {r.cached_tokens}  命中率 = {fmt_ratio(hit)}")
    else:
        print("  ℹ️ 本次冷请求未返回缓存命中字段；connectivity 仅验证连通性，"
              "短输入可能不进入缓存，不能据此判断服务端不支持 Cache")


# ----------------------------------------------------- prefix-repetition

async def cmd_prefix_repetition(args: argparse.Namespace) -> None:
    start_time = datetime.datetime.now()
    _check_positive(args, "prefix_len", "suffix_len", "num_prefixes",
                    "num_requests", "max_concurrency",
                    cmd="prefix-repetition")
    if args.num_requests % args.num_prefixes != 0:
        print(f"[prefix-repetition] ⚠️ num_requests={args.num_requests} 不是 "
              f"num_prefixes="
              f"{args.num_prefixes} 的整数倍，各前缀复用次数不等，"
              f"按复用序号的命中率/时延曲线末尾样本量会偏少", file=sys.stderr)
    # 语料池按需扩容：随机旋转起点需要 2 倍取材量，保证 prefix 区 + suffix 区
    # 不回绕（回绕会造成跨请求额外缓存命中），且每轮起点有充分随机空间
    need_tokens = (args.num_prefixes * args.prefix_len + args.num_requests * args.suffix_len) * 2
    need_chars = int(need_tokens * args.chars_per_token * 1.2) + 1_000_000
    pool = datasets.ensure_corpus(pool_chars=need_chars)
    slicer = datasets.TokenSlicer(args.tokenizer, args.chars_per_token)
    rng, dataset_seed = new_dataset_rng(args.seed)
    message_lists, prefix_ids = datasets.build_prefix_requests(
        pool, slicer, args.num_prefixes, args.prefix_len, args.suffix_len,
        args.num_requests, rng=rng)
    if args.dump_data:  # 只看数据形态，不发请求（也不需要 API key）
        dump_prefix_repetition_data(
            args.dump_data, args, slicer, message_lists, prefix_ids, dataset_seed)
        return
    targets = build_targets(args)
    tags = [{"prefix_id": pid, "reuse_n": reuse_n}
            for pid, reuse_n in zip(prefix_ids, _prefix_reuse_numbers(prefix_ids))]

    # 与 usage 口径对齐：prompt_tokens 每请求都计完整 prompt（前缀复用 N 次就计 N 次），
    # 另有每请求 ≤max_completion_tokens 的输出
    est_input = args.num_requests * (args.prefix_len + args.suffix_len)
    est_output = args.num_requests * targets[0].engine.max_completion_tokens  # 输出上限，实测通常低于
    est = est_input + est_output
    print(f"[prefix-repetition] model={args.model} num_requests={args.num_requests} "
          f"num_prefixes={args.num_prefixes} prefix_len={args.prefix_len} "
          f"suffix_len={args.suffix_len} max_concurrency={args.max_concurrency} "
          f"(总 token 估算 ~{est} = 语料净输入 ~{est_input} + 输出上限 ~{est_output}；"
          f"不含分隔符/协议开销，多模型对比时为单模型口径)")
    print_targets(targets)

    t0 = time.perf_counter()
    # 多模型共用同一份 message_lists：样本数据逐字节一致；各自独立并发池同步发起。
    # 某个目标触发 fail-fast 早停时，其余目标一并取消后退出（配置问题须先修）
    try:
        timed_runs = await _gather_cancel_on_error(
            _timed(t.engine.run_requests(t.model, message_lists,
                                         args.max_concurrency, tags, label=t.label))
            for t in targets)
    except EarlyAbortError as e:
        print(e, file=sys.stderr)
        sys.exit(1)
    wall_elapsed = time.perf_counter() - t0
    print(f"[prefix-repetition] finished in {wall_elapsed:.1f}s")
    all_results = [timed_run[0] for timed_run in timed_runs]

    if len(targets) == 1:
        results, elapsed = timed_runs[0]
        m = collect_metrics(results)
        params = common_params(args, slicer,
                               targets[0].engine.max_completion_tokens,
                               dataset_seed) | {
            "prefix_len（语料净长度）": str(args.prefix_len),
            "suffix_len（语料净长度）": str(args.suffix_len),
            "num_prefixes": str(args.num_prefixes), "num_requests": str(args.num_requests),
            "max_concurrency": str(args.max_concurrency),
        }
        rep = build_report("prefix-repetition", args.model, m, params, start_time,
                           elapsed,
                           targets[0].engine.peak_concurrency,
                           targets[0].label,
                           _thinking_desc(targets[0].engine))
        w = warn_reasoning(args.label or args.model, results)
        if w:
            rep.warnings.append(w)
        report_path = args.report or default_report_path(
            rep.mode, rep.start_time,
            [(targets[0].label, targets[0].model)])
        rep.images.extend(report_lib.prefix_charts(
            results, report_path, targets[0].label, targets[0].model))
        emit_report(rep, args, report_path)
        dump_output(
            args.output or report_lib.default_output_path(
                report_path, "prefix-repetition"),
            "prefix-repetition", results)
        return

    # ---- 多模型对比报告
    rep = ComparisonReport("prefix-repetition",
                           common_params(args, slicer,
                                         targets[0].engine.max_completion_tokens,
                                         dataset_seed) | {
                               "prefix_len（语料净长度）": str(args.prefix_len),
                               "suffix_len（语料净长度）": str(args.suffix_len),
                               "num_prefixes": str(args.num_prefixes),
                               "num_requests": str(args.num_requests),
                               "max_concurrency": str(args.max_concurrency),
                           }, start_time)
    report_path = args.report or default_report_path(
        "prefix-repetition", rep.start_time,
        [(t.label, t.model) for t in targets])
    series = []
    for t, (results, elapsed) in zip(targets, timed_runs):
        w = warn_reasoning(t.label, results)
        if w:
            rep.warnings.append(w)
        rep.add(t.label, t.model, collect_metrics(results),
                elapsed, t.engine.peak_concurrency,
                _thinking_desc(t.engine), t.engine.chat_url,
                t.engine.max_completion_tokens)
        series.append((t.label, t.model, results))
    rep.images.extend(report_lib.prefix_charts_compare(series, report_path))
    rep.render_console()
    rep.write_markdown(report_path)
    dump_output(args.output or report_lib.default_output_path(
                    report_path, "prefix-repetition"),
                "prefix-repetition", all_results[0],
                targets=[(t.label, t.model, rs)
                         for t, rs in zip(targets, all_results)])


# -------------------------------------------------------------- multi-turn

# 429 限流 / 瞬时连接错误重试：这两类是配额/网络问题而非服务端性能，
# 重试保住会话连续性（一次失败就截断会话会让轮次曲线尾部样本悄悄变少）。
# timeout / HTTP 5xx 不重试：可能是真实性能故障，重试会掩盖问题、拉偏统计。
TRANSIENT_RETRIES = 3
TRANSIENT_BACKOFF_SEC = (2.0, 4.0, 8.0)


def _transient_error(r: RequestResult) -> bool:
    """可重试的瞬时错误：429 限流 或 连接建立失败（DNS/TCP/TLS 抖动）。"""
    return r.throttled or r.error.startswith("connection error")


async def run_multi_turn(engine: Engine, model: str, max_concurrency: int,
                         sessions_questions: List[List[str]],
                         progress: bool = True,
                         log_tag: str = "multi-turn",
                         verbose: bool = False
                         ) -> "tuple[List[RequestResult], int]":
    """滚动调度：会话为并发单位，跑完一个立刻补位。
    全体会话共享一个 keep-alive 连接池会话（TCPConnector limit=并发数）。

    429 / 连接错误每轮最多重试 TRANSIENT_RETRIES 次（指数退避）；重试期间该会话
    阻塞等待，不发起新请求，重试成功的请求只保留成功那次（不污染时延统计）。
    进度日志：缺省每 PROGRESS_EVERY 轮打一行摘要（多模型交错时逐请求日志太吵），
    verbose=True 时保留逐请求明细。失败/重试始终即时打印到 stderr。
    返回 (results, transient_retries)：retries 为实际发生的重试次数。"""
    PROGRESS_EVERY = 20
    sem = asyncio.Semaphore(max_concurrency)
    results: List[RequestResult] = []
    transient_retries = 0
    done_turns = 0
    total_turns = sum(len(qs) for qs in sessions_questions)
    ff = FailFast(log_tag, engine)

    async def one_session(sid: int, questions: List[str],
                          sess: aiohttp.ClientSession) -> None:
        nonlocal transient_retries, done_turns
        messages: List[Dict[str, str]] = []
        async with sem:
            for turn, q in enumerate(questions, start=1):
                if ff.triggered:  # 已判定配置性失败：剩余会话/轮次不再发出
                    raise ff.error()
                messages = messages + [{"role": "user", "content": q}]
                for attempt in range(TRANSIENT_RETRIES + 1):
                    async with engine.inflight():
                        r = await send_chat(sess, engine.chat_url, engine.headers, model,
                                            messages, engine.timeout_sec,
                                            engine.reasoning_effort,
                                            engine.max_completion_tokens,
                                            engine.output_cap_field,
                                            engine.thinking)
                    if not _transient_error(r) or attempt >= TRANSIENT_RETRIES:
                        break
                    transient_retries += 1
                    backoff = TRANSIENT_BACKOFF_SEC[attempt]
                    kind = "429 限流" if r.throttled else "连接错误"
                    print(f"[{log_tag}] session#{sid} turn{turn} {kind}"
                          f"（{r.error[:80]}），{backoff:.0f}s 后重试"
                          f"（第 {transient_retries} 次重试）",
                          file=sys.stderr, flush=True)
                    await asyncio.sleep(backoff)
                r.session_id = sid
                r.turn = turn
                ff.record(r)
                if ff.triggered:
                    raise ff.error()
                results.append(r)
                if not r.ok:
                    reason = "429 限流重试后仍未恢复" if r.throttled else (
                        "连接错误重试后仍未恢复" if _transient_error(r) else "失败")
                    print(f"[{log_tag}] session#{sid} turn{turn} {reason}: "
                          f"{r.error}，会话提前终止", file=sys.stderr)
                    break
                if progress:
                    if verbose:
                        think = (f" think={fmt_ms(r.ttft_content - r.ttft)}ms"
                                 if r.had_reasoning else "")
                        print(f"[{log_tag}] session#{sid} turn{turn} "
                              f"ttft={fmt_ms(r.ttft)}ms{think} e2e={fmt_ms(r.e2e)}ms "
                              f"cached={r.cached_tokens}/{r.prompt_tokens}", flush=True)
                    done_turns += 1
                    if done_turns % PROGRESS_EVERY == 0 or done_turns == total_turns:
                        fails = sum(1 for x in results if not x.ok)
                        print(f"[{log_tag}] 进度 {done_turns}/{total_turns} 轮"
                              f"（失败 {fails}）", flush=True)
                # 性能基线以关闭思考为主，只回填可见回答。部分厂商的
                # preserved-thinking 模式要求同时回填 reasoning_content/details；
                # 当前未实现该厂商特定协议，开启思考时不保证多轮上下文完整。
                messages = messages + [{"role": "assistant", "content": r.completion_text}]

    async with engine.client(max_concurrency) as sess:
        tasks = [asyncio.create_task(one_session(sid, qs, sess))
                 for sid, qs in enumerate(sessions_questions)]
        try:
            await asyncio.gather(*tasks)
        except EarlyAbortError:
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
    return results, transient_retries


async def cmd_multi_turn(args: argparse.Namespace) -> None:
    start_time = datetime.datetime.now()
    _check_positive(args, "initial_len", "question_len", "num_sessions",
                    "max_turns", "max_concurrency",
                    cmd="multi-turn")
    slicer = datasets.TokenSlicer(args.tokenizer, args.chars_per_token)

    # 每会话消耗 ~ initial_len + (max_turns-1)*question_len tokens 的语料；
    # 池按需扩容保证不同会话取材互不相同（相同会造成跨会话缓存命中，虚高命中率）
    per_session = args.initial_len + max(0, args.max_turns - 1) * args.question_len
    need_chars = int(args.num_sessions * per_session * args.chars_per_token * 1.2) + 1_000_000
    pool = datasets.ensure_corpus(pool_chars=need_chars)
    rng, dataset_seed = new_dataset_rng(args.seed)
    step = slicer.chars_for(per_session)
    span = max(1, len(pool) - step)
    sessions_questions = []
    if args.num_sessions > span:
        # 语料池太小放不下互不重叠的取材窗：允许重叠（首 token 不同即不产生
        # 前缀命中），但仍去重起点，避免两 session 逐字节相同、命中率虚高
        print(f"[multi-turn] ⚠️ num_sessions={args.num_sessions} 超过可用起点数 "
              f"{span}，会话间取材将部分重叠", file=sys.stderr)
        offsets = rng.sample(range(span), span) if span > 1 else [0] * args.num_sessions
        while len(offsets) < args.num_sessions:
            offsets.append(rng.randrange(0, span))
    else:
        # rng.sample 保证起点互不重复：相同起点的两个 session 语料逐字节相同，
        # 会造成跨会话前缀命中、该会话命中率虚高
        offsets = rng.sample(range(span), args.num_sessions)
    for offset in offsets:
        # 各会话起点随机且互不相同（对齐 vLLM 随机采样）：起点不同则首个 token 不同，
        # prefix cache 跨会话必然无法命中；内容允许重叠，不影响命中口径。
        # 顺序推进（s*step）在重复压测时取材固定，会被上一轮的 cache 污染。
        sessions_questions.append(datasets.build_session_questions(
            pool, slicer, args.max_turns, args.initial_len, args.question_len, offset))
    # 包装成真实提问形态后再 dump/发送，保证预览看到的就是实际内容
    sessions_questions = wrap_session_questions(sessions_questions)

    if args.dump_data:  # 只看数据形态，不发请求（也不需要 API key）
        dump_multi_turn_data(args.dump_data, args, slicer, sessions_questions,
                             dataset_seed)
        return
    targets = build_targets(args)

    print(f"[multi-turn] model={args.model} num_conversations={args.num_sessions} "
          f"max_turns={args.max_turns} initial_len={args.initial_len} "
          f"question_len={args.question_len} "
          f"max_completion_tokens={targets[0].engine.max_completion_tokens} "
          f"max_concurrency={args.max_concurrency} "
          f"(长度口径: {slicer.tokenizer_label}，"
          f"{'估算 ~%.1f chars/token' % args.chars_per_token if slicer.approx else '按构造 tokenizer 切片'})")
    est_final = args.initial_len + (args.max_turns - 1) * (
        args.question_len + targets[0].engine.max_completion_tokens)
    # 与 usage 口径对齐的总量估算：每轮 prompt 都计完整累积上下文（含历史回答），
    # 回答按 max_completion_tokens 上限近似（实测因提前 stop 通常低于上限）
    T = args.max_turns
    per_session_input = T * args.initial_len + T * (T - 1) // 2 * (
        args.question_len + targets[0].engine.max_completion_tokens)
    est_input = args.num_sessions * per_session_input
    est_output = args.num_sessions * T * targets[0].engine.max_completion_tokens
    est = est_input + est_output
    print(f"[multi-turn] 长度形态: Turn1 语料 {args.initial_len} tokens（长文档），"
          f"Turn2+ 追问语料各 {args.question_len} tokens，"
          f"每轮回答 ≤{targets[0].engine.max_completion_tokens} tokens；"
          f"最终轮语料+回答上限估算 ≈ {est_final} tokens（不含固定指令/协议开销）")
    print(f"[multi-turn] 总 token 估算 ~{est} = 语料+历史回答输入 ~{est_input}"
          f"（每轮计完整累积上下文，不含固定指令/协议开销）"
          f"+ 输出上限 ~{est_output}（单模型口径）")
    print_targets(targets)
    t0 = time.perf_counter()
    # 多模型共用同一份 sessions_questions（提问逐字节一致）；历史由各自回答累积，
    # 上下文长度会随模型输出风格自然分化（属多轮真实形态）。
    # 某个目标触发 fail-fast 早停时，其余目标一并取消后退出（配置问题须先修）
    try:
        timed_runs = await _gather_cancel_on_error(
            _timed(run_multi_turn(t.engine, t.model, args.max_concurrency,
                                  sessions_questions, log_tag=f"multi-turn/{t.label}",
                                  verbose=args.verbose))
            for t in targets)
    except EarlyAbortError as e:
        print(e, file=sys.stderr)
        sys.exit(1)
    wall_elapsed = time.perf_counter() - t0
    print(f"[multi-turn] finished in {wall_elapsed:.1f}s")
    runs = [timed_run[0] for timed_run in timed_runs]

    if len(targets) == 1:
        (results, retries), elapsed = timed_runs[0]
        m = collect_metrics(results)
        params = common_params(args, slicer,
                               targets[0].engine.max_completion_tokens,
                               dataset_seed) | {
            "initial_len（语料净长度）": str(args.initial_len),
            "question_len（语料净长度）": str(args.question_len),
            "num_conversations": str(args.num_sessions), "max_turns": str(args.max_turns),
            "max_concurrency": str(args.max_concurrency),
            "429/网络重试次数": str(retries),
        }
        if retries:  # 触发过重试必须在报告头部披露：重试改变了测量形态
            params["⚠️ 重试提示"] = (
                f"触发 429 限流/连接错误重试 {retries} 次"
                "（重试尝试不计入逻辑请求数，仅保留每个逻辑请求的最终结果；"
                "重试等待不计入时延统计）；429 说明压测强度超服务配额（TPM/RPM），"
                "建议提额后重测再作结论")
        rep = build_report("multi-turn", args.model, m, params, start_time,
                           elapsed,
                           targets[0].engine.peak_concurrency,
                           targets[0].label,
                           _thinking_desc(targets[0].engine))
        w = warn_reasoning(args.label or args.model, results)
        if w:
            rep.warnings.append(w)
        report_path = args.report or default_report_path(
            rep.mode, rep.start_time,
            [(targets[0].label, targets[0].model)])
        rep.images.extend(report_lib.multi_turn_charts(
            results, report_path, targets[0].label, targets[0].model))
        emit_report(rep, args, report_path)
        dump_output(
            args.output or report_lib.default_output_path(report_path, "multi-turn"),
            "multi-turn", results)
        return

    # ---- 多模型对比报告
    rep = ComparisonReport("multi-turn",
                           common_params(args, slicer,
                                         targets[0].engine.max_completion_tokens,
                                         dataset_seed) | {
                               "initial_len（语料净长度）": str(args.initial_len),
                               "question_len（语料净长度）": str(args.question_len),
                               "num_conversations": str(args.num_sessions),
                               "max_turns": str(args.max_turns),
                               "max_concurrency": str(args.max_concurrency),
                               "429/网络重试次数": "; ".join(
                                   f"{t.label}={r[1]}"
                                   for t, r in zip(targets, runs)),
                               "口径说明": "各模型提问逐字节一致；多轮历史由各自回答累积，"
                                          "上下文长度随模型输出风格自然分化",
                           }, start_time)
    retried = [f"{t.label} {run[1]} 次"
               for t, run in zip(targets, runs) if run[1]]
    if retried:  # 任一侧触发过重试必须在报告头部披露（标注是哪一侧）
        rep.params["⚠️ 重试提示"] = (
            "；".join(retried) + " 触发 429 限流/连接错误重试，退避重试后恢复"
            "（重试等待不计入时延统计）；429 说明对应侧压测强度超服务配额（TPM/RPM），"
            "建议提额后重测再作结论")
    report_path = args.report or default_report_path(
        "multi-turn", rep.start_time,
        [(t.label, t.model) for t in targets])
    series = []
    for t, (run, elapsed) in zip(targets, timed_runs):
        results = run[0]
        w = warn_reasoning(t.label, results)
        if w:
            rep.warnings.append(w)
        rep.add(t.label, t.model, collect_metrics(results),
                elapsed, t.engine.peak_concurrency,
                _thinking_desc(t.engine), t.engine.chat_url,
                t.engine.max_completion_tokens)
        series.append((t.label, t.model, results))
    rep.images.extend(report_lib.multi_turn_charts_compare(series, report_path))
    rep.render_console()
    rep.write_markdown(report_path)
    dump_output(args.output or report_lib.default_output_path(report_path, "multi-turn"),
                "multi-turn", runs[0][0],
                targets=[(t.label, t.model, run[0])
                         for t, run in zip(targets, runs)])


# ---------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser(description="轻量级 LLM Benchmark Serve 工具")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_conn = sub.add_parser("connectivity", help="连通性自检")
    add_common_args(p_conn)

    p_prefix_repetition = sub.add_parser(
        "prefix-repetition",
        help="固定前缀池重复 + 可变 suffix（对标 vLLM prefix_repetition）")
    add_common_args(p_prefix_repetition)
    add_compare_args(p_prefix_repetition)
    p_prefix_repetition.add_argument(
        "--prefix-len", type=_positive_int, default=12000, help="前缀长度（token）")
    p_prefix_repetition.add_argument(
        "--suffix-len", type=_positive_int, default=2000, help="后缀长度（token）")
    p_prefix_repetition.add_argument(
        "--num-prefixes", type=_positive_int, default=10, help="前缀池个数")
    p_prefix_repetition.add_argument(
        "--num-requests", type=_positive_int, default=200, help="总请求数")
    p_prefix_repetition.add_argument(
        "--max-concurrency", type=_positive_int, default=5, help="并发上限")
    p_prefix_repetition.add_argument(
        "--seed", type=int, default=None,
        help="语料采样 seed；缺省随机生成，传报告中的值可复现输入")

    p_multi_turn = sub.add_parser(
        "multi-turn", help="多轮对话动态 Cache 命中率")
    add_common_args(p_multi_turn)
    add_compare_args(p_multi_turn)
    p_multi_turn.add_argument(
        "--initial-len", type=_positive_int, default=3000,
        help="Turn1 长输入的 token 数（模拟首轮塞入的文档/长上下文）")
    p_multi_turn.add_argument(
        "--question-len", type=_positive_int, default=256,
        help="Turn2+ 每轮短追问的 token 数（真实用户提问通常很短）")
    p_multi_turn.add_argument("--num-sessions", type=_positive_int, default=10,
                              help="总会话数")
    p_multi_turn.add_argument(
        "--max-turns", type=_positive_int, default=20,
        help="每会话轮数（Turn1 为 initial_len 长输入，Turn2+ 每轮 question_len 短追问）")
    p_multi_turn.add_argument("--max-concurrency", type=_positive_int, default=5,
                              help="并发会话数上限")
    p_multi_turn.add_argument(
        "--seed", type=int, default=None,
        help="语料采样 seed；缺省随机生成，传报告中的值可复现输入")

    args = ap.parse_args()
    fn = {"connectivity": cmd_connectivity,
          "prefix-repetition": cmd_prefix_repetition,
          "multi-turn": cmd_multi_turn}[args.cmd]
    asyncio.run(fn(args))


if __name__ == "__main__":
    main()
