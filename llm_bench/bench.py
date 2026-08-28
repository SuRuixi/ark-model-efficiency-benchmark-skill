#!/usr/bin/env python3
"""轻量级 LLM Benchmark Serve 工具入口。

子命令：
  connectivity  快速连通性自检（发 1 个请求打印 TTFT/usage）
  prefix        模式 A：固定前缀池复用 + 新 suffix，全并发
  multiturn     模式 B：会话内逐轮累加历史，会话内串行 / 会话间并发
"""
from __future__ import annotations

import argparse
import asyncio
import datetime
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import aiohttp

import datasets
import report as report_lib
from engine import MAX_COMPLETION_TOKENS, Engine, RequestResult, send_chat
from metrics import RequestMetrics, fmt_ms, fmt_ratio
from report import ComparisonReport, Report, default_report_path

ENV_BASE_URL = "ARK_BASE_URL"
ENV_API_KEY = "ARK_API_KEY"
# OpenAI 兼容命名作为回退（部分客户习惯 export OPENAI_*）
ENV_BASE_URL_FALLBACK = "OPENAI_BASE_URL"
ENV_API_KEY_FALLBACK = "OPENAI_API_KEY"

# multiturn 语料切片本身不是问句，裸发会让模型偶尔“不接话”（输出只有几个 token，
# 上下文不累积、命中率形态失真）。包一层指令把它变成真实提问形态。
# 指令文本所有 session 相同，但位于每轮新提问的开头（上下文末尾），不影响前缀 cache 口径。
MULTITURN_TURN1_TEMPLATE = "请仔细阅读以下材料，之后我会基于它连续提问：\n\n{}"
MULTITURN_FOLLOWUP_TEMPLATE = "基于前面的材料，继续分析下面这段内容并给出你的解读：\n\n{}"


def wrap_session_questions(sessions_questions: List[List[str]]) -> List[List[str]]:
    """把语料切片包装成真实提问形态（Turn1 阅读指令，Turn2+ 追问指令）。"""
    return [[(MULTITURN_TURN1_TEMPLATE if t == 1 else MULTITURN_FOLLOWUP_TEMPLATE).format(q)
             for t, q in enumerate(qs, start=1)]
            for qs in sessions_questions]


def _require_nonempty(value: str) -> str:
    """拦下空字符串：required=True 只拦「没传」，拦不住 --model "$EMPTY" 展开为空的情况，
    空值会一路发到服务端才报 400，提前在这里拦下。"""
    if not value or not value.strip():
        raise argparse.ArgumentTypeError("值不能为空（多半是 --model \"$MODEL\" 而 $MODEL 未定义）")
    return value


def add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", required=True, type=_require_nonempty,
                   help="服务商模型 ID")
    p.add_argument("--base-url",
                   default=os.environ.get(ENV_BASE_URL) or os.environ.get(ENV_BASE_URL_FALLBACK, ""),
                   help=f"API 地址（默认取环境变量 {ENV_BASE_URL} 或 {ENV_BASE_URL_FALLBACK}）")
    p.add_argument("--api-key",
                   default=os.environ.get(ENV_API_KEY) or os.environ.get(ENV_API_KEY_FALLBACK, ""),
                   help=f"鉴权密钥（默认取环境变量 {ENV_API_KEY} 或 {ENV_API_KEY_FALLBACK}）")
    p.add_argument("--tokenizer", default=None,
                   help="可选：transformers tokenizer 名称，用于精确 token 切片；缺省按字符估算")
    p.add_argument("--chars-per-token", type=float, default=datasets.DEFAULT_CHARS_PER_TOKEN,
                   help=f"无 tokenizer 时的字符/token 估算系数（默认 {datasets.DEFAULT_CHARS_PER_TOKEN}）")
    p.add_argument("--timeout", type=float, default=600.0, help="单请求超时秒数")
    p.add_argument("--max-completion-tokens", type=int, default=None,
                   help="模型单次输出上限（回答+思维链）；prefix 默认 512，multiturn 默认 1024")
    p.add_argument("--reasoning-effort", default="none",
                   help="思考深度参数（Ark reasoning_effort），none=关闭思维链；传空字符串则不发送该参数")
    # 厂商方言思考开关：主目标直连第三方（如阿里 enable_thinking / DeepSeek thinking）
    # 时用；缺省 None 不发送，方舟路径走 --reasoning-effort none 即可，
    # 对比目标仍走 --compare spec / peers/*.env 配置
    p.add_argument("--thinking", default=None, choices=["enabled", "disabled"],
                   help="DeepSeek 风格思考开关（disabled=关闭思维链）；直连该类厂商时用，缺省不发送")
    p.add_argument("--enable-thinking", default=None, choices=["true", "false"],
                   help="阿里 DashScope 风格思考开关（false=关闭思维链）；直连该类厂商时用，缺省不发送")
    p.add_argument("--output-param", default=None,
                   help="输出封顶字段名（缺省 max_completion_tokens）；直连只认 max_tokens "
                        "的厂商（如 DeepSeek 官方）时设为 max_tokens，否则输出不被封顶、对比失真")
    p.add_argument("--output", default=None,
                   help="可选：结果 JSON 落盘路径（逐请求数据）；"
                        "缺省写入本次运行的 reports/<mode>_<时间戳>/result_<mode>.json")
    p.add_argument("--report", default=None,
                   help="可选：Markdown 报告路径；"
                        "缺省写入 reports/<mode>_<时间戳>/report_<mode>.md")
    p.add_argument("--dump-data", default=None,
                   help="可选：只生成测试数据预览 JSON（每请求/每轮的 token 数+首尾预览）"
                        "并退出，不发送任何请求，无需 API key；用于压测前检查数据形态")


def add_compare_args(p: argparse.ArgumentParser) -> None:
    """多模型同步对比参数（prefix / multiturn 可用，connectivity 不适用）。"""
    p.add_argument("--label", default=None,
                   help="可选：主模型在报告/图例中的显示名（缺省用模型 ID）")
    p.add_argument("--compare", action="append", default=None, metavar="SPEC",
                   help="同步对比的友商模型（可重复传入多个）。SPEC 为分号分隔的 "
                        "key=value：label=显示名（缺省用 model）、model=模型 ID（必填）、"
                        "base_url=API 地址（缺省沿用主模型）、api_key=密钥 或 "
                        "api_key_env=环境变量名（二选一）、reasoning_effort / "
                        "max_completion_tokens（缺省沿用主模型，reasoning_effort 传空表示不发送）、"
                        "output_param=输出封顶字段名（缺省 max_completion_tokens；"
                        "DeepSeek 等只认 max_tokens 的厂商须设为 max_tokens，否则输出不封顶）、"
                        "thinking=DeepSeek 风格思考开关 disabled/enabled（缺省不发送）、"
                        "enable_thinking=阿里 DashScope 风格思考开关 true/false（缺省不发送；"
                        "阿里侧 DeepSeek-V4 默认开思考，对齐关闭口径须显式 false）。"
                        "示例：--compare 'label=友商A;model=qwen-max;"
                        "base_url=https://dashscope.example.com/v1;api_key_env=DASHSCOPE_KEY'")


# ---------------------------------------------------------------- 对比目标

@dataclass
class Target:
    """一个压测目标：主模型或某个对比模型，各自独立的 URL/密钥/参数。"""
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
                     f"reasoning_effort/max_completion_tokens/output_param/thinking/"
                     f"enable_thinking）")
        fields[k.strip()] = v.strip()
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
    # 缺省沿用主模型参数；reasoning_effort 传空值表示对该模型不发送该参数
    reasoning = (fields["reasoning_effort"] if "reasoning_effort" in fields
                 else args.reasoning_effort)
    try:
        mct = int(fields["max_completion_tokens"]) if fields.get("max_completion_tokens") else default_mct
    except ValueError:
        sys.exit(f"[compare] max_completion_tokens 需为整数，"
                 f"收到 {fields['max_completion_tokens']!r}（对比目标 {model}）")
    # 输出封顶参数字段名：DeepSeek 等厂商只认 max_tokens（不认则静默忽略封顶，对比失真）
    cap_field = fields.get("output_param") or "max_completion_tokens"
    # DeepSeek 风格思考开关：thinking=disabled/enabled（缺省不发送）
    thinking = fields.get("thinking") or None
    # 阿里 DashScope 风格思考开关：enable_thinking=true/false（缺省不发送）。
    # 阿里侧 DeepSeek-V4 默认开思考，对齐 reasoning 关闭口径须显式 false
    et = fields.get("enable_thinking", "")
    enable_thinking = (et.lower() == "true") if et in ("true", "false") else None
    if et and enable_thinking is None:
        sys.exit(f"[compare] enable_thinking 仅接受 true/false，收到 {et!r}")
    return Target(fields.get("label") or model, model,
                  Engine(peer_url, peer_key, timeout_sec=args.timeout,
                         reasoning_effort=reasoning,
                         max_completion_tokens=mct,
                         output_cap_field=cap_field,
                         thinking=thinking,
                         enable_thinking=enable_thinking))


def build_targets(args: argparse.Namespace) -> List[Target]:
    """主模型 + 全部 --compare 对比目标。所有目标共用同一样本数据与并发上限，
    由调用方 asyncio.gather 同步并发发起。"""
    if not args.base_url or not args.api_key:
        sys.exit(f"请先 export {ENV_API_KEY}（或 {ENV_API_KEY_FALLBACK}）和 "
                 f"{ENV_BASE_URL}（或用 --base-url/--api-key 传入）")
    mct = args.max_completion_tokens
    if mct is None:
        mct = 1024 if args.cmd == "multiturn" else MAX_COMPLETION_TOKENS
    # 主目标（方舟）只走 reasoning_effort=none；直连第三方时可显式给方言开关
    # （--thinking / --enable-thinking），对比目标仍走 --compare spec（peers/*.env）
    targets = [Target(args.label or args.model, args.model,
                      Engine(args.base_url, args.api_key, timeout_sec=args.timeout,
                             reasoning_effort=args.reasoning_effort,
                             max_completion_tokens=mct,
                             output_cap_field=args.output_param or "max_completion_tokens",
                             thinking=args.thinking,
                             enable_thinking=(args.enable_thinking == "true"
                                               if args.enable_thinking is not None
                                               else None)))]
    for spec in args.compare or []:
        t = parse_compare_spec(spec, args, mct)
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
                  f"reasoning_effort={t.engine.reasoning_effort or '（不发送）'}")
        print("[compare] 所有模型共用同一份样本数据与并发上限，同一时间窗内并发发起")


def check_env(args: argparse.Namespace) -> Engine:
    if not args.base_url or not args.api_key:
        sys.exit(f"请先 export {ENV_API_KEY}（或 {ENV_API_KEY_FALLBACK}）和 "
                 f"{ENV_BASE_URL}（或用 --base-url/--api-key 传入）")
    mct = args.max_completion_tokens
    if mct is None:
        mct = 1024 if args.cmd == "multiturn" else MAX_COMPLETION_TOKENS
    return Engine(args.base_url, args.api_key, timeout_sec=args.timeout,
                  reasoning_effort=args.reasoning_effort, max_completion_tokens=mct,
                  output_cap_field=args.output_param or "max_completion_tokens",
                  thinking=args.thinking,
                  enable_thinking=(args.enable_thinking == "true"
                                   if args.enable_thinking is not None else None))


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
                 "请检查该目标的思考关闭参数方言是否被接受")
    msg = "\n".join(lines)
    print(msg, file=sys.stderr)
    return msg


def collect_metrics(results: List[RequestResult]) -> RequestMetrics:
    m = RequestMetrics()
    for r in results:
        if r.ok:
            m.add_success(r.ttft, r.e2e, r.output_tokens, r.prompt_tokens,
                          r.cached_tokens, r.has_cache_field)
        elif r.throttled:
            m.add_failure()  # 429 计为失败（对齐 vLLM：影响成功率）；时延统计仅含成功请求
        else:
            m.add_failure()
    return m


def build_report(mode: str, model: str, m: RequestMetrics,
                 params: Dict[str, str],
                 start_time, elapsed: float, peak_concurrency: int) -> Report:
    """汇总指标，返回 Report（可渲染控制台/Markdown）。"""
    rep = Report(mode, model, params, start_time)
    rep.metrics = m
    rep.elapsed_sec = elapsed
    rep.peak_concurrency = peak_concurrency
    return rep


def emit_report(rep: Report, args: argparse.Namespace, path: Optional[str] = None) -> None:
    """控制台打印 + Markdown 落盘（path 可指定，缺省自动生成）。"""
    rep.render_console()
    path = path or args.report or default_report_path(rep.mode, rep.start_time)
    rep.write_markdown(path)


def new_dataset_rng() -> "tuple[random.Random, int]":
    """每轮压测生成独立的数据 seed（时间熵，无 CLI 参数）。

    对齐 vLLM benchmark_serve 的随机采样思想：语料切片起点由 seed 随机化，
    多轮压测之间取材不同，避免上一轮的 KV cache 污染本轮冷启动测量。
    seed 打印并写进报告参数，指标异常时可用同 seed 复现当时的确切数据。
    """
    seed = random.randrange(1 << 32)
    print(f"[dataset] 本轮语料采样 seed={seed}（写入报告，可用于复现数据）")
    return random.Random(seed), seed


def _check_positive(args: argparse.Namespace, *names: str, cmd: str) -> None:
    """压测规模参数必须为正整数：0/负数会在深处炸出裸 traceback
    （如 num_prefixes=0 触发 ZeroDivisionError），提前拦下给出可读报错。"""
    for n in names:
        if getattr(args, n) < 1:
            sys.exit(f"[{cmd}] --{n.replace('_', '-')} 必须为正整数，收到 {getattr(args, n)}")


def _thinking_desc(e: Engine) -> str:
    """一个目标实际发送的思考控制参数（对比报告逐目标展示用）。"""
    parts = []
    if e.reasoning_effort:
        parts.append(f"reasoning_effort={e.reasoning_effort}")
    if e.thinking:
        parts.append(f"thinking={e.thinking}")
    if e.enable_thinking is not None:
        parts.append(f"enable_thinking={'true' if e.enable_thinking else 'false'}")
    return "; ".join(parts) if parts else "未发送思考参数"


def compare_thinking_params(targets: List[Target]) -> str:
    """对比报告用：逐目标列出实际生效的思考口径，避免读者误以为
    头部的 reasoning_effort（主目标值）适用于所有对比目标。"""
    return "；".join(f"{t.label}: {_thinking_desc(t.engine)}" for t in targets)


def common_params(args: argparse.Namespace, slicer: "datasets.TokenSlicer",
                  max_completion_tokens: int, dataset_seed: int) -> Dict[str, str]:
    return {
        "API 地址": args.base_url,
        "长度口径": f"{slicer.backend}（精确切片）" if not slicer.approx
                   else f"估算 ~{args.chars_per_token} chars/token",
        "max_completion_tokens": str(max_completion_tokens),
        "reasoning_effort": args.reasoning_effort or "（未发送）",
        "语料采样 seed": str(dataset_seed),
    }


def _result_dict(r: RequestResult) -> Dict:
    return {
        "ok": r.ok, "error": r.error, "ttft_s": r.ttft, "e2e_s": r.e2e,
        "output_tokens": r.output_tokens, "prompt_tokens": r.prompt_tokens,
        "cached_tokens": r.cached_tokens, "has_cache_field": r.has_cache_field,
        "finish_reason": r.finish_reason,
        "turn": r.turn, "session_id": r.session_id, "prefix_id": r.prefix_id,
        "throttled": r.throttled,
    }


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
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"\n[output] 逐请求数据已写入 {path}")


def _preview(text: str, n: int = 120) -> Dict[str, str]:
    return {"head": text[:n], "tail": text[-n:] if len(text) > n else ""}


def dump_multiturn_data(path: str, args: argparse.Namespace,
                        slicer: "datasets.TokenSlicer",
                        sessions_questions: List[List[str]],
                        dataset_seed: int) -> None:
    """multiturn 数据预览：每 session 每轮的 token 数 / 字符数 / 首尾预览。"""
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
        "mode": "multiturn",
        "params": {"initial_len": args.initial_len, "question_len": args.question_len,
                   "num_conversations": args.num_sessions, "max_turns": args.max_turns,
                   "长度口径": slicer.backend, "语料采样 seed": dataset_seed},
        "sessions": sessions,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"[dump-data] multiturn 数据预览已写入 {path}"
          f"（{args.num_sessions} conversations × {args.max_turns} turns，未发送任何请求）")


def dump_prefix_data(path: str, args: argparse.Namespace,
                     slicer: "datasets.TokenSlicer",
                     message_lists: List[List[Dict[str, str]]],
                     prefix_ids: List[int], dataset_seed: int) -> None:
    """prefix 数据预览：每请求的 prefix/suffix token 数与首尾预览。"""
    requests = []
    for i, (msgs, pid) in enumerate(zip(message_lists, prefix_ids)):
        content = msgs[0]["content"]
        requests.append({"request_id": i, "prefix_id": pid,
                         "tokens": slicer.count_tokens(content), "chars": len(content),
                         **_preview(content)})
    data = {
        "mode": "prefix",
        "params": {"prefix_len": args.prefix_len, "suffix_len": args.suffix_len,
                   "num_prefixes": args.num_prefixes, "num_requests": args.num_requests,
                   "长度口径": slicer.backend, "语料采样 seed": dataset_seed},
        "requests": requests,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"[dump-data] prefix 数据预览已写入 {path}"
          f"（{len(requests)} 请求，未发送任何请求）")


# ---------------------------------------------------------------- connectivity

async def cmd_connectivity(args: argparse.Namespace) -> None:
    engine = check_env(args)
    payload_msgs = [{"role": "user", "content": "你好，请回复：连通性正常"}]
    async with aiohttp.ClientSession() as sess:
        # 用 engine 的实际配置发请求：连通性自检的 payload 形态须与正式压测一致，
        # 否则对严格校验字段的第三方厂商会误报失败
        r = await send_chat(sess, engine.chat_url, engine.headers, args.model,
                            payload_msgs, timeout_sec=args.timeout,
                            reasoning_effort=engine.reasoning_effort,
                            max_completion_tokens=engine.max_completion_tokens,
                            output_cap_field=engine.output_cap_field,
                            thinking=engine.thinking,
                            enable_thinking=engine.enable_thinking)
    if not r.ok:
        sys.exit(f"连通性失败: {r.error}")
    print(f"连通性 OK  model={args.model}")
    print(f"  TTFT = {fmt_ms(r.ttft)} ms   E2E = {fmt_ms(r.e2e)} ms")
    print(f"  prompt_tokens = {r.prompt_tokens}  completion_tokens = {r.output_tokens}")
    if r.had_reasoning:
        # 思考关不掉会让 TTFT/E2E 混入思维链时延，口径直接失真
        print("  ⚠️ 检测到 reasoning_content：思考未关闭！该厂商忽略了当前关闭参数，"
              "请换用其方言（--reasoning-effort none / --thinking disabled / "
              "--enable-thinking false，直连时按厂商选一个）后重试；"
              "强行压测口径不可比")
        sys.exit(1)
    print("  ✓ 未出现 reasoning_content（思考已关闭）")
    if r.has_cache_field:
        hit = r.cached_tokens / r.prompt_tokens if r.prompt_tokens else 0.0
        print(f"  cached_tokens = {r.cached_tokens}  命中率 = {fmt_ratio(hit)}")
    else:
        print("  ⚠️ 未读到 usage.prompt_tokens_details.cached_tokens 字段（冷请求可能为 0/缺失）")


# ---------------------------------------------------------------- prefix

async def cmd_prefix(args: argparse.Namespace) -> None:
    start_time = datetime.datetime.now()
    _check_positive(args, "num_prefixes", "num_requests", cmd="prefix")
    if args.num_requests % args.num_prefixes != 0:
        print(f"[prefix] ⚠️ num_requests={args.num_requests} 不是 num_prefixes="
              f"{args.num_prefixes} 的整数倍，各前缀复用次数不等，"
              f"按复用序号的命中率/时延曲线末尾样本量会偏少", file=sys.stderr)
    # 语料池按需扩容：随机旋转起点需要 2 倍取材量，保证 prefix 区 + suffix 区
    # 不回绕（回绕会造成跨请求额外缓存命中），且每轮起点有充分随机空间
    need_tokens = (args.num_prefixes * args.prefix_len + args.num_requests * args.suffix_len) * 2
    need_chars = int(need_tokens * args.chars_per_token * 1.2) + 1_000_000
    pool = datasets.ensure_corpus(pool_chars=need_chars)
    slicer = datasets.TokenSlicer(args.tokenizer, args.chars_per_token)
    rng, dataset_seed = new_dataset_rng()
    message_lists, prefix_ids = datasets.build_prefix_requests(
        pool, slicer, args.num_prefixes, args.prefix_len, args.suffix_len,
        args.num_requests, rng=rng)
    if args.dump_data:  # 只看数据形态，不发请求（也不需要 API key）
        dump_prefix_data(args.dump_data, args, slicer, message_lists, prefix_ids,
                         dataset_seed)
        return
    targets = build_targets(args)
    tags = [{"prefix_id": pid} for pid in prefix_ids]

    # 与 usage 口径对齐：prompt_tokens 每请求都计完整 prompt（前缀复用 N 次就计 N 次），
    # 另有每请求 ≤max_completion_tokens 的输出
    est_input = args.num_requests * (args.prefix_len + args.suffix_len)
    est_output = args.num_requests * targets[0].engine.max_completion_tokens  # 输出上限，实测通常低于
    est = est_input + est_output
    print(f"[prefix] model={args.model} num_requests={args.num_requests} "
          f"num_prefixes={args.num_prefixes} prefix_len={args.prefix_len} "
          f"suffix_len={args.suffix_len} max_concurrency={args.max_concurrency} "
          f"(总 token 估算 ~{est} = 输入 ~{est_input} + 输出上限 ~{est_output}，"
          f"多模型对比时为单模型口径)")
    print_targets(targets)

    t0 = time.perf_counter()
    # 多模型共用同一份 message_lists：样本数据逐字节一致；各自独立并发池同步发起
    all_results = await asyncio.gather(*(
        t.engine.run_requests(t.model, message_lists,
                              args.max_concurrency, tags)
        for t in targets))
    print(f"[prefix] finished in {time.perf_counter() - t0:.1f}s")

    if len(targets) == 1:
        results = all_results[0]
        m = collect_metrics(results)
        params = common_params(args, slicer,
                               targets[0].engine.max_completion_tokens,
                               dataset_seed) | {
            "prefix_len": str(args.prefix_len), "suffix_len": str(args.suffix_len),
            "num_prefixes": str(args.num_prefixes), "num_requests": str(args.num_requests),
            "max_concurrency": str(args.max_concurrency),
        }
        rep = build_report("prefix", args.model, m, params, start_time,
                           time.perf_counter() - t0,
                           targets[0].engine.peak_concurrency)
        w = warn_reasoning(args.label or args.model, results)
        if w:
            rep.warnings.append(w)
        report_path = args.report or default_report_path(rep.mode, rep.start_time)
        rep.images.extend(report_lib.prefix_charts(results, report_path,
                                                  args.label or args.model))
        emit_report(rep, args, report_path)
        dump_output(args.output or report_lib.default_output_path(report_path, "prefix"),
                    "prefix", results)
        return

    # ---- 多模型对比报告
    rep = ComparisonReport("prefix",
                           common_params(args, slicer,
                                         targets[0].engine.max_completion_tokens,
                                         dataset_seed) | {
                               "prefix_len": str(args.prefix_len),
                               "suffix_len": str(args.suffix_len),
                               "num_prefixes": str(args.num_prefixes),
                               "num_requests": str(args.num_requests),
                               "max_concurrency": str(args.max_concurrency),
                               # 逐目标列出实际生效的思考口径：头部 reasoning_effort
                               # 是主目标值，各对比目标可能各自覆盖（如阿里用
                               # enable_thinking=false、DeepSeek 不发送），不标注会误读
                               "各目标思考口径": compare_thinking_params(targets),
                           }, start_time)
    report_path = args.report or default_report_path("prefix", rep.start_time)
    series = []
    for t, results in zip(targets, all_results):
        w = warn_reasoning(t.label, results)
        if w:
            rep.warnings.append(w)
        rep.add(t.label, t.model, collect_metrics(results),
                time.perf_counter() - t0, t.engine.peak_concurrency)
        series.append((t.label, results))
    rep.images.extend(report_lib.prefix_charts_compare(series, report_path))
    rep.render_console()
    rep.write_markdown(report_path)
    dump_output(args.output or report_lib.default_output_path(report_path, "prefix"),
                "prefix", all_results[0],
                targets=[(t.label, t.model, rs)
                         for t, rs in zip(targets, all_results)])


# ---------------------------------------------------------------- multiturn

# 429 限流 / 瞬时连接错误重试：这两类是配额/网络问题而非服务端性能，
# 重试保住会话连续性（一次失败就截断会话会让轮次曲线尾部样本悄悄变少）。
# timeout / HTTP 5xx 不重试：可能是真实性能故障，重试会掩盖问题、拉偏统计。
TRANSIENT_RETRIES = 3
TRANSIENT_BACKOFF_SEC = (2.0, 4.0, 8.0)


def _transient_error(r: RequestResult) -> bool:
    """可重试的瞬时错误：429 限流 或 连接建立失败（DNS/TCP/TLS 抖动）。"""
    return r.throttled or r.error.startswith("connection error")


async def run_multiturn(engine: Engine, model: str, max_concurrency: int,
                        sessions_questions: List[List[str]],
                        progress: bool = True,
                        log_tag: str = "multiturn"
                        ) -> "tuple[List[RequestResult], int]":
    """滚动调度：会话为并发单位，跑完一个立刻补位。
    全体会话共享一个 keep-alive 连接池会话（TCPConnector limit=并发数）。

    429 / 连接错误每轮最多重试 TRANSIENT_RETRIES 次（指数退避）；重试期间该会话
    阻塞等待，不发起新请求，重试成功的请求只保留最终成功那次（不污染时延统计）。
    多模型对比时 log_tag 加模型标签，区分交错的进度日志。
    返回 (results, transient_retries)：retries 为实际发生的重试次数。"""
    sem = asyncio.Semaphore(max_concurrency)
    results: List[RequestResult] = []
    transient_retries = 0

    async def one_session(sid: int, questions: List[str],
                          sess: aiohttp.ClientSession) -> None:
        nonlocal transient_retries
        messages: List[Dict[str, str]] = []
        async with sem:
            for turn, q in enumerate(questions, start=1):
                messages = messages + [{"role": "user", "content": q}]
                for attempt in range(TRANSIENT_RETRIES + 1):
                    async with engine.inflight():
                        r = await send_chat(sess, engine.chat_url, engine.headers, model,
                                            messages, engine.timeout_sec,
                                            engine.reasoning_effort,
                                            engine.max_completion_tokens,
                                            engine.output_cap_field,
                                            engine.thinking,
                                            engine.enable_thinking)
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
                results.append(r)
                if not r.ok:
                    reason = "429 限流重试后仍未恢复" if r.throttled else (
                        "连接错误重试后仍未恢复" if _transient_error(r) else "失败")
                    print(f"[{log_tag}] session#{sid} turn{turn} {reason}: "
                          f"{r.error}，会话提前终止", file=sys.stderr)
                    break
                if progress:
                    print(f"[{log_tag}] session#{sid} turn{turn} "
                          f"ttft={fmt_ms(r.ttft)}ms e2e={fmt_ms(r.e2e)}ms "
                          f"cached={r.cached_tokens}/{r.prompt_tokens}", flush=True)
                # 历史追加模型真实返回，保证下一轮前缀天然稳定且真实
                messages = messages + [{"role": "assistant", "content": r.completion_text}]

    async with engine.client(max_concurrency) as sess:
        await asyncio.gather(*(one_session(sid, qs, sess)
                               for sid, qs in enumerate(sessions_questions)))
    return results, transient_retries


async def cmd_multiturn(args: argparse.Namespace) -> None:
    start_time = datetime.datetime.now()
    _check_positive(args, "num_sessions", "max_turns", cmd="multiturn")
    slicer = datasets.TokenSlicer(args.tokenizer, args.chars_per_token)

    # 每会话消耗 ~ initial_len + (max_turns-1)*question_len tokens 的语料；
    # 池按需扩容保证不同会话取材互不相同（相同会造成跨会话缓存命中，虚高命中率）
    per_session = args.initial_len + max(0, args.max_turns - 1) * args.question_len
    need_chars = int(args.num_sessions * per_session * args.chars_per_token * 1.2) + 1_000_000
    pool = datasets.ensure_corpus(pool_chars=need_chars)
    rng, dataset_seed = new_dataset_rng()
    step = slicer.chars_for(per_session)
    span = max(1, len(pool) - step)
    sessions_questions = []
    if args.num_sessions > span:
        # 语料池太小放不下互不重叠的取材窗：允许重叠（首 token 不同即不产生
        # 前缀命中），但仍去重起点，避免两 session 逐字节相同、命中率虚高
        print(f"[multiturn] ⚠️ num_sessions={args.num_sessions} 超过可用起点数 "
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
        dump_multiturn_data(args.dump_data, args, slicer, sessions_questions,
                            dataset_seed)
        return
    targets = build_targets(args)

    print(f"[multiturn] model={args.model} num_conversations={args.num_sessions} "
          f"max_turns={args.max_turns} initial_len={args.initial_len} "
          f"question_len={args.question_len} "
          f"max_completion_tokens={targets[0].engine.max_completion_tokens} "
          f"max_concurrency={args.max_concurrency} "
          f"(长度口径: {slicer.backend}，"
          f"{'估算 ~%.1f chars/token' % args.chars_per_token if slicer.approx else '精确切片'})")
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
    print(f"[multiturn] 长度形态: Turn1 输入 {args.initial_len} tokens（长文档），"
          f"Turn2+ 追问各 {args.question_len} tokens，每轮回答 ≤{targets[0].engine.max_completion_tokens} tokens；"
          f"上下文主要由回答驱动累积，最终轮 prompt 估算 ≈ {est_final} tokens")
    print(f"[multiturn] 总 token 估算 ~{est} = 输入 ~{est_input}"
          f"（每轮计完整累积上下文）+ 输出上限 ~{est_output}（单模型口径）")
    print_targets(targets)
    t0 = time.perf_counter()
    # 多模型共用同一份 sessions_questions（提问逐字节一致）；历史由各自回答累积，
    # 上下文长度会随模型输出风格自然分化（属多轮真实形态）
    runs = await asyncio.gather(*(
        run_multiturn(t.engine, t.model, args.max_concurrency,
                      sessions_questions, log_tag=f"multiturn/{t.label}")
        for t in targets))
    print(f"[multiturn] finished in {time.perf_counter() - t0:.1f}s")

    if len(targets) == 1:
        results, retries = runs[0]
        m = collect_metrics(results)
        params = common_params(args, slicer,
                               targets[0].engine.max_completion_tokens,
                               dataset_seed) | {
            "initial_len": str(args.initial_len), "question_len": str(args.question_len),
            "num_conversations": str(args.num_sessions), "max_turns": str(args.max_turns),
            "max_concurrency": str(args.max_concurrency),
            "429/网络重试次数": str(retries),
            "长度形态": f"Turn1 输入 {args.initial_len} tokens，追问各 {args.question_len} tokens，"
                       f"回答各 ≤{targets[0].engine.max_completion_tokens} tokens，最终轮 ≈ {est_final} tokens",
        }
        if retries:  # 触发过重试必须在报告头部披露：重试改变了测量形态
            params["⚠️ 重试提示"] = (
                f"触发 429 限流/连接错误重试 {retries} 次，退避重试后恢复"
                "（重试等待不计入时延统计）；429 说明压测强度超服务配额（TPM/RPM），"
                "建议提额后重测再作结论")
        rep = build_report("multiturn", args.model, m, params, start_time,
                           time.perf_counter() - t0,
                           targets[0].engine.peak_concurrency)
        w = warn_reasoning(args.label or args.model, results)
        if w:
            rep.warnings.append(w)
        report_path = args.report or default_report_path(rep.mode, rep.start_time)
        rep.images.extend(report_lib.multiturn_charts(results, report_path,
                                                     args.label or args.model))
        emit_report(rep, args, report_path)
        dump_output(args.output or report_lib.default_output_path(report_path, "multiturn"),
                    "multiturn", results)
        return

    # ---- 多模型对比报告
    rep = ComparisonReport("multiturn",
                           common_params(args, slicer,
                                         targets[0].engine.max_completion_tokens,
                                         dataset_seed) | {
                               "initial_len": str(args.initial_len),
                               "question_len": str(args.question_len),
                               "num_conversations": str(args.num_sessions),
                               "max_turns": str(args.max_turns),
                               "max_concurrency": str(args.max_concurrency),
                               "长度形态": f"Turn1 输入 {args.initial_len} tokens，"
                                          f"追问各 {args.question_len} tokens，"
                                          f"回答各 ≤{targets[0].engine.max_completion_tokens} tokens，"
                                          f"最终轮 ≈ {est_final} tokens",
                               "各目标思考口径": compare_thinking_params(targets),
                               "429/网络重试次数": "; ".join(
                                   f"{t.label}={r[1]}"
                                   for t, r in zip(targets, runs)),
                               "口径说明": "各模型提问逐字节一致；多轮历史由各自回答累积，"
                                          "上下文长度随模型输出风格自然分化",
                           }, start_time)
    retried = [f"{t.label} {r[1]} 次" for t, r in zip(targets, runs) if r[1]]
    if retried:  # 任一侧触发过重试必须在报告头部披露（标注是哪一侧）
        rep.params["⚠️ 重试提示"] = (
            "；".join(retried) + " 触发 429 限流/连接错误重试，退避重试后恢复"
            "（重试等待不计入时延统计）；429 说明对应侧压测强度超服务配额（TPM/RPM），"
            "建议提额后重测再作结论")
    report_path = args.report or default_report_path("multiturn", rep.start_time)
    series = []
    for t, (results, _retries) in zip(targets, runs):
        w = warn_reasoning(t.label, results)
        if w:
            rep.warnings.append(w)
        rep.add(t.label, t.model, collect_metrics(results),
                time.perf_counter() - t0, t.engine.peak_concurrency)
        series.append((t.label, results))
    rep.images.extend(report_lib.multiturn_charts_compare(series, report_path))
    rep.render_console()
    rep.write_markdown(report_path)
    dump_output(args.output or report_lib.default_output_path(report_path, "multiturn"),
                "multiturn", runs[0][0],
                targets=[(t.label, t.model, rs)
                         for t, (rs, _retries) in zip(targets, runs)])


# ---------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser(description="轻量级 LLM Benchmark Serve 工具")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_conn = sub.add_parser("connectivity", help="连通性自检")
    add_common_args(p_conn)

    p_prefix = sub.add_parser("prefix", help="模式 A：固定前缀池复用（对标 vLLM prefix_repetition）")
    add_common_args(p_prefix)
    add_compare_args(p_prefix)
    p_prefix.add_argument("--prefix-len", type=int, default=12000, help="前缀长度（token）")
    p_prefix.add_argument("--suffix-len", type=int, default=2000, help="后缀长度（token）")
    p_prefix.add_argument("--num-prefixes", type=int, default=10, help="前缀池个数")
    p_prefix.add_argument("--num-requests", type=int, default=200, help="总请求数")
    p_prefix.add_argument("--max-concurrency", type=int, default=5, help="并发上限")

    p_mt = sub.add_parser("multiturn", help="模式 B：多轮对话动态 Cache 命中率")
    add_common_args(p_mt)
    add_compare_args(p_mt)
    p_mt.add_argument("--initial-len", type=int, default=3000,
                      help="Turn1 长输入的 token 数（模拟首轮塞入的文档/长上下文）")
    p_mt.add_argument("--question-len", type=int, default=256,
                      help="Turn2+ 每轮短追问的 token 数（真实用户提问通常很短）")
    p_mt.add_argument("--num-sessions", type=int, default=10, help="总会话数")
    p_mt.add_argument("--max-turns", type=int, default=20,
                      help="每会话轮数（Turn1 为 initial_len 长输入，Turn2+ 每轮 question_len 短追问）")
    p_mt.add_argument("--max-concurrency", type=int, default=5, help="并发会话数上限")

    args = ap.parse_args()
    fn = {"connectivity": cmd_connectivity, "prefix": cmd_prefix,
          "multiturn": cmd_multiturn}[args.cmd]
    asyncio.run(fn(args))


if __name__ == "__main__":
    main()
