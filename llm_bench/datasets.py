"""ShareGPT 数据：本地池裁剪拼装 / tokenize 降级。完全离线，不联网下载。"""
from __future__ import annotations

import os
import random
import sys
from typing import List, Optional

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
POOL_PATH = os.path.join(DATA_DIR, "sharegpt_pool.txt")

# 无 tokenizer 时的字符近似：1 token ~= 4 chars（英文为主的 ShareGPT 经验值）
DEFAULT_CHARS_PER_TOKEN = 4.0

# 语料池默认字符数（约几 MB 的文本样本；不够用时按设计循环拼接补足）
DEFAULT_POOL_CHARS = 8_000_000


def ensure_corpus(pool_chars: int = DEFAULT_POOL_CHARS) -> str:
    """返回满足 pool_chars 需求的本地语料池内容。完全离线，不做任何下载。

    - 池存在且够大 -> 直接复用
    - 不够大 -> 循环拼接现有池补足（回绕部分可能引入跨会话/跨请求的额外
      缓存命中，属已知折衷，发生时告警）
    - 池缺失 -> 直接报错（语料池随工具自带，缺失说明安装不完整）
    """
    if not os.path.exists(POOL_PATH):
        sys.exit(f"[dataset] 语料池不存在: {POOL_PATH}（随工具自带，"
                 f"请检查安装/分发是否完整；不支持联网下载）")
    with open(POOL_PATH, "r", encoding="utf-8") as f:
        existing = f.read()
    if not existing:
        sys.exit(f"[dataset] 语料池为空: {POOL_PATH}（语料文件损坏，请重新获取）")
    if len(existing) >= pool_chars:
        return existing
    reps = -(-pool_chars // len(existing))
    print(f"[dataset] ⚠️ 现有池 {len(existing)} chars < 需求 {pool_chars}，"
          f"循环拼接 {reps} 份补足（回绕可能引入跨会话的额外缓存命中）",
          file=sys.stderr, flush=True)
    return (existing * reps)[:pool_chars]


class TokenSlicer:
    """按构造 tokenizer 切片。

    显式指定 tokenizer 时使用 transformers 且加载失败即报错；否则使用本地
    tiktoken o200k_base，缺失时才退化为字符估算。这里的 token 长度只对构造
    tokenizer 精确，服务端实际长度仍以 usage.prompt_tokens 为准。
    """

    def __init__(self, tokenizer_name: Optional[str] = None,
                 chars_per_token: float = DEFAULT_CHARS_PER_TOKEN,
                 encoding_name: str = "o200k_base") -> None:
        self.chars_per_token = chars_per_token
        self.backend = "approx"
        self.tokenizer_label = f"approx:{chars_per_token:g}-chars-per-token"
        self._enc = None
        self._tok = None
        if tokenizer_name:
            try:
                from transformers import AutoTokenizer  # type: ignore
                self._tok = AutoTokenizer.from_pretrained(tokenizer_name)
            except Exception as e:  # noqa: BLE001
                raise RuntimeError(
                    f"[dataset] 无法加载指定 tokenizer {tokenizer_name!r}: {e}"
                ) from e
            self.backend = "transformers"
            self.tokenizer_label = f"transformers:{tokenizer_name}"
            print(f"[dataset] using tokenizer: {tokenizer_name}", flush=True)
            return
        try:
            import tiktoken  # type: ignore
            self._enc = tiktoken.get_encoding(encoding_name)
            self.backend = "tiktoken"
            self.tokenizer_label = f"tiktoken:{encoding_name}"
            print(f"[dataset] using tiktoken encoding: {encoding_name}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[dataset] tiktoken 不可用 ({e})，"
                  f"按 ~{chars_per_token} chars/token 估算", file=sys.stderr, flush=True)
        if self.backend == "approx":
            print(f"[dataset] 注意：长度为估算值 (~{chars_per_token} chars/token)", flush=True)

    @property
    def approx(self) -> bool:
        return self.backend == "approx"

    def encode(self, text: str) -> List[int]:
        """整段编码为 token id（精确模式；approx 模式返回空）。
        disallowed_special=()：语料里可能含 <|endoftext|> 等字面特殊 token，
        一律按普通文本编码，否则 tiktoken 直接抛错。"""
        if self._enc is not None:
            return self._enc.encode(text, disallowed_special=())
        if self._tok is not None:
            return self._tok.encode(text, add_special_tokens=False)
        return []

    def decode(self, ids: List[int]) -> str:
        if self._enc is not None:
            return self._enc.decode(ids)
        if self._tok is not None:
            return self._tok.decode(ids)
        return ""

    def count_tokens(self, text: str) -> int:
        """校验用：数一段文本的 token 数。"""
        if self._enc is not None:
            return len(self._enc.encode(text, disallowed_special=()))
        if self._tok is not None:
            return len(self._tok.encode(text, add_special_tokens=False))
        return max(1, int(len(text) / self.chars_per_token))

    def chars_for(self, n_tokens: int) -> int:
        return max(1, int(n_tokens * self.chars_per_token))

    def slice_text(self, text: str, start_token: int, n_tokens: int) -> str:
        """从 text 的 start_token 起切 n_tokens 个 token。"""
        # 统一走 self.encode()：与 count_tokens 一致地带 disallowed_special=()
        # （tiktoken 遇到语料里的 <...> 字面特殊 token 会直接抛错，不能裸调 encode）
        if self._enc is not None or self._tok is not None:
            ids = self.encode(text)
            seg = ids[start_token:start_token + n_tokens]
            if not seg:  # 越界取尾部
                seg = ids[-n_tokens:]
            return self.decode(seg)
        c0 = self.chars_for(start_token)
        c1 = self.chars_for(start_token + n_tokens)
        if c0 >= len(text):
            c0 = max(0, len(text) - (c1 - c0))
            c1 = len(text)
        return text[c0:min(c1, len(text))]

    def verify_equal_lens(self, pieces: List[str], expected: int, what: str) -> None:
        """校验一组切片 token 长度是否都等于 expected，不等则告警。

        容差 ±1：切片在 token id 层面是构造性精确的，但校验是对 decode 出的
        文本重新 encode，BPE 在任意切点可能有往返不守恒（切点两侧 token 重编码
        时合并/拆分，±1）。偏差超过 1 才是真正的异常（如语料回绕切错位）。"""
        if self.approx:
            return
        lens = [self.count_tokens(p) for p in pieces]
        bad = [(i, l) for i, l in enumerate(lens) if abs(l - expected) > 1]
        if bad:
            print(f"[dataset] ⚠️ {what} 长度校验不一致（期望 {expected}，容差 ±1）: {bad[:5]}",
                  file=sys.stderr, flush=True)
        else:
            drift = sum(1 for l in lens if l != expected)
            note = f"（{drift} 段有 ±1 BPE 往返抖动，属正常）" if drift else ""
            print(f"[dataset] {what} 长度校验通过: {len(pieces)} 段 × {expected} tokens{note}",
                  flush=True)


def tile_pool(pool: str, need_chars: int) -> str:
    """语料池不足时循环拼接补足。"""
    if len(pool) >= need_chars:
        return pool
    reps = (need_chars + len(pool) - 1) // len(pool)
    return (pool * reps)[:need_chars]


def build_prefix_requests(pool: str, slicer: TokenSlicer, num_prefixes: int,
                          prefix_len: int, suffix_len: int,
                          num_requests: int,
                          rng: Optional[random.Random] = None) -> "tuple":
    """返回 (message_lists, prefix_ids)。prefix 专用区在前，suffix 专用区在后，互不重叠。

    rng 不为 None 时，整段取材区在语料池上随机旋转起点（vLLM 随机采样式去重）：
    分区结构不变（prefix 命中形态仍可预测），但每轮取到的具体文本不同，
    避免重复压测时上一轮的 KV cache 污染本轮冷启动测量。
    """
    prefixes: List[str]
    suffixes: List[str]

    if not slicer.approx:
        # 精确模式：encode 一次，按 token id 切片，保证每段长度严格相等
        total_tokens = num_prefixes * prefix_len + num_requests * suffix_len
        ids = slicer.encode(pool)
        if ids:  # 语料不足时循环拼接补足
            # 随机旋转需要 2 倍余量，保证起点有随机空间（池够大时通常天然满足）
            need = total_tokens * 2 if rng is not None else total_tokens
            reps = -(-need // len(ids))
            if reps > 1:
                ids = (ids * reps)[:need]
            else:
                ids = ids[:need]
            if rng is not None and len(ids) > total_tokens:
                start = rng.randrange(0, len(ids) - total_tokens)
                ids = ids[start:start + total_tokens]
        # 前 num_prefixes*prefix_len 个 token 为 prefix 专用区，其后为 suffix 专用区
        prefix_zone, suffix_zone = ids[:num_prefixes * prefix_len], ids[num_prefixes * prefix_len:]
        prefixes = [slicer.decode(prefix_zone[i * prefix_len:(i + 1) * prefix_len])
                    for i in range(num_prefixes)]
        cursor = 0
        suffixes = []
        for _ in range(num_requests):
            suffixes.append(slicer.decode(suffix_zone[cursor:cursor + suffix_len]))
            cursor += suffix_len
        slicer.verify_equal_lens(prefixes, prefix_len, "prefixes")
        slicer.verify_equal_lens(suffixes, suffix_len, "suffixes")
    else:
        total_prefix_chars = slicer.chars_for(num_prefixes * prefix_len)
        total_chars = total_prefix_chars + slicer.chars_for(num_requests * suffix_len)
        padded = tile_pool(pool, total_chars * 2 if rng is not None else total_chars)
        if rng is not None and len(padded) > total_chars:
            start = rng.randrange(0, len(padded) - total_chars)
            padded = padded[start:start + total_chars]
        prefix_zone, suffix_zone = padded[:total_prefix_chars], padded[total_prefix_chars:]
        prefixes = [slicer.slice_text(prefix_zone, i * prefix_len, prefix_len)
                    for i in range(num_prefixes)]
        cursor = 0
        suffixes = []
        suffix_zone_tokens = len(suffix_zone) // slicer.chars_per_token
        for _ in range(num_requests):
            if cursor + suffix_len > suffix_zone_tokens:
                cursor = 0  # 滚动游标，越界回到起点
            suffixes.append(slicer.slice_text(suffix_zone, cursor, suffix_len))
            cursor += suffix_len

    message_lists, prefix_ids = [], []
    for i in range(num_requests):
        pid = i % num_prefixes  # 均匀分配，命中率可预测可复现
        prefix_ids.append(pid)
        message_lists.append([
            {"role": "user", "content": prefixes[pid] + "\n\n" + suffixes[i]},
        ])
    return message_lists, prefix_ids


def build_session_questions(pool: str, slicer: TokenSlicer, max_turns: int,
                            initial_len: int, question_len: int,
                            session_chars_offset: int) -> List[str]:
    """为一个会话构造各轮用户提问（贴近真实使用形态）。

    - Turn1: initial_len tokens 的长输入（模拟首轮塞入的文档/长上下文）
    - Turn2..N: question_len tokens 的短追问（用户真实提问通常很短）
    - 累积上下文的增长主要由模型回答驱动（每轮回答 ≤ max_completion_tokens）
    - 各段从语料池连续切出，不同会话通过 session_chars_offset 错开，内容互不相同
    """
    lens = [initial_len] + [question_len] * (max_turns - 1) if max_turns > 1 else [initial_len]
    total_tokens = sum(lens)
    if not slicer.approx:
        ids = slicer.encode(pool)
        chars_per_token = max(1, int(slicer.chars_per_token))
        start = (session_chars_offset // chars_per_token) % max(1, len(ids) // 2)
        # 平铺份数必须覆盖 start 偏移后的整段窗口，否则小语料池上末尾轮次切出空串
        reps = -(-(start + total_tokens) // len(ids)) if ids else 1
        zone = (ids * reps)[start:start + total_tokens]
        questions, cursor = [], 0
        for n in lens:
            questions.append(slicer.decode(zone[cursor:cursor + n]))
            cursor += n
        return questions
    questions, cursor = [], session_chars_offset
    padded = tile_pool(pool, session_chars_offset + slicer.chars_for(total_tokens))
    for n in lens:
        c = slicer.chars_for(n)
        questions.append(padded[cursor:cursor + c])
        cursor += c
    return questions
