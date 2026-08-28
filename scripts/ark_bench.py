#!/usr/bin/env python3
"""Ark CLI adapter for the bundled llm-bench implementation."""
from __future__ import annotations

import argparse
import difflib
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


SKILL_DIR = Path(__file__).resolve().parents[1]
BENCH_SCRIPT = SKILL_DIR / "llm_bench" / "bench.py"
DEFAULT_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"


class BenchmarkError(RuntimeError):
    pass


@dataclass
class Target:
    model: str
    profile: str
    profile_type: str
    base_url: str = ""


def run_json(argv: list[str]) -> dict[str, Any]:
    proc = subprocess.run(argv, text=True, capture_output=True, check=False)
    if proc.returncode:
        message = proc.stderr.strip() or proc.stdout.strip()
        raise BenchmarkError(f"Command failed: {' '.join(argv[:3])}: {message}")
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise BenchmarkError(f"Invalid JSON from {' '.join(argv[:3])}") from exc


def check_auth() -> dict[str, Any]:
    status = run_json(["arkcli", "auth", "status", "--format", "json"])
    if not status.get("logged_in"):
        raise BenchmarkError(
            "Ark CLI is not authenticated. Run `arkcli auth login volc-sso` first."
        )
    return status


def available_profiles(auth_status: dict[str, Any]) -> list[dict[str, Any]]:
    profiles = auth_status.get("profiles_summary", [])
    if not profiles:
        profiles = run_json(
            ["arkcli", "profile", "list", "--format", "json"]
        ).get("profiles", [])
    return profiles


def select_platform_profile(
    auth_status: dict[str, Any], profile_filter: str | None = None
) -> dict[str, Any]:
    profiles = available_profiles(auth_status)
    if profile_filter:
        profiles = [item for item in profiles if item.get("name") == profile_filter]
        if not profiles:
            raise BenchmarkError(f"Ark CLI profile not found: {profile_filter}")
        if profiles[0].get("type") != "platform":
            raise BenchmarkError(
                f"Profile {profile_filter} is type={profiles[0].get('type')}; "
                "postpaid benchmarking requires a platform profile."
            )
        return profiles[0]

    platform_profiles = [
        profile for profile in profiles if profile.get("type") == "platform"
    ]
    if not platform_profiles:
        raise BenchmarkError(
            "No Ark CLI platform profile is available for postpaid API access."
        )
    return next(
        (profile for profile in platform_profiles if profile.get("is_default")),
        platform_profiles[0],
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


def model_search_query(value: str) -> str:
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
    return re.sub(r"[^a-z0-9]+", "-", value).strip("-")


def candidate_score(query: str, candidate: str) -> float:
    q = normalize_model_name(query)
    c = normalize_model_name(candidate)
    if q == c:
        return 1.0
    if q and (q in c or c in q):
        return 0.94 - min(abs(len(q) - len(c)), 20) / 200
    return difflib.SequenceMatcher(None, q, c).ratio()


def callable_model_id(item: dict[str, Any]) -> str:
    if item.get("callable_model_id"):
        return item["callable_model_id"]
    name = item.get("name", "")
    version = item.get("primary_version")
    if version and version != "latest-version":
        return f"{name}-{version}"
    return name


def discover_model_candidates(query: str, profile: str) -> list[dict[str, Any]]:
    search_query = model_search_query(query)
    if not search_query:
        raise BenchmarkError(f"Could not derive a model search term from {query!r}.")
    response = run_json(
        [
            "arkcli",
            "models",
            "search",
            search_query,
            "--profile",
            profile,
            "--modality",
            "text",
            "--size",
            "30",
            "--format",
            "json",
        ]
    )
    candidates: list[dict[str, Any]] = []
    for item in response.get("items", []):
        model_id = callable_model_id(item)
        if not model_id:
            continue
        aliases = [model_id, item.get("name", ""), item.get("display_name", "")]
        candidates.append(
            {
                "id": model_id,
                "name": item.get("name", ""),
                "display_name": item.get("display_name", ""),
                "score": max(candidate_score(query, alias) for alias in aliases),
            }
        )
    return candidates


def resolve_target(
    query: str,
    auth_status: dict[str, Any],
    profile_filter: str | None = None,
) -> Target:
    profile = select_platform_profile(auth_status, profile_filter)
    query = query.strip()
    if re.fullmatch(r"ep-[a-z0-9-]+", query, flags=re.IGNORECASE) or re.fullmatch(
        r"[a-z][a-z0-9]*(?:[-_.][a-z0-9]+)+-\d{6}",
        query,
        flags=re.IGNORECASE,
    ):
        return Target(query, profile["name"], "platform")

    ranked = sorted(
        discover_model_candidates(query, profile["name"]),
        key=lambda item: item["score"],
        reverse=True,
    )
    if not ranked:
        raise BenchmarkError(
            f"No callable text model matched {query!r} in the Ark model catalog."
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
                    "name": item["display_name"] or item["name"],
                    "score": round(item["score"], 3),
                }
                for item in ranked[:5]
            ],
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2), file=sys.stderr)
        raise SystemExit(2)
    return Target(best["id"], profile["name"], "platform")


def parse_api_key(raw: str) -> str:
    value = raw.strip()
    if not value:
        return ""
    if not value.startswith("{"):
        return value
    try:
        payload = json.loads(value)
    except json.JSONDecodeError:
        return ""
    if not isinstance(payload, dict):
        return ""
    return payload.get("api_key") or (payload.get("data") or {}).get("api_key") or ""


def get_api_key(profile: str) -> str:
    base_command = [
        "arkcli",
        "profile",
        "apikey",
        "get",
        "--profile",
        profile,
    ]
    errors: list[str] = []
    for output_flag in (["--plain"], ["--format", "json"]):
        proc = subprocess.run(
            base_command + output_flag,
            text=True,
            capture_output=True,
            check=False,
        )
        key = parse_api_key(proc.stdout)
        if proc.returncode == 0 and key:
            return key
        if proc.stderr.strip():
            errors.append(proc.stderr.strip())
    detail = errors[-1] if errors else "empty credential response"
    detail = re.sub(r"ark-[A-Za-z0-9-]+", "<redacted>", detail)
    raise BenchmarkError(
        f"Ark CLI could not provide a key for profile {profile}: {detail}. "
        "Refresh authentication with `arkcli auth login volc-sso`."
    )


def load_profile_access(profile: str) -> tuple[str, str]:
    with ThreadPoolExecutor(max_workers=2) as executor:
        profile_future = executor.submit(
            run_json,
            ["arkcli", "profile", "show", profile, "--format", "json"],
        )
        key_future = executor.submit(get_api_key, profile)
        profile_data = profile_future.result()
        api_key = key_future.result()
    return profile_data.get("base_url") or DEFAULT_BASE_URL, api_key


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
            "initial_len": 3000,
            "question_len": 256,
            "num_sessions": 10,
            "max_turns": 20,
            "max_concurrency": 5,
        },
    }
    for key, value in presets[args.preset].items():
        if getattr(args, key) is None:
            setattr(args, key, value)


def scenario_output_tokens(args: argparse.Namespace, scenario: str) -> int:
    if args.max_output_tokens is not None:
        return args.max_output_tokens
    return 1024 if scenario == "multiturn" else 512


def run_process(
    command: list[str],
    env: dict[str, str],
    log_path: Path,
    secret: str,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(
        command,
        cwd=Path.cwd(),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    lines: list[str] = []
    assert proc.stdout is not None
    for raw_line in proc.stdout:
        line = raw_line.replace(secret, "<redacted>")
        print(line, end="", flush=True)
        lines.append(line)
    proc.stdout.close()
    return_code = proc.wait()
    log_path.write_text("".join(lines), encoding="utf-8")
    if return_code:
        raise BenchmarkError(
            f"llm-bench exited with code {return_code}; log: {log_path}"
        )


def command_base(args: argparse.Namespace, target: Target, scenario: str) -> list[str]:
    return [
        sys.executable,
        str(BENCH_SCRIPT),
        scenario,
        "--model",
        target.model,
        "--reasoning-effort",
        args.reasoning_effort,
        "--timeout",
        str(args.timeout),
        "--max-completion-tokens",
        str(scenario_output_tokens(args, scenario)),
    ]


def execute(args: argparse.Namespace) -> dict[str, Any]:
    if not BENCH_SCRIPT.exists():
        raise BenchmarkError(f"Bundled llm-bench is missing: {BENCH_SCRIPT}")
    auth_status = check_auth()
    target = resolve_target(args.model, auth_status, args.profile)
    target.base_url, api_key = load_profile_access(target.profile)

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
                "engine": "llm-bench",
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

    env = os.environ.copy()
    env["ARK_BASE_URL"] = target.base_url
    env["ARK_API_KEY"] = api_key
    warning_filter = "ignore:::urllib3"
    env["PYTHONWARNINGS"] = ",".join(
        item for item in (env.get("PYTHONWARNINGS"), warning_filter) if item
    )
    output_root = Path(args.output_dir).expanduser().resolve()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_paths: list[str] = []

    for scenario in scenarios:
        run_dir = output_root / f"{scenario}_{timestamp}"
        command = command_base(args, target, scenario)
        if scenario == "connectivity":
            log_path = run_dir / "connectivity.log"
            run_process(command, env, log_path, api_key)
            report_path = run_dir / "report.md"
            log = log_path.read_text(encoding="utf-8")
            report_path.write_text(
                "# LLM Benchmark Connectivity\n\n"
                f"- Model: `{target.model}`\n"
                f"- Profile: `{target.profile}` (`platform`, postpaid)\n"
                f"- API: `{target.base_url}`\n\n"
                "```text\n"
                f"{log.rstrip()}\n"
                "```\n",
                encoding="utf-8",
            )
        elif scenario == "prefix":
            report_path = run_dir / "report_prefix.md"
            command.extend(
                [
                    "--prefix-len",
                    str(args.prefix_len),
                    "--suffix-len",
                    str(args.suffix_len),
                    "--num-prefixes",
                    str(args.num_prefixes),
                    "--num-requests",
                    str(args.num_requests),
                    "--max-concurrency",
                    str(args.max_concurrency),
                    "--report",
                    str(report_path),
                    "--output",
                    str(run_dir / "result_prefix.json"),
                ]
            )
            run_process(command, env, run_dir / "run.log", api_key)
        else:
            report_path = run_dir / "report_multiturn.md"
            command.extend(
                [
                    "--initial-len",
                    str(args.initial_len),
                    "--question-len",
                    str(args.question_len),
                    "--num-sessions",
                    str(args.num_sessions),
                    "--max-turns",
                    str(args.max_turns),
                    "--max-concurrency",
                    str(args.max_concurrency),
                    "--report",
                    str(report_path),
                    "--output",
                    str(run_dir / "result_multiturn.json"),
                ]
            )
            run_process(command, env, run_dir / "run.log", api_key)
        report_paths.append(str(report_path))
        print(
            json.dumps(
                {
                    "event": "scenario_complete",
                    "scenario": scenario,
                    "report": str(report_path),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    return {
        "ok": True,
        "engine": "llm-bench",
        "model": target.model,
        "profile": target.profile,
        "reports": report_paths,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run bundled llm-bench with Ark CLI-managed postpaid credentials."
    )
    parser.add_argument("--model", required=True, help="Model name, alias, or endpoint ID")
    parser.add_argument("--profile", help="Ark CLI platform profile")
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
    return parser


def main() -> int:
    args = build_parser().parse_args()
    apply_preset(args)
    try:
        result = execute(args)
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
