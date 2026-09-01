import importlib.util
import io
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


SKILL_DIR = Path(__file__).parents[1]
MODULE_PATH = SKILL_DIR / "scripts" / "ark_bench.py"
SPEC = importlib.util.spec_from_file_location("ark_bench", MODULE_PATH)
ark_bench = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ark_bench
SPEC.loader.exec_module(ark_bench)

LLM_BENCH_DIR = SKILL_DIR / "llm_bench"
sys.path.insert(0, str(LLM_BENCH_DIR))
import datasets as llm_datasets
import engine as llm_engine
import metrics as llm_metrics


class ModelResolutionTests(unittest.TestCase):
    def test_normalizes_chinese_aliases(self):
        actual = ark_bench.normalize_model_name("豆包 Seed 2.0 迷你模型")
        self.assertEqual(actual, "doubaoseed20mini")
        self.assertEqual(
            ark_bench.model_search_query("豆包 Seed 2.0 迷你模型"),
            "doubao-seed-2-0-mini",
        )

    def test_exact_candidate_has_highest_score(self):
        query = "豆包 seed 2.0 mini"
        exact = ark_bench.candidate_score(
            query, "doubao-seed-2-0-mini-260428"
        )
        other = ark_bench.candidate_score(
            query, "doubao-seed-2-0-lite-260428"
        )
        self.assertGreater(exact, other)

    def test_selects_platform_profile_instead_of_agent_plan(self):
        auth_status = {
            "profiles_summary": [
                {
                    "name": "agent-plan_cn-beijing_personal",
                    "type": "agent-plan",
                    "is_default": False,
                },
                {
                    "name": "platform_cn-beijing_accountwide",
                    "type": "platform",
                    "is_default": True,
                },
            ]
        }
        profile = ark_bench.select_platform_profile(auth_status)
        self.assertEqual(profile["name"], "platform_cn-beijing_accountwide")

    def test_rejects_explicit_agent_plan_profile(self):
        auth_status = {
            "profiles_summary": [
                {
                    "name": "agent-plan_cn-beijing_personal",
                    "type": "agent-plan",
                }
            ]
        }
        with self.assertRaisesRegex(
            ark_bench.BenchmarkError, "postpaid benchmarking requires"
        ):
            ark_bench.select_platform_profile(
                auth_status, "agent-plan_cn-beijing_personal"
            )

    def test_resolves_catalog_model_to_postpaid_model_id(self):
        auth_status = {
            "profiles_summary": [
                {
                    "name": "platform_cn-beijing_accountwide",
                    "type": "platform",
                    "is_default": True,
                }
            ]
        }
        catalog = {
            "items": [
                {
                    "name": "doubao-seed-2-0-mini",
                    "display_name": "Doubao-Seed-2.0-mini",
                    "primary_version": "260428",
                },
                {
                    "name": "doubao-seed-2-0-lite",
                    "display_name": "Doubao-Seed-2.0-lite",
                    "primary_version": "260428",
                },
            ]
        }
        with patch.object(ark_bench, "run_json", return_value=catalog):
            target = ark_bench.resolve_target(
                "豆包 Seed 2.0 Mini", auth_status
            )
        self.assertEqual(target.model, "doubao-seed-2-0-mini-260428")
        self.assertEqual(target.profile_type, "platform")

    def test_explicit_model_id_skips_catalog_lookup(self):
        auth_status = {
            "profiles_summary": [
                {
                    "name": "platform_cn-beijing_accountwide",
                    "type": "platform",
                }
            ]
        }
        with patch.object(ark_bench, "run_json") as mocked:
            target = ark_bench.resolve_target(
                "doubao-seed-2-0-mini-260428", auth_status
            )
        mocked.assert_not_called()
        self.assertEqual(target.model, "doubao-seed-2-0-mini-260428")


class AdapterTests(unittest.TestCase):
    def test_missing_sso_error_includes_internal_login_flow(self):
        with patch.object(
            ark_bench, "run_json", return_value={"logged_in": False}
        ):
            with self.assertRaises(ark_bench.BenchmarkError) as caught:
                ark_bench.check_auth()
        message = str(caught.exception)
        self.assertIn("babi.bytedance.net/finance/basic/volcManage/", message)
        self.assertIn("arkcli auth login volc-sso", message)

    def test_api_key_plain_output(self):
        result = SimpleNamespace(returncode=0, stdout="test-key\n", stderr="")
        with patch.object(ark_bench.subprocess, "run", return_value=result) as mocked:
            key = ark_bench.get_api_key("platform-profile")
        self.assertEqual(key, "test-key")
        self.assertIn("--plain", mocked.call_args.args[0])

    def test_api_key_falls_back_to_json_when_plain_is_unsupported(self):
        unsupported = SimpleNamespace(
            returncode=1,
            stdout="",
            stderr="unknown flag: --plain",
        )
        json_result = SimpleNamespace(
            returncode=0,
            stdout='{"api_key":"test-key","profile":"platform-profile"}',
            stderr="",
        )
        with patch.object(
            ark_bench.subprocess,
            "run",
            side_effect=[unsupported, json_result],
        ) as mocked:
            key = ark_bench.get_api_key("platform-profile")
        self.assertEqual(key, "test-key")
        self.assertEqual(mocked.call_count, 2)
        self.assertEqual(mocked.call_args.args[0][-2:], ["--format", "json"])

    def test_standard_preset_matches_bundled_llm_bench(self):
        args = ark_bench.build_parser().parse_args(["--model", "example"])
        ark_bench.apply_preset(args)
        self.assertEqual(args.prefix_len, 12000)
        self.assertEqual(args.suffix_len, 2000)
        self.assertEqual(args.num_requests, 200)
        self.assertEqual(args.initial_len, 3000)
        self.assertEqual(args.num_sessions, 10)
        self.assertEqual(args.max_turns, 20)
        self.assertEqual(args.max_concurrency, 5)
        self.assertEqual(
            ark_bench.scenario_output_tokens(args, "multiturn"), 1024
        )

    def test_quick_preset_is_low_cost(self):
        args = ark_bench.build_parser().parse_args(
            ["--model", "example", "--preset", "quick"]
        )
        ark_bench.apply_preset(args)
        self.assertEqual(args.num_requests, 6)
        self.assertEqual(args.num_sessions * args.max_turns, 6)
        self.assertEqual(args.max_output_tokens, 64)

    def test_command_uses_bundled_bench_without_api_key_argument(self):
        args = ark_bench.build_parser().parse_args(["--model", "example"])
        ark_bench.apply_preset(args)
        target = ark_bench.Target(
            "doubao-seed-2-0-mini-260428",
            "platform_cn-beijing_accountwide",
            "platform",
            "https://ark.cn-beijing.volces.com/api/v3",
        )
        command = ark_bench.command_base(args, target, "prefix")
        self.assertIn(str(LLM_BENCH_DIR / "bench.py"), command)
        self.assertIn("prefix-repetition", command)
        self.assertNotIn("prefix", command)
        self.assertNotIn("--api-key", command)
        self.assertNotIn("--reasoning-effort", command)
        self.assertEqual(command[command.index("--thinking") + 1], "disabled")

    def test_process_log_redacts_secret(self):
        with tempfile.TemporaryDirectory() as directory:
            log_path = Path(directory) / "run.log"
            with redirect_stdout(io.StringIO()):
                ark_bench.run_process(
                    [sys.executable, "-c", "print('ark-secret-value')"],
                    os.environ.copy(),
                    log_path,
                    "ark-secret-value",
                )
            self.assertNotIn("ark-secret-value", log_path.read_text())
            self.assertIn("<redacted>", log_path.read_text())


class BundledLlmBenchTests(unittest.TestCase):
    def test_chat_payload_requests_stream_usage(self):
        payload = llm_engine.build_payload(
            "model-id",
            [{"role": "user", "content": "hello"}],
            reasoning_effort="",
            max_completion_tokens=64,
        )
        self.assertTrue(payload["stream"])
        self.assertEqual(payload["stream_options"], {"include_usage": True})
        self.assertEqual(payload["max_completion_tokens"], 64)
        self.assertNotIn("reasoning_effort", payload)
        self.assertEqual(payload["thinking"], {"type": "disabled"})

    def test_weighted_cache_and_tpot_metrics(self):
        metrics = llm_metrics.RequestMetrics()
        metrics.add_success(
            ttft=0.1,
            e2e=0.5,
            output_tokens=21,
            prompt_tokens=100,
            cached_tokens=50,
            has_cache_field=True,
        )
        metrics.add_success(
            ttft=0.2,
            e2e=0.8,
            output_tokens=21,
            prompt_tokens=300,
            cached_tokens=250,
            has_cache_field=True,
        )
        self.assertAlmostEqual(metrics.weighted_cache_hit, 0.75)
        self.assertEqual(len(metrics.tpot), 2)

    def test_sharegpt_corpus_is_bundled(self):
        path = Path(llm_datasets.POOL_PATH)
        self.assertTrue(path.is_file())
        self.assertGreater(path.stat().st_size, 10_000_000)


if __name__ == "__main__":
    unittest.main()
