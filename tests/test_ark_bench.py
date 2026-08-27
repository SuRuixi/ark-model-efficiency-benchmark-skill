import importlib.util
import sys
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "scripts" / "ark_bench.py"
SPEC = importlib.util.spec_from_file_location("ark_bench", MODULE_PATH)
ark_bench = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ark_bench
SPEC.loader.exec_module(ark_bench)


class ModelResolutionTests(unittest.TestCase):
    def test_normalizes_chinese_aliases(self):
        actual = ark_bench.normalize_model_name("豆包 Seed 2.0 迷你模型")
        self.assertEqual(actual, "doubaoseed20mini")

    def test_exact_candidate_has_highest_score(self):
        query = "豆包 seed 2.0 mini"
        exact = ark_bench.candidate_score(
            query, "doubao-seed-2-0-mini-260215"
        )
        other = ark_bench.candidate_score(
            query, "doubao-seed-2-0-lite-260215"
        )
        self.assertGreater(exact, other)
        self.assertGreater(exact, 0.8)


class MetricTests(unittest.TestCase):
    def test_recognizes_ark_dot_separated_token_delta(self):
        self.assertTrue(
            ark_bench.is_token_delta("response.output_text.delta", "token")
        )
        self.assertFalse(
            ark_bench.is_token_delta("response.output_text.done", "token")
        )

    def test_length_limited_response_is_valid_for_benchmarking(self):
        self.assertTrue(
            ark_bench.is_success_response(
                {
                    "status": "incomplete",
                    "incomplete_details": {"reason": "length"},
                }
            )
        )
        self.assertFalse(
            ark_bench.is_success_response(
                {
                    "status": "incomplete",
                    "incomplete_details": {"reason": "content_filter"},
                }
            )
        )

    def test_extracts_responses_api_usage(self):
        actual = ark_bench.extract_usage(
            {
                "usage": {
                    "input_tokens": 100,
                    "output_tokens": 20,
                    "input_tokens_details": {"cached_tokens": 80},
                }
            }
        )
        self.assertEqual(actual, (100, 20, 80))

    def test_weighted_cache_hit_rate(self):
        metrics = [
            ark_bench.RequestMetric(
                scenario="prefix",
                request_id="1",
                success=True,
                ttft_ms=100,
                e2e_ms=300,
                tpot_ms=10,
                input_tokens=100,
                output_tokens=21,
                cached_tokens=50,
            ),
            ark_bench.RequestMetric(
                scenario="prefix",
                request_id="2",
                success=True,
                ttft_ms=200,
                e2e_ms=500,
                tpot_ms=15,
                input_tokens=300,
                output_tokens=21,
                cached_tokens=250,
            ),
        ]
        summary = ark_bench.summarize(metrics)
        self.assertEqual(summary["successful_requests"], 2)
        self.assertAlmostEqual(summary["cache_hit_rate"], 0.75)
        self.assertAlmostEqual(summary["ttft_ms"]["p50"], 150)

    def test_failures_are_excluded_from_latency(self):
        metrics = [
            ark_bench.RequestMetric(
                scenario="prefix",
                request_id="ok",
                success=True,
                ttft_ms=100,
                e2e_ms=200,
                input_tokens=10,
                output_tokens=2,
            ),
            ark_bench.RequestMetric(
                scenario="prefix",
                request_id="failed",
                success=False,
                error="timeout",
            ),
        ]
        summary = ark_bench.summarize(metrics)
        self.assertEqual(summary["failed_requests"], 1)
        self.assertEqual(summary["ttft_ms"]["mean"], 100)


class PresetTests(unittest.TestCase):
    def test_cache_can_be_enabled_for_reuse_scenarios(self):
        payload = ark_bench.make_payload(
            "example", "input", 64, "none", enable_cache=True
        )
        self.assertEqual(
            payload["caching"], {"type": "enabled", "prefix": True}
        )

    def test_standard_preset_matches_document(self):
        args = ark_bench.build_parser().parse_args(["--model", "example"])
        ark_bench.apply_preset(args)
        self.assertEqual(args.prefix_len, 12000)
        self.assertEqual(args.suffix_len, 2000)
        self.assertEqual(args.num_requests, 200)
        self.assertEqual(args.initial_len, 7000)
        self.assertEqual(args.max_turns, 30)
        self.assertEqual(args.max_concurrency, 5)


if __name__ == "__main__":
    unittest.main()
