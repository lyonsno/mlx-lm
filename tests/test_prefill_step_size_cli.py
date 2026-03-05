# Copyright © 2026 Apple Inc.

import argparse
import contextlib
import io
import sys
import unittest
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import patch

import mlx_lm.benchmark as benchmark_module
from mlx_lm.benchmark import setup_arg_parser as setup_benchmark_arg_parser
from mlx_lm.cli_utils import coerce_positive_int, positive_int
from mlx_lm.server import setup_arg_parser as setup_server_arg_parser


class TestPrefillStepSizeCLI(unittest.TestCase):
    def _assert_prefill_step_size_parse_error(self, parser, value):
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            with self.assertRaises(SystemExit) as cm:
                parser.parse_args(["--prefill-step-size", value])
        self.assertEqual(cm.exception.code, 2)
        err = stderr.getvalue()
        self.assertIn("--prefill-step-size", err)
        self.assertIn("positive integer", err)

    def test_server_parser_rejects_non_positive_prefill_step_size(self):
        parser = setup_server_arg_parser()
        self._assert_prefill_step_size_parse_error(parser, "0")
        self._assert_prefill_step_size_parse_error(parser, "-1")
        self._assert_prefill_step_size_parse_error(parser, "1.5")

    def test_benchmark_parser_rejects_non_positive_prefill_step_size(self):
        parser = setup_benchmark_arg_parser()
        self._assert_prefill_step_size_parse_error(parser, "0")
        self._assert_prefill_step_size_parse_error(parser, "-1")
        self._assert_prefill_step_size_parse_error(parser, "1.5")

    def test_parsers_accept_positive_prefill_step_size(self):
        server_parser = setup_server_arg_parser()
        server_step_size = server_parser.parse_args(
            ["--prefill-step-size", "16"]
        ).prefill_step_size
        self.assertIsInstance(server_step_size, int)
        self.assertEqual(server_step_size, 16)

        benchmark_parser = setup_benchmark_arg_parser()
        benchmark_step_size = benchmark_parser.parse_args(
            ["--prefill-step-size", "32"]
        ).prefill_step_size
        self.assertIsInstance(benchmark_step_size, int)
        self.assertEqual(benchmark_step_size, 32)


class TestBenchmarkPrefillStepSizeForwarding(unittest.TestCase):
    class _DummyGroup:
        def rank(self):
            return 0

        def size(self):
            return 1

    def test_benchmark_single_mode_forwards_prefill_step_size(self):
        args = [
            "benchmark.py",
            "--prefill-step-size",
            "37",
            "--num-trials",
            "1",
            "--prompt-tokens",
            "4",
            "--generation-tokens",
            "2",
            "--batch-size",
            "1",
        ]
        response = SimpleNamespace(prompt_tps=1.0, generation_tps=2.0, peak_memory=3.0)

        with patch.object(sys, "argv", args), patch(
            "mlx_lm.benchmark.mx.distributed.init",
            return_value=self._DummyGroup(),
        ), patch(
            "mlx_lm.benchmark.load",
            return_value=(
                object(),
                SimpleNamespace(_eos_token_ids={}),
                {"vocab_size": 32},
            ),
        ), patch(
            "mlx_lm.benchmark.stream_generate",
            side_effect=lambda *a, **k: iter([response]),
        ) as stream_generate_mock, patch(
            "mlx_lm.benchmark.batch_generate"
        ) as batch_generate_mock:
            benchmark_module.main()

        self.assertFalse(batch_generate_mock.called)
        self.assertGreaterEqual(stream_generate_mock.call_count, 2)
        for call in stream_generate_mock.call_args_list:
            self.assertEqual(call.kwargs["prefill_step_size"], 37)

    def test_benchmark_batch_mode_forwards_prefill_step_size(self):
        args = [
            "benchmark.py",
            "--prefill-step-size",
            "73",
            "--num-trials",
            "1",
            "--prompt-tokens",
            "4",
            "--generation-tokens",
            "2",
            "--batch-size",
            "2",
        ]
        response = SimpleNamespace(prompt_tps=1.0, generation_tps=2.0, peak_memory=3.0)

        with patch.object(sys, "argv", args), patch(
            "mlx_lm.benchmark.mx.distributed.init",
            return_value=self._DummyGroup(),
        ), patch(
            "mlx_lm.benchmark.load",
            return_value=(
                object(),
                SimpleNamespace(_eos_token_ids={}),
                {"vocab_size": 32},
            ),
        ), patch(
            "mlx_lm.benchmark.batch_generate",
            return_value=SimpleNamespace(stats=response),
        ) as batch_generate_mock, patch(
            "mlx_lm.benchmark.stream_generate"
        ) as stream_generate_mock:
            benchmark_module.main()

        self.assertFalse(stream_generate_mock.called)
        self.assertGreaterEqual(batch_generate_mock.call_count, 2)
        for call in batch_generate_mock.call_args_list:
            self.assertEqual(call.kwargs["prefill_step_size"], 73)


class TestPositiveIntValidation(unittest.TestCase):
    def test_runtime_coerce_positive_int_accepts_only_positive_integers(self):
        self.assertEqual(coerce_positive_int(1, field_name="prefill_step_size"), 1)
        self.assertEqual(coerce_positive_int(64, field_name="prefill_step_size"), 64)

    def test_runtime_coerce_positive_int_rejects_invalid_inputs(self):
        for value in (0, -1, 1.5, True, "8", "1.5"):
            with self.assertRaisesRegex(
                ValueError, "prefill_step_size must be a positive integer"
            ):
                coerce_positive_int(value, field_name="prefill_step_size")

    def test_argparse_positive_int_wrapper_preserves_int_coercion_compatibility(self):
        self.assertEqual(positive_int("16"), 16)
        self.assertEqual(positive_int(Decimal("2")), 2)
        self.assertEqual(positive_int(1.5), 1)


if __name__ == "__main__":
    unittest.main()
