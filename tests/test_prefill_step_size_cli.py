# Copyright © 2026 Apple Inc.

import contextlib
import io
import unittest

from mlx_lm.benchmark import setup_arg_parser as setup_benchmark_arg_parser
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

    def test_benchmark_parser_rejects_non_positive_prefill_step_size(self):
        parser = setup_benchmark_arg_parser()
        self._assert_prefill_step_size_parse_error(parser, "0")
        self._assert_prefill_step_size_parse_error(parser, "-1")

    def test_parsers_accept_positive_prefill_step_size(self):
        server_parser = setup_server_arg_parser()
        self.assertEqual(
            server_parser.parse_args(["--prefill-step-size", "16"]).prefill_step_size,
            16,
        )

        benchmark_parser = setup_benchmark_arg_parser()
        self.assertEqual(
            benchmark_parser.parse_args(
                ["--prefill-step-size", "32"]
            ).prefill_step_size,
            32,
        )


if __name__ == "__main__":
    unittest.main()
