# Copyright © 2026 Apple Inc.

import importlib
import unittest
from itertools import islice

import mlx.core as mx

generate_lib = importlib.import_module("mlx_lm.generate")


class _TrimmableCache:
    def __init__(self):
        self.offset = 0
        self.trim_history = []

    def is_trimmable(self):
        return True

    def trim(self, n):
        self.trim_history.append(n)
        self.offset = max(0, self.offset - n)
        return n

    @property
    def state(self):
        return []


class _HistoryTrimmableCache(_TrimmableCache):
    def __init__(self):
        super().__init__()
        self.token_history = []

    def append_tokens(self, tokens: list[int]):
        self.token_history.extend(tokens)
        self.offset += len(tokens)

    def trim(self, n):
        self.trim_history.append(n)
        n = min(n, len(self.token_history))
        if n > 0:
            self.token_history = self.token_history[:-n]
        self.offset = max(0, self.offset - n)
        return n


class _NonTrimmableCache:
    def __init__(self):
        self.offset = 0

    def is_trimmable(self):
        return False

    @property
    def state(self):
        return []


class _FakeMTPModel:
    def __init__(
        self,
        draft_tokens: list[int],
        verify_tokens: list[int],
        vocab_size: int = 16,
        max_prefetch_overread: int = 1,
    ):
        if len(draft_tokens) == 0 or len(verify_tokens) == 0:
            raise ValueError("Fake token streams must be non-empty.")
        if max_prefetch_overread < 0:
            raise ValueError("max_prefetch_overread must be >= 0.")
        self.draft_tokens = draft_tokens
        self.verify_tokens = verify_tokens
        self.vocab_size = vocab_size
        self.max_prefetch_overread = max_prefetch_overread
        self.hidden_size = 4
        self.layers = [object()]
        self._main_cache = [_TrimmableCache()]
        self._mtp_cache = [_TrimmableCache()]
        self._draft_idx = 0
        self._verify_idx = 0
        self._draft_overreads = 0
        self._verify_overreads = 0

    def make_cache(self):
        return self._main_cache

    def make_mtp_cache(self):
        return self._mtp_cache

    def _next_verify_token(self) -> int:
        if self._verify_idx >= len(self.verify_tokens):
            self._verify_overreads += 1
            if self._verify_overreads > self.max_prefetch_overread:
                raise AssertionError(
                    "verify token stream overread budget exceeded; unexpected "
                    "extra verification passes"
                )
            return self.verify_tokens[-1]
        token = self.verify_tokens[self._verify_idx]
        self._verify_idx += 1
        return token

    def _next_draft_token(self) -> int:
        if self._draft_idx >= len(self.draft_tokens):
            self._draft_overreads += 1
            if self._draft_overreads > self.max_prefetch_overread:
                raise AssertionError(
                    "draft token stream overread budget exceeded; unexpected extra "
                    "draft passes"
                )
            return self.draft_tokens[-1]
        token = self.draft_tokens[self._draft_idx]
        self._draft_idx += 1
        return token

    def __call__(self, tokens: mx.array, cache=None):
        logits, _ = self.forward_with_hidden(tokens, cache)
        return logits

    def forward_with_hidden(self, tokens: mx.array, cache):
        for c in cache:
            c.offset += tokens.shape[1]
        hidden = mx.zeros(
            (tokens.shape[0], tokens.shape[1], self.hidden_size), dtype=mx.float32
        )
        logits = mx.full(
            (tokens.shape[0], tokens.shape[1], self.vocab_size),
            -1e9,
            dtype=mx.float32,
        )
        logits[:, -1, self._next_verify_token()] = 0.0
        return logits, hidden

    def mtp_logits(self, hidden_states: mx.array, draft_tokens: mx.array, cache):
        del hidden_states
        for c in cache:
            c.offset += draft_tokens.shape[1]
        logits = mx.full(
            (draft_tokens.shape[0], draft_tokens.shape[1], self.vocab_size),
            -1e9,
            dtype=mx.float32,
        )
        logits[:, -1, self._next_draft_token()] = 0.0
        return logits


class _ContextualRejectingMTPModel:
    """
    A context-sensitive toy model where MTP drafts are always wrong.

    This model is used to assert that when every draft is rejected,
    `mtp_generate_step` emits the same token sequence as `generate_step`.
    """

    def __init__(self, vocab_size: int = 17):
        self.vocab_size = vocab_size
        self.hidden_size = 4
        self.layers = [object()]
        self._main_cache = [_HistoryTrimmableCache()]
        self._mtp_cache = [_TrimmableCache()]

    def make_cache(self):
        return self._main_cache

    def make_mtp_cache(self):
        return self._mtp_cache

    def __call__(self, tokens: mx.array, cache=None):
        logits, _ = self.forward_with_hidden(tokens, cache)
        return logits

    def _verify_token(self, history: list[int]) -> int:
        # Depend on multi-token context and rolling history statistics so state
        # drift from duplicated prefetching is observable.
        last = history[-1]
        prev = history[-2] if len(history) > 1 else 0
        hist_sum = sum(history) % self.vocab_size
        hist_xor = 0
        for t in history[-6:]:
            hist_xor ^= t
        return (
            prev * 11 + last * 7 + hist_sum * 3 + hist_xor + len(history)
        ) % self.vocab_size

    def forward_with_hidden(self, tokens: mx.array, cache):
        token_history = [int(t) for t in tokens.reshape(-1).tolist()]
        for c in cache:
            if hasattr(c, "append_tokens"):
                c.append_tokens(token_history)
            else:
                c.offset += len(token_history)
        verify_token = self._verify_token(cache[0].token_history)

        hidden = mx.zeros(
            (tokens.shape[0], tokens.shape[1], self.hidden_size), dtype=mx.float32
        )
        # Encode verify choice so mtp_logits can deterministically draft the wrong token.
        hidden[:, -1, 0] = float(verify_token)

        logits = mx.full(
            (tokens.shape[0], tokens.shape[1], self.vocab_size),
            -1e9,
            dtype=mx.float32,
        )
        logits[:, -1, verify_token] = 0.0
        return logits, hidden

    def mtp_logits(self, hidden_states: mx.array, draft_tokens: mx.array, cache):
        del draft_tokens
        for c in cache:
            c.offset += 1
        verify_hint = int(hidden_states[:, -1, 0].item())
        draft_token = (verify_hint + 1) % self.vocab_size
        logits = mx.full(
            (hidden_states.shape[0], hidden_states.shape[1], self.vocab_size),
            -1e9,
            dtype=mx.float32,
        )
        logits[:, -1, draft_token] = 0.0
        return logits


class _SameArgmaxDifferentDistributionModel:
    """
    A toy model where verifier and draft agree on argmax but disagree on distribution.

    Used to ensure mtp_generate_step returns verifier-side logprobs even when
    `from_draft=True`.
    """

    def __init__(self):
        self.vocab_size = 7
        self.hidden_size = 4
        self.layers = [object()]
        self._main_cache = [_TrimmableCache()]
        self._mtp_cache = [_TrimmableCache()]
        # Argmax at token 2, but intentionally soft.
        self.verify_row = mx.array(
            [-2.0, -1.0, 0.0, -0.3, -1.5, -2.5, -3.0], dtype=mx.float32
        )
        # Same argmax at token 2, but near one-hot.
        self.draft_row = mx.array(
            [-9.0, -9.0, 0.0, -9.0, -9.0, -9.0, -9.0], dtype=mx.float32
        )

    def make_cache(self):
        return self._main_cache

    def make_mtp_cache(self):
        return self._mtp_cache

    def __call__(self, tokens: mx.array, cache=None):
        logits, _ = self.forward_with_hidden(tokens, cache)
        return logits

    def forward_with_hidden(self, tokens: mx.array, cache):
        for c in cache:
            c.offset += tokens.shape[1]
        hidden = mx.zeros(
            (tokens.shape[0], tokens.shape[1], self.hidden_size), dtype=mx.float32
        )
        logits = mx.full(
            (tokens.shape[0], tokens.shape[1], self.vocab_size),
            -1e9,
            dtype=mx.float32,
        )
        logits[:, -1, :] = self.verify_row
        return logits, hidden

    def mtp_logits(self, hidden_states: mx.array, draft_tokens: mx.array, cache):
        del hidden_states, draft_tokens
        for c in cache:
            c.offset += 1
        logits = mx.full((1, 1, self.vocab_size), -1e9, dtype=mx.float32)
        logits[:, -1, :] = self.draft_row
        return logits


class TestMTPGenerateStep(unittest.TestCase):
    def test_generate_module_exposes_callable_mtp_generate_step(self):
        self.assertTrue(
            callable(getattr(generate_lib, "mtp_generate_step", None)),
            "Expected mlx_lm.generate.mtp_generate_step to exist and be callable.",
        )

    def _run_mtp_generate(
        self, prompt: mx.array, model: _FakeMTPModel, max_tokens: int
    ):
        mtp_generate_step = getattr(generate_lib, "mtp_generate_step", None)
        if not callable(mtp_generate_step):
            self.skipTest(
                "mtp_generate_step is missing; behavior tests are skipped while API "
                "contract test enforces presence."
            )

        token_gen = mtp_generate_step(
            prompt,
            model,
            max_tokens=max_tokens,
            prompt_cache=model.make_cache(),
            mtp_cache=model.make_mtp_cache(),
        )
        results = list(islice(token_gen, max_tokens + 1))

        self.assertLessEqual(
            len(results),
            max_tokens,
            "mtp_generate_step yielded more than max_tokens items.",
        )
        if len(results) == max_tokens:
            self.assertIsNone(
                next(token_gen, None),
                "mtp_generate_step should be exhausted after max_tokens items.",
            )
        return results

    def _assert_logprobs_are_valid(self, results, vocab_size: int):
        for token, logprobs, _ in results:
            self.assertEqual(
                tuple(logprobs.shape),
                (vocab_size,),
                "Each yielded logprobs entry should be a vocab-sized vector.",
            )
            self.assertTrue(
                not bool(mx.any(mx.isnan(logprobs)).item()),
                "Logprobs should not contain NaNs.",
            )
            self.assertEqual(
                int(mx.argmax(logprobs).item()),
                int(token),
                "With default argmax sampling, yielded token should match argmax(logprobs).",
            )
            prob_sum = float(mx.sum(mx.exp(logprobs)).item())
            self.assertAlmostEqual(
                prob_sum,
                1.0,
                places=4,
                msg="exp(logprobs) should sum to 1.",
            )

    def _assert_rewind_events_match_rejections(
        self, main_trim_history, mtp_trim_history, rejected: int
    ):
        main_nonzero = [int(t) for t in main_trim_history if int(t) > 0]
        mtp_nonzero = [int(t) for t in mtp_trim_history if int(t) > 0]

        self.assertEqual(
            sum(main_nonzero),
            0,
            "Main cache should not be rewound for MTP-1 rejections in this decode path.",
        )
        self.assertEqual(
            sum(mtp_nonzero),
            rejected,
            "MTP cache total positive rewind should equal rejected drafts.",
        )
        self.assertEqual(
            len(main_nonzero),
            0,
            "Main cache should not have positive rewind events.",
        )
        if rejected == 0:
            self.assertEqual(
                len(mtp_nonzero),
                0,
                "MTP cache should not have positive rewind events when nothing is rejected.",
            )
        else:
            self.assertGreater(
                len(mtp_nonzero),
                0,
                "MTP cache should have at least one positive rewind event.",
            )
            self.assertLessEqual(
                len(mtp_nonzero),
                rejected,
                "MTP cache should not have more positive rewind events than rejections.",
            )

    def _assert_cache_offsets_track_emitted_tokens(
        self, prompt, results, drafted, model
    ):
        main_offset = int(model._main_cache[0].offset)
        mtp_offset = int(model._mtp_cache[0].offset)
        emitted = len(results)
        accepted = sum(1 for d in drafted if d)
        rejected = emitted - accepted
        prompt_len = int(prompt.size)

        # Exact lower bound for this fake model: main cache must at least include
        # prompt plus emitted tokens.
        self.assertGreaterEqual(
            main_offset,
            prompt_len + emitted,
            "Main cache offset should account for prompt plus all emitted tokens.",
        )
        # Upper bound prevents silent double-advancement drift.
        self.assertLessEqual(
            main_offset,
            prompt_len + emitted + rejected + int(model._verify_overreads),
            "Main cache offset advanced more than expected for emitted/rejected tokens.",
        )

        self.assertGreaterEqual(
            mtp_offset,
            accepted,
            "MTP cache offset should retain accepted drafts.",
        )
        self.assertLessEqual(
            mtp_offset,
            accepted + rejected + int(model._draft_overreads),
            "MTP cache offset advanced more than expected for accepted/rejected drafts.",
        )

    def _assert_prefetch_overreads_within_budget(self, model):
        self.assertLessEqual(
            int(model._verify_overreads),
            int(model.max_prefetch_overread),
            "Verification lookahead exceeded allowed prefetch overread budget.",
        )
        self.assertLessEqual(
            int(model._draft_overreads),
            int(model.max_prefetch_overread),
            "Draft lookahead exceeded allowed prefetch overread budget.",
        )

    def test_matching_mtp_drafts_are_accepted_across_iterations(self):
        model = _FakeMTPModel(draft_tokens=[7, 8, 9], verify_tokens=[7, 8, 9])
        prompt = mx.array([1], dtype=mx.uint32)

        results = self._run_mtp_generate(prompt, model, max_tokens=3)
        tokens = [t for t, _, _ in results]
        drafted = [d for _, _, d in results]

        self.assertEqual(tokens, [7, 8, 9])
        self.assertEqual(drafted, [True, True, True])
        self._assert_logprobs_are_valid(results, model.vocab_size)

        self.assertEqual(sum(model._main_cache[0].trim_history), 0)
        self.assertEqual(sum(model._mtp_cache[0].trim_history), 0)
        self.assertTrue(all(t >= 0 for t in model._main_cache[0].trim_history))
        self.assertTrue(all(t >= 0 for t in model._mtp_cache[0].trim_history))
        self._assert_cache_offsets_track_emitted_tokens(prompt, results, drafted, model)
        self._assert_prefetch_overreads_within_budget(model)

    def test_reject_then_accept_rewinds_mtp_and_keeps_main_cache_aligned(self):
        model = _FakeMTPModel(
            draft_tokens=[3, 4, 5],
            verify_tokens=[9, 4, 5],
        )
        prompt = mx.array([1], dtype=mx.uint32)

        results = self._run_mtp_generate(prompt, model, max_tokens=3)
        tokens = [t for t, _, _ in results]
        drafted = [d for _, _, d in results]

        self.assertEqual(tokens, [9, 4, 5])
        self.assertEqual(drafted, [False, True, True])
        self._assert_logprobs_are_valid(results, model.vocab_size)

        main_trim_total = sum(model._main_cache[0].trim_history)
        mtp_trim_total = sum(model._mtp_cache[0].trim_history)
        self.assertEqual(
            mtp_trim_total,
            1,
            "A single rejected MTP-1 draft should rewind one token total in MTP cache.",
        )
        self.assertEqual(
            main_trim_total,
            0,
            "Main cache should not be rewound on MTP rejection in this decode path.",
        )
        self.assertTrue(all(t >= 0 for t in model._main_cache[0].trim_history))
        self.assertTrue(all(t >= 0 for t in model._mtp_cache[0].trim_history))
        rejected = len(results) - sum(1 for d in drafted if d)
        self.assertEqual(
            main_trim_total,
            0,
            "Main cache trim total should remain zero.",
        )
        self.assertEqual(
            mtp_trim_total,
            rejected,
            "MTP cache trim total should equal the number of rejected drafts.",
        )
        self._assert_rewind_events_match_rejections(
            model._main_cache[0].trim_history,
            model._mtp_cache[0].trim_history,
            rejected,
        )
        self._assert_cache_offsets_track_emitted_tokens(prompt, results, drafted, model)
        self._assert_prefetch_overreads_within_budget(model)

    def test_accept_then_reject_path_rewinds_once(self):
        model = _FakeMTPModel(
            draft_tokens=[7, 8],
            verify_tokens=[7, 4],
        )
        prompt = mx.array([1], dtype=mx.uint32)

        results = self._run_mtp_generate(prompt, model, max_tokens=2)
        tokens = [t for t, _, _ in results]
        drafted = [d for _, _, d in results]

        self.assertEqual(tokens, [7, 4])
        self.assertEqual(drafted, [True, False])
        self._assert_logprobs_are_valid(results, model.vocab_size)

        main_trim_total = sum(model._main_cache[0].trim_history)
        mtp_trim_total = sum(model._mtp_cache[0].trim_history)
        rejected = len(results) - sum(1 for d in drafted if d)
        self.assertEqual(rejected, 1)
        self.assertEqual(main_trim_total, 0)
        self.assertEqual(mtp_trim_total, rejected)
        self._assert_rewind_events_match_rejections(
            model._main_cache[0].trim_history,
            model._mtp_cache[0].trim_history,
            rejected,
        )
        self._assert_cache_offsets_track_emitted_tokens(prompt, results, drafted, model)
        self._assert_prefetch_overreads_within_budget(model)

    def test_all_rejected_mtp_matches_autoregressive_generation(self):
        prompt = mx.array([2, 1], dtype=mx.uint32)
        max_tokens = 6

        baseline_model = _ContextualRejectingMTPModel()
        mtp_model = _ContextualRejectingMTPModel()

        baseline_results = list(
            generate_lib.generate_step(
                prompt,
                baseline_model,
                max_tokens=max_tokens,
                prompt_cache=baseline_model.make_cache(),
            )
        )
        baseline_tokens = [t for t, _ in baseline_results]

        mtp_results = self._run_mtp_generate(prompt, mtp_model, max_tokens=max_tokens)
        mtp_tokens = [t for t, _, _ in mtp_results]
        drafted = [d for _, _, d in mtp_results]

        self.assertEqual(
            drafted,
            [False] * max_tokens,
            "This regression model is constructed so every draft is rejected.",
        )
        self.assertEqual(
            mtp_tokens,
            baseline_tokens,
            "When all drafts are rejected, mtp_generate_step should match "
            "generate_step token-for-token.",
        )

    def test_all_rejected_mtp_matches_autoregressive_across_prompts(self):
        prompts = [
            [2, 1],
            [4, 3, 2],
            [7, 0, 6, 1],
        ]
        max_tokens = 7

        for prompt_tokens in prompts:
            with self.subTest(prompt=prompt_tokens):
                prompt = mx.array(prompt_tokens, dtype=mx.uint32)
                baseline_model = _ContextualRejectingMTPModel()
                mtp_model = _ContextualRejectingMTPModel()

                baseline_results = list(
                    generate_lib.generate_step(
                        prompt,
                        baseline_model,
                        max_tokens=max_tokens,
                        prompt_cache=baseline_model.make_cache(),
                    )
                )
                baseline_out = [t for t, _ in baseline_results]

                mtp_results = self._run_mtp_generate(
                    prompt, mtp_model, max_tokens=max_tokens
                )
                mtp_out = [t for t, _, _ in mtp_results]
                drafted = [d for _, _, d in mtp_results]

                self.assertEqual(drafted, [False] * max_tokens)
                self.assertEqual(mtp_out, baseline_out)

    def test_mtp_reject_path_requires_trimmable_caches(self):
        mtp_generate_step = getattr(generate_lib, "mtp_generate_step", None)
        if not callable(mtp_generate_step):
            self.skipTest(
                "mtp_generate_step missing; trimmable-cache contract not testable."
            )

        prompt = mx.array([1], dtype=mx.uint32)
        model = _FakeMTPModel(
            draft_tokens=[3, 3],
            verify_tokens=[9, 9],
            max_prefetch_overread=0,
        )

        with self.assertRaisesRegex(ValueError, "trimmable"):
            token_gen = mtp_generate_step(
                prompt,
                model,
                max_tokens=1,
                prompt_cache=model.make_cache(),
                mtp_cache=[_NonTrimmableCache()],
            )
            next(token_gen)

    def test_reject_path_allows_nontrimmable_prompt_cache(self):
        mtp_generate_step = getattr(generate_lib, "mtp_generate_step", None)
        if not callable(mtp_generate_step):
            self.skipTest(
                "mtp_generate_step missing; prompt-cache contract not testable."
            )

        prompt = mx.array([1], dtype=mx.uint32)
        model = _FakeMTPModel(
            draft_tokens=[3],
            verify_tokens=[9],
            max_prefetch_overread=1,
        )

        # Rejection path should only require mtp_cache rewinding.
        token, _, from_draft = next(
            mtp_generate_step(
                prompt,
                model,
                max_tokens=1,
                prompt_cache=[_NonTrimmableCache()],
                mtp_cache=model.make_mtp_cache(),
            )
        )
        self.assertEqual(token, 9)
        self.assertFalse(from_draft)

    def test_accepted_draft_reports_verifier_logprobs_not_draft_logprobs(self):
        model = _SameArgmaxDifferentDistributionModel()
        prompt = mx.array([1], dtype=mx.uint32)

        token, logprobs, from_draft = next(
            generate_lib.mtp_generate_step(
                prompt,
                model,
                max_tokens=1,
                prompt_cache=model.make_cache(),
                mtp_cache=model.make_mtp_cache(),
            )
        )
        self.assertTrue(from_draft)
        self.assertEqual(token, 2)

        expected = model.verify_row - mx.logsumexp(
            model.verify_row, axis=-1, keepdims=True
        )
        unexpected = model.draft_row - mx.logsumexp(
            model.draft_row, axis=-1, keepdims=True
        )
        self.assertTrue(
            bool(mx.allclose(logprobs, expected, rtol=1e-5, atol=1e-6).item()),
            "Accepted tokens should report verifier/main-model logprobs.",
        )
        self.assertFalse(
            bool(mx.allclose(logprobs, unexpected, rtol=1e-5, atol=1e-6).item()),
            "Accepted tokens should not report draft-head logprobs.",
        )


if __name__ == "__main__":
    unittest.main()
