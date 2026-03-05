# Copyright © 2026 Apple Inc.

import importlib
import unittest
from itertools import islice
from unittest.mock import patch

import mlx.core as mx

from mlx_lm.tokenizer_utils import TokenizerWrapper

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


class _MinimalHFTokenizer:
    chat_template = None
    eos_token_id = 9999
    bos_token = None
    clean_up_tokenization_spaces = False

    def get_vocab(self):
        return {}

    def encode(self, text, add_special_tokens=True):
        del text, add_special_tokens
        return [5, 7]

    def decode(self, tokens):
        return "".join(str(int(t)) for t in tokens)


class _PassthroughDetokenizer:
    def __init__(self, tokenizer):
        del tokenizer
        self.last_segment = ""

    def add_token(self, token):
        self.last_segment = str(int(token))

    def finalize(self):
        pass


class _RoutingMTPModel:
    def __init__(self):
        self.layers = [object()]
        self._main_cache = [_TrimmableCache()]
        self._mtp_cache = [_TrimmableCache()]

    def make_cache(self):
        return self._main_cache

    def make_mtp_cache(self):
        return self._mtp_cache

    def mtp_logits(self, *args, **kwargs):
        del args, kwargs
        raise AssertionError("Routing tests should mock mtp_generate_step.")


class _NonMTPModel:
    def __init__(self):
        self.layers = [object()]


class _MTPModelWithoutCacheFactory:
    def __init__(self):
        self.layers = [object()]
        self._main_cache = [_TrimmableCache()]

    def make_cache(self):
        return self._main_cache

    def mtp_logits(self, *args, **kwargs):
        del args, kwargs
        raise AssertionError(
            "mtp_logits should not be called when mtp_cache is missing and "
            "model lacks make_mtp_cache."
        )


class _RunnableMTPModelWithoutCacheFactory:
    def __init__(self, token: int = 3, vocab_size: int = 16):
        self.layers = [object()]
        self._main_cache = [_TrimmableCache()]
        self.token = token
        self.vocab_size = vocab_size
        self.hidden_size = 4

    def make_cache(self):
        return self._main_cache

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
        logits[:, -1, self.token] = 0.0
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
        logits[:, -1, self.token] = 0.0
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

    def test_stream_generate_use_mtp_routes_to_mtp_generate_step(self):
        model = _RoutingMTPModel()
        tokenizer = TokenizerWrapper(
            _MinimalHFTokenizer(),
            detokenizer_class=_PassthroughDetokenizer,
            eos_token_ids=[9999],
        )
        prompt = mx.array([5, 7], dtype=mx.uint32)

        verifier_logits = mx.array(
            [-2.0, -1.4, -0.2, -3.0, -2.1, -4.0], dtype=mx.float32
        )
        logprobs = verifier_logits - mx.logsumexp(
            verifier_logits, axis=-1, keepdims=True
        )
        mtp_out = iter([(2, logprobs, True)])
        vanilla_out = iter([(1, logprobs)])

        with (
            patch.object(
                generate_lib, "mtp_generate_step", autospec=True, return_value=mtp_out
            ) as mtp,
            patch.object(
                generate_lib, "generate_step", return_value=vanilla_out
            ) as vanilla,
            patch.object(generate_lib, "speculative_generate_step") as speculative,
        ):
            results = list(
                generate_lib.stream_generate(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    max_tokens=1,
                    use_mtp=True,
                )
            )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].token, 2)
        self.assertTrue(results[0].from_draft)
        mtp.assert_called_once()
        self.assertNotIn(
            "use_mtp",
            mtp.call_args.kwargs,
            "use_mtp is a stream_generate control flag and must not be forwarded.",
        )
        vanilla.assert_not_called()
        speculative.assert_not_called()

    def test_stream_generate_rejects_use_mtp_with_draft_model(self):
        model = _RoutingMTPModel()
        tokenizer = TokenizerWrapper(
            _MinimalHFTokenizer(),
            detokenizer_class=_PassthroughDetokenizer,
            eos_token_ids=[9999],
        )
        prompt = mx.array([5, 7], dtype=mx.uint32)

        with (
            patch.object(
                generate_lib,
                "generate_step",
                side_effect=AssertionError(
                    "generate_step should not be entered on conflicting flags."
                ),
            ) as vanilla,
            patch.object(
                generate_lib,
                "speculative_generate_step",
                side_effect=AssertionError(
                    "speculative_generate_step should not be entered on conflicting flags."
                ),
            ) as speculative,
            patch.object(
                generate_lib,
                "mtp_generate_step",
                side_effect=AssertionError(
                    "mtp_generate_step should not be entered on conflicting flags."
                ),
            ) as mtp,
        ):
            with self.assertRaisesRegex(ValueError, "use_mtp"):
                list(
                    generate_lib.stream_generate(
                        model=model,
                        tokenizer=tokenizer,
                        prompt=prompt,
                        max_tokens=1,
                        use_mtp=True,
                        draft_model=object(),
                    )
                )

        vanilla.assert_not_called()
        speculative.assert_not_called()
        mtp.assert_not_called()

    def test_stream_generate_rejects_non_positive_max_tokens_on_vanilla_route(self):
        model = _RoutingMTPModel()
        tokenizer = TokenizerWrapper(
            _MinimalHFTokenizer(),
            detokenizer_class=_PassthroughDetokenizer,
            eos_token_ids=[9999],
        )
        prompt = mx.array([5, 7], dtype=mx.uint32)

        with (
            patch.object(
                generate_lib,
                "generate_step",
                side_effect=AssertionError(
                    "generate_step should not be entered when max_tokens <= 0."
                ),
            ) as vanilla,
            patch.object(
                generate_lib,
                "speculative_generate_step",
                side_effect=AssertionError(
                    "speculative_generate_step should not be entered when "
                    "max_tokens <= 0."
                ),
            ) as speculative,
            patch.object(
                generate_lib,
                "mtp_generate_step",
                side_effect=AssertionError(
                    "mtp_generate_step should not be entered when max_tokens <= 0."
                ),
            ) as mtp,
        ):
            for bad_max_tokens in (0, -1):
                with self.subTest(max_tokens=bad_max_tokens):
                    with self.assertRaisesRegex(ValueError, "max_tokens"):
                        list(
                            generate_lib.stream_generate(
                                model=model,
                                tokenizer=tokenizer,
                                prompt=prompt,
                                max_tokens=bad_max_tokens,
                            )
                        )

    def test_stream_generate_use_mtp_rejects_non_positive_max_tokens(self):
        model = _RoutingMTPModel()
        tokenizer = TokenizerWrapper(
            _MinimalHFTokenizer(),
            detokenizer_class=_PassthroughDetokenizer,
            eos_token_ids=[9999],
        )
        prompt = mx.array([5, 7], dtype=mx.uint32)

        with (
            patch.object(
                generate_lib,
                "generate_step",
                side_effect=AssertionError(
                    "generate_step should not be entered when max_tokens <= 0."
                ),
            ) as vanilla,
            patch.object(
                generate_lib,
                "speculative_generate_step",
                side_effect=AssertionError(
                    "speculative_generate_step should not be entered when "
                    "max_tokens <= 0."
                ),
            ) as speculative,
            patch.object(
                generate_lib,
                "mtp_generate_step",
                side_effect=AssertionError(
                    "mtp_generate_step should not be entered when max_tokens <= 0."
                ),
            ) as mtp,
        ):
            for bad_max_tokens in (0, -1):
                with self.subTest(max_tokens=bad_max_tokens):
                    with self.assertRaisesRegex(ValueError, "max_tokens"):
                        list(
                            generate_lib.stream_generate(
                                model=model,
                                tokenizer=tokenizer,
                                prompt=prompt,
                                max_tokens=bad_max_tokens,
                                use_mtp=True,
                            )
                        )

    def test_stream_generate_speculative_route_rejects_non_positive_max_tokens(self):
        model = _RoutingMTPModel()
        draft_model = object()
        tokenizer = TokenizerWrapper(
            _MinimalHFTokenizer(),
            detokenizer_class=_PassthroughDetokenizer,
            eos_token_ids=[9999],
        )
        prompt = mx.array([5, 7], dtype=mx.uint32)

        with (
            patch.object(
                generate_lib,
                "generate_step",
                side_effect=AssertionError(
                    "generate_step should not be entered when max_tokens <= 0."
                ),
            ) as vanilla,
            patch.object(
                generate_lib,
                "speculative_generate_step",
                side_effect=AssertionError(
                    "speculative_generate_step should not be entered when "
                    "max_tokens <= 0."
                ),
            ) as speculative,
            patch.object(
                generate_lib,
                "mtp_generate_step",
                side_effect=AssertionError(
                    "mtp_generate_step should not be entered when max_tokens <= 0."
                ),
            ) as mtp,
        ):
            for bad_max_tokens in (0, -1):
                with self.subTest(max_tokens=bad_max_tokens):
                    with self.assertRaisesRegex(ValueError, "max_tokens"):
                        list(
                            generate_lib.stream_generate(
                                model=model,
                                tokenizer=tokenizer,
                                prompt=prompt,
                                max_tokens=bad_max_tokens,
                                draft_model=draft_model,
                            )
                        )

    def test_stream_generate_rejects_boolean_max_tokens(self):
        model = _RoutingMTPModel()
        tokenizer = TokenizerWrapper(
            _MinimalHFTokenizer(),
            detokenizer_class=_PassthroughDetokenizer,
            eos_token_ids=[9999],
        )
        prompt = mx.array([5, 7], dtype=mx.uint32)

        with (
            patch.object(
                generate_lib,
                "generate_step",
                side_effect=AssertionError(
                    "generate_step should not be entered when max_tokens is boolean."
                ),
            ) as vanilla,
            patch.object(
                generate_lib,
                "speculative_generate_step",
                side_effect=AssertionError(
                    "speculative_generate_step should not be entered when max_tokens is "
                    "boolean."
                ),
            ) as speculative,
            patch.object(
                generate_lib,
                "mtp_generate_step",
                side_effect=AssertionError(
                    "mtp_generate_step should not be entered when max_tokens is boolean."
                ),
            ) as mtp,
        ):
            for bad_max_tokens in (True, False):
                with self.subTest(max_tokens=bad_max_tokens):
                    with self.assertRaisesRegex(ValueError, "max_tokens"):
                        list(
                            generate_lib.stream_generate(
                                model=model,
                                tokenizer=tokenizer,
                                prompt=prompt,
                                max_tokens=bad_max_tokens,
                            )
                        )

    def test_stream_generate_rejects_non_bool_use_mtp(self):
        model = _RoutingMTPModel()
        tokenizer = TokenizerWrapper(
            _MinimalHFTokenizer(),
            detokenizer_class=_PassthroughDetokenizer,
            eos_token_ids=[9999],
        )
        prompt = mx.array([5, 7], dtype=mx.uint32)

        with (
            patch.object(
                generate_lib,
                "generate_step",
                side_effect=AssertionError(
                    "generate_step should not be entered when use_mtp is invalid."
                ),
            ) as vanilla,
            patch.object(
                generate_lib,
                "speculative_generate_step",
                side_effect=AssertionError(
                    "speculative_generate_step should not be entered when use_mtp is "
                    "invalid."
                ),
            ) as speculative,
            patch.object(
                generate_lib,
                "mtp_generate_step",
                side_effect=AssertionError(
                    "mtp_generate_step should not be entered when use_mtp is invalid."
                ),
            ) as mtp,
        ):
            for bad_use_mtp in ("true", 1, None):
                with self.subTest(use_mtp=bad_use_mtp):
                    with self.assertRaisesRegex(
                        ValueError, "use_mtp must be a boolean"
                    ):
                        list(
                            generate_lib.stream_generate(
                                model=model,
                                tokenizer=tokenizer,
                                prompt=prompt,
                                max_tokens=1,
                                use_mtp=bad_use_mtp,
                            )
                        )

    def test_stream_generate_default_route_stays_vanilla_for_mtp_capable_model(self):
        model = _RoutingMTPModel()
        tokenizer = TokenizerWrapper(
            _MinimalHFTokenizer(),
            detokenizer_class=_PassthroughDetokenizer,
            eos_token_ids=[9999],
        )
        prompt = mx.array([5, 7], dtype=mx.uint32)

        verifier_logits = mx.array(
            [-2.0, -1.4, -0.2, -3.0, -2.1, -4.0], dtype=mx.float32
        )
        logprobs = verifier_logits - mx.logsumexp(
            verifier_logits, axis=-1, keepdims=True
        )
        vanilla_out = iter([(1, logprobs)])

        with (
            patch.object(
                generate_lib, "generate_step", autospec=True, return_value=vanilla_out
            ) as vanilla,
            patch.object(generate_lib, "mtp_generate_step") as mtp,
            patch.object(generate_lib, "speculative_generate_step") as speculative,
        ):
            results = list(
                generate_lib.stream_generate(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    max_tokens=1,
                )
            )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].token, 1)
        self.assertFalse(results[0].from_draft)
        vanilla.assert_called_once()
        mtp.assert_not_called()
        speculative.assert_not_called()

    def test_stream_generate_use_mtp_false_routes_to_vanilla(self):
        model = _RoutingMTPModel()
        tokenizer = TokenizerWrapper(
            _MinimalHFTokenizer(),
            detokenizer_class=_PassthroughDetokenizer,
            eos_token_ids=[9999],
        )
        prompt = mx.array([5, 7], dtype=mx.uint32)
        mtp_cache = [_TrimmableCache()]

        verifier_logits = mx.array(
            [-2.0, -1.4, -0.2, -3.0, -2.1, -4.0], dtype=mx.float32
        )
        logprobs = verifier_logits - mx.logsumexp(
            verifier_logits, axis=-1, keepdims=True
        )
        vanilla_out = iter([(1, logprobs)])

        with (
            patch.object(
                generate_lib, "generate_step", autospec=True, return_value=vanilla_out
            ) as vanilla,
            patch.object(generate_lib, "mtp_generate_step") as mtp,
            patch.object(generate_lib, "speculative_generate_step") as speculative,
        ):
            results = list(
                generate_lib.stream_generate(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    max_tokens=1,
                    use_mtp=False,
                    num_draft_tokens=3,
                    mtp_cache=mtp_cache,
                )
            )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].token, 1)
        self.assertFalse(results[0].from_draft)
        vanilla.assert_called_once()
        self.assertNotIn(
            "use_mtp",
            vanilla.call_args.kwargs,
            "use_mtp=False should not be forwarded to generate_step.",
        )
        self.assertNotIn(
            "num_draft_tokens",
            vanilla.call_args.kwargs,
            "num_draft_tokens should be stripped when draft_model is not provided.",
        )
        self.assertNotIn(
            "mtp_cache",
            vanilla.call_args.kwargs,
            "mtp_cache should not be forwarded on the vanilla route.",
        )
        mtp.assert_not_called()
        speculative.assert_not_called()

    def test_stream_generate_use_mtp_filters_invalid_kwargs_before_mtp_call(self):
        model = _RoutingMTPModel()
        tokenizer = TokenizerWrapper(
            _MinimalHFTokenizer(),
            detokenizer_class=_PassthroughDetokenizer,
            eos_token_ids=[9999],
        )
        prompt = mx.array([5, 7], dtype=mx.uint32)
        sampler = lambda x: mx.argmax(x, axis=-1)
        prompt_cache = [_TrimmableCache()]
        mtp_cache = [_TrimmableCache()]
        input_embeddings = mx.zeros((2, 4), dtype=mx.float32)
        kv_bits = 4
        kv_group_size = 32
        quantized_kv_start = 11

        def logits_processor(tokens, logits):
            del tokens
            return logits

        verifier_logits = mx.array(
            [-2.0, -1.4, -0.2, -3.0, -2.1, -4.0], dtype=mx.float32
        )
        logprobs = verifier_logits - mx.logsumexp(
            verifier_logits, axis=-1, keepdims=True
        )
        mtp_out = iter([(2, logprobs, True)])
        vanilla_out = iter([(1, logprobs)])

        def progress_callback(processed: int, total: int):
            del processed, total

        with (
            patch.object(
                generate_lib, "mtp_generate_step", autospec=True, return_value=mtp_out
            ) as mtp,
            patch.object(
                generate_lib, "generate_step", return_value=vanilla_out
            ) as vanilla,
            patch.object(generate_lib, "speculative_generate_step") as speculative,
        ):
            results = list(
                generate_lib.stream_generate(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    max_tokens=1,
                    use_mtp=True,
                    num_draft_tokens=3,
                    max_kv_size=128,
                    prompt_progress_callback=progress_callback,
                    prefill_step_size=17,
                    sampler=sampler,
                    prompt_cache=prompt_cache,
                    mtp_cache=mtp_cache,
                    kv_bits=kv_bits,
                    kv_group_size=kv_group_size,
                    quantized_kv_start=quantized_kv_start,
                    input_embeddings=input_embeddings,
                    logits_processors=[logits_processor],
                )
            )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].token, 2)
        self.assertTrue(results[0].from_draft)
        mtp.assert_called_once()
        self.assertNotIn("use_mtp", mtp.call_args.kwargs)
        self.assertNotIn("num_draft_tokens", mtp.call_args.kwargs)
        self.assertNotIn("max_kv_size", mtp.call_args.kwargs)
        self.assertNotIn("prompt_progress_callback", mtp.call_args.kwargs)
        self.assertNotIn("input_embeddings", mtp.call_args.kwargs)
        self.assertNotIn("logits_processors", mtp.call_args.kwargs)
        self.assertEqual(
            mtp.call_args.kwargs.get("max_tokens"),
            1,
            "max_tokens should be forwarded to mtp_generate_step.",
        )
        self.assertEqual(
            mtp.call_args.kwargs.get("prefill_step_size"),
            17,
            "Valid MTP kwargs should be preserved when routing.",
        )
        self.assertIs(
            mtp.call_args.kwargs.get("sampler"),
            sampler,
            "Sampler should be forwarded unchanged to mtp_generate_step.",
        )
        self.assertIs(
            mtp.call_args.kwargs.get("prompt_cache"),
            prompt_cache,
            "prompt_cache should be forwarded unchanged to mtp_generate_step.",
        )
        self.assertIs(
            mtp.call_args.kwargs.get("mtp_cache"),
            mtp_cache,
            "mtp_cache should be forwarded unchanged to mtp_generate_step.",
        )
        self.assertEqual(
            mtp.call_args.kwargs.get("kv_bits"),
            kv_bits,
            "kv_bits should be preserved on the MTP route.",
        )
        self.assertEqual(
            mtp.call_args.kwargs.get("kv_group_size"),
            kv_group_size,
            "kv_group_size should be preserved on the MTP route.",
        )
        self.assertEqual(
            mtp.call_args.kwargs.get("quantized_kv_start"),
            quantized_kv_start,
            "quantized_kv_start should be preserved on the MTP route.",
        )
        vanilla.assert_not_called()
        speculative.assert_not_called()

    def test_stream_generate_draft_model_with_use_mtp_false_routes_speculative(self):
        model = _RoutingMTPModel()
        tokenizer = TokenizerWrapper(
            _MinimalHFTokenizer(),
            detokenizer_class=_PassthroughDetokenizer,
            eos_token_ids=[9999],
        )
        prompt = mx.array([5, 7], dtype=mx.uint32)
        draft_model = object()
        num_draft_tokens = 5
        input_embeddings = mx.zeros((2, 4), dtype=mx.float32)
        mtp_cache = [_TrimmableCache()]

        def progress_callback(processed: int, total: int):
            del processed, total

        verifier_logits = mx.array(
            [-2.0, -1.4, -0.2, -3.0, -2.1, -4.0], dtype=mx.float32
        )
        logprobs = verifier_logits - mx.logsumexp(
            verifier_logits, axis=-1, keepdims=True
        )
        speculative_out = iter([(3, logprobs, True)])

        with (
            patch.object(generate_lib, "generate_step") as vanilla,
            patch.object(generate_lib, "mtp_generate_step") as mtp,
            patch.object(
                generate_lib,
                "speculative_generate_step",
                autospec=True,
                return_value=speculative_out,
            ) as speculative,
        ):
            results = list(
                generate_lib.stream_generate(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    max_tokens=1,
                    draft_model=draft_model,
                    num_draft_tokens=num_draft_tokens,
                    use_mtp=False,
                    max_kv_size=64,
                    prompt_progress_callback=progress_callback,
                    input_embeddings=input_embeddings,
                    mtp_cache=mtp_cache,
                )
            )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].token, 3)
        self.assertTrue(results[0].from_draft)
        speculative.assert_called_once()
        called_draft_model = (
            speculative.call_args.args[2]
            if len(speculative.call_args.args) > 2
            else speculative.call_args.kwargs.get("draft_model")
        )
        self.assertIs(
            called_draft_model,
            draft_model,
            "draft_model should be forwarded unchanged to speculative_generate_step.",
        )
        self.assertEqual(
            speculative.call_args.kwargs.get("num_draft_tokens"),
            num_draft_tokens,
            "num_draft_tokens should be preserved on the speculative route.",
        )
        self.assertEqual(
            speculative.call_args.kwargs.get("max_tokens"),
            1,
            "max_tokens should be forwarded to speculative_generate_step.",
        )
        self.assertNotIn(
            "use_mtp",
            speculative.call_args.kwargs,
            "use_mtp flag should not be forwarded to speculative_generate_step.",
        )
        self.assertNotIn(
            "max_kv_size",
            speculative.call_args.kwargs,
            "max_kv_size should be stripped on speculative route.",
        )
        self.assertNotIn(
            "prompt_progress_callback",
            speculative.call_args.kwargs,
            "prompt_progress_callback should be stripped on speculative route.",
        )
        self.assertNotIn(
            "input_embeddings",
            speculative.call_args.kwargs,
            "input_embeddings should not be forwarded to speculative_generate_step.",
        )
        self.assertNotIn(
            "mtp_cache",
            speculative.call_args.kwargs,
            "mtp_cache should not be forwarded on the speculative route.",
        )
        vanilla.assert_not_called()
        mtp.assert_not_called()

    def test_stream_generate_use_mtp_routes_with_list_prompt(self):
        model = _RoutingMTPModel()
        tokenizer = TokenizerWrapper(
            _MinimalHFTokenizer(),
            detokenizer_class=_PassthroughDetokenizer,
            eos_token_ids=[9999],
        )
        prompt = [5, 7]

        verifier_logits = mx.array(
            [-2.0, -1.4, -0.2, -3.0, -2.1, -4.0], dtype=mx.float32
        )
        logprobs = verifier_logits - mx.logsumexp(
            verifier_logits, axis=-1, keepdims=True
        )
        mtp_out = iter([(2, logprobs, True)])
        vanilla_out = iter([(1, logprobs)])

        with (
            patch.object(
                generate_lib, "mtp_generate_step", autospec=True, return_value=mtp_out
            ) as mtp,
            patch.object(
                generate_lib, "generate_step", return_value=vanilla_out
            ) as vanilla,
            patch.object(generate_lib, "speculative_generate_step") as speculative,
        ):
            results = list(
                generate_lib.stream_generate(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    max_tokens=1,
                    use_mtp=True,
                )
            )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].token, 2)
        self.assertTrue(results[0].from_draft)
        mtp.assert_called_once()
        prompt_arg = mtp.call_args.args[0]
        self.assertIsInstance(
            prompt_arg,
            type(mx.array([0], dtype=mx.uint32)),
            "List prompts should be normalized to mx.array before MTP routing.",
        )
        self.assertEqual(
            [int(t) for t in prompt_arg.tolist()],
            [5, 7],
            "List prompt token values should be preserved through normalization.",
        )
        vanilla.assert_not_called()
        speculative.assert_not_called()

    def test_stream_generate_use_mtp_routes_with_string_prompt(self):
        model = _RoutingMTPModel()
        tokenizer = TokenizerWrapper(
            _MinimalHFTokenizer(),
            detokenizer_class=_PassthroughDetokenizer,
            eos_token_ids=[9999],
        )
        prompt = "hello"

        verifier_logits = mx.array(
            [-2.0, -1.4, -0.2, -3.0, -2.1, -4.0], dtype=mx.float32
        )
        logprobs = verifier_logits - mx.logsumexp(
            verifier_logits, axis=-1, keepdims=True
        )
        mtp_out = iter([(2, logprobs, True)])
        vanilla_out = iter([(1, logprobs)])

        with (
            patch.object(
                generate_lib, "mtp_generate_step", autospec=True, return_value=mtp_out
            ) as mtp,
            patch.object(
                generate_lib, "generate_step", return_value=vanilla_out
            ) as vanilla,
            patch.object(generate_lib, "speculative_generate_step") as speculative,
        ):
            results = list(
                generate_lib.stream_generate(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    max_tokens=1,
                    use_mtp=True,
                )
            )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].token, 2)
        self.assertTrue(results[0].from_draft)
        mtp.assert_called_once()
        prompt_arg = mtp.call_args.args[0]
        self.assertIsInstance(
            prompt_arg,
            type(mx.array([0], dtype=mx.uint32)),
            "String prompts should be tokenized and normalized to mx.array for MTP routing.",
        )
        self.assertEqual(
            [int(t) for t in prompt_arg.tolist()],
            [5, 7],
            "Tokenized string prompt should be forwarded to MTP routing.",
        )
        vanilla.assert_not_called()
        speculative.assert_not_called()

    def test_stream_generate_use_mtp_wraps_plain_tokenizer(self):
        model = _RoutingMTPModel()
        tokenizer = _MinimalHFTokenizer()
        prompt = "hello"

        verifier_logits = mx.array(
            [-2.0, -1.4, -0.2, -3.0, -2.1, -4.0], dtype=mx.float32
        )
        logprobs = verifier_logits - mx.logsumexp(
            verifier_logits, axis=-1, keepdims=True
        )
        mtp_out = iter([(2, logprobs, True)])
        vanilla_out = iter([(1, logprobs)])

        with (
            patch.object(
                generate_lib, "mtp_generate_step", autospec=True, return_value=mtp_out
            ) as mtp,
            patch.object(
                generate_lib, "generate_step", return_value=vanilla_out
            ) as vanilla,
            patch.object(generate_lib, "speculative_generate_step") as speculative,
        ):
            results = list(
                generate_lib.stream_generate(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    max_tokens=1,
                    use_mtp=True,
                )
            )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].token, 2)
        self.assertTrue(results[0].from_draft)
        mtp.assert_called_once()
        prompt_arg = mtp.call_args.args[0]
        self.assertIsInstance(
            prompt_arg,
            type(mx.array([0], dtype=mx.uint32)),
            "Plain tokenizer input should be wrapped and normalized to mx.array.",
        )
        self.assertEqual(
            [int(t) for t in prompt_arg.tolist()],
            [5, 7],
            "Auto-wrapped tokenizer should preserve encoded prompt token ids.",
        )
        vanilla.assert_not_called()
        speculative.assert_not_called()

    def test_stream_generate_use_mtp_plain_tokenizer_smoke_stops_on_eos(self):
        model = _FakeMTPModel(
            draft_tokens=[4],
            verify_tokens=[4],
        )
        tokenizer = _MinimalHFTokenizer()
        tokenizer.eos_token_id = 4

        results = list(
            generate_lib.stream_generate(
                model=model,
                tokenizer=tokenizer,
                prompt="hello",
                max_tokens=3,
                use_mtp=True,
                prompt_cache=model.make_cache(),
                mtp_cache=model.make_mtp_cache(),
            )
        )

        # EOS on first token should produce only the terminal response, even
        # when stream_generate wraps a plain tokenizer internally.
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].token, 4)
        self.assertTrue(results[0].from_draft)
        self.assertEqual(results[0].finish_reason, "stop")

    def test_stream_generate_use_mtp_smoke_uses_real_mtp_generator(self):
        model = _FakeMTPModel(
            draft_tokens=[7, 8],
            verify_tokens=[7, 4],
        )
        tokenizer = TokenizerWrapper(
            _MinimalHFTokenizer(),
            detokenizer_class=_PassthroughDetokenizer,
            eos_token_ids=[9999],
        )

        results = list(
            generate_lib.stream_generate(
                model=model,
                tokenizer=tokenizer,
                prompt=[1],
                max_tokens=2,
                use_mtp=True,
                prompt_cache=model.make_cache(),
                mtp_cache=model.make_mtp_cache(),
            )
        )

        # stream_generate yields interim chunks plus a final response.
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].token, 7)
        self.assertTrue(results[0].from_draft)
        self.assertIsNone(results[0].finish_reason)

        self.assertEqual(results[1].token, 4)
        self.assertFalse(results[1].from_draft)
        self.assertEqual(results[1].finish_reason, "length")

    def test_stream_generate_use_mtp_smoke_uses_default_caches(self):
        model = _FakeMTPModel(
            draft_tokens=[7, 8],
            verify_tokens=[7, 4],
        )
        tokenizer = TokenizerWrapper(
            _MinimalHFTokenizer(),
            detokenizer_class=_PassthroughDetokenizer,
            eos_token_ids=[9999],
        )

        results = list(
            generate_lib.stream_generate(
                model=model,
                tokenizer=tokenizer,
                prompt=[1],
                max_tokens=2,
                use_mtp=True,
            )
        )

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].token, 7)
        self.assertTrue(results[0].from_draft)
        self.assertIsNone(results[0].finish_reason)

        self.assertEqual(results[1].token, 4)
        self.assertFalse(results[1].from_draft)
        self.assertEqual(results[1].finish_reason, "length")
        self.assertGreater(
            model._main_cache[0].offset,
            0,
            "Default prompt cache creation should advance model main cache state.",
        )
        self.assertGreater(
            model._mtp_cache[0].offset,
            0,
            "Default MTP cache creation should advance model MTP cache state.",
        )

    def test_stream_generate_use_mtp_smoke_stops_on_eos(self):
        model = _FakeMTPModel(
            draft_tokens=[4],
            verify_tokens=[4],
        )
        tokenizer = TokenizerWrapper(
            _MinimalHFTokenizer(),
            detokenizer_class=_PassthroughDetokenizer,
            eos_token_ids=[4],
        )

        results = list(
            generate_lib.stream_generate(
                model=model,
                tokenizer=tokenizer,
                prompt=[1],
                max_tokens=3,
                use_mtp=True,
                prompt_cache=model.make_cache(),
                mtp_cache=model.make_mtp_cache(),
            )
        )

        # EOS on first token should produce only the terminal response.
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].token, 4)
        self.assertTrue(results[0].from_draft)
        self.assertEqual(results[0].finish_reason, "stop")

    def test_mtp_generate_requires_make_mtp_cache_or_explicit_mtp_cache(self):
        model = _MTPModelWithoutCacheFactory()
        mtp_generate_step = getattr(generate_lib, "mtp_generate_step", None)
        self.assertTrue(
            callable(mtp_generate_step),
            "Expected callable mlx_lm.generate.mtp_generate_step for behavior tests.",
        )

        prompt = mx.array([1], dtype=mx.uint32)
        with self.assertRaisesRegex(ValueError, "mtp_cache is required"):
            list(
                islice(
                    mtp_generate_step(
                        prompt,
                        model,
                        max_tokens=1,
                    ),
                    1,
                )
            )

    def test_mtp_generate_accepts_explicit_mtp_cache_without_make_mtp_cache(self):
        model = _RunnableMTPModelWithoutCacheFactory(token=3)
        explicit_mtp_cache = [_TrimmableCache()]
        mtp_generate_step = getattr(generate_lib, "mtp_generate_step", None)
        self.assertTrue(
            callable(mtp_generate_step),
            "Expected callable mlx_lm.generate.mtp_generate_step for behavior tests.",
        )

        prompt = mx.array([1], dtype=mx.uint32)
        results = list(
            islice(
                mtp_generate_step(
                    prompt,
                    model,
                    max_tokens=1,
                    prompt_cache=model.make_cache(),
                    mtp_cache=explicit_mtp_cache,
                ),
                1,
            )
        )
        self.assertEqual(len(results), 1)
        token, _, from_draft = results[0]
        self.assertEqual(token, 3)
        self.assertTrue(from_draft)
        self.assertGreater(
            explicit_mtp_cache[0].offset,
            0,
            "Explicit mtp_cache should be accepted and advanced.",
        )

    def test_stream_generate_use_mtp_accepts_explicit_mtp_cache_without_factory(self):
        model = _RunnableMTPModelWithoutCacheFactory(token=3)
        tokenizer = TokenizerWrapper(
            _MinimalHFTokenizer(),
            detokenizer_class=_PassthroughDetokenizer,
            eos_token_ids=[9999],
        )
        explicit_mtp_cache = [_TrimmableCache()]

        results = list(
            generate_lib.stream_generate(
                model=model,
                tokenizer=tokenizer,
                prompt=[1],
                max_tokens=1,
                use_mtp=True,
                prompt_cache=model.make_cache(),
                mtp_cache=explicit_mtp_cache,
            )
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].token, 3)
        self.assertTrue(results[0].from_draft)
        self.assertEqual(results[0].finish_reason, "length")
        self.assertGreater(
            explicit_mtp_cache[0].offset,
            0,
            "stream_generate should forward explicit mtp_cache to MTP generation.",
        )

    def test_stream_generate_use_mtp_on_non_mtp_model_raises_clear_error(self):
        model = _NonMTPModel()
        tokenizer = TokenizerWrapper(
            _MinimalHFTokenizer(),
            detokenizer_class=_PassthroughDetokenizer,
            eos_token_ids=[9999],
        )
        prompt = mx.array([5, 7], dtype=mx.uint32)

        with (
            patch.object(
                generate_lib,
                "generate_step",
                side_effect=AssertionError(
                    "generate_step should not be entered when use_mtp=True."
                ),
            ) as vanilla,
            patch.object(
                generate_lib,
                "speculative_generate_step",
                side_effect=AssertionError(
                    "speculative_generate_step should not be entered when use_mtp=True."
                ),
            ) as speculative,
        ):
            with self.assertRaisesRegex(
                ValueError,
                r"(?i)(does not (expose|support).*(mtp|mtp_logits)|required for mtp)",
            ):
                list(
                    generate_lib.stream_generate(
                        model=model,
                        tokenizer=tokenizer,
                        prompt=prompt,
                        max_tokens=1,
                        use_mtp=True,
                    )
                )

        vanilla.assert_not_called()
        speculative.assert_not_called()


if __name__ == "__main__":
    unittest.main()
