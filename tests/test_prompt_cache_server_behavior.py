# Copyright © 2024 Apple Inc.

import copy
import unittest

import mlx.core as mx

from mlx_lm.models.cache import CacheList, KVCache, RotatingKVCache
from mlx_lm.server import LRUPromptCache
from tests.prompt_cache_test_utils import (
    DeepcopyShouldNotRunLayer,
    LegacyTrimLayer,
    RewindRecorderLayer,
    UnknownLayerWithoutLegacyHooks,
    UnknownNonTrimmableLayer,
    UnknownNonTrimmableNoDeepcopy,
    build_real_rotating_cache,
    make_tiny_step3p5_model,
)


class MockCache:
    def __init__(self, value):
        self.value = value

    @property
    def nbytes(self):
        return len(self.value)

    def __eq__(self, other):
        return other.value == self.value


class TestLRUPromptCacheBehavior(unittest.TestCase):
    def test_caching(self):
        cache = LRUPromptCache(max_size=10)

        def get_kv(n):
            keys = mx.arange(n).reshape(1, 1, n, 1)
            return keys, keys

        model = ("test", None, None)
        tokens = [10] * 24

        c, t = cache.fetch_nearest_cache(model, tokens)
        self.assertIsNone(c)
        self.assertEqual(t, tokens)

        c = [KVCache()]
        c[0].update_and_fetch(*get_kv(24))
        cache.insert_cache(model, t, c)

        tokens = tokens + [20] * 5
        c, t = cache.fetch_nearest_cache(model, tokens)
        k, v = c[0].state
        self.assertTrue((k == v).all().item())
        self.assertTrue((k.flatten() == mx.arange(24)).all().item())
        self.assertEqual(t, [20] * 5)

        tokens = tokens + [30] * 3
        c[0].update_and_fetch(*get_kv(8))
        cache.insert_cache(model, tokens, c)

        tokens = tokens[:26] + [40] * 8
        c, t = cache.fetch_nearest_cache(model, tokens)
        k, v = c[0].state
        self.assertTrue((k == v).all().item())
        self.assertTrue(
            (k.flatten() == mx.concatenate([mx.arange(24), mx.arange(2)])).all().item()
        )
        self.assertEqual(t, [40] * 8)

        # The 32-token entry should still be reusable after the prefix-match
        # extraction (rewind operates on a copy, not the original).
        full_tokens = [10] * 24 + [20] * 5 + [30] * 3
        c2, t2 = cache.fetch_nearest_cache(model, full_tokens)
        self.assertIsNotNone(c2)
        self.assertEqual(t2, [])

    def test_lru(self):
        cache = LRUPromptCache(max_size=2)
        model = ("test", None, None)
        cache.insert_cache(model, [1, 2], [MockCache("test1")])
        cache.insert_cache(model, [1, 2], [MockCache("test1")])

        c, t = cache.fetch_nearest_cache(model, [1, 2])
        self.assertEqual(c, [MockCache("test1")])
        self.assertEqual(t, [])
        c, t = cache.fetch_nearest_cache(model, [1, 2])
        self.assertEqual(c, [MockCache("test1")])
        self.assertEqual(t, [])
        c, t = cache.fetch_nearest_cache(model, [1, 2])
        self.assertIsNone(c)
        self.assertEqual(t, [1, 2])

        cache.insert_cache(model, [1, 2], [MockCache("test1")])
        cache.insert_cache(model, [2, 3], [MockCache("test2")])
        cache.insert_cache(model, [3, 4], [MockCache("test3")])

        c, t = cache.fetch_nearest_cache(model, [1, 2])
        self.assertIsNone(c)
        self.assertEqual(t, [1, 2])
        c, t = cache.fetch_nearest_cache(model, [2, 3])
        self.assertEqual(c, [MockCache("test2")])
        self.assertEqual(t, [])
        c, t = cache.fetch_nearest_cache(model, [3, 4])
        self.assertEqual(c, [MockCache("test3")])
        self.assertEqual(t, [])

    def test_lru_bytes(self):
        cache = LRUPromptCache(max_size=100, max_bytes=10)
        model = ("test", None, None)

        cache.insert_cache(model, [1, 2], [MockCache("aaa")])
        cache.insert_cache(model, [3, 4], [MockCache("bbb")])
        cache.insert_cache(model, [4, 5], [MockCache("ccc")])
        cache.insert_cache(model, [6, 7], [MockCache("ddd")])

        self.assertEqual(len(cache), 3)
        self.assertEqual(cache.nbytes, 9)

        cache.trim_to(n_bytes=7)
        self.assertEqual(len(cache), 2)
        self.assertEqual(cache.nbytes, 6)

        c, t = cache.fetch_nearest_cache(model, [1, 2])
        self.assertIsNone(c)
        self.assertEqual(t, [1, 2])
        c, t = cache.fetch_nearest_cache(model, [3, 4])
        self.assertIsNone(c)
        self.assertEqual(t, [3, 4])

    def test_fast_trim_path_fails_closed_on_partial_trim(self):
        lru = LRUPromptCache(max_size=10)
        model = ("fast-trim", None, None)
        long_tokens = [1, 2, 3, 4]
        shorter_tokens = [1, 2]
        expected_num_to_trim = len(long_tokens) - (len(shorter_tokens) - 1)

        partial_layer = RewindRecorderLayer(max_rewind=4, rewind_result=False)
        lru.insert_cache(model, long_tokens, [partial_layer])

        reused_cache, remaining = lru.fetch_nearest_cache(model, shorter_tokens)
        self.assertIsNone(reused_cache)
        self.assertEqual(remaining, shorter_tokens)
        self.assertEqual(partial_layer.rewind_calls, [expected_num_to_trim])

        exact_cache, exact_remaining = lru.fetch_nearest_cache(model, long_tokens)
        self.assertIsNotNone(exact_cache)
        self.assertEqual(exact_remaining, [])
        self.assertEqual(exact_cache[0].offset, 4)

    def test_unknown_layer_safe_miss_variants(self):
        scenarios = [
            (
                "unknown_non_trimmable_refcounted",
                UnknownNonTrimmableLayer,
                2,
            ),
            (
                "unknown_non_trimmable_no_deepcopy",
                UnknownNonTrimmableNoDeepcopy,
                1,
            ),
            (
                "unknown_no_legacy_hooks_no_deepcopy",
                UnknownLayerWithoutLegacyHooks,
                1,
            ),
        ]

        long_tokens = [1, 2, 3, 4]
        shorter_tokens = [1, 2]

        for name, layer_factory, insert_count in scenarios:
            with self.subTest(name=name):
                lru = LRUPromptCache(max_size=10)
                model = (f"{name}", None, None)

                for _ in range(insert_count):
                    lru.insert_cache(model, long_tokens, [layer_factory()])

                reused_cache, remaining = lru.fetch_nearest_cache(model, shorter_tokens)
                self.assertIsNone(reused_cache)
                self.assertEqual(remaining, shorter_tokens)

                for _ in range(insert_count):
                    exact_cache, exact_remaining = lru.fetch_nearest_cache(
                        model, long_tokens
                    )
                    self.assertIsNotNone(exact_cache)
                    self.assertEqual(exact_remaining, [])

                miss, miss_remaining = lru.fetch_nearest_cache(model, long_tokens)
                self.assertIsNone(miss)
                self.assertEqual(miss_remaining, long_tokens)

    def test_legacy_trimmable_layer_without_rewind_api_still_reuses(self):
        trim_calls = []
        deepcopy_calls = []

        lru = LRUPromptCache(max_size=10)
        model = ("legacy-trim-layer", None, None)
        long_tokens = [1, 2, 3, 4]
        shorter_tokens = [1, 2]
        expected_num_to_trim = len(long_tokens) - (len(shorter_tokens) - 1)

        lru.insert_cache(
            model,
            long_tokens,
            [
                LegacyTrimLayer(
                    trim_calls=trim_calls,
                    deepcopy_calls=deepcopy_calls,
                )
            ],
        )

        reused_cache, remaining = lru.fetch_nearest_cache(model, shorter_tokens)
        self.assertIsNotNone(reused_cache)
        self.assertEqual(remaining, shorter_tokens[-1:])
        self.assertEqual(trim_calls, [expected_num_to_trim])
        self.assertEqual(reused_cache[0].offset, 1)

    def test_legacy_partial_trim_fails_closed_and_preserves_exact_entry(self):
        trim_calls = []
        deepcopy_calls = []

        lru = LRUPromptCache(max_size=10)
        model = ("legacy-partial-trim", None, None)
        long_tokens = [1, 2, 3, 4]
        shorter_tokens = [1, 2]
        expected_num_to_trim = len(long_tokens) - (len(shorter_tokens) - 1)

        lru.insert_cache(
            model,
            long_tokens,
            [
                LegacyTrimLayer(
                    trim_shortfall=1,
                    trim_calls=trim_calls,
                    deepcopy_calls=deepcopy_calls,
                )
            ],
        )

        reused_cache, remaining = lru.fetch_nearest_cache(model, shorter_tokens)
        self.assertIsNone(reused_cache)
        self.assertEqual(remaining, shorter_tokens)
        self.assertEqual(trim_calls, [expected_num_to_trim])
        self.assertEqual(len(deepcopy_calls), 1)

        exact_cache, exact_remaining = lru.fetch_nearest_cache(model, long_tokens)
        self.assertIsNotNone(exact_cache)
        self.assertEqual(exact_remaining, [])
        self.assertEqual(exact_cache[0].offset, 4)

    def test_legacy_rewind_only_layer_without_trim_still_reuses(self):
        class LegacyRewindOnlyLayer:
            def __init__(self):
                self.offset = 4

            @property
            def nbytes(self):
                return 1

            def is_trimmable(self):
                return True

            def rewind(self, n):
                if n > self.offset:
                    return False
                self.offset -= n
                return True

        lru = LRUPromptCache(max_size=10)
        model = ("legacy-rewind-only", None, None)
        long_tokens = [1, 2, 3, 4]
        shorter_tokens = [1, 2]

        lru.insert_cache(model, long_tokens, [LegacyRewindOnlyLayer()])
        reused_cache, remaining = lru.fetch_nearest_cache(model, shorter_tokens)
        self.assertIsNotNone(reused_cache)
        self.assertEqual(remaining, shorter_tokens[-1:])
        self.assertEqual(reused_cache[0].offset, 1)

        exact_cache, exact_remaining = lru.fetch_nearest_cache(model, long_tokens)
        self.assertIsNotNone(exact_cache)
        self.assertEqual(exact_remaining, [])
        self.assertEqual(exact_cache[0].offset, 4)

    def test_can_rewind_only_layer_without_rewind_path_safe_miss_skips_deepcopy(self):
        class CanRewindOnlyNoExecutionLayer:
            def __init__(self):
                self.offset = 4

            @property
            def nbytes(self):
                return 1

            def can_rewind(self, n):
                return True

            def __deepcopy__(self, memo):
                raise AssertionError(
                    "deepcopy should be skipped when can_rewind layer cannot execute rewind"
                )

        lru = LRUPromptCache(max_size=10)
        model = ("can-rewind-only-no-execution", None, None)
        long_tokens = [1, 2, 3, 4]
        shorter_tokens = [1, 2]

        layer = CanRewindOnlyNoExecutionLayer()
        lru.insert_cache(model, long_tokens, [layer])

        reused_cache, remaining = lru.fetch_nearest_cache(model, shorter_tokens)
        self.assertIsNone(reused_cache)
        self.assertEqual(remaining, shorter_tokens)

        exact_cache, exact_remaining = lru.fetch_nearest_cache(model, long_tokens)
        self.assertIsNotNone(exact_cache)
        self.assertEqual(exact_remaining, [])
        self.assertEqual(exact_cache[0].offset, 4)

    def test_legacy_offset_insufficient_safe_miss_skips_deepcopy(self):
        class LegacyOffsetLimitedLayer:
            def __init__(self):
                self.offset = 2
                self.trim_calls = []

            @property
            def nbytes(self):
                return 1

            def is_trimmable(self):
                return True

            def trim(self, n):
                self.trim_calls.append(n)
                trimmed = min(n, self.offset)
                self.offset -= trimmed
                return trimmed

            def __deepcopy__(self, memo):
                raise AssertionError(
                    "deepcopy should be skipped for offset-bounded legacy miss"
                )

        lru = LRUPromptCache(max_size=10)
        model = ("legacy-offset-insufficient", None, None)
        long_tokens = [1, 2, 3, 4]
        shorter_tokens = [1, 2]

        layer = LegacyOffsetLimitedLayer()
        lru.insert_cache(model, long_tokens, [layer])

        reused_cache, remaining = lru.fetch_nearest_cache(model, shorter_tokens)
        self.assertIsNone(reused_cache)
        self.assertEqual(remaining, shorter_tokens)
        self.assertEqual(layer.trim_calls, [])
        self.assertEqual(layer.offset, 2)

        exact_cache, exact_remaining = lru.fetch_nearest_cache(model, long_tokens)
        self.assertIsNotNone(exact_cache)
        self.assertEqual(exact_remaining, [])
        self.assertEqual(exact_cache[0].offset, 2)

    def test_composite_partial_trim_safe_miss_keeps_exact_entry_available(self):
        lru = LRUPromptCache(max_size=10)
        model = ("composite-partial", None, None)
        long_tokens = [1, 2, 3, 4]
        shorter_tokens = [1, 2]
        expected_num_to_trim = len(long_tokens) - (len(shorter_tokens) - 1)

        full = RewindRecorderLayer(max_rewind=4, rewind_result=True)
        partial = RewindRecorderLayer(max_rewind=4, rewind_result=False)
        composite = CacheList(full, partial)
        lru.insert_cache(model, long_tokens, [composite])

        reused_cache, remaining = lru.fetch_nearest_cache(model, shorter_tokens)
        self.assertIsNone(reused_cache)
        self.assertEqual(remaining, shorter_tokens)
        self.assertEqual(full.rewind_calls, [expected_num_to_trim])
        self.assertEqual(partial.rewind_calls, [expected_num_to_trim])

        exact_cache, exact_remaining = lru.fetch_nearest_cache(model, long_tokens)
        self.assertIsNotNone(exact_cache)
        self.assertEqual(exact_remaining, [])
        self.assertEqual(exact_cache[0][0].offset, 4)
        self.assertEqual(exact_cache[0][1].offset, 4)

    def test_cachelist_partial_child_failure_is_safe_miss(self):
        scenarios = [
            (
                "first_child_partial",
                lambda: CacheList(
                    RewindRecorderLayer(max_rewind=10, rewind_result=False),
                    RewindRecorderLayer(max_rewind=10, rewind_result=True),
                ),
            ),
            (
                "nested_partial",
                lambda: CacheList(
                    CacheList(
                        RewindRecorderLayer(max_rewind=10, rewind_result=False),
                        RewindRecorderLayer(max_rewind=10, rewind_result=True),
                    ),
                    RewindRecorderLayer(max_rewind=10, rewind_result=True),
                ),
            ),
            (
                "second_child_partial",
                lambda: CacheList(
                    RewindRecorderLayer(max_rewind=10, rewind_result=True),
                    RewindRecorderLayer(max_rewind=10, rewind_result=False),
                ),
            ),
        ]

        long_tokens = list(range(1, 11))
        shorter_tokens = long_tokens[:5]

        for name, builder in scenarios:
            with self.subTest(name=name):
                lru = LRUPromptCache(max_size=10)
                model = (f"cachelist-partial-{name}", None, None)
                composite = builder()
                lru.insert_cache(model, long_tokens, [composite])

                reused, remaining = lru.fetch_nearest_cache(model, shorter_tokens)
                self.assertIsNone(reused)
                self.assertEqual(remaining, shorter_tokens)

                exact, exact_rem = lru.fetch_nearest_cache(model, long_tokens)
                self.assertIsNotNone(exact)
                self.assertEqual(exact_rem, [])

    def test_broken_rotating_cache_fails_closed_even_when_trimmable(self):
        cache = RotatingKVCache(max_size=8)
        kv = mx.arange(4, dtype=mx.float32).reshape(1, 1, 4, 1)
        cache.update_and_fetch(kv, kv)
        mx.eval(cache.keys, cache.values)

        self.assertTrue(cache.is_trimmable())

        lru = LRUPromptCache(max_size=10)
        model = ("broken-rotating-guard", None, None)
        long_tokens = [1, 2, 3, 4]
        shorter_tokens = [1, 2]

        lru.insert_cache(model, long_tokens, [cache])

        # Corrupt the cache after insertion to simulate a broken state
        # that is_trimmable() still reports as trimmable.
        cache.values = None

        reused, remaining = lru.fetch_nearest_cache(model, shorter_tokens)
        self.assertIsNone(reused)
        self.assertEqual(remaining, shorter_tokens)

        exact, exact_rem = lru.fetch_nearest_cache(model, long_tokens)
        self.assertIsNotNone(exact)
        self.assertEqual(exact_rem, [])

    def test_mixed_cache_longer_prefix_reuse_when_rotating_cache_is_full(self):
        lru = LRUPromptCache(max_size=10)
        model_key = ("step3p5-tiny", None, None)
        model = make_tiny_step3p5_model()

        long_tokens = list(range(1, 13))
        shorter_tokens = long_tokens[:8]
        continuation_tokens = [42, 43, 44, 45]

        long_array = mx.array([long_tokens], dtype=mx.int32)
        shorter_array = mx.array([shorter_tokens], dtype=mx.int32)
        remaining_array = mx.array([shorter_tokens[-1:]], dtype=mx.int32)

        long_cache = model.make_cache()
        mx.eval(model(long_array, cache=long_cache))
        stored_snapshot = copy.deepcopy(long_cache)

        lru.insert_cache(model_key, long_tokens, long_cache)
        reused_cache, remaining = lru.fetch_nearest_cache(model_key, shorter_tokens)

        self.assertIsNotNone(reused_cache)
        self.assertEqual(remaining, shorter_tokens[-1:])

        exact_cache, exact_remaining = lru.fetch_nearest_cache(model_key, long_tokens)
        self.assertIsNotNone(exact_cache)
        self.assertEqual(exact_remaining, [])

        self.assertEqual(
            [c.offset for c in exact_cache],
            [c.offset for c in stored_snapshot],
        )

        baseline_cache = model.make_cache()
        mx.eval(model(shorter_array, cache=baseline_cache))

        mx.eval(model(remaining_array, cache=reused_cache))
        self.assertEqual(
            [c.offset for c in reused_cache], [c.offset for c in baseline_cache]
        )

        for tok in continuation_tokens:
            tok_array = mx.array([[tok]], dtype=mx.int32)
            reused_logits = model(tok_array, cache=reused_cache)
            baseline_logits = model(tok_array, cache=baseline_cache)
            mx.eval(reused_logits, baseline_logits)
            self.assertTrue(
                mx.allclose(reused_logits, baseline_logits, rtol=1e-5, atol=1e-5)
            )
            self.assertEqual(
                [c.offset for c in reused_cache], [c.offset for c in baseline_cache]
            )

    def test_mixed_cache_longer_prefix_reuse_after_chunked_prefill(self):
        lru = LRUPromptCache(max_size=10)
        model_key = ("step3p5-tiny-chunked", None, None)
        model = make_tiny_step3p5_model()

        long_tokens = list(range(1, 13))
        shorter_tokens = long_tokens[:10]
        continuation_tokens = [42, 43, 44]

        long_array = mx.array([long_tokens], dtype=mx.int32)
        shorter_array = mx.array([shorter_tokens], dtype=mx.int32)
        remaining_array = mx.array([shorter_tokens[-1:]], dtype=mx.int32)

        long_cache = model.make_cache()
        mx.eval(model(long_array[:, :8], cache=long_cache))
        mx.eval(model(long_array[:, 8:], cache=long_cache))

        rotating_layers = [c for c in long_cache if isinstance(c, RotatingKVCache)]
        self.assertGreater(len(rotating_layers), 0)
        for sliding in rotating_layers:
            self.assertGreater(sliding.offset, sliding.keys.shape[2])

        lru.insert_cache(model_key, long_tokens, long_cache)
        reused_cache, remaining = lru.fetch_nearest_cache(model_key, shorter_tokens)
        self.assertIsNotNone(reused_cache)
        self.assertEqual(remaining, shorter_tokens[-1:])

        baseline_cache = model.make_cache()
        mx.eval(model(shorter_array, cache=baseline_cache))

        mx.eval(model(remaining_array, cache=reused_cache))
        self.assertEqual(
            [c.offset for c in reused_cache], [c.offset for c in baseline_cache]
        )

        for tok in continuation_tokens:
            tok_array = mx.array([[tok]], dtype=mx.int32)
            reused_logits = model(tok_array, cache=reused_cache)
            baseline_logits = model(tok_array, cache=baseline_cache)
            mx.eval(reused_logits, baseline_logits)
            self.assertTrue(
                mx.allclose(reused_logits, baseline_logits, rtol=1e-5, atol=1e-5)
            )

    def test_longer_hit_unrecoverable_rotating_miss_skips_deepcopy(self):
        lru = LRUPromptCache(max_size=10)
        model = ("skip-deepcopy", None, None)
        long_tokens = [1, 2, 3, 4]
        shorter_tokens = [1, 2]

        unrecoverable = build_real_rotating_cache()
        unrecoverable.offset = unrecoverable.keys.shape[2] + 1
        unrecoverable._idx = unrecoverable.keys.shape[2]

        lru.insert_cache(
            model,
            long_tokens,
            [DeepcopyShouldNotRunLayer(), unrecoverable],
        )

        reused_cache, remaining = lru.fetch_nearest_cache(model, shorter_tokens)
        self.assertIsNone(reused_cache)
        self.assertEqual(remaining, shorter_tokens)

        exact_cache, exact_remaining = lru.fetch_nearest_cache(model, long_tokens)
        self.assertIsNotNone(exact_cache)
        self.assertEqual(exact_remaining, [])

    def test_mixed_cache_longer_prefix_reuse_preserves_refcounted_long_entry(self):
        lru = LRUPromptCache(max_size=10)
        model_key = ("step3p5-tiny", None, None)
        model = make_tiny_step3p5_model()

        long_tokens = list(range(1, 13))
        shorter_tokens = long_tokens[:8]

        long_array = mx.array([long_tokens], dtype=mx.int32)
        long_cache = model.make_cache()
        mx.eval(model(long_array, cache=long_cache))

        lru.insert_cache(model_key, long_tokens, long_cache)
        lru.insert_cache(model_key, long_tokens, long_cache)

        reused_cache, remaining = lru.fetch_nearest_cache(model_key, shorter_tokens)
        self.assertIsNotNone(reused_cache)
        self.assertEqual(remaining, shorter_tokens[-1:])

        hit1, rem1 = lru.fetch_nearest_cache(model_key, long_tokens)
        self.assertIsNotNone(hit1)
        self.assertEqual(rem1, [])

        hit2, rem2 = lru.fetch_nearest_cache(model_key, long_tokens)
        self.assertIsNotNone(hit2)
        self.assertEqual(rem2, [])

        miss, rem3 = lru.fetch_nearest_cache(model_key, long_tokens)
        self.assertIsNone(miss)
        self.assertEqual(rem3, long_tokens)

    def test_mixed_cache_longer_prefix_reuse_misses_after_decode_rotation(self):
        lru = LRUPromptCache(max_size=10)
        model_key = ("step3p5-tiny", None, None)
        model = make_tiny_step3p5_model()

        prompt_tokens = list(range(1, 13))
        decoded_tokens = prompt_tokens + [99]
        shorter_tokens = prompt_tokens[:8]

        prompt_array = mx.array([prompt_tokens], dtype=mx.int32)
        decode_array = mx.array([[decoded_tokens[-1]]], dtype=mx.int32)

        long_cache = model.make_cache()
        mx.eval(model(prompt_array, cache=long_cache))
        mx.eval(model(decode_array, cache=long_cache))

        rotating_layers = [c for c in long_cache if isinstance(c, RotatingKVCache)]
        self.assertGreater(len(rotating_layers), 0)
        for sliding in rotating_layers:
            self.assertGreater(sliding.offset, sliding.keys.shape[2])

        lru.insert_cache(model_key, decoded_tokens, long_cache)
        reused_cache, remaining = lru.fetch_nearest_cache(model_key, shorter_tokens)
        self.assertIsNone(reused_cache)
        self.assertEqual(remaining, shorter_tokens)

        exact_cache, exact_remaining = lru.fetch_nearest_cache(
            model_key, decoded_tokens
        )
        self.assertIsNotNone(exact_cache)
        self.assertEqual(exact_remaining, [])

    def test_mixed_cache_decode_rotation_safe_miss_preserves_exact_snapshot(self):
        lru = LRUPromptCache(max_size=10)
        model_key = ("step3p5-tiny", None, None)
        model = make_tiny_step3p5_model()

        prompt_tokens = list(range(1, 13))
        decode_tokens = [99, 100, 101]
        decoded_tokens = prompt_tokens + decode_tokens
        shorter_tokens = prompt_tokens[:8]

        prompt_array = mx.array([prompt_tokens], dtype=mx.int32)

        long_cache = model.make_cache()
        mx.eval(model(prompt_array, cache=long_cache))
        for tok in decode_tokens:
            tok_array = mx.array([[tok]], dtype=mx.int32)
            mx.eval(model(tok_array, cache=long_cache))
        stored_snapshot = copy.deepcopy(long_cache)

        lru.insert_cache(model_key, decoded_tokens, long_cache)
        lru.insert_cache(model_key, decoded_tokens, long_cache)
        reused_cache, remaining = lru.fetch_nearest_cache(model_key, shorter_tokens)
        self.assertIsNone(reused_cache)
        self.assertEqual(remaining, shorter_tokens)

        def assert_matches_snapshot(cache):
            self.assertEqual(
                [c.offset for c in cache], [c.offset for c in stored_snapshot]
            )
            for fetched, snap in zip(cache, stored_snapshot):
                fk, fv = fetched.state
                sk, sv = snap.state
                self.assertTrue(mx.array_equal(fk, sk))
                self.assertTrue(mx.array_equal(fv, sv))

        hit1, rem1 = lru.fetch_nearest_cache(model_key, decoded_tokens)
        self.assertIsNotNone(hit1)
        self.assertEqual(rem1, [])
        assert_matches_snapshot(hit1)

        hit2, rem2 = lru.fetch_nearest_cache(model_key, decoded_tokens)
        self.assertIsNotNone(hit2)
        self.assertEqual(rem2, [])
        assert_matches_snapshot(hit2)

        miss, rem3 = lru.fetch_nearest_cache(model_key, decoded_tokens)
        self.assertIsNone(miss)
        self.assertEqual(rem3, decoded_tokens)


if __name__ == "__main__":
    unittest.main()
