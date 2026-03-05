# Copyright © 2024 Apple Inc.

import unittest

import mlx.core as mx

from mlx_lm.models.cache import CacheList, RotatingKVCache
from mlx_lm.server import LRUPromptCache
from tests.prompt_cache_test_utils import (
    RewindRecorderLayer,
    build_real_rotating_cache,
    snapshot_cache_arrays,
)


class TestLRUPromptCacheRewindInternals(unittest.TestCase):
    def test_rewind_prompt_cache_fails_closed_on_cachelist_partial_trim_matrix(self):
        def build_first_child_partial():
            partial = RewindRecorderLayer(max_rewind=10, rewind_result=False)
            full = RewindRecorderLayer(max_rewind=10, rewind_result=True)
            return CacheList(partial, full), {
                "partial": partial,
                "full": full,
            }

        def build_nested_partial():
            partial = RewindRecorderLayer(max_rewind=10, rewind_result=False)
            inner_full = RewindRecorderLayer(max_rewind=10, rewind_result=True)
            outer_full = RewindRecorderLayer(max_rewind=10, rewind_result=True)
            nested = CacheList(CacheList(partial, inner_full), outer_full)
            return nested, {
                "partial": partial,
                "inner_full": inner_full,
                "outer_full": outer_full,
            }

        def build_second_child_partial():
            first_full = RewindRecorderLayer(max_rewind=10, rewind_result=True)
            second_partial = RewindRecorderLayer(max_rewind=10, rewind_result=False)
            return CacheList(first_full, second_partial), {
                "first_full": first_full,
                "second_partial": second_partial,
            }

        scenarios = [
            (
                "first_child_partial",
                build_first_child_partial,
                {
                    "partial": [5],
                    "full": [],
                },
            ),
            (
                "nested_partial",
                build_nested_partial,
                {
                    "partial": [5],
                    "inner_full": [],
                    "outer_full": [],
                },
            ),
            (
                "second_child_partial",
                build_second_child_partial,
                {
                    "first_full": [5],
                    "second_partial": [5],
                },
            ),
        ]

        for name, builder, expected_calls in scenarios:
            with self.subTest(name=name):
                lru = LRUPromptCache(max_size=10)
                composite, layers = builder()
                ok = lru._rewind_prompt_cache([composite], 5)
                self.assertFalse(ok)
                for layer_name, expected_rewind_calls in expected_calls.items():
                    self.assertEqual(
                        layers[layer_name].rewind_calls,
                        expected_rewind_calls,
                        msg=f"unexpected rewind calls for {layer_name}",
                    )

    def test_rewind_rotating_cache_failure_paths_preserve_state(self):
        def missing_values(cache):
            cache.values = None
            return 2

        def exceeds_offset(cache):
            return cache.offset + 1

        def unrecoverable_history(cache):
            cache.offset = cache.keys.shape[2] + 1
            cache._idx = cache.keys.shape[2]
            return 1

        def exceeds_idx(cache):
            cache._idx = 0
            return 1

        scenarios = [
            ("missing_values", missing_values),
            ("trim_exceeds_offset", exceeds_offset),
            ("history_unrecoverable", unrecoverable_history),
            ("trim_exceeds_idx", exceeds_idx),
        ]

        for name, configure in scenarios:
            with self.subTest(name=name):
                cache = build_real_rotating_cache()
                num_to_trim = configure(cache)
                original_offset = cache.offset
                original_idx = cache._idx
                original_keys, original_values = snapshot_cache_arrays(cache)

                self.assertFalse(cache.can_rewind(num_to_trim))
                self.assertFalse(cache.rewind(num_to_trim))
                self.assertEqual(cache.offset, original_offset)
                self.assertEqual(cache._idx, original_idx)
                self.assertTrue(mx.array_equal(cache.keys, original_keys))
                if original_values is None:
                    self.assertIsNone(cache.values)
                else:
                    self.assertTrue(mx.array_equal(cache.values, original_values))

    def test_rewind_rotating_cache_materializes_state_for_single_token_updates(self):
        total_tokens = 8
        trim_tokens = 3
        expected_prefix = total_tokens - trim_tokens

        rewound = build_real_rotating_cache(total_tokens=total_tokens)
        self.assertTrue(rewound.can_rewind(trim_tokens))
        self.assertTrue(rewound.rewind(trim_tokens))

        next_tok = mx.array([[[[99.0]]]], dtype=mx.float32)
        rewound.update_and_fetch(next_tok, next_tok)
        mx.eval(rewound.keys, rewound.values)

        baseline = build_real_rotating_cache(total_tokens=expected_prefix)
        baseline.update_and_fetch(next_tok, next_tok)
        mx.eval(baseline.keys, baseline.values)

        self.assertEqual(rewound.offset, baseline.offset)
        self.assertEqual(rewound._idx, baseline._idx)
        self.assertTrue(mx.array_equal(rewound.keys, baseline.keys))
        self.assertTrue(mx.array_equal(rewound.values, baseline.values))

    def test_rewind_rotating_cache_zero_trim_is_noop(self):
        cache = build_real_rotating_cache(total_tokens=4)
        original_offset = cache.offset
        original_idx = cache._idx
        original_keys, original_values = snapshot_cache_arrays(cache)

        self.assertTrue(cache.can_rewind(0))
        self.assertTrue(cache.rewind(0))
        self.assertEqual(cache.offset, original_offset)
        self.assertEqual(cache._idx, original_idx)
        self.assertTrue(mx.array_equal(cache.keys, original_keys))
        self.assertTrue(mx.array_equal(cache.values, original_values))

    def test_rewind_rotating_cache_clamps_idx_after_state_materialization(self):
        cache = RotatingKVCache(max_size=8)
        for tok in range(6):
            kv = mx.array([[[[float(tok)]]]], dtype=mx.float32)
            cache.update_and_fetch(kv, kv)
        mx.eval(cache.keys, cache.values)

        self.assertLess(cache.offset, cache.keys.shape[2])
        cache._idx = cache.keys.shape[2]
        self.assertGreater(cache._idx, cache.offset)

        self.assertTrue(cache.can_rewind(1))
        self.assertTrue(cache.rewind(1))
        self.assertEqual(cache.offset, 5)
        self.assertEqual(cache.keys.shape[2], 5)
        self.assertEqual(cache._idx, cache.keys.shape[2])

    def test_rewind_prompt_cache_uses_strict_rotating_guard_even_when_trimmable(self):
        cache = RotatingKVCache(max_size=8)
        kv = mx.arange(4, dtype=mx.float32).reshape(1, 1, 4, 1)
        cache.update_and_fetch(kv, kv)
        mx.eval(cache.keys, cache.values)

        self.assertTrue(cache.is_trimmable())
        cache.values = None
        original_offset = cache.offset
        original_idx = cache._idx
        original_keys, _ = snapshot_cache_arrays(cache)

        lru = LRUPromptCache(max_size=10)
        ok = lru._rewind_prompt_cache([cache], 1)
        self.assertFalse(ok)
        self.assertEqual(cache.offset, original_offset)
        self.assertEqual(cache._idx, original_idx)
        self.assertTrue(mx.array_equal(cache.keys, original_keys))
        self.assertIsNone(cache.values)


if __name__ == "__main__":
    unittest.main()
