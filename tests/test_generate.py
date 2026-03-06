# Copyright © 2024 Apple Inc.

import importlib
import random
import sys
import unittest
from typing import List
from unittest.mock import patch

import mlx.core as mx

from mlx_lm.generate import (
    BatchGenerator,
    GenerationResponse,
    batch_generate,
    can_rewind_prompt_cache,
    generate,
    generate_step,
    maybe_quantize_kv_cache,
    rewind_prompt_cache,
    speculative_generate_step,
    stream_generate,
)
from mlx_lm.models.cache import (
    CacheList,
    KVCache,
    QuantizedKVCache,
    RotatingKVCache,
    can_trim_prompt_cache,
)
from mlx_lm.sample_utils import make_logits_processors, make_sampler
from mlx_lm.tokenizer_utils import TokenizerWrapper
from mlx_lm.utils import load

generate_module = importlib.import_module("mlx_lm.generate")


class TestKVBitsCoverage(unittest.TestCase):
    @staticmethod
    def _make_tiny_step3p5_model():
        from mlx_lm.models import step3p5

        args = step3p5.ModelArgs.from_dict(
            {
                "model_type": "step3p5",
                "hidden_size": 128,
                "num_hidden_layers": 4,
                "vocab_size": 256,
                "num_attention_heads": 4,
                "num_attention_groups": 2,
                "head_dim": 32,
                "intermediate_size": 256,
                "rms_norm_eps": 1e-5,
                "rope_theta": [10000.0, 10000.0, 10000.0, 10000.0],
                "sliding_window": 4,
                "layer_types": [
                    "full_attention",
                    "sliding_attention",
                    "sliding_attention",
                    "full_attention",
                ],
                "partial_rotary_factors": [1.0, 1.0, 1.0, 1.0],
                "attention_other_setting": {
                    "num_attention_heads": 4,
                    "num_attention_groups": 2,
                },
                "use_head_wise_attn_gate": True,
                "moe_num_experts": 4,
                "moe_top_k": 2,
                "moe_intermediate_size": 128,
                "share_expert_dim": 128,
                "moe_layers_enum": "1,2,3",
            }
        )
        return step3p5.Model(args)

    def test_maybe_quantize_kv_cache_honors_threshold_and_none_bits(self):
        class QuantizedCache:
            def __init__(self, bits, group_size):
                self.bits = bits
                self.group_size = group_size

        class QuantizableCache:
            def __init__(self, offset):
                self.offset = offset
                self.calls = []

            def to_quantized(self, group_size, bits):
                self.calls.append((group_size, bits))
                return QuantizedCache(bits=bits, group_size=group_size)

        class NonQuantizableCache:
            def __init__(self, offset):
                self.offset = offset

        low_offset = QuantizableCache(offset=3)
        high_offset = QuantizableCache(offset=5)
        no_method = NonQuantizableCache(offset=99)
        prompt_cache = [low_offset, high_offset, no_method]

        maybe_quantize_kv_cache(
            prompt_cache,
            quantized_kv_start=5,
            kv_group_size=32,
            kv_bits=None,
        )
        self.assertEqual(low_offset.calls, [])
        self.assertEqual(high_offset.calls, [])
        self.assertIs(prompt_cache[0], low_offset)
        self.assertIs(prompt_cache[1], high_offset)
        self.assertIs(prompt_cache[2], no_method)

        maybe_quantize_kv_cache(
            prompt_cache,
            quantized_kv_start=5,
            kv_group_size=32,
            kv_bits=4,
        )
        self.assertEqual(low_offset.calls, [])
        self.assertEqual(high_offset.calls, [(32, 4)])
        self.assertIs(prompt_cache[0], low_offset)
        self.assertIs(prompt_cache[2], no_method)
        self.assertIsInstance(prompt_cache[1], QuantizedCache)
        self.assertEqual(prompt_cache[1].bits, 4)
        self.assertEqual(prompt_cache[1].group_size, 32)

    def test_generate_step_step3p5_kv_bits_skips_rotating_cache_quantization(self):
        model = self._make_tiny_step3p5_model()
        prompt_cache = model.make_cache()

        prompt = mx.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=mx.uint32)
        next(
            generate_step(
                prompt=prompt,
                model=model,
                prompt_cache=prompt_cache,
                max_tokens=1,
                kv_bits=4,
                kv_group_size=32,
                quantized_kv_start=0,
            )
        )

        self.assertGreater(
            sum(isinstance(c, QuantizedKVCache) for c in prompt_cache), 0
        )
        self.assertGreater(sum(isinstance(c, RotatingKVCache) for c in prompt_cache), 0)

    def test_rewind_prompt_cache_rewinds_step3p5_mixed_saturated_cache(self):
        model = self._make_tiny_step3p5_model()
        prompt_cache = model.make_cache()

        prompt = mx.array([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=mx.int32)
        mx.eval(model(prompt, cache=prompt_cache))

        self.assertFalse(can_trim_prompt_cache(prompt_cache))
        offsets_before = [c.offset for c in prompt_cache]
        self.assertTrue(rewind_prompt_cache(prompt_cache, 1))
        self.assertEqual(
            [c.offset for c in prompt_cache], [o - 1 for o in offsets_before]
        )

    def test_rewind_prompt_cache_fails_closed_without_partial_mutation(self):
        class RewindLayer:
            def __init__(self, *, offset, can_rewind_result, rewind_result):
                self.offset = offset
                self.can_rewind_result = can_rewind_result
                self.rewind_result = rewind_result
                self.can_rewind_calls = []
                self.rewind_calls = []

            def can_rewind(self, n):
                self.can_rewind_calls.append(n)
                return self.can_rewind_result

            def rewind(self, n):
                self.rewind_calls.append(n)
                self.offset -= n
                return self.rewind_result

        first = RewindLayer(offset=9, can_rewind_result=True, rewind_result=True)
        second = RewindLayer(offset=9, can_rewind_result=False, rewind_result=False)
        prompt_cache = [first, second]

        self.assertFalse(rewind_prompt_cache(prompt_cache, 2))
        self.assertEqual(first.offset, 9)
        self.assertEqual(second.offset, 9)
        self.assertEqual(first.rewind_calls, [])
        self.assertEqual(second.rewind_calls, [])

    def test_maybe_quantize_kv_cache_caches_non_quantizable_layers(self):
        class UnsupportedQuantCache:
            def __init__(self, offset):
                self.offset = offset
                self.calls = 0

            def to_quantized(self, group_size, bits):
                self.calls += 1
                raise NotImplementedError("quantization unsupported")

        unsupported = UnsupportedQuantCache(offset=32)
        prompt_cache = [unsupported]

        for _ in range(3):
            maybe_quantize_kv_cache(
                prompt_cache,
                quantized_kv_start=0,
                kv_group_size=32,
                kv_bits=4,
            )

        self.assertEqual(unsupported.calls, 1)

    def test_maybe_quantize_kv_cache_retries_for_different_quantization_config(self):
        class ConfigSensitiveQuantCache:
            def __init__(self, offset):
                self.offset = offset
                self.calls = []

            def to_quantized(self, group_size, bits):
                self.calls.append((group_size, bits))
                if bits == 4:
                    raise NotImplementedError("4-bit unsupported")
                return QuantizedKVCache(bits=bits, group_size=group_size)

        cache_layer = ConfigSensitiveQuantCache(offset=32)
        prompt_cache = [cache_layer]

        maybe_quantize_kv_cache(
            prompt_cache,
            quantized_kv_start=0,
            kv_group_size=32,
            kv_bits=4,
        )
        maybe_quantize_kv_cache(
            prompt_cache,
            quantized_kv_start=0,
            kv_group_size=32,
            kv_bits=8,
        )

        self.assertEqual(cache_layer.calls, [(32, 4), (32, 8)])
        self.assertIsInstance(prompt_cache[0], QuantizedKVCache)
        self.assertEqual(prompt_cache[0].bits, 8)
        self.assertEqual(prompt_cache[0].group_size, 32)

    def test_can_rewind_prompt_cache_requires_callable_rewind_or_trim(self):
        class RewindableLayer:
            def __init__(self, offset):
                self.offset = offset
                self.rewind_calls = []

            def can_rewind(self, n):
                return True

            def rewind(self, n):
                self.rewind_calls.append(n)
                self.offset -= n
                return True

        class CanOnlyLayer:
            def can_rewind(self, n):
                return True

        first = RewindableLayer(offset=9)
        second = CanOnlyLayer()
        prompt_cache = [first, second]

        self.assertFalse(can_rewind_prompt_cache(prompt_cache, 2))
        self.assertFalse(rewind_prompt_cache(prompt_cache, 2))
        self.assertEqual(first.offset, 9)
        self.assertEqual(first.rewind_calls, [])

    def test_rewind_prompt_cache_rolls_back_nested_mutable_state(self):
        class NestedStateLayer:
            def __init__(self):
                self.state = [[10]]

            def can_rewind(self, n):
                return True

            def rewind(self, n):
                self.state[0][0] -= n
                return False

        layer = NestedStateLayer()
        self.assertFalse(rewind_prompt_cache([layer], 2))
        self.assertEqual(layer.state, [[10]])

    def test_rewind_prompt_cache_rolls_back_nested_custom_object_state(self):
        class Box:
            def __init__(self, value):
                self.value = value

        class CustomObjectStateLayer:
            def __init__(self):
                self.state = {"box": Box(10)}

            def can_rewind(self, n):
                return True

            def rewind(self, n):
                self.state["box"].value -= n
                return False

        layer = CustomObjectStateLayer()
        self.assertFalse(rewind_prompt_cache([layer], 2))
        self.assertEqual(layer.state["box"].value, 10)

    def test_can_rewind_prompt_cache_requires_snapshotable_layers(self):
        class UnsnapshotableLayer:
            __slots__ = ()

            def can_rewind(self, n):
                return True

            def rewind(self, n):
                return True

        layer = UnsnapshotableLayer()
        self.assertFalse(can_rewind_prompt_cache([layer], 1))
        self.assertFalse(rewind_prompt_cache([layer], 1))

    def test_rewind_prompt_cache_cyclic_state_fails_closed_without_exception(self):
        class CyclicStateLayer:
            def __init__(self):
                self.offset = 10
                self.state = []
                self.state.append(self.state)

            def can_rewind(self, n):
                return True

            def rewind(self, n):
                self.offset -= n
                return True

        layer = CyclicStateLayer()
        self.assertFalse(can_rewind_prompt_cache([layer], 1))
        try:
            result = rewind_prompt_cache([layer], 1)
        except RecursionError as exc:
            self.fail(f"rewind_prompt_cache raised RecursionError: {exc}")
        self.assertFalse(result)
        self.assertEqual(layer.offset, 10)

    def test_rewind_prompt_cache_hybrid_slots_and_dict_state_fails_closed(self):
        class HybridStateLayer:
            __slots__ = ("offset", "slot_state", "__dict__")

            def __init__(self):
                self.offset = 10
                self.slot_state = {"value": 7}
                self.dict_state = [5]

            def can_rewind(self, n):
                return True

            def rewind(self, n):
                self.offset -= n
                self.slot_state["value"] -= n
                self.dict_state[0] -= n
                return False

        layer = HybridStateLayer()
        self.assertFalse(can_rewind_prompt_cache([layer], 1))
        self.assertFalse(rewind_prompt_cache([layer], 1))
        self.assertEqual(layer.offset, 10)
        self.assertEqual(layer.slot_state, {"value": 7})
        self.assertEqual(layer.dict_state, [5])

    def test_rewind_prompt_cache_hybrid_slots_declared_but_unset_fails_closed(self):
        class HybridUnsetSlotLayer:
            __slots__ = ("offset", "slot_state", "__dict__")

            def __init__(self):
                self.offset = 10
                self.dict_state = [5]

            def can_rewind(self, n):
                return True

            def rewind(self, n):
                self.offset -= n
                self.slot_state = {"value": 7}
                self.dict_state[0] -= n
                return False

        layer = HybridUnsetSlotLayer()
        self.assertFalse(hasattr(layer, "slot_state"))
        self.assertFalse(can_rewind_prompt_cache([layer], 1))
        self.assertFalse(rewind_prompt_cache([layer], 1))
        self.assertEqual(layer.offset, 10)
        self.assertEqual(layer.dict_state, [5])
        self.assertFalse(hasattr(layer, "slot_state"))

    def test_rewind_prompt_cache_slot_only_new_slot_created_on_failure_is_removed(
        self,
    ):
        class SlotOnlyLayer:
            __slots__ = ("offset", "slot_state")

            def __init__(self):
                self.offset = 10

            def can_rewind(self, n):
                return True

            def rewind(self, n):
                self.offset -= n
                self.slot_state = {"value": 7}
                return False

        layer = SlotOnlyLayer()
        self.assertFalse(hasattr(layer, "slot_state"))
        self.assertTrue(can_rewind_prompt_cache([layer], 1))
        self.assertFalse(rewind_prompt_cache([layer], 1))
        self.assertEqual(layer.offset, 10)
        self.assertFalse(hasattr(layer, "slot_state"))

    def test_generate_step_quantizes_eligible_cache_layer(self):
        class QuantizedCache:
            def __init__(self, bits, group_size):
                self.bits = bits
                self.group_size = group_size

        class QuantizableCache:
            def __init__(self, offset):
                self.offset = offset
                self.calls = []

            def to_quantized(self, group_size, bits):
                self.calls.append((group_size, bits))
                return QuantizedCache(bits=bits, group_size=group_size)

        class SimpleModel:
            layers = [object()]

            def __call__(self, input_tokens, cache=None, input_embeddings=None):
                batch, seq_len = input_tokens.shape
                vocab_size = 4
                return mx.zeros((batch, seq_len, vocab_size), dtype=mx.float32)

        cache_layer = QuantizableCache(offset=7)
        prompt_cache = [cache_layer]
        prompt = mx.array([1], dtype=mx.uint32)

        token, _ = next(
            generate_step(
                prompt=prompt,
                model=SimpleModel(),
                prompt_cache=prompt_cache,
                max_tokens=1,
                kv_bits=6,
                kv_group_size=16,
                quantized_kv_start=4,
            )
        )

        self.assertEqual(token, 0)
        self.assertEqual(cache_layer.calls, [(16, 6)])
        self.assertIsInstance(prompt_cache[0], QuantizedCache)
        self.assertEqual(prompt_cache[0].bits, 6)
        self.assertEqual(prompt_cache[0].group_size, 16)

    def test_speculative_generate_step_raises_on_rewind_failure(self):
        class FlakyRewindCache:
            offset = 8

            def can_rewind(self, n):
                return True

            def rewind(self, n):
                self.offset -= n
                return False

        class FixedTokenModel:
            layers = [object()]

            def __init__(self, forced_token):
                self.forced_token = forced_token

            def make_cache(self):
                return [FlakyRewindCache()]

            def __call__(self, input_tokens, cache=None, input_embeddings=None):
                batch, seq_len = input_tokens.shape
                vocab_size = 4
                token_logits = -1000.0 * mx.ones((vocab_size,), dtype=mx.float32)
                token_logits = token_logits + (
                    2000.0 * (mx.arange(vocab_size) == self.forced_token)
                )
                return mx.broadcast_to(token_logits, (batch, seq_len, vocab_size))

        prompt = mx.array([1, 2, 3], dtype=mx.uint32)
        gen = speculative_generate_step(
            prompt=prompt,
            model=FixedTokenModel(forced_token=0),
            draft_model=FixedTokenModel(forced_token=1),
            max_tokens=2,
            num_draft_tokens=2,
        )
        next(gen)
        with self.assertRaisesRegex(
            RuntimeError, "Speculative decoding cache rewind failed"
        ):
            next(gen)
        gen.close()

    def test_speculative_generate_step_rewind_failure_keeps_caller_cache_unchanged(
        self,
    ):
        class RewindLayer:
            def __init__(self, *, offset, can_rewind_result, rewind_result):
                self.offset = offset
                self.can_rewind_result = can_rewind_result
                self.rewind_result = rewind_result
                self.rewind_calls = []

            def can_rewind(self, n):
                return self.can_rewind_result

            def rewind(self, n):
                self.rewind_calls.append(n)
                if self.rewind_result:
                    self.offset -= n
                return self.rewind_result

        class FixedTokenModel:
            def __init__(self, forced_token, num_layers):
                self.forced_token = forced_token
                self.layers = [object() for _ in range(num_layers)]

            def __call__(self, input_tokens, cache=None, input_embeddings=None):
                batch, seq_len = input_tokens.shape
                vocab_size = 4
                token_logits = -1000.0 * mx.ones((vocab_size,), dtype=mx.float32)
                token_logits = token_logits + (
                    2000.0 * (mx.arange(vocab_size) == self.forced_token)
                )
                return mx.broadcast_to(token_logits, (batch, seq_len, vocab_size))

        model_cache_ok = RewindLayer(
            offset=12, can_rewind_result=True, rewind_result=True
        )
        model_cache_fail = RewindLayer(
            offset=12, can_rewind_result=False, rewind_result=False
        )
        draft_cache = RewindLayer(offset=12, can_rewind_result=True, rewind_result=True)
        prompt_cache = [model_cache_ok, model_cache_fail, draft_cache]

        prompt = mx.array([1, 2, 3], dtype=mx.uint32)
        gen = speculative_generate_step(
            prompt=prompt,
            model=FixedTokenModel(forced_token=0, num_layers=2),
            draft_model=FixedTokenModel(forced_token=1, num_layers=1),
            prompt_cache=prompt_cache,
            max_tokens=2,
            num_draft_tokens=2,
        )
        next(gen)
        with self.assertRaisesRegex(
            RuntimeError, "Speculative decoding cache rewind failed"
        ):
            next(gen)

        self.assertEqual(model_cache_ok.offset, 12)
        self.assertEqual(model_cache_fail.offset, 12)
        self.assertEqual(draft_cache.offset, 12)
        self.assertEqual(model_cache_ok.rewind_calls, [])
        self.assertEqual(model_cache_fail.rewind_calls, [])
        self.assertEqual(draft_cache.rewind_calls, [])

    def test_speculative_generate_step_preserves_primary_exception_on_cleanup_failure(
        self,
    ):
        class NonRewindCache:
            offset = 8

            def can_rewind(self, n):
                return False

            def rewind(self, n):
                return False

        class TargetModel:
            layers = [object()]

            def make_cache(self):
                return [NonRewindCache()]

            def __call__(self, input_tokens, cache=None, input_embeddings=None):
                batch, seq_len = input_tokens.shape
                vocab_size = 4
                token_logits = -1000.0 * mx.ones((vocab_size,), dtype=mx.float32)
                token_logits = token_logits + (2000.0 * (mx.arange(vocab_size) == 0))
                return mx.broadcast_to(token_logits, (batch, seq_len, vocab_size))

        class FailingDraftModel:
            layers = [object()]

            def make_cache(self):
                return [NonRewindCache()]

            def __call__(self, input_tokens, cache=None, input_embeddings=None):
                raise ValueError("draft boom")

        prompt = mx.array([1, 2, 3], dtype=mx.uint32)
        with self.assertRaisesRegex(ValueError, "draft boom"):
            next(
                speculative_generate_step(
                    prompt=prompt,
                    model=TargetModel(),
                    draft_model=FailingDraftModel(),
                    max_tokens=2,
                    num_draft_tokens=2,
                )
            )

    def test_speculative_generate_step_close_triggers_best_effort_rewind(self):
        class TrackingLayer:
            def __init__(self, offset):
                self.offset = offset
                self.rewind_calls = []

            def can_rewind(self, n):
                return n <= self.offset

            def rewind(self, n):
                self.rewind_calls.append(n)
                self.offset -= n
                return True

        class FixedTokenModel:
            def __init__(self, forced_token, num_layers):
                self.forced_token = forced_token
                self.layers = [object() for _ in range(num_layers)]

            def __call__(self, input_tokens, cache=None, input_embeddings=None):
                batch, seq_len = input_tokens.shape
                vocab_size = 4
                token_logits = -1000.0 * mx.ones((vocab_size,), dtype=mx.float32)
                token_logits = token_logits + (
                    2000.0 * (mx.arange(vocab_size) == self.forced_token)
                )
                return mx.broadcast_to(token_logits, (batch, seq_len, vocab_size))

        model_cache = TrackingLayer(offset=12)
        draft_cache = TrackingLayer(offset=12)
        prompt_cache = [model_cache, draft_cache]

        prompt = mx.array([1, 2, 3], dtype=mx.uint32)
        gen = speculative_generate_step(
            prompt=prompt,
            model=FixedTokenModel(forced_token=0, num_layers=1),
            draft_model=FixedTokenModel(forced_token=1, num_layers=1),
            prompt_cache=prompt_cache,
            max_tokens=2,
            num_draft_tokens=2,
        )
        next(gen)
        self.assertEqual(model_cache.rewind_calls, [])
        self.assertEqual(draft_cache.rewind_calls, [])
        gen.close()
        self.assertEqual(model_cache.rewind_calls, [2])
        self.assertEqual(draft_cache.rewind_calls, [1])

    def test_speculative_generate_step_draft_preflight_failure_does_not_mutate_model(
        self,
    ):
        class TrackingLayer:
            def __init__(self, *, offset, can_rewind_result, rewind_result):
                self.offset = offset
                self.can_rewind_result = can_rewind_result
                self.rewind_result = rewind_result
                self.rewind_calls = []

            def can_rewind(self, n):
                return self.can_rewind_result

            def rewind(self, n):
                self.rewind_calls.append(n)
                if self.rewind_result:
                    self.offset -= n
                return self.rewind_result

        class FixedTokenModel:
            def __init__(self, forced_token):
                self.forced_token = forced_token
                self.layers = [object()]

            def __call__(self, input_tokens, cache=None, input_embeddings=None):
                batch, seq_len = input_tokens.shape
                vocab_size = 4
                token_logits = -1000.0 * mx.ones((vocab_size,), dtype=mx.float32)
                token_logits = token_logits + (
                    2000.0 * (mx.arange(vocab_size) == self.forced_token)
                )
                return mx.broadcast_to(token_logits, (batch, seq_len, vocab_size))

        model_cache = TrackingLayer(
            offset=12, can_rewind_result=True, rewind_result=True
        )
        draft_cache = TrackingLayer(
            offset=12, can_rewind_result=False, rewind_result=False
        )
        prompt_cache = [model_cache, draft_cache]

        prompt = mx.array([1, 2, 3], dtype=mx.uint32)
        gen = speculative_generate_step(
            prompt=prompt,
            model=FixedTokenModel(forced_token=0),
            draft_model=FixedTokenModel(forced_token=1),
            prompt_cache=prompt_cache,
            max_tokens=2,
            num_draft_tokens=2,
        )
        next(gen)
        with self.assertRaisesRegex(
            RuntimeError, "Speculative decoding cache rewind failed for draft cache"
        ):
            next(gen)

        self.assertEqual(model_cache.offset, 12)
        self.assertEqual(model_cache.rewind_calls, [])
        self.assertEqual(draft_cache.offset, 12)
        self.assertEqual(draft_cache.rewind_calls, [])

    def test_speculative_generate_step_draft_rewind_failure_does_not_mutate_model(
        self,
    ):
        class TrackingLayer:
            def __init__(self, *, offset, can_rewind_result, rewind_result):
                self.offset = offset
                self.can_rewind_result = can_rewind_result
                self.rewind_result = rewind_result
                self.rewind_calls = []

            def can_rewind(self, n):
                return self.can_rewind_result

            def rewind(self, n):
                self.rewind_calls.append(n)
                if self.rewind_result:
                    self.offset -= n
                return self.rewind_result

        class FixedTokenModel:
            def __init__(self, forced_token):
                self.forced_token = forced_token
                self.layers = [object()]

            def __call__(self, input_tokens, cache=None, input_embeddings=None):
                batch, seq_len = input_tokens.shape
                vocab_size = 4
                token_logits = -1000.0 * mx.ones((vocab_size,), dtype=mx.float32)
                token_logits = token_logits + (
                    2000.0 * (mx.arange(vocab_size) == self.forced_token)
                )
                return mx.broadcast_to(token_logits, (batch, seq_len, vocab_size))

        model_cache = TrackingLayer(
            offset=12, can_rewind_result=True, rewind_result=True
        )
        draft_cache = TrackingLayer(
            offset=12, can_rewind_result=True, rewind_result=False
        )
        prompt_cache = [model_cache, draft_cache]

        prompt = mx.array([1, 2, 3], dtype=mx.uint32)
        gen = speculative_generate_step(
            prompt=prompt,
            model=FixedTokenModel(forced_token=0),
            draft_model=FixedTokenModel(forced_token=1),
            prompt_cache=prompt_cache,
            max_tokens=2,
            num_draft_tokens=2,
        )
        next(gen)
        with self.assertRaisesRegex(
            RuntimeError, "Speculative decoding cache rewind failed for draft cache"
        ):
            next(gen)

        self.assertEqual(model_cache.offset, 12)
        self.assertEqual(model_cache.rewind_calls, [])
        self.assertEqual(draft_cache.offset, 12)
        self.assertEqual(draft_cache.rewind_calls, [])

    def test_speculative_generate_step_draft_snapshot_failure_reports_draft_cache(
        self,
    ):
        class ModelLayer:
            def __init__(self, offset):
                self.offset = offset

            def can_rewind(self, n):
                return True

            def rewind(self, n):
                self.offset -= n
                return True

        class UnsnapshotableDraftLayer:
            __slots__ = ()

            def can_rewind(self, n):
                return True

            def rewind(self, n):
                return True

        class FixedTokenModel:
            def __init__(self, forced_token):
                self.forced_token = forced_token
                self.layers = [object()]

            def __call__(self, input_tokens, cache=None, input_embeddings=None):
                batch, seq_len = input_tokens.shape
                vocab_size = 4
                token_logits = -1000.0 * mx.ones((vocab_size,), dtype=mx.float32)
                token_logits = token_logits + (
                    2000.0 * (mx.arange(vocab_size) == self.forced_token)
                )
                return mx.broadcast_to(token_logits, (batch, seq_len, vocab_size))

        prompt = mx.array([1, 2, 3], dtype=mx.uint32)
        gen = speculative_generate_step(
            prompt=prompt,
            model=FixedTokenModel(forced_token=0),
            draft_model=FixedTokenModel(forced_token=1),
            prompt_cache=[ModelLayer(offset=12), UnsnapshotableDraftLayer()],
            max_tokens=2,
            num_draft_tokens=2,
        )
        next(gen)
        with self.assertRaisesRegex(
            RuntimeError, "Speculative decoding cache rewind failed for draft cache"
        ):
            next(gen)

    def test_rewind_prompt_cache_legacy_trim_only_layer_still_rewinds(self):
        class LegacyTrimOnlyLayer:
            def __init__(self):
                self.trim_calls = []

            def is_trimmable(self):
                return True

            def trim(self, n):
                self.trim_calls.append(n)
                return n

        layer = LegacyTrimOnlyLayer()
        self.assertTrue(rewind_prompt_cache([layer], 3))
        self.assertEqual(layer.trim_calls, [3])

    def test_speculative_generate_step_surfaces_cleanup_rewind_failure_on_success_path(
        self,
    ):
        class FlakyRewindCache:
            offset = 8

            def can_rewind(self, n):
                return True

            def rewind(self, n):
                self.offset -= n
                return False

        class FixedTokenModel:
            layers = [object()]

            def __init__(self, forced_token):
                self.forced_token = forced_token

            def __call__(self, input_tokens, cache=None, input_embeddings=None):
                batch, seq_len = input_tokens.shape
                vocab_size = 4
                token_logits = -1000.0 * mx.ones((vocab_size,), dtype=mx.float32)
                token_logits = token_logits + (
                    2000.0 * (mx.arange(vocab_size) == self.forced_token)
                )
                return mx.broadcast_to(token_logits, (batch, seq_len, vocab_size))

        prompt = mx.array([1, 2, 3], dtype=mx.uint32)
        gen = speculative_generate_step(
            prompt=prompt,
            model=FixedTokenModel(forced_token=0),
            draft_model=FixedTokenModel(forced_token=1),
            prompt_cache=[FlakyRewindCache(), FlakyRewindCache()],
            max_tokens=1,
            num_draft_tokens=1,
        )
        next(gen)
        with self.assertRaisesRegex(
            RuntimeError, "Speculative decoding cache rewind failed"
        ):
            next(gen)

    def test_speculative_generate_step_zero_trim_rewind_skips_snapshot_requirements(
        self,
    ):
        class UnsnapshotableLayer:
            __slots__ = ()

            def can_rewind(self, n):
                return True

            def rewind(self, n):
                return True

        class FixedTokenModel:
            def __init__(self, forced_token):
                self.forced_token = forced_token
                self.layers = [object()]

            def __call__(self, input_tokens, cache=None, input_embeddings=None):
                batch, seq_len = input_tokens.shape
                vocab_size = 4
                token_logits = -1000.0 * mx.ones((vocab_size,), dtype=mx.float32)
                token_logits = token_logits + (
                    2000.0 * (mx.arange(vocab_size) == self.forced_token)
                )
                return mx.broadcast_to(token_logits, (batch, seq_len, vocab_size))

        prompt = mx.array([1, 2, 3], dtype=mx.uint32)
        gen = speculative_generate_step(
            prompt=prompt,
            model=FixedTokenModel(forced_token=0),
            draft_model=FixedTokenModel(forced_token=0),
            prompt_cache=[UnsnapshotableLayer(), UnsnapshotableLayer()],
            max_tokens=1,
            num_draft_tokens=1,
        )
        token, _logprobs, from_draft = next(gen)
        self.assertEqual(token, 0)
        self.assertTrue(from_draft)
        with self.assertRaises(StopIteration):
            next(gen)

    def test_stream_generate_speculative_eos_break_rewinds_caller_prompt_cache(self):
        class TrackingCache:
            def __init__(self, offset):
                self.offset = offset
                self.rewind_calls = []

            def can_rewind(self, n):
                return n <= self.offset

            def rewind(self, n):
                self.rewind_calls.append(n)
                self.offset -= n
                return True

        class FixedTokenModel:
            def __init__(self, forced_token):
                self.forced_token = forced_token
                self.layers = [object()]

            def __call__(self, input_tokens, cache=None, input_embeddings=None):
                if cache is not None:
                    for layer_cache in cache:
                        layer_cache.offset += input_tokens.shape[1]
                batch, seq_len = input_tokens.shape
                vocab_size = 4
                token_logits = -1000.0 * mx.ones((vocab_size,), dtype=mx.float32)
                token_logits = token_logits + (
                    2000.0 * (mx.arange(vocab_size) == self.forced_token)
                )
                return mx.broadcast_to(token_logits, (batch, seq_len, vocab_size))

        class StubTokenizer:
            eos_token_id = 0
            chat_template = None

            @staticmethod
            def get_vocab():
                return {}

        class StubDetokenizer:
            def __init__(self, _tokenizer):
                self.reset()

            def reset(self):
                self.tokens = []
                self.text = ""
                self.offset = 0

            def add_token(self, token):
                self.tokens.append(token)
                self.text += "x"

            def finalize(self):
                return None

            @property
            def last_segment(self):
                segment = self.text[self.offset :]
                self.offset = len(self.text)
                return segment

        tokenizer = TokenizerWrapper(
            StubTokenizer(),
            detokenizer_class=StubDetokenizer,
            eos_token_ids=[0],
        )

        model_cache = TrackingCache(offset=20)
        draft_cache = TrackingCache(offset=20)
        prompt_cache = [model_cache, draft_cache]
        prompt = mx.array([1, 2, 3], dtype=mx.uint32)

        responses = list(
            stream_generate(
                model=FixedTokenModel(forced_token=0),
                tokenizer=tokenizer,
                prompt=prompt,
                max_tokens=4,
                draft_model=FixedTokenModel(forced_token=1),
                prompt_cache=prompt_cache,
                num_draft_tokens=2,
            )
        )

        self.assertEqual(len(responses), 1)
        self.assertEqual(responses[0].finish_reason, "stop")
        self.assertEqual(responses[0].token, 0)
        self.assertEqual(model_cache.rewind_calls, [2])
        self.assertEqual(draft_cache.rewind_calls, [1])
        self.assertEqual(model_cache.offset, 23)
        self.assertEqual(draft_cache.offset, 23)

    def test_rewind_prompt_cache_snapshots_once_per_call(self):
        class CountedBox:
            copies = 0

            def __init__(self, value):
                self.value = value

            def __deepcopy__(self, memo):
                type(self).copies += 1
                copied = CountedBox(self.value)
                memo[id(self)] = copied
                return copied

        class RewindLayer:
            def __init__(self):
                self.offset = 8
                self.box = CountedBox(10)

            def can_rewind(self, n):
                return n <= self.offset

            def rewind(self, n):
                self.offset -= n
                return True

        CountedBox.copies = 0
        layer = RewindLayer()
        self.assertTrue(rewind_prompt_cache([layer], 1))
        self.assertEqual(CountedBox.copies, 1)

    def test_speculative_generate_step_rewind_snapshots_once_per_cycle(self):
        class CountedBox:
            copies = 0

            def __init__(self, value):
                self.value = value

            def __deepcopy__(self, memo):
                type(self).copies += 1
                copied = CountedBox(self.value)
                memo[id(self)] = copied
                return copied

        class RewindLayer:
            def __init__(self):
                self.offset = 12
                self.box = CountedBox(10)

            def can_rewind(self, n):
                return n <= self.offset

            def rewind(self, n):
                self.offset -= n
                return True

        class FixedTokenModel:
            def __init__(self, forced_token):
                self.forced_token = forced_token
                self.layers = [object()]

            def __call__(self, input_tokens, cache=None, input_embeddings=None):
                batch, seq_len = input_tokens.shape
                vocab_size = 4
                token_logits = -1000.0 * mx.ones((vocab_size,), dtype=mx.float32)
                token_logits = token_logits + (
                    2000.0 * (mx.arange(vocab_size) == self.forced_token)
                )
                return mx.broadcast_to(token_logits, (batch, seq_len, vocab_size))

        CountedBox.copies = 0
        prompt = mx.array([1, 2, 3], dtype=mx.uint32)
        prompt_cache = [RewindLayer(), RewindLayer()]

        gen = speculative_generate_step(
            prompt=prompt,
            model=FixedTokenModel(forced_token=0),
            draft_model=FixedTokenModel(forced_token=1),
            prompt_cache=prompt_cache,
            max_tokens=4,
            num_draft_tokens=1,
        )
        token, _logprobs, from_draft = next(gen)
        self.assertEqual(token, 0)
        self.assertFalse(from_draft)
        gen.close()

        self.assertEqual(CountedBox.copies, 1)

    def test_speculative_generate_step_known_cache_fast_path_skips_snapshotting(self):
        class CacheUpdatingFixedTokenModel:
            def __init__(self, forced_token):
                self.forced_token = forced_token
                self.layers = [object()]

            def __call__(self, input_tokens, cache=None, input_embeddings=None):
                if cache is not None:
                    for layer_cache in cache:
                        kv = mx.zeros(
                            (1, 1, input_tokens.shape[1], 1), dtype=mx.float32
                        )
                        layer_cache.update_and_fetch(kv, kv)
                batch, seq_len = input_tokens.shape
                vocab_size = 4
                token_logits = -1000.0 * mx.ones((vocab_size,), dtype=mx.float32)
                token_logits = token_logits + (
                    2000.0 * (mx.arange(vocab_size) == self.forced_token)
                )
                return mx.broadcast_to(token_logits, (batch, seq_len, vocab_size))

        prompt = mx.array([1, 2, 3], dtype=mx.uint32)
        model_cache = KVCache()
        draft_cache = KVCache()
        prompt_cache = [model_cache, draft_cache]
        rewind_calls = {id(model_cache): [], id(draft_cache): []}
        original_kv_rewind = KVCache.rewind

        def tracking_kv_rewind(cache_obj, num_to_trim):
            if id(cache_obj) in rewind_calls:
                rewind_calls[id(cache_obj)].append(num_to_trim)
            return original_kv_rewind(cache_obj, num_to_trim)

        # Known in-tree cache types should be able to rewind without going through
        # generic snapshot materialization.
        with patch.object(
            generate_module,
            "_snapshot_prompt_cache",
            side_effect=AssertionError(
                "_snapshot_prompt_cache should not run for known cache fast-path"
            ),
        ), patch.object(KVCache, "rewind", tracking_kv_rewind):
            outputs = []
            for token, _logprobs, from_draft in speculative_generate_step(
                prompt=prompt,
                model=CacheUpdatingFixedTokenModel(forced_token=0),
                draft_model=CacheUpdatingFixedTokenModel(forced_token=1),
                prompt_cache=prompt_cache,
                max_tokens=2,
                num_draft_tokens=2,
            ):
                outputs.append(
                    (
                        int(token),
                        bool(from_draft),
                        model_cache.offset,
                        draft_cache.offset,
                    )
                )

        self.assertEqual(len(outputs), 2)
        self.assertTrue(all(not from_draft for _, from_draft, _, _ in outputs))
        self.assertEqual(rewind_calls[id(model_cache)], [2, 1])
        self.assertEqual(rewind_calls[id(draft_cache)], [1])
        self.assertLess(model_cache.offset, max(o[2] for o in outputs))
        self.assertEqual(model_cache.offset, draft_cache.offset)

    def test_speculative_generate_step_known_mixed_rotating_fast_path_skips_snapshotting(
        self,
    ):
        class CacheUpdatingFixedTokenModel:
            def __init__(self, forced_token):
                self.forced_token = forced_token
                self.layers = [object(), object()]

            def __call__(self, input_tokens, cache=None, input_embeddings=None):
                if cache is not None:
                    for layer_cache in cache:
                        kv = mx.zeros(
                            (1, 1, input_tokens.shape[1], 1), dtype=mx.float32
                        )
                        layer_cache.update_and_fetch(kv, kv)
                batch, seq_len = input_tokens.shape
                vocab_size = 4
                token_logits = -1000.0 * mx.ones((vocab_size,), dtype=mx.float32)
                token_logits = token_logits + (
                    2000.0 * (mx.arange(vocab_size) == self.forced_token)
                )
                return mx.broadcast_to(token_logits, (batch, seq_len, vocab_size))

        prompt = mx.array([1, 2, 3], dtype=mx.uint32)
        prompt_cache = [
            KVCache(),
            RotatingKVCache(max_size=64),
            KVCache(),
            RotatingKVCache(max_size=64),
        ]
        model_kv, model_rot, draft_kv, draft_rot = prompt_cache
        kv_rewind_calls = {id(model_kv): [], id(draft_kv): []}
        rot_rewind_calls = {id(model_rot): [], id(draft_rot): []}
        original_kv_rewind = KVCache.rewind
        original_rot_rewind = RotatingKVCache.rewind

        def tracking_kv_rewind(cache_obj, num_to_trim):
            if id(cache_obj) in kv_rewind_calls:
                kv_rewind_calls[id(cache_obj)].append(num_to_trim)
            return original_kv_rewind(cache_obj, num_to_trim)

        def tracking_rot_rewind(cache_obj, num_to_trim):
            if id(cache_obj) in rot_rewind_calls:
                rot_rewind_calls[id(cache_obj)].append(num_to_trim)
            return original_rot_rewind(cache_obj, num_to_trim)

        with patch.object(
            generate_module,
            "_snapshot_prompt_cache",
            side_effect=AssertionError(
                "_snapshot_prompt_cache should not run for known mixed fast-path"
            ),
        ), patch.object(KVCache, "rewind", tracking_kv_rewind), patch.object(
            RotatingKVCache, "rewind", tracking_rot_rewind
        ):
            outputs = list(
                speculative_generate_step(
                    prompt=prompt,
                    model=CacheUpdatingFixedTokenModel(forced_token=0),
                    draft_model=CacheUpdatingFixedTokenModel(forced_token=1),
                    prompt_cache=prompt_cache,
                    max_tokens=2,
                    num_draft_tokens=2,
                )
            )

        self.assertEqual(len(outputs), 2)
        self.assertTrue(all(not from_draft for _, _, from_draft in outputs))
        self.assertEqual(kv_rewind_calls[id(model_kv)], [2, 1])
        self.assertEqual(rot_rewind_calls[id(model_rot)], [2, 1])
        self.assertEqual(kv_rewind_calls[id(draft_kv)], [1])
        self.assertEqual(rot_rewind_calls[id(draft_rot)], [1])
        self.assertEqual([layer.offset for layer in prompt_cache], [4, 4, 4, 4])

    def test_speculative_generate_step_zero_trim_full_accept_keeps_speculating(self):
        class CacheUpdatingFixedTokenModel:
            def __init__(self, forced_token):
                self.forced_token = forced_token
                self.layers = [object()]

            def __call__(self, input_tokens, cache=None, input_embeddings=None):
                if cache is not None:
                    for layer_cache in cache:
                        kv = mx.zeros(
                            (1, 1, input_tokens.shape[1], 1), dtype=mx.float32
                        )
                        layer_cache.update_and_fetch(kv, kv)
                batch, seq_len = input_tokens.shape
                vocab_size = 4
                token_logits = -1000.0 * mx.ones((vocab_size,), dtype=mx.float32)
                token_logits = token_logits + (
                    2000.0 * (mx.arange(vocab_size) == self.forced_token)
                )
                return mx.broadcast_to(token_logits, (batch, seq_len, vocab_size))

        prompt = mx.array([1, 2, 3], dtype=mx.uint32)
        prompt_cache = [KVCache(), KVCache()]
        outputs = list(
            speculative_generate_step(
                prompt=prompt,
                model=CacheUpdatingFixedTokenModel(forced_token=0),
                draft_model=CacheUpdatingFixedTokenModel(forced_token=0),
                prompt_cache=prompt_cache,
                max_tokens=4,
                num_draft_tokens=1,
            )
        )

        self.assertEqual(
            [bool(from_draft) for _, _, from_draft in outputs],
            [True, False, True, False],
        )

    def test_speculative_generate_step_rotating_rewind_miss_degrades_without_crashing(
        self,
    ):
        class OffsetTokenModel:
            def __init__(self, shift):
                self.shift = shift
                self.layers = [object()]

            def __call__(self, input_tokens, cache=None, input_embeddings=None):
                if cache is not None:
                    for layer_cache in cache:
                        kv = mx.zeros(
                            (1, 1, input_tokens.shape[1], 1), dtype=mx.float32
                        )
                        layer_cache.update_and_fetch(kv, kv)
                    token_id = (int(cache[0].offset) + self.shift) % 7
                else:
                    token_id = self.shift % 7
                batch, seq_len = input_tokens.shape
                vocab_size = 7
                token_logits = -1000.0 * mx.ones((vocab_size,), dtype=mx.float32)
                token_logits = token_logits + (
                    2000.0 * (mx.arange(vocab_size) == token_id)
                )
                return mx.broadcast_to(token_logits, (batch, seq_len, vocab_size))

        prompt = mx.array(list(range(1, 12)), dtype=mx.uint32)
        baseline_tokens = [
            int(token)
            for token, _ in generate_step(
                prompt=prompt,
                model=OffsetTokenModel(shift=0),
                prompt_cache=[RotatingKVCache(max_size=8)],
                max_tokens=3,
            )
        ]

        outputs = list(
            speculative_generate_step(
                prompt=prompt,
                model=OffsetTokenModel(shift=0),
                draft_model=OffsetTokenModel(shift=1),
                prompt_cache=[RotatingKVCache(max_size=8), RotatingKVCache(max_size=8)],
                max_tokens=3,
                num_draft_tokens=2,
            )
        )

        # Rewind miss should gracefully fall back instead of aborting generation.
        self.assertEqual(len(outputs), 3)
        self.assertTrue(all(not from_draft for _, _, from_draft in outputs))
        self.assertEqual([int(token) for token, _, _ in outputs], baseline_tokens)

    def test_speculative_generate_step_short_kv_prompt_does_not_false_fail_precheck(
        self,
    ):
        class CacheUpdatingFixedTokenModel:
            def __init__(self, forced_token):
                self.forced_token = forced_token
                self.layers = [object()]

            def __call__(self, input_tokens, cache=None, input_embeddings=None):
                if cache is not None:
                    for layer_cache in cache:
                        kv = mx.zeros(
                            (1, 1, input_tokens.shape[1], 1), dtype=mx.float32
                        )
                        layer_cache.update_and_fetch(kv, kv)
                batch, seq_len = input_tokens.shape
                vocab_size = 4
                token_logits = -1000.0 * mx.ones((vocab_size,), dtype=mx.float32)
                token_logits = token_logits + (
                    2000.0 * (mx.arange(vocab_size) == self.forced_token)
                )
                return mx.broadcast_to(token_logits, (batch, seq_len, vocab_size))

        outputs = list(
            speculative_generate_step(
                prompt=mx.array([1], dtype=mx.uint32),
                model=CacheUpdatingFixedTokenModel(forced_token=0),
                draft_model=CacheUpdatingFixedTokenModel(forced_token=1),
                prompt_cache=[KVCache(), KVCache()],
                max_tokens=3,
                num_draft_tokens=2,
            )
        )
        self.assertEqual(len(outputs), 3)
        self.assertTrue(all(not from_draft for _, _, from_draft in outputs))

    def test_speculative_generate_step_rotating_boundary_prompt_len_equals_window_degrades(
        self,
    ):
        class OffsetTokenModel:
            def __init__(self, shift):
                self.shift = shift
                self.layers = [object()]

            def __call__(self, input_tokens, cache=None, input_embeddings=None):
                if cache is not None:
                    for layer_cache in cache:
                        kv = mx.zeros(
                            (1, 1, input_tokens.shape[1], 1), dtype=mx.float32
                        )
                        layer_cache.update_and_fetch(kv, kv)
                    token_id = (int(cache[0].offset) + self.shift) % 7
                else:
                    token_id = self.shift % 7
                batch, seq_len = input_tokens.shape
                vocab_size = 7
                token_logits = -1000.0 * mx.ones((vocab_size,), dtype=mx.float32)
                token_logits = token_logits + (
                    2000.0 * (mx.arange(vocab_size) == token_id)
                )
                return mx.broadcast_to(token_logits, (batch, seq_len, vocab_size))

        prompt = mx.array(list(range(1, 9)), dtype=mx.uint32)
        baseline_tokens = [
            int(token)
            for token, _ in generate_step(
                prompt=prompt,
                model=OffsetTokenModel(shift=0),
                prompt_cache=[RotatingKVCache(max_size=8)],
                max_tokens=3,
            )
        ]

        outputs = list(
            speculative_generate_step(
                prompt=prompt,
                model=OffsetTokenModel(shift=0),
                draft_model=OffsetTokenModel(shift=1),
                prompt_cache=[RotatingKVCache(max_size=8), RotatingKVCache(max_size=8)],
                max_tokens=3,
                num_draft_tokens=2,
            )
        )
        self.assertEqual(len(outputs), 3)
        self.assertTrue(all(not from_draft for _, _, from_draft in outputs))
        self.assertEqual([int(token) for token, _, _ in outputs], baseline_tokens)

    def test_speculative_generate_step_rotating_boundary_single_draft_token_keeps_speculation(
        self,
    ):
        class CacheUpdatingFixedTokenModel:
            def __init__(self, forced_token):
                self.forced_token = forced_token
                self.layers = [object()]

            def __call__(self, input_tokens, cache=None, input_embeddings=None):
                if cache is not None:
                    for layer_cache in cache:
                        kv = mx.zeros(
                            (1, 1, input_tokens.shape[1], 1), dtype=mx.float32
                        )
                        layer_cache.update_and_fetch(kv, kv)
                batch, seq_len = input_tokens.shape
                vocab_size = 4
                token_logits = -1000.0 * mx.ones((vocab_size,), dtype=mx.float32)
                token_logits = token_logits + (
                    2000.0 * (mx.arange(vocab_size) == self.forced_token)
                )
                return mx.broadcast_to(token_logits, (batch, seq_len, vocab_size))

        outputs = list(
            speculative_generate_step(
                prompt=mx.array(list(range(1, 9)), dtype=mx.uint32),
                model=CacheUpdatingFixedTokenModel(forced_token=0),
                draft_model=CacheUpdatingFixedTokenModel(forced_token=0),
                prompt_cache=[RotatingKVCache(max_size=8), RotatingKVCache(max_size=8)],
                max_tokens=4,
                num_draft_tokens=1,
            )
        )

        # Boundary prompt length should not disable safe single-token speculation.
        self.assertTrue(any(bool(from_draft) for _, _, from_draft in outputs))

    def test_speculative_generate_step_rotating_chunked_prefill_degrades_correctly(
        self,
    ):
        class OffsetTokenModel:
            def __init__(self, shift):
                self.shift = shift
                self.layers = [object()]

            def __call__(self, input_tokens, cache=None, input_embeddings=None):
                if cache is not None:
                    for layer_cache in cache:
                        kv = mx.zeros(
                            (1, 1, input_tokens.shape[1], 1), dtype=mx.float32
                        )
                        layer_cache.update_and_fetch(kv, kv)
                    token_id = (int(cache[0].offset) + self.shift) % 7
                else:
                    token_id = self.shift % 7
                batch, seq_len = input_tokens.shape
                vocab_size = 7
                token_logits = -1000.0 * mx.ones((vocab_size,), dtype=mx.float32)
                token_logits = token_logits + (
                    2000.0 * (mx.arange(vocab_size) == token_id)
                )
                return mx.broadcast_to(token_logits, (batch, seq_len, vocab_size))

        prompt = mx.arange(600, dtype=mx.uint32) + 1
        baseline_tokens = [
            int(token)
            for token, _ in generate_step(
                prompt=prompt,
                model=OffsetTokenModel(shift=0),
                prompt_cache=[RotatingKVCache(max_size=512)],
                max_tokens=3,
                prefill_step_size=512,
            )
        ]

        outputs = list(
            speculative_generate_step(
                prompt=prompt,
                model=OffsetTokenModel(shift=0),
                draft_model=OffsetTokenModel(shift=1),
                prompt_cache=[
                    RotatingKVCache(max_size=512),
                    RotatingKVCache(max_size=512),
                ],
                max_tokens=3,
                num_draft_tokens=2,
                prefill_step_size=512,
            )
        )
        self.assertEqual(len(outputs), 3)
        self.assertTrue(all(not from_draft for _, _, from_draft in outputs))
        self.assertEqual([int(token) for token, _, _ in outputs], baseline_tokens)

    def test_speculative_generate_step_rotating_presence_does_not_mask_non_rotating_failure(
        self,
    ):
        class NonRotatingFailLayer:
            offset = 16

            def can_rewind(self, n):
                return False

            def rewind(self, n):
                return False

        class CacheUpdatingFixedTokenModel:
            def __init__(self, forced_token):
                self.forced_token = forced_token
                self.layers = [object()]

            def __call__(self, input_tokens, cache=None, input_embeddings=None):
                if cache is not None:
                    for layer_cache in cache:
                        caches = (
                            layer_cache.caches
                            if isinstance(layer_cache, CacheList)
                            else (layer_cache,)
                        )
                        for cache_entry in caches:
                            if hasattr(cache_entry, "update_and_fetch"):
                                kv = mx.zeros(
                                    (1, 1, input_tokens.shape[1], 1), dtype=mx.float32
                                )
                                cache_entry.update_and_fetch(kv, kv)
                batch, seq_len = input_tokens.shape
                vocab_size = 4
                token_logits = -1000.0 * mx.ones((vocab_size,), dtype=mx.float32)
                token_logits = token_logits + (
                    2000.0 * (mx.arange(vocab_size) == self.forced_token)
                )
                return mx.broadcast_to(token_logits, (batch, seq_len, vocab_size))

        prompt = mx.array([1, 2, 3], dtype=mx.uint32)
        model_cache = CacheList(RotatingKVCache(max_size=16), NonRotatingFailLayer())
        prompt_cache = [model_cache, KVCache()]
        gen = speculative_generate_step(
            prompt=prompt,
            model=CacheUpdatingFixedTokenModel(forced_token=0),
            draft_model=CacheUpdatingFixedTokenModel(forced_token=1),
            prompt_cache=prompt_cache,
            max_tokens=3,
            num_draft_tokens=2,
        )
        next(gen)
        with self.assertRaisesRegex(
            RuntimeError, "Speculative decoding cache rewind failed for model cache"
        ):
            next(gen)

    def test_speculative_generate_step_known_fast_path_failure_rolls_back_atomically(
        self,
    ):
        class CacheUpdatingFixedTokenModel:
            def __init__(self, forced_token):
                self.forced_token = forced_token
                self.layers = [object()]

            def __call__(self, input_tokens, cache=None, input_embeddings=None):
                if cache is not None:
                    for layer_cache in cache:
                        kv = mx.zeros(
                            (1, 1, input_tokens.shape[1], 1), dtype=mx.float32
                        )
                        layer_cache.update_and_fetch(kv, kv)
                batch, seq_len = input_tokens.shape
                vocab_size = 4
                token_logits = -1000.0 * mx.ones((vocab_size,), dtype=mx.float32)
                token_logits = token_logits + (
                    2000.0 * (mx.arange(vocab_size) == self.forced_token)
                )
                return mx.broadcast_to(token_logits, (batch, seq_len, vocab_size))

        prompt = mx.array([1, 2, 3], dtype=mx.uint32)
        model_cache = KVCache()
        draft_cache = KVCache()
        prompt_cache = [model_cache, draft_cache]
        original_kv_rewind = KVCache.rewind

        def fail_draft_rewind(cache_obj, num_to_trim):
            if cache_obj is draft_cache:
                # Simulate a buggy rewind implementation that mutates before fail.
                cache_obj.offset -= num_to_trim
                return False
            return original_kv_rewind(cache_obj, num_to_trim)

        gen = speculative_generate_step(
            prompt=prompt,
            model=CacheUpdatingFixedTokenModel(forced_token=0),
            draft_model=CacheUpdatingFixedTokenModel(forced_token=1),
            prompt_cache=prompt_cache,
            max_tokens=2,
            num_draft_tokens=2,
        )
        with patch.object(KVCache, "rewind", fail_draft_rewind):
            next(gen)
            offsets_before_failure = (model_cache.offset, draft_cache.offset)
            with self.assertRaisesRegex(
                RuntimeError, "Speculative decoding cache rewind failed for draft cache"
            ):
                next(gen)

        self.assertEqual(
            (model_cache.offset, draft_cache.offset), offsets_before_failure
        )

    def test_known_fast_rewind_handles_quantized_and_nested_cachelist(self):
        kv_shape = (1, 1, 3, 32)

        kv = KVCache()
        kv.update_and_fetch(
            mx.zeros(kv_shape, dtype=mx.float32),
            mx.zeros(kv_shape, dtype=mx.float32),
        )

        rotating = RotatingKVCache(max_size=16)
        rotating.update_and_fetch(
            mx.zeros(kv_shape, dtype=mx.float32),
            mx.zeros(kv_shape, dtype=mx.float32),
        )

        quantized = QuantizedKVCache(bits=4, group_size=32)
        quantized.update_and_fetch(
            mx.zeros(kv_shape, dtype=mx.float32),
            mx.zeros(kv_shape, dtype=mx.float32),
        )

        nested_known = CacheList(CacheList(kv, rotating), quantized)
        self.assertTrue(generate_module._can_fast_rewind_layers([nested_known], 1))

        class UnknownLayer:
            offset = 3

            def can_rewind(self, n):
                return True

            def rewind(self, n):
                return True

        nested_mixed = CacheList(CacheList(KVCache(), UnknownLayer()), quantized)
        self.assertFalse(generate_module._can_fast_rewind_layers([nested_mixed], 1))

    def test_speculative_generate_step_custom_cache_path_still_uses_snapshot(self):
        class CustomRewindLayer:
            def __init__(self, offset, *, fail_rewind=False):
                self.offset = offset
                self.fail_rewind = fail_rewind
                self.rewind_calls = []

            def can_rewind(self, n):
                return n <= self.offset

            def rewind(self, n):
                self.rewind_calls.append(n)
                self.offset -= n
                return not self.fail_rewind

        class CacheUpdatingFixedTokenModel:
            def __init__(self, forced_token):
                self.forced_token = forced_token
                self.layers = [object()]

            def __call__(self, input_tokens, cache=None, input_embeddings=None):
                if cache is not None:
                    for layer_cache in cache:
                        layer_cache.offset += input_tokens.shape[1]
                batch, seq_len = input_tokens.shape
                vocab_size = 4
                token_logits = -1000.0 * mx.ones((vocab_size,), dtype=mx.float32)
                token_logits = token_logits + (
                    2000.0 * (mx.arange(vocab_size) == self.forced_token)
                )
                return mx.broadcast_to(token_logits, (batch, seq_len, vocab_size))

        prompt = mx.array([1, 2, 3], dtype=mx.uint32)
        model_layer = CustomRewindLayer(offset=20)
        draft_layer = CustomRewindLayer(offset=20, fail_rewind=True)
        prompt_cache = [model_layer, draft_layer]
        snapshot_calls = []
        original_snapshot = generate_module._snapshot_prompt_cache

        def tracking_snapshot(cache_layers):
            snapshot_calls.append(len(cache_layers))
            return original_snapshot(cache_layers)

        with patch.object(
            generate_module,
            "_snapshot_prompt_cache",
            side_effect=tracking_snapshot,
        ):
            gen = speculative_generate_step(
                prompt=prompt,
                model=CacheUpdatingFixedTokenModel(forced_token=0),
                draft_model=CacheUpdatingFixedTokenModel(forced_token=1),
                prompt_cache=prompt_cache,
                max_tokens=2,
                num_draft_tokens=2,
            )
            next(gen)
            offsets_before_failure = [layer.offset for layer in prompt_cache]
            with self.assertRaisesRegex(
                RuntimeError, "Speculative decoding cache rewind failed for draft cache"
            ):
                next(gen)

        self.assertGreaterEqual(len(snapshot_calls), 2)
        self.assertEqual(snapshot_calls[0], 1)
        self.assertEqual(snapshot_calls[1], 1)
        self.assertEqual(
            [layer.offset for layer in prompt_cache], offsets_before_failure
        )

    def test_speculative_generate_step_mixed_known_custom_failure_is_atomic(self):
        class CustomRewindLayer:
            def __init__(self, offset, *, fail_rewind=False, call_log=None):
                self.offset = offset
                self.fail_rewind = fail_rewind
                self.call_log = call_log

            def can_rewind(self, n):
                return n <= self.offset

            def rewind(self, n):
                if self.call_log is not None:
                    self.call_log.append(n)
                self.offset -= n
                return not self.fail_rewind

        class CacheUpdatingFixedTokenModel:
            def __init__(self, forced_token):
                self.forced_token = forced_token
                self.layers = [object(), object()]

            def __call__(self, input_tokens, cache=None, input_embeddings=None):
                if cache is not None:
                    for layer_cache in cache:
                        if isinstance(layer_cache, KVCache):
                            kv = mx.zeros(
                                (1, 1, input_tokens.shape[1], 1), dtype=mx.float32
                            )
                            layer_cache.update_and_fetch(kv, kv)
                        else:
                            layer_cache.offset += input_tokens.shape[1]
                batch, seq_len = input_tokens.shape
                vocab_size = 4
                token_logits = -1000.0 * mx.ones((vocab_size,), dtype=mx.float32)
                token_logits = token_logits + (
                    2000.0 * (mx.arange(vocab_size) == self.forced_token)
                )
                return mx.broadcast_to(token_logits, (batch, seq_len, vocab_size))

        prompt = mx.array([1, 2, 3], dtype=mx.uint32)
        model_known = KVCache()
        model_custom_calls = []
        draft_custom_fail_calls = []
        model_custom = CustomRewindLayer(offset=20, call_log=model_custom_calls)
        draft_known = KVCache()
        draft_custom_fail = CustomRewindLayer(
            offset=20, fail_rewind=True, call_log=draft_custom_fail_calls
        )
        prompt_cache = [model_known, model_custom, draft_known, draft_custom_fail]
        known_rewind_calls = {id(model_known): [], id(draft_known): []}
        original_kv_rewind = KVCache.rewind

        def tracking_kv_rewind(cache_obj, num_to_trim):
            if id(cache_obj) in known_rewind_calls:
                known_rewind_calls[id(cache_obj)].append(num_to_trim)
            return original_kv_rewind(cache_obj, num_to_trim)

        with patch.object(KVCache, "rewind", tracking_kv_rewind):
            gen = speculative_generate_step(
                prompt=prompt,
                model=CacheUpdatingFixedTokenModel(forced_token=0),
                draft_model=CacheUpdatingFixedTokenModel(forced_token=1),
                prompt_cache=prompt_cache,
                max_tokens=2,
                num_draft_tokens=2,
            )
            next(gen)
            offsets_before_failure = [
                cache_layer.offset for cache_layer in prompt_cache
            ]
            with self.assertRaisesRegex(
                RuntimeError, "Speculative decoding cache rewind failed for draft cache"
            ):
                next(gen)

        self.assertGreater(len(known_rewind_calls[id(model_known)]), 0)
        self.assertGreater(len(known_rewind_calls[id(draft_known)]), 0)
        self.assertEqual(model_custom_calls, [2])
        self.assertEqual(draft_custom_fail_calls, [1])
        self.assertEqual(
            [cache_layer.offset for cache_layer in prompt_cache],
            offsets_before_failure,
        )

    def test_generate_main_raises_for_prompt_cache_kv_bits_mismatch(self):
        with patch.object(
            sys,
            "argv",
            [
                "generate.py",
                "--prompt-cache-file",
                "dummy_cache.safetensors",
                "--kv-bits",
                "8",
            ],
        ), patch(
            "mlx_lm.generate.load_prompt_cache",
            return_value=(
                [QuantizedKVCache(bits=4, group_size=32)],
                {"tokenizer_config": "{}", "model": "dummy-model"},
            ),
        ), patch(
            "mlx_lm.generate.load"
        ) as load_mock:
            with self.assertRaisesRegex(
                ValueError,
                "--kv-bits does not match the kv cache loaded from --prompt-cache-file.",
            ):
                generate_module.main()
        self.assertFalse(load_mock.called)

    def test_generate_main_raises_for_prompt_cache_kv_group_size_mismatch(self):
        with patch.object(
            sys,
            "argv",
            [
                "generate.py",
                "--prompt-cache-file",
                "dummy_cache.safetensors",
                "--kv-bits",
                "4",
                "--kv-group-size",
                "16",
            ],
        ), patch(
            "mlx_lm.generate.load_prompt_cache",
            return_value=(
                [QuantizedKVCache(bits=4, group_size=32)],
                {"tokenizer_config": "{}", "model": "dummy-model"},
            ),
        ), patch(
            "mlx_lm.generate.load"
        ) as load_mock:
            with self.assertRaisesRegex(
                ValueError,
                "--kv-group-size does not match the kv cache loaded from --prompt-cache-file.",
            ):
                generate_module.main()
        self.assertFalse(load_mock.called)


class TestGenerate(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.HF_MODEL_PATH = "mlx-community/Qwen1.5-0.5B-Chat-4bit"
        cls.model, cls.tokenizer = load(cls.HF_MODEL_PATH)
        cls.model.set_dtype(mx.float32)

    def test_generate(self):
        # Simple test that generation runs
        text = generate(
            self.model, self.tokenizer, "hello", max_tokens=5, verbose=False
        )

    def test_generate_with_logit_bias(self):
        logit_bias = {0: 2000.0, 1: -20.0}
        text = generate(
            self.model,
            self.tokenizer,
            "hello",
            max_tokens=5,
            logits_processors=make_logits_processors(logit_bias),
            verbose=False,
        )
        self.assertEqual(text, "!!!!!")

    def test_stream_generate_max_tokens(self):
        prompt = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": "Write a story about Einstein"}],
            tokenize=True,
            add_generation_prompt=True,
        )

        tokens = []
        for response in stream_generate(
            self.model,
            self.tokenizer,
            prompt,
            max_tokens=4,
        ):
            tokens.append(response.token)
        self.assertEqual(len(tokens), 4)

    def test_generate_with_processor(self):
        init_toks = self.tokenizer.encode("hello")

        all_toks = None

        def logits_processor(toks, logits):
            nonlocal all_toks
            all_toks = toks
            return logits

        generate(
            self.model,
            self.tokenizer,
            "hello",
            max_tokens=5,
            verbose=False,
            logits_processors=[logits_processor],
        )
        self.assertEqual(len(all_toks), len(init_toks) + 5)

    def test_stream_generate_speculative(self):
        # Use same model as draft model, this is not a speed test
        draft_model = self.model

        results: List[GenerationResponse] = []
        drafted: List[bool] = []

        # make a determinate sampler
        sampler = make_sampler(temp=0.0)
        messages = [{"role": "user", "content": "hello"}]
        prompt = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
        )

        for generation_result in stream_generate(
            model=self.model,
            tokenizer=self.tokenizer,
            prompt=prompt,
            max_tokens=5,
            draft_model=draft_model,
            num_draft_tokens=2,
            sampler=sampler,
        ):
            drafted.append(generation_result.from_draft)
            results.append(generation_result)

        self.assertEqual(len(results), 5)
        # since num_draft_tokens is 2 and draft model is the same, the
        # first 2 generations should be drafts, the third should come
        # from the target model. The final two tokens may remain speculative
        # or fall back to target-model tokens if speculative rewind becomes
        # unavailable for the model/cache combination.
        self.assertEqual(drafted[:3], [True, True, False])
        self.assertIn(drafted[3:], ([True, True], [False, False]))

    def test_stream_generate_input_embeddings(self):
        sampler = make_sampler(temp=0.0)  # determinate sampler

        # get prompt embeddings
        messages = [{"role": "user", "content": "Say 'TEST' and nothing else"}]
        prompt = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
        )
        prompt_embeddings = self.model.model.embed_tokens(prompt)

        response = ""
        for generation_result in stream_generate(
            model=self.model,
            tokenizer=self.tokenizer,
            prompt=prompt,
            max_tokens=5,
            sampler=sampler,
            input_embeddings=prompt_embeddings,
        ):
            response += generation_result.text

        self.assertEqual("TEST", response)

    def test_stream_generate_input_embeddings_prefill(self):
        sampler = make_sampler(temp=0.0)  # determinate sampler

        # get prompt embeddings
        messages = [{"role": "user", "content": "Say 'TEST' and nothing else"}]
        prompt = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
        )
        prompt_embeddings = self.model.model.embed_tokens(prompt)

        # setup prompt progress callback to track batched prefill
        num_prompt_processing_callbacks = 0

        def progress_callback(processed: int, total: int) -> None:
            nonlocal num_prompt_processing_callbacks
            num_prompt_processing_callbacks += 1

        # generate
        prefill_step_size = 5
        response = ""
        for generation_result in stream_generate(
            model=self.model,
            tokenizer=self.tokenizer,
            prompt=prompt,
            max_tokens=5,
            sampler=sampler,
            input_embeddings=prompt_embeddings,
            prefill_step_size=prefill_step_size,
            prompt_progress_callback=progress_callback,
        ):
            response += generation_result.text

        self.assertEqual("TEST", response)
        num_embeddings = prompt_embeddings.shape[0]
        self.assertTrue(
            num_embeddings / prefill_step_size < num_prompt_processing_callbacks
        )

    def test_generate_step_rejects_invalid_prefill_step_size(self):
        class FailFastModel:
            layers = [object()]

            def make_cache(self):
                return [KVCache()]

            def __call__(self, input_tokens, cache=None, input_embeddings=None):
                raise RuntimeError("model-call-should-not-happen")

        prompt = mx.array([1], dtype=mx.uint32)
        model = FailFastModel()
        for bad_step_size in (0, -1, 1.5, "8", True):
            with self.assertRaisesRegex(
                ValueError, "prefill_step_size must be a positive integer"
            ):
                next(
                    generate_step(
                        prompt=prompt,
                        model=model,
                        max_tokens=1,
                        prefill_step_size=bad_step_size,
                    )
                )

    def test_speculative_generate_step_rejects_invalid_prefill_step_size(self):
        class FailFastModel:
            layers = [object()]

            def make_cache(self):
                return [KVCache()]

            def __call__(self, input_tokens, cache=None, input_embeddings=None):
                raise RuntimeError("model-call-should-not-happen")

        prompt = mx.array([1], dtype=mx.uint32)
        model = FailFastModel()
        for bad_step_size in (0, -1, 1.5, "8", True):
            with self.assertRaisesRegex(
                ValueError, "prefill_step_size must be a positive integer"
            ):
                next(
                    speculative_generate_step(
                        prompt=prompt,
                        model=model,
                        draft_model=model,
                        max_tokens=1,
                        prefill_step_size=bad_step_size,
                    )
                )

    def test_batch_generator_rejects_non_positive_prefill_step_size(self):
        for bad_step_size in (0, -8):
            with self.assertRaisesRegex(
                ValueError, "prefill_step_size must be a positive integer"
            ):
                BatchGenerator(self.model, prefill_step_size=bad_step_size)

    def test_batch_generator_rejects_non_integer_prefill_step_size(self):
        for bad_step_size in (1.5, "8", True):
            with self.assertRaisesRegex(
                ValueError, "prefill_step_size must be a positive integer"
            ):
                BatchGenerator(self.model, prefill_step_size=bad_step_size)

    def test_batch_matches_single(self):

        prompts = [
            "Write a story about Einstein",
            "Hi",
            "What time is it?",
            "How tall is Mt Everest?",
        ]
        prompts = [
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": p}],
                tokenize=True,
                add_generation_prompt=True,
            )
            for p in prompts
        ]

        gen = BatchGenerator(
            self.model, stop_tokens=self.tokenizer.eos_token_ids, max_tokens=1
        )
        uids = gen.insert(prompts)
        batch_responses = {r.uid: r for r in gen.next()}

        # Do a test for each prompt the logits are close
        for e, prompt in enumerate(prompts):

            for response in stream_generate(
                self.model, self.tokenizer, prompt, max_tokens=1
            ):
                blp = batch_responses[uids[e]].logprobs
                lp = response.logprobs
                self.assertTrue(mx.allclose(blp, lp))
                break

    def test_many_batches(self):

        prompts = [
            "Write a story about Einstein",
            "Hi",
            "What time is it?",
            "How tall is Mt Everest?",
        ]
        prompts = [
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": p}],
                tokenize=True,
                add_generation_prompt=True,
            )
            for p in prompts
        ]

        gen = BatchGenerator(
            self.model,
            stop_tokens=self.tokenizer.eos_token_ids,
            max_tokens=1,
            prefill_batch_size=2,
            prefill_step_size=8,
            completion_batch_size=3,
        )
        uids = gen.insert(prompts)
        batch_responses = {}
        not_in = True
        iters = 0
        while responses := gen.next():
            for r in responses:
                not_in &= r.uid not in batch_responses
                batch_responses[r.uid] = r
            iters += 1
        # only one token per prompt means only one response per prompt
        self.assertTrue(not_in)

        # completion batch size is too small for a single iteration
        self.assertTrue(iters > 1)

        # Do a test for each prompt the logits are close
        for e, prompt in enumerate(prompts):

            for response in stream_generate(
                self.model, self.tokenizer, prompt, max_tokens=1
            ):
                blp = batch_responses[uids[e]].logprobs
                lp = response.logprobs
                self.assertTrue(mx.allclose(blp, lp))
                break

    def test_batch_unique_max_toks(self):
        prompts = [
            "Write a story about Einstein",
            "Hi",
            "What time is it?",
            "How tall is Mt Everest?",
        ]
        prompts = [
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": p}],
                tokenize=True,
                add_generation_prompt=True,
            )
            for p in prompts
        ]

        gen = BatchGenerator(
            self.model,
            stop_tokens=self.tokenizer.eos_token_ids,
            prefill_batch_size=2,
            prefill_step_size=8,
            completion_batch_size=3,
        )
        num_toks = [2, 3, 4, 5]
        uids = gen.insert(prompts, max_tokens=num_toks)
        batch_responses = {uid: [] for uid in uids}
        while responses := gen.next():
            for r in responses:
                batch_responses[r.uid].append(r.token)

        # Do a test for each prompt the logits are close
        for e, prompt in enumerate(prompts):

            tokens = []
            for response in stream_generate(
                self.model,
                self.tokenizer,
                prompt,
                max_tokens=num_toks[e],
            ):
                tokens.append(response.token)

            batch_tokens = batch_responses[uids[e]]
            self.assertEqual(tokens, batch_tokens)

    def test_batch_sliding_window(self):
        prompts = [
            "Write a story about Einstein",
            "Hi",
            "What time is it?",
            "How tall is Mt Everest?",
        ]
        prompts = [
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": p}],
                tokenize=True,
                add_generation_prompt=True,
            )
            for p in prompts
        ]

        self.model.make_cache = lambda: [
            RotatingKVCache(max_size=4) for _ in self.model.layers
        ]
        batch_gen = BatchGenerator(
            self.model,
            stop_tokens=self.tokenizer.eos_token_ids,
            max_tokens=10,
            prefill_batch_size=1,
            prefill_step_size=8,
            completion_batch_size=2,
        )
        uids = batch_gen.insert(prompts)
        batch_responses = {uid: [] for uid in uids}
        while responses := batch_gen.next():
            for r in responses:
                batch_responses[r.uid].append(r.logprobs)

        for e, uid in enumerate(uids):
            for i, response in enumerate(
                stream_generate(
                    self.model,
                    self.tokenizer,
                    prompts[e],
                    max_tokens=10,
                )
            ):
                batch_logprobs = batch_responses[uid][i]
                logprobs = response.logprobs
                self.assertTrue(
                    mx.allclose(batch_logprobs, logprobs, rtol=1e-4, atol=1e-4)
                )

        del self.model.make_cache

    def test_batch_generate_with_logits_processors(self):
        """Test that batch_generate with logits_processors produces correct results."""
        logit_bias = {0: 2000.0, 1: -2000.0}
        processors = make_logits_processors(logit_bias)

        batch_gen = BatchGenerator(
            self.model,
            max_tokens=1,
            logits_processors=processors,
        )
        prompt = self.tokenizer.encode("hello")
        uids = batch_gen.insert([prompt])
        response = batch_gen.next()[0]
        logprobs = response.logprobs
        self.assertEqual(logprobs[0].item(), 0.0)
        self.assertEqual(logprobs.argmin().item(), 1)

        del batch_gen

        logit_bias = {0: 2000.0}
        processors = make_logits_processors(logit_bias)
        batch_gen = BatchGenerator(
            self.model,
            max_tokens=1,
            logits_processors=processors,
        )

        (uid0,) = batch_gen.insert([prompt])

        logit_bias = {1: 2000.0}
        processors = make_logits_processors(logit_bias)
        (uid1,) = batch_gen.insert([prompt], logits_processors=[processors])

        logit_bias = {2: 2000.0}
        processors = make_logits_processors(logit_bias)
        (uid2,) = batch_gen.insert([prompt], logits_processors=[processors])

        responses = batch_gen.next()
        responses = {response.uid: response for response in responses}
        self.assertEqual(responses[uid0].logprobs[0].item(), 0.0)
        self.assertEqual(responses[uid1].logprobs[1].item(), 0.0)
        self.assertEqual(responses[uid2].logprobs[2].item(), 0.0)

    def test_batch_generate_with_samplers(self):
        """Test that batch_generate with logits_processors produces correct results."""
        batch_gen = BatchGenerator(
            self.model,
            max_tokens=1,
            sampler=lambda _: mx.array([1]),
        )
        prompt = self.tokenizer.encode("hello")
        uids = batch_gen.insert([prompt])
        response = batch_gen.next()[0]
        self.assertEqual(response.token, 1)

        del batch_gen

        batch_gen = BatchGenerator(
            self.model,
            max_tokens=1,
            sampler=lambda _: mx.array([1]),
        )

        (uid0,) = batch_gen.insert([prompt])
        uid1, uid2 = batch_gen.insert(
            [prompt, prompt],
            samplers=[lambda _: mx.array([2]), lambda _: mx.array([3])],
        )

        responses = batch_gen.next()
        responses = {response.uid: response for response in responses}
        self.assertEqual(responses[uid0].token, 1)
        self.assertEqual(responses[uid1].token, 2)
        self.assertEqual(responses[uid2].token, 3)

    def test_batch_continued_generation(self):
        for rotating in [False, True]:
            if rotating:
                self.model.make_cache = lambda: [
                    RotatingKVCache(max_size=4) for _ in self.model.layers
                ]

            # Make the prompts
            prompts_a = [
                "Write a story about Einstein",
                "Hi",
                "What time is it?",
                "How tall is Mt Everest?",
            ]
            prompts_a = [
                self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": p}],
                    tokenize=True,
                    add_generation_prompt=True,
                )
                for p in prompts_a
            ]
            prompts_b = [
                "Another one",
                "sup?",
                "And how about the date?",
                "Mt Olympus?",
            ]
            prompts_b = [
                self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": p}],
                    tokenize=True,
                    add_generation_prompt=True,
                )
                for p in prompts_b
            ]

            # Generate once
            batch_gen = BatchGenerator(
                self.model,
                stop_tokens=self.tokenizer.eos_token_ids,
                max_tokens=10,
                prefill_batch_size=4,
                prefill_step_size=8,
                completion_batch_size=2,
            )
            uids = batch_gen.insert(prompts_a)
            caches = {uid: None for uid in uids}
            while responses := batch_gen.next():
                for r in responses:
                    if r.finish_reason is not None:
                        caches[r.uid] = r.prompt_cache
            caches = [caches[uid] for uid in uids]

            # Generate the 2nd time
            uids = batch_gen.insert(prompts_b, caches=caches)
            batch_responses = {uid: [] for uid in uids}
            while responses := batch_gen.next():
                for r in responses:
                    batch_responses[r.uid].append(r.logprobs)

            for e, uid in enumerate(uids):
                for i, response in enumerate(
                    stream_generate(
                        self.model,
                        self.tokenizer,
                        prompts_b[e],
                        max_tokens=10,
                        prompt_cache=caches[e],
                    )
                ):
                    batch_logprobs = batch_responses[uid][i]
                    logprobs = response.logprobs
                    self.assertTrue(
                        mx.allclose(batch_logprobs, logprobs, rtol=1e-4, atol=1e-4)
                    )

            if rotating:
                del self.model.make_cache

    def _continued_generation_test_helper(self, model):
        def rand_prompt(n):
            return [random.randint(0, 1000) for _ in range(n)]

        # Make the prompts
        prompts_a = [
            rand_prompt(5),
            rand_prompt(3),
            rand_prompt(8),
            rand_prompt(1),
        ]
        prompts_b = [
            rand_prompt(2),
            rand_prompt(7),
            rand_prompt(4),
            rand_prompt(6),
        ]

        # Generate once
        batch_gen = BatchGenerator(
            model,
            stop_tokens={},
            max_tokens=10,
            prefill_batch_size=4,
            prefill_step_size=32,
            completion_batch_size=2,
        )

        uids = batch_gen.insert(prompts_a)
        caches = {uid: None for uid in uids}
        while responses := batch_gen.next():
            for r in responses:
                if r.finish_reason is not None:
                    caches[r.uid] = r.prompt_cache

        caches = [caches[uid] for uid in uids]

        # Generate the 2nd time
        uids = batch_gen.insert(prompts_b, caches=caches)
        batch_responses = {uid: [] for uid in uids}
        while responses := batch_gen.next():
            for r in responses:
                batch_responses[r.uid].append(r.logprobs)

        for e, uid in enumerate(uids):
            for i, (_, logprobs) in enumerate(
                generate_step(
                    mx.array(prompts_b[e]),
                    model,
                    max_tokens=10,
                    prompt_cache=caches[e],
                )
            ):
                batch_logprobs = batch_responses[uid][i]
                self.assertTrue(
                    mx.allclose(batch_logprobs, logprobs, rtol=1e-4, atol=1e-4)
                )

    def test_batch_continued_generation_ssm(self):
        from mlx_lm.models import mamba2

        random.seed(0)
        mx.random.seed(4)

        # Make a small SSM model
        args = mamba2.ModelArgs(
            model_type="mamba2",
            num_heads=8,
            head_dim=16,
            vocab_size=1000,
            hidden_size=128,
            intermediate_size=128,
            state_size=32,
            num_hidden_layers=4,
            layer_norm_epsilon=1e-4,
            conv_kernel=3,
            n_groups=4,
            use_bias=False,
            use_conv_bias=False,
            tie_word_embeddings=True,
            time_step_limit=(0.01, 10),
            time_step_rank="auto",
        )
        model = mamba2.Model(args)
        self._continued_generation_test_helper(model)

    def test_batch_continued_generation_gated_delta(self):
        from mlx_lm.models import qwen3_next

        random.seed(0)
        mx.random.seed(4)
        args = qwen3_next.ModelArgs(
            model_type="qwen3_next",
            hidden_size=128,
            num_hidden_layers=4,
            intermediate_size=128,
            num_attention_heads=8,
            num_key_value_heads=4,
            vocab_size=1000,
            linear_num_value_heads=4,
            linear_num_key_heads=4,
            linear_key_head_dim=32,
            linear_value_head_dim=32,
            linear_conv_kernel_dim=3,
            num_experts=4,
            num_experts_per_tok=2,
            decoder_sparse_step=1,
            shared_expert_intermediate_size=128,
            mlp_only_layers=[0],
            moe_intermediate_size=128,
            rms_norm_eps=1e-5,
            head_dim=64,
            rope_theta=1000.0,
            partial_rotary_factor=0.5,
            max_position_embeddings=1000,
        )
        model = qwen3_next.Model(args)
        self._continued_generation_test_helper(model)


if __name__ == "__main__":
    unittest.main()
