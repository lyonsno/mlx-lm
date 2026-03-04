# Copyright © 2024 Apple Inc.

import copy
import http
import io
import json
import threading
import unittest

import mlx.core as mx
import requests

from mlx_lm.models.cache import KVCache, RotatingKVCache
from mlx_lm.server import APIHandler, LRUPromptCache, ResponseGenerator
from mlx_lm.utils import load


class DummyModelProvider:
    def __init__(self, with_draft=False):
        HF_MODEL_PATH = "mlx-community/Qwen1.5-0.5B-Chat-4bit"
        self.model, self.tokenizer = load(HF_MODEL_PATH)
        self.model_key = (HF_MODEL_PATH, None)
        self.is_batchable = True

        # Add draft model support
        self.draft_model = None
        self.draft_model_key = None
        self.cli_args = type(
            "obj",
            (object,),
            {
                "adapter_path": None,
                "chat_template": None,
                "use_default_chat_template": False,
                "trust_remote_code": False,
                "draft_model": None,
                "num_draft_tokens": 3,
                "temp": 0.0,
                "top_p": 1.0,
                "top_k": 0,
                "min_p": 0.0,
                "max_tokens": 512,
                "chat_template_args": {},
                "model": None,
                "decode_concurrency": 32,
                "prompt_concurrency": 8,
                "prefill_step_size": 2048,
                "prompt_cache_size": 10,
                "prompt_cache_bytes": 1 << 63,
                "prompt_cache_total_bytes": None,
            },
        )

        if with_draft:
            # Use the same model as the draft model for testing
            self.draft_model, _ = load(HF_MODEL_PATH)
            self.draft_model_key = HF_MODEL_PATH
            self.cli_args.draft_model = HF_MODEL_PATH

    def load(self, model, adapter=None, draft_model=None):
        assert model in ["default_model", "chat_model"]
        return self.model, self.tokenizer


class MockCache:
    def __init__(self, value):
        self.value = value

    @property
    def nbytes(self):
        return len(self.value)

    def __eq__(self, other):
        return other.value == self.value


class TestServer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.response_generator = ResponseGenerator(
            DummyModelProvider(), LRUPromptCache()
        )
        cls.server_address = ("localhost", 0)
        cls.httpd = http.server.HTTPServer(
            cls.server_address,
            lambda *args, **kwargs: APIHandler(cls.response_generator, *args, **kwargs),
        )
        cls.port = cls.httpd.server_port
        cls.server_thread = threading.Thread(target=cls.httpd.serve_forever)
        cls.server_thread.daemon = True
        cls.server_thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()
        cls.server_thread.join()
        cls.response_generator.stop_and_join()

    def test_handle_completions(self):
        url = f"http://localhost:{self.port}/v1/completions"

        post_data = {
            "model": "default_model",
            "prompt": "Once upon a time",
            "max_tokens": 10,
            "temperature": 0.5,
            "top_p": 0.9,
            "repetition_penalty": 1.1,
            "repetition_context_size": 20,
            "seed": 999,
            "stop": "stop sequence",
        }

        response = requests.post(url, json=post_data)

        response_body = json.loads(response.text)

        self.assertIn("id", response_body)
        self.assertIn("choices", response_body)
        first_text = response_body["choices"][0]["text"]
        self.assertEqual(
            first_text,
            json.loads(requests.post(url, json=post_data).text)["choices"][0]["text"],
        )

    def test_handle_chat_completions(self):
        url = f"http://localhost:{self.port}/v1/chat/completions"
        chat_post_data = {
            "model": "chat_model",
            "max_tokens": 10,
            "temperature": 0.7,
            "top_p": 0.85,
            "repetition_penalty": 1.2,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello!"},
            ],
        }
        response = requests.post(url, json=chat_post_data)
        response_body = response.text
        self.assertIn("id", response_body)
        self.assertIn("choices", response_body)

    def test_handle_chat_completions_with_content_fragments(self):
        url = f"http://localhost:{self.port}/v1/chat/completions"
        chat_post_data = {
            "model": "chat_model",
            "max_tokens": 10,
            "temperature": 0.7,
            "top_p": 0.85,
            "repetition_penalty": 1.2,
            "messages": [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": "You are a helpful assistant."}
                    ],
                },
                {"role": "user", "content": [{"type": "text", "text": "Hello!"}]},
            ],
        }
        response = requests.post(url, json=chat_post_data)
        response_body = response.text
        self.assertIn("id", response_body)
        self.assertIn("choices", response_body)

    def test_handle_chat_completions_with_null_tool_content(self):
        url = f"http://localhost:{self.port}/v1/chat/completions"
        chat_post_data = {
            "model": "chat_model",
            "max_tokens": 10,
            "temperature": 0.7,
            "top_p": 0.85,
            "repetition_penalty": 1.2,
            "messages": [
                {"role": "user", "content": "what is 2+3?"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "type": "function",
                            "id": "123",
                            "function": {
                                "name": "add",
                                "arguments": '{"a": 2, "b": 3}',
                            },
                        }
                    ],
                },
                {"role": "tool", "content": "5", "tool_call_id": "123"},
            ],
        }
        response = requests.post(url, json=chat_post_data)
        response_body = response.text
        self.assertIn("id", response_body)
        self.assertIn("choices", response_body)

    def test_handle_models(self):
        url = f"http://localhost:{self.port}/v1/models"
        response = requests.get(url)
        self.assertEqual(response.status_code, 200)
        response_body = json.loads(response.text)
        self.assertEqual(response_body["object"], "list")
        self.assertIsInstance(response_body["data"], list)
        self.assertGreater(len(response_body["data"]), 0)
        model = response_body["data"][0]
        self.assertIn("id", model)
        self.assertEqual(model["object"], "model")
        self.assertIn("created", model)

    def test_sequence_overlap(self):
        from mlx_lm.server import sequence_overlap

        self.assertTrue(sequence_overlap([1], [1]))
        self.assertTrue(sequence_overlap([1, 2], [1, 2]))
        self.assertTrue(sequence_overlap([1, 3], [3, 4]))
        self.assertTrue(sequence_overlap([1, 2, 3], [2, 3]))

        self.assertFalse(sequence_overlap([1], [2]))
        self.assertFalse(sequence_overlap([1, 2], [3, 4]))
        self.assertFalse(sequence_overlap([1, 2, 3], [4, 1, 2, 3]))


class TestServerWithDraftModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.response_generator = ResponseGenerator(
            DummyModelProvider(with_draft=True), LRUPromptCache()
        )
        cls.server_address = ("localhost", 0)
        cls.httpd = http.server.HTTPServer(
            cls.server_address,
            lambda *args, **kwargs: APIHandler(cls.response_generator, *args, **kwargs),
        )
        cls.port = cls.httpd.server_port
        cls.server_thread = threading.Thread(target=cls.httpd.serve_forever)
        cls.server_thread.daemon = True
        cls.server_thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()
        cls.server_thread.join()
        cls.response_generator.stop_and_join()

    def test_handle_completions_with_draft_model(self):
        url = f"http://localhost:{self.port}/v1/completions"

        post_data = {
            "model": "default_model",
            "prompt": "Once upon a time",
            "max_tokens": 10,
            "temperature": 0.0,
            "top_p": 1.0,
        }

        response = requests.post(url, json=post_data)
        self.assertEqual(response.status_code, 200)

        response_body = json.loads(response.text)
        self.assertIn("id", response_body)
        self.assertIn("choices", response_body)
        self.assertIn("usage", response_body)

        # Check that tokens were generated
        self.assertTrue(response_body["usage"]["completion_tokens"] > 0)

    def test_handle_chat_completions_with_draft_model(self):
        url = f"http://localhost:{self.port}/v1/chat/completions"

        chat_post_data = {
            "model": "chat_model",
            "max_tokens": 10,
            "temperature": 0.0,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello!"},
            ],
        }

        response = requests.post(url, json=chat_post_data)
        self.assertEqual(response.status_code, 200)

        response_body = json.loads(response.text)
        self.assertIn("id", response_body)
        self.assertIn("choices", response_body)
        self.assertIn("usage", response_body)

        # Check that tokens were generated
        self.assertTrue(response_body["usage"]["completion_tokens"] > 0)

    def test_streaming_with_draft_model(self):
        url = f"http://localhost:{self.port}/v1/chat/completions"

        chat_post_data = {
            "model": "chat_model",
            "max_tokens": 10,
            "temperature": 0.0,
            "stream": True,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello!"},
            ],
        }

        response = requests.post(url, json=chat_post_data, stream=True)
        self.assertEqual(response.status_code, 200)

        chunk_count = 0
        for chunk in response.iter_lines():
            if chunk:
                data = chunk.decode("utf-8")
                if data.startswith("data: ") and data != "data: [DONE]":
                    chunk_data = json.loads(data[6:])  # Skip the "data: " prefix
                    self.assertIn("choices", chunk_data)
                    self.assertEqual(len(chunk_data["choices"]), 1)
                    self.assertIn("delta", chunk_data["choices"][0])
                    chunk_count += 1

        # Make sure we got some streaming chunks
        self.assertGreater(chunk_count, 0)

    def test_prompt_cache_with_draft_model(self):
        url = f"http://localhost:{self.port}/v1/chat/completions"

        # First request to initialize cache
        chat_post_data = {
            "model": "chat_model",
            "max_tokens": 5,
            "temperature": 0.0,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Tell me a story about"},
            ],
        }

        first_response = requests.post(url, json=chat_post_data)
        self.assertEqual(first_response.status_code, 200)

        # Second request with same prefix should use cache
        chat_post_data = {
            "model": "chat_model",
            "max_tokens": 5,
            "temperature": 0.0,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Tell me a story about dragons."},
            ],
        }

        second_response = requests.post(url, json=chat_post_data)
        self.assertEqual(second_response.status_code, 200)

        # Both responses should have content
        first_response_body = json.loads(first_response.text)
        second_response_body = json.loads(second_response.text)

        self.assertIn("choices", first_response_body)
        self.assertIn("choices", second_response_body)
        self.assertIn("message", first_response_body["choices"][0])
        self.assertIn("message", second_response_body["choices"][0])
        self.assertIn("content", first_response_body["choices"][0]["message"])
        self.assertIn("content", second_response_body["choices"][0]["message"])

        # Ensure both generated content
        self.assertIsNotNone(first_response_body["choices"][0]["message"]["content"])
        self.assertIsNotNone(second_response_body["choices"][0]["message"]["content"])


class TestKeepalive(unittest.TestCase):
    def test_keepalive_callback(self):
        """Test keepalive callback sends SSE comments and handles errors"""
        from unittest.mock import Mock

        # Mock handler
        mock_wfile = io.BytesIO()
        handler = Mock()
        handler.wfile = mock_wfile

        # Test callback logic (same as in server.py)
        def keepalive_callback(processed_tokens, total_tokens):
            if handler.stream:
                try:
                    handler.wfile.write(
                        f": keepalive {processed_tokens}/{total_tokens}\n\n".encode()
                    )
                    handler.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, OSError):
                    pass

        # Test streaming enabled
        handler.stream = True
        keepalive_callback(1024, 4096)

        output = mock_wfile.getvalue().decode("utf-8")
        self.assertEqual(output, ": keepalive 1024/4096\n\n")

        # Test streaming disabled
        handler.stream = False
        mock_wfile.seek(0)
        mock_wfile.truncate(0)
        keepalive_callback(2048, 4096)

        output = mock_wfile.getvalue().decode("utf-8")
        self.assertEqual(output, "")

        # Test error handling
        handler.stream = True
        handler.wfile = Mock()
        handler.wfile.write.side_effect = BrokenPipeError("Connection broken")

        # Should not raise exception
        try:
            keepalive_callback(3072, 4096)
        except Exception as e:
            self.fail(f"Callback should handle BrokenPipeError: {e}")


class TestLRUPromptCache(unittest.TestCase):
    def _make_tiny_step3p5_model(self):
        from mlx_lm.models import step3p5

        # Keep this config minimal and centralized so schema churn in one place
        # does not ripple through multiple cache behavior assertions.
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

    def test_caching(self):
        cache = LRUPromptCache(max_size=10)

        def get_kv(n):
            keys = mx.arange(n).reshape(1, 1, n, 1)
            return keys, keys

        model = ("test", None, None)
        tokens = [10] * 24

        c, t = cache.fetch_nearest_cache(model, tokens)
        self.assertTrue(c is None)
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
        self.assertEqual(len(cache._lru), 0)

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
        self.assertEqual(len(cache._lru), 1)

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
        self.assertEqual(c, None)
        self.assertEqual(t, [1, 2])

        cache.insert_cache(model, [1, 2], [MockCache("test1")])
        cache.insert_cache(model, [2, 3], [MockCache("test2")])
        cache.insert_cache(model, [3, 4], [MockCache("test3")])

        c, t = cache.fetch_nearest_cache(model, [1, 2])
        self.assertEqual(c, None)
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
        self.assertEqual(c, None)
        self.assertEqual(t, [1, 2])
        c, t = cache.fetch_nearest_cache(model, [3, 4])
        self.assertEqual(c, None)
        self.assertEqual(t, [3, 4])

    def test_mixed_cache_longer_prefix_reuse_when_rotating_cache_is_full(self):
        lru = LRUPromptCache(max_size=10)
        model_key = ("step3p5-tiny", None, None)

        model = self._make_tiny_step3p5_model()

        long_tokens = list(range(1, 13))
        shorter_tokens = long_tokens[:8]
        expected_cached_prefix = len(shorter_tokens) - 1
        continuation_tokens = [42, 43, 44, 45]

        long_array = mx.array([long_tokens], dtype=mx.int32)
        shorter_array = mx.array([shorter_tokens], dtype=mx.int32)
        remaining_array = mx.array([shorter_tokens[-1:]], dtype=mx.int32)

        # Rotating cache is saturated and non-trimmable via its public contract.
        long_cache = model.make_cache()
        mx.eval(model(long_array, cache=long_cache))
        stored_snapshot = copy.deepcopy(long_cache)

        rotating_layers = [c for c in long_cache if isinstance(c, RotatingKVCache)]
        self.assertGreater(len(rotating_layers), 0)
        for sliding in rotating_layers:
            self.assertEqual(sliding.size(), sliding.max_size)
            self.assertFalse(sliding.is_trimmable())

        lru.insert_cache(model_key, long_tokens, long_cache)
        reused_cache, remaining = lru.fetch_nearest_cache(model_key, shorter_tokens)

        # Desired behavior for mixed-cache models:
        # still return a reusable cache object and a short suffix to process,
        # rather than dropping to a full miss.
        self.assertIsNotNone(reused_cache)
        self.assertEqual(len(reused_cache), len(model.layers))
        self.assertEqual(remaining, shorter_tokens[-1:])

        # Longer-prefix lookups use read-only access today, so the long entry
        # should still be reusable as an exact match after a shorter lookup.
        exact_cache, exact_remaining = lru.fetch_nearest_cache(model_key, long_tokens)
        self.assertIsNotNone(exact_cache)
        self.assertEqual(exact_remaining, [])

        # Exact long hit should remain intact compared to original stored state.
        self.assertEqual(
            [c.offset for c in exact_cache], [c.offset for c in stored_snapshot]
        )
        for fetched, snap in zip(exact_cache, stored_snapshot):
            self.assertIsInstance(fetched, (KVCache, RotatingKVCache))
            self.assertIsInstance(snap, (KVCache, RotatingKVCache))
            if isinstance(fetched, RotatingKVCache):
                self.assertEqual(fetched.max_size, snap.max_size)
                self.assertEqual(fetched.keep, snap.keep)
            if isinstance(fetched, (KVCache, RotatingKVCache)):
                fk, fv = fetched.state
                sk, sv = snap.state
                self.assertTrue(mx.array_equal(fk, sk))
                self.assertTrue(mx.array_equal(fv, sv))

        # If a cache is returned, it should represent the common prefix
        # (all shorter tokens except the final carry token).
        self.assertEqual(
            [c.offset for c in reused_cache],
            [expected_cached_prefix] * len(reused_cache),
        )

        # Full-attention caches must match a direct prefix prefill baseline.
        prefix_array = mx.array([shorter_tokens[:-1]], dtype=mx.int32)
        baseline_prefix_cache = model.make_cache()
        mx.eval(model(prefix_array, cache=baseline_prefix_cache))
        full_attention_indices = [
            i for i, layer in enumerate(model.layers) if not layer.is_sliding
        ]
        for idx in full_attention_indices:
            rk, rv = reused_cache[idx].state
            bk, bv = baseline_prefix_cache[idx].state
            self.assertEqual(rk.shape[2], expected_cached_prefix)
            self.assertEqual(rv.shape[2], expected_cached_prefix)
            self.assertTrue(mx.allclose(rk, bk, rtol=1e-5, atol=1e-5))
            self.assertTrue(mx.allclose(rv, bv, rtol=1e-5, atol=1e-5))

        # Sliding cache should preserve a bounded, non-empty window.
        sliding_indices = [
            i for i, layer in enumerate(model.layers) if layer.is_sliding
        ]
        for idx in sliding_indices:
            self.assertGreater(reused_cache[idx].size(), 0)
            self.assertLessEqual(reused_cache[idx].size(), reused_cache[idx].max_size)

        # Behavior-level check: short multi-token continuation from the reused
        # cache should match a baseline cache built from the shorter prompt.
        baseline_cache = model.make_cache()
        mx.eval(model(shorter_array, cache=baseline_cache))

        # Process the remaining carry token to align with the full shorter prompt.
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

    def test_mixed_cache_longer_prefix_reuse_preserves_refcounted_long_entry(self):
        lru = LRUPromptCache(max_size=10)
        model_key = ("step3p5-tiny", None, None)
        model = self._make_tiny_step3p5_model()

        long_tokens = list(range(1, 13))
        shorter_tokens = long_tokens[:8]

        long_array = mx.array([long_tokens], dtype=mx.int32)
        long_cache = model.make_cache()
        mx.eval(model(long_array, cache=long_cache))

        # Store two references to the same long cache entry.
        lru.insert_cache(model_key, long_tokens, long_cache)
        lru.insert_cache(model_key, long_tokens, long_cache)

        # Mixed longer->shorter lookup should return a reusable cache.
        # It should not consume one of the long-entry references.
        reused_cache, remaining = lru.fetch_nearest_cache(model_key, shorter_tokens)
        self.assertIsNotNone(reused_cache)
        self.assertEqual(remaining, shorter_tokens[-1:])

        # If longer->shorter lookup is non-destructive, we should still get
        # two exact hits from the refcounted long entry.
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
        model = self._make_tiny_step3p5_model()

        prompt_tokens = list(range(1, 13))
        decoded_tokens = prompt_tokens + [99]
        shorter_tokens = prompt_tokens[:8]

        prompt_array = mx.array([prompt_tokens], dtype=mx.int32)
        decode_array = mx.array([[decoded_tokens[-1]]], dtype=mx.int32)

        long_cache = model.make_cache()
        mx.eval(model(prompt_array, cache=long_cache))
        mx.eval(model(decode_array, cache=long_cache))

        # After decode-time in-place rotation, offsets can exceed backing
        # storage length, so older history is unrecoverable for rewind.
        rotating_layers = [c for c in long_cache if isinstance(c, RotatingKVCache)]
        self.assertGreater(len(rotating_layers), 0)
        for sliding in rotating_layers:
            self.assertGreater(sliding.offset, sliding.keys.shape[2])
            self.assertEqual(sliding.size(), sliding.max_size)
            self.assertFalse(sliding.is_trimmable())

        lru.insert_cache(model_key, decoded_tokens, long_cache)
        reused_cache, remaining = lru.fetch_nearest_cache(model_key, shorter_tokens)
        self.assertIsNone(reused_cache)
        self.assertEqual(remaining, shorter_tokens)

        # The longer entry itself should remain available as an exact hit.
        exact_cache, exact_remaining = lru.fetch_nearest_cache(
            model_key, decoded_tokens
        )
        self.assertIsNotNone(exact_cache)
        self.assertEqual(exact_remaining, [])


if __name__ == "__main__":
    unittest.main()
