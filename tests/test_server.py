# Copyright © 2024 Apple Inc.

import copy
import http
import io
import json
import threading
import unittest
from queue import Queue
from unittest.mock import patch

import mlx.core as mx
import requests

from mlx_lm.models.cache import CacheList, KVCache, RotatingKVCache
from mlx_lm.server import (
    APIHandler,
    CompletionRequest,
    GenerationArguments,
    LogitsProcessorArguments,
    LRUPromptCache,
    ModelDescription,
    ResponseGenerator,
    SamplingArguments,
)
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


class TestResponseGeneratorPrefillStepSizeForwarding(unittest.TestCase):
    @staticmethod
    def _generation_args():
        return GenerationArguments(
            model=ModelDescription("default_model", None, None),
            sampling=SamplingArguments(0.0, 1.0, 0, 0.0, 0.0, 0.0),
            logits=LogitsProcessorArguments(None, 1.0, 20),
            stop_words=[],
            max_tokens=2,
            num_draft_tokens=3,
            logprobs=False,
            top_logprobs=0,
            seed=None,
            chat_template_kwargs=None,
        )

    @staticmethod
    def _make_text_request(prompt="hello"):
        return CompletionRequest(
            request_type="text",
            prompt=prompt,
            messages=[],
            tools=None,
            role_mapping=None,
        )

    def _build_response_generator(self, *, prefill_step_size):
        class FakeModel:
            def make_cache(self):
                return [KVCache()]

        class FakeTokenizer:
            has_tool_calling = False
            tool_call_start = ""
            tool_call_end = ""
            tool_parser = staticmethod(lambda text, _: {})
            has_thinking = False
            think_start_id = 0
            think_end_id = 0
            think_end = ""
            eos_token_id = 0
            eos_token_ids = set()

            def encode(self, text, add_special_tokens=False):
                return [1, 2, 3]

        class FakeProvider:
            is_batchable = True

            def __init__(self):
                self.cli_args = type(
                    "obj",
                    (object,),
                    {
                        "decode_concurrency": 4,
                        "prompt_concurrency": 2,
                        "prefill_step_size": prefill_step_size,
                        "prompt_cache_bytes": None,
                    },
                )
                self.model = FakeModel()
                self.tokenizer = FakeTokenizer()
                self.draft_model = None
                self.model_key = ("fake-model", None, None)

            def load(self, model, adapter=None, draft_model=None):
                return self.model, self.tokenizer

        generator = ResponseGenerator.__new__(ResponseGenerator)
        generator.model_provider = FakeProvider()
        generator.prompt_cache = LRUPromptCache(max_size=10)
        generator.requests = Queue()
        generator._is_distributed = False
        generator._rank = 0
        generator._stop = False
        generator._time_budget = []
        return generator

    def test_serve_single_forwards_prefill_step_size(self):
        expected_prefill_step_size = 77
        generator = self._build_response_generator(
            prefill_step_size=expected_prefill_step_size
        )

        gen_result = type(
            "GenResult",
            (),
            {
                "text": "x",
                "token": 0,
                "logprobs": mx.array([0.0], dtype=mx.float32),
                "finish_reason": "stop",
            },
        )()

        with patch(
            "mlx_lm.server.stream_generate", return_value=iter([gen_result])
        ) as stream_generate_mock:
            generator._serve_single(
                (Queue(), self._make_text_request(), self._generation_args())
            )

        self.assertEqual(stream_generate_mock.call_count, 1)
        self.assertEqual(
            stream_generate_mock.call_args.kwargs["prefill_step_size"],
            expected_prefill_step_size,
        )

    def test_generate_batch_mode_forwards_prefill_step_size(self):
        expected_prefill_step_size = 91
        generator = self._build_response_generator(
            prefill_step_size=expected_prefill_step_size
        )
        request_queue = Queue()
        request_args = self._generation_args()
        request = self._make_text_request()
        request_seen = False

        def next_request(timeout=None):
            nonlocal request_seen
            if request_seen:
                return None
            request_seen = True
            return (request_queue, request, request_args)

        generator._next_request = next_request

        def fake_batch_generator(*args, **kwargs):
            generator._stop = True
            return object()

        with patch(
            "mlx_lm.server.BatchGenerator", side_effect=fake_batch_generator
        ) as batch_generator_mock:
            generator._generate()

        self.assertEqual(batch_generator_mock.call_count, 1)
        self.assertEqual(
            batch_generator_mock.call_args.kwargs["prefill_step_size"],
            expected_prefill_step_size,
        )


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

    def _build_real_rotating_cache(self, *, max_size=4, total_tokens=4):
        cache = RotatingKVCache(max_size=max_size)
        kv = mx.arange(total_tokens, dtype=mx.float32).reshape(1, 1, total_tokens, 1)
        cache.update_and_fetch(kv, kv)
        mx.eval(cache.keys, cache.values)
        return cache

    def _snapshot_cache_arrays(self, cache):
        keys = mx.array(cache.keys) if cache.keys is not None else None
        values = mx.array(cache.values) if cache.values is not None else None
        if keys is not None:
            mx.eval(keys)
        if values is not None:
            mx.eval(values)
        return keys, values

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

    def test_rewind_prompt_cache_fails_closed_on_partial_trim(self):
        class PartialTrimCache:
            def __init__(self):
                self.calls = []

            def is_trimmable(self):
                return True

            def trim(self, n):
                self.calls.append(n)
                return n - 1

        class FullTrimCache:
            def __init__(self):
                self.calls = []

            def is_trimmable(self):
                return True

            def trim(self, n):
                self.calls.append(n)
                return n

        lru = LRUPromptCache(max_size=10)
        partial = PartialTrimCache()
        full = FullTrimCache()
        composite = CacheList(partial, full)

        ok = lru._rewind_prompt_cache([composite], 5)
        self.assertFalse(ok)
        self.assertEqual(partial.calls, [5])

    def test_rewind_prompt_cache_fails_closed_on_nested_cachelist_partial_trim(self):
        class PartialTrimCache:
            def __init__(self):
                self.calls = []

            def is_trimmable(self):
                return True

            def trim(self, n):
                self.calls.append(n)
                return n - 1

        class FullTrimCache:
            def __init__(self):
                self.calls = []

            def is_trimmable(self):
                return True

            def trim(self, n):
                self.calls.append(n)
                return n

        lru = LRUPromptCache(max_size=10)
        partial = PartialTrimCache()
        inner_full = FullTrimCache()
        outer_full = FullTrimCache()
        nested = CacheList(CacheList(partial, inner_full), outer_full)

        ok = lru._rewind_prompt_cache([nested], 5)
        self.assertFalse(ok)
        self.assertEqual(partial.calls, [5])

    def test_rewind_prompt_cache_fails_closed_on_non_first_cachelist_child(self):
        class PartialTrimCache:
            def __init__(self):
                self.calls = []

            def is_trimmable(self):
                return True

            def trim(self, n):
                self.calls.append(n)
                return n - 1

        class FullTrimCache:
            def __init__(self):
                self.calls = []

            def is_trimmable(self):
                return True

            def trim(self, n):
                self.calls.append(n)
                return n

        lru = LRUPromptCache(max_size=10)
        first_full = FullTrimCache()
        second_partial = PartialTrimCache()
        composite = CacheList(first_full, second_partial)

        ok = lru._rewind_prompt_cache([composite], 5)
        self.assertFalse(ok)
        self.assertEqual(first_full.calls, [5])
        self.assertEqual(second_partial.calls, [5])

    def test_fast_trim_path_fails_closed_on_partial_trim(self):
        class PartialTrimLayer:
            total_calls = 0
            trim_args = []

            def __init__(self):
                self.offset = 4

            @property
            def nbytes(self):
                return 1

            def is_trimmable(self):
                return True

            def trim(self, n):
                type(self).total_calls += 1
                type(self).trim_args.append(n)
                trimmed = max(0, n - 1)
                self.offset = max(0, self.offset - trimmed)
                return trimmed

        PartialTrimLayer.total_calls = 0
        PartialTrimLayer.trim_args = []

        lru = LRUPromptCache(max_size=10)
        model = ("fast-trim", None, None)
        long_tokens = [1, 2, 3, 4]
        shorter_tokens = [1, 2]
        expected_num_to_trim = len(long_tokens) - (len(shorter_tokens) - 1)
        layer = PartialTrimLayer()

        lru.insert_cache(model, long_tokens, [layer])
        reused_cache, remaining = lru.fetch_nearest_cache(model, shorter_tokens)

        # Even on the all-trimmable fast path, partial trims must fail closed.
        self.assertIsNone(reused_cache)
        self.assertEqual(remaining, shorter_tokens)
        self.assertEqual(PartialTrimLayer.total_calls, 1)
        self.assertEqual(PartialTrimLayer.trim_args, [expected_num_to_trim])

        # Safe miss should not consume or mutate the stored longer entry.
        exact_cache, exact_remaining = lru.fetch_nearest_cache(model, long_tokens)
        self.assertIsNotNone(exact_cache)
        self.assertEqual(exact_remaining, [])
        self.assertEqual(exact_cache[0].offset, 4)

    def test_rewind_rotating_cache_fails_without_values_and_keeps_state(self):
        missing_values = self._build_real_rotating_cache()
        missing_values.values = None
        original_offset = missing_values.offset
        original_idx = missing_values._idx
        original_keys, _ = self._snapshot_cache_arrays(missing_values)
        self.assertFalse(LRUPromptCache._rewind_rotating_cache(missing_values, 2))
        self.assertEqual(missing_values.offset, original_offset)
        self.assertEqual(missing_values._idx, original_idx)
        self.assertTrue(mx.array_equal(missing_values.keys, original_keys))
        self.assertIsNone(missing_values.values)

    def test_rewind_rotating_cache_fails_when_trim_exceeds_offset_and_keeps_state(self):
        """Regression guard: fail path preserves state when trim exceeds offset."""
        insufficient_offset = self._build_real_rotating_cache()
        original_offset = insufficient_offset.offset
        original_idx = insufficient_offset._idx
        original_keys, original_values = self._snapshot_cache_arrays(
            insufficient_offset
        )
        self.assertFalse(
            LRUPromptCache._rewind_rotating_cache(
                insufficient_offset, insufficient_offset.offset + 1
            )
        )
        self.assertEqual(insufficient_offset.offset, original_offset)
        self.assertEqual(insufficient_offset._idx, original_idx)
        self.assertTrue(mx.array_equal(insufficient_offset.keys, original_keys))
        self.assertTrue(mx.array_equal(insufficient_offset.values, original_values))

    def test_rewind_rotating_cache_fails_when_history_unrecoverable_and_keeps_state(
        self,
    ):
        """Regression guard: unrecoverable history must fail closed without mutation."""
        unrecoverable_history = self._build_real_rotating_cache()
        unrecoverable_history.offset = unrecoverable_history.keys.shape[2] + 1
        original_offset = unrecoverable_history.offset
        original_idx = unrecoverable_history._idx
        original_keys, original_values = self._snapshot_cache_arrays(
            unrecoverable_history
        )
        self.assertFalse(
            LRUPromptCache._rewind_rotating_cache(unrecoverable_history, 1)
        )
        self.assertEqual(unrecoverable_history.offset, original_offset)
        self.assertEqual(unrecoverable_history._idx, original_idx)
        self.assertTrue(mx.array_equal(unrecoverable_history.keys, original_keys))
        self.assertTrue(mx.array_equal(unrecoverable_history.values, original_values))

    def test_rewind_rotating_cache_fails_when_trim_exceeds_idx_and_keeps_state(self):
        insufficient_idx = self._build_real_rotating_cache()
        insufficient_idx._idx = 0
        original_offset = insufficient_idx.offset
        original_idx = insufficient_idx._idx
        original_keys, original_values = self._snapshot_cache_arrays(insufficient_idx)
        self.assertFalse(LRUPromptCache._rewind_rotating_cache(insufficient_idx, 1))
        self.assertEqual(insufficient_idx.offset, original_offset)
        self.assertEqual(insufficient_idx._idx, original_idx)
        self.assertTrue(mx.array_equal(insufficient_idx.keys, original_keys))
        self.assertTrue(mx.array_equal(insufficient_idx.values, original_values))

    def test_rewind_rotating_cache_materializes_state_for_single_token_updates(self):
        """Regression guard: rewound cache must match direct-prefix continuation."""
        total_tokens = 8
        trim_tokens = 3
        expected_prefix = total_tokens - trim_tokens

        rewound = self._build_real_rotating_cache(total_tokens=total_tokens)
        self.assertTrue(LRUPromptCache._rewind_rotating_cache(rewound, trim_tokens))

        next_tok = mx.array([[[[99.0]]]], dtype=mx.float32)
        rewound.update_and_fetch(next_tok, next_tok)
        mx.eval(rewound.keys, rewound.values)

        baseline = self._build_real_rotating_cache(total_tokens=expected_prefix)
        baseline.update_and_fetch(next_tok, next_tok)
        mx.eval(baseline.keys, baseline.values)

        self.assertEqual(rewound.offset, baseline.offset)
        self.assertEqual(rewound._idx, baseline._idx)
        self.assertTrue(mx.array_equal(rewound.keys, baseline.keys))
        self.assertTrue(mx.array_equal(rewound.values, baseline.values))

    def test_rewind_rotating_cache_zero_trim_is_noop(self):
        cache = self._build_real_rotating_cache(total_tokens=4)
        original_offset = cache.offset
        original_idx = cache._idx
        original_keys, original_values = self._snapshot_cache_arrays(cache)

        self.assertTrue(LRUPromptCache._rewind_rotating_cache(cache, 0))
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

        self.assertTrue(LRUPromptCache._rewind_rotating_cache(cache, 1))
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
        original_keys, _ = self._snapshot_cache_arrays(cache)

        lru = LRUPromptCache(max_size=10)
        ok = lru._rewind_prompt_cache([cache], 1)
        self.assertFalse(ok)
        self.assertEqual(cache.offset, original_offset)
        self.assertEqual(cache._idx, original_idx)
        self.assertTrue(mx.array_equal(cache.keys, original_keys))
        self.assertIsNone(cache.values)

    def test_unknown_non_trimmable_layer_type_fails_closed_and_keeps_exact_entry(self):
        """Regression guard: unknown non-trimmable layers safe-miss and preserve exact."""

        class UnknownNonTrimmableLayer:
            @property
            def nbytes(self):
                return 1

            def is_trimmable(self):
                return False

        lru = LRUPromptCache(max_size=10)
        model = ("unknown-layer", None, None)
        long_tokens = [1, 2, 3, 4]
        shorter_tokens = [1, 2]

        layer = UnknownNonTrimmableLayer()
        lru.insert_cache(model, long_tokens, [layer])
        lru.insert_cache(model, long_tokens, [layer])

        # Exercise real layer-type dispatch in _rewind_layer_cache: unknown,
        # non-trimmable, non-rotating layers must fail closed to a miss.
        reused_cache, remaining = lru.fetch_nearest_cache(model, shorter_tokens)
        self.assertIsNone(reused_cache)
        self.assertEqual(remaining, shorter_tokens)

        # Safe miss should be non-destructive to all exact longer references.
        hit1, rem1 = lru.fetch_nearest_cache(model, long_tokens)
        self.assertIsNotNone(hit1)
        self.assertEqual(rem1, [])

        hit2, rem2 = lru.fetch_nearest_cache(model, long_tokens)
        self.assertIsNotNone(hit2)
        self.assertEqual(rem2, [])

        miss, rem3 = lru.fetch_nearest_cache(model, long_tokens)
        self.assertIsNone(miss)
        self.assertEqual(rem3, long_tokens)

    def test_unknown_non_trimmable_layer_safe_miss_skips_deepcopy(self):
        class UnknownNonTrimmableNoDeepcopy:
            @property
            def nbytes(self):
                return 1

            def is_trimmable(self):
                return False

            def __deepcopy__(self, memo):
                raise AssertionError(
                    "deepcopy should be skipped for unknown non-trimmable layers"
                )

        lru = LRUPromptCache(max_size=10)
        model = ("unknown-no-deepcopy", None, None)
        long_tokens = [1, 2, 3, 4]
        shorter_tokens = [1, 2]

        lru.insert_cache(model, long_tokens, [UnknownNonTrimmableNoDeepcopy()])
        reused_cache, remaining = lru.fetch_nearest_cache(model, shorter_tokens)
        self.assertIsNone(reused_cache)
        self.assertEqual(remaining, shorter_tokens)

        exact_cache, exact_remaining = lru.fetch_nearest_cache(model, long_tokens)
        self.assertIsNotNone(exact_cache)
        self.assertEqual(exact_remaining, [])

    def test_unknown_layer_without_is_trimmable_fails_closed_without_deepcopy(self):
        class UnknownLayerNoIsTrimmable:
            @property
            def nbytes(self):
                return 1

            def __deepcopy__(self, memo):
                raise AssertionError(
                    "deepcopy should be skipped when layer lacks is_trimmable"
                )

        lru = LRUPromptCache(max_size=10)
        model = ("unknown-missing-is-trimmable", None, None)
        long_tokens = [1, 2, 3, 4]
        shorter_tokens = [1, 2]

        lru.insert_cache(model, long_tokens, [UnknownLayerNoIsTrimmable()])
        reused_cache, remaining = lru.fetch_nearest_cache(model, shorter_tokens)
        self.assertIsNone(reused_cache)
        self.assertEqual(remaining, shorter_tokens)

        exact_cache, exact_remaining = lru.fetch_nearest_cache(model, long_tokens)
        self.assertIsNotNone(exact_cache)
        self.assertEqual(exact_remaining, [])

    def test_composite_partial_trim_safe_miss_keeps_exact_entry_available(self):
        class PartialTrimLeaf:
            total_calls = 0
            trim_args = []

            def __init__(self):
                self.offset = 4

            @property
            def nbytes(self):
                return 1

            def is_trimmable(self):
                return True

            def trim(self, n):
                type(self).total_calls += 1
                type(self).trim_args.append(n)
                trimmed = max(0, n - 1)
                self.offset = max(0, self.offset - trimmed)
                return trimmed

        class FullTrimLeaf:
            total_calls = 0

            def __init__(self):
                self.offset = 4

            @property
            def nbytes(self):
                return 1

            def is_trimmable(self):
                return True

            def trim(self, n):
                type(self).total_calls += 1
                self.offset = max(0, self.offset - n)
                return n

        PartialTrimLeaf.total_calls = 0
        PartialTrimLeaf.trim_args = []
        FullTrimLeaf.total_calls = 0

        lru = LRUPromptCache(max_size=10)
        model = ("composite-partial", None, None)
        long_tokens = [1, 2, 3, 4]
        shorter_tokens = [1, 2]
        expected_num_to_trim = len(long_tokens) - (len(shorter_tokens) - 1)

        full = FullTrimLeaf()
        partial = PartialTrimLeaf()
        composite = CacheList(full, partial)
        lru.insert_cache(model, long_tokens, [composite])

        reused_cache, remaining = lru.fetch_nearest_cache(model, shorter_tokens)
        self.assertIsNone(reused_cache)
        self.assertEqual(remaining, shorter_tokens)
        self.assertEqual(FullTrimLeaf.total_calls, 1)
        self.assertEqual(PartialTrimLeaf.total_calls, 1)
        self.assertEqual(PartialTrimLeaf.trim_args, [expected_num_to_trim])

        # Safe miss should be non-destructive to the longer exact entry.
        exact_cache, exact_remaining = lru.fetch_nearest_cache(model, long_tokens)
        self.assertIsNotNone(exact_cache)
        self.assertEqual(exact_remaining, [])
        self.assertEqual(exact_cache[0][0].offset, 4)
        self.assertEqual(exact_cache[0][1].offset, 4)

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

    def test_mixed_cache_longer_prefix_reuse_after_chunked_prefill(self):
        lru = LRUPromptCache(max_size=10)
        model_key = ("step3p5-tiny-chunked", None, None)
        model = self._make_tiny_step3p5_model()

        long_tokens = list(range(1, 13))
        shorter_tokens = long_tokens[:10]
        continuation_tokens = [42, 43, 44]

        long_array = mx.array([long_tokens], dtype=mx.int32)
        shorter_array = mx.array([shorter_tokens], dtype=mx.int32)
        remaining_array = mx.array([shorter_tokens[-1:]], dtype=mx.int32)

        long_cache = model.make_cache()
        # Simulate chunked prefill to reproduce the real server path where
        # rotating caches can have offset > backing length before decode.
        mx.eval(model(long_array[:, :8], cache=long_cache))
        mx.eval(model(long_array[:, 8:], cache=long_cache))

        rotating_layers = [c for c in long_cache if isinstance(c, RotatingKVCache)]
        self.assertGreater(len(rotating_layers), 0)
        for sliding in rotating_layers:
            self.assertGreater(sliding.offset, sliding.keys.shape[2])
            self.assertEqual(sliding._idx, sliding.keys.shape[2])
            self.assertGreater(sliding.keys.shape[2], sliding.max_size)

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
        class DeepcopyShouldNotRunLayer:
            @property
            def nbytes(self):
                return 1

            def is_trimmable(self):
                return True

            def trim(self, n):
                return n

            def __deepcopy__(self, memo):
                raise AssertionError("deepcopy should be skipped on known-safe miss")

        lru = LRUPromptCache(max_size=10)
        model = ("skip-deepcopy", None, None)
        long_tokens = [1, 2, 3, 4]
        shorter_tokens = [1, 2]

        # Decode-style rotating state with offset beyond materialized backing.
        # This should be detected as unrecoverable before any deepcopy occurs.
        unrecoverable = self._build_real_rotating_cache()
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

        # Safe miss should remain non-destructive for the exact longer entry.
        exact_cache, exact_remaining = lru.fetch_nearest_cache(model, long_tokens)
        self.assertIsNotNone(exact_cache)
        self.assertEqual(exact_remaining, [])

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

    def test_mixed_cache_decode_rotation_safe_miss_preserves_exact_snapshot(self):
        """Regression guard: decode-rotated safe miss preserves exact snapshot."""
        lru = LRUPromptCache(max_size=10)
        model_key = ("step3p5-tiny", None, None)
        model = self._make_tiny_step3p5_model()

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

        rotating_layers = [c for c in long_cache if isinstance(c, RotatingKVCache)]
        self.assertGreater(len(rotating_layers), 0)
        for sliding in rotating_layers:
            # Real decode-time rotation leaves history unrecoverable for
            # rewinding to much shorter prefixes.
            self.assertEqual(sliding.size(), sliding.max_size)
            self.assertFalse(sliding.is_trimmable())
            self.assertGreater(sliding.offset, sliding.keys.shape[2])

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

        # Safe miss must not mutate or consume the exact longer entry.
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
