# Copyright © 2024 Apple Inc.

import http
import io
import json
import threading
import types
import unittest
from queue import Queue
from types import SimpleNamespace
from unittest.mock import Mock, patch

import mlx.core as mx
import requests

from mlx_lm.models.cache import KVCache, RotatingKVCache
from mlx_lm.server import (
    APIHandler,
    CompletionRequest,
    GenerationArguments,
    LogitsProcessorArguments,
    LRUPromptCache,
    ModelDescription,
    Response,
    ResponseGenerator,
    SamplingArguments,
    _process_control_tokens,
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
                "allowed_origins": ["*"],
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

    def load_default(self):
        return self.load("default_model", None, "default_model")


class MockCache:
    def __init__(self, value, is_trimmable: bool = True):
        self.value = value
        self._is_trimmable = is_trimmable

    @property
    def nbytes(self):
        return len(self.value)

    def __eq__(self, other):
        return other.value == self.value

    def is_trimmable(self):
        return self._is_trimmable

    def trim(self, n):
        assert self._is_trimmable
        return n


class TestProcessControlTokens(unittest.TestCase):
    @staticmethod
    def _r(text, state, match=None):
        return Response(text, 0, state, match, 0.0, None, ())

    def test_single_tool_call_passes_body_with_open_and_close_crossings(self):
        r = self._r
        stream = [
            r("hi ", "normal"),
            r("<tool_call>", "tool", match=(0,)),
            r("body", "tool"),
            r("</tool_call>", "normal", match=(1,)),
            r(" bye", "normal"),
        ]
        ctx = types.SimpleNamespace(
            sequences={(0,): "<tool_call>", (1,): "</tool_call>"}
        )
        out = list(_process_control_tokens(ctx, iter(stream)))

        self.assertEqual("".join(t.text for t in out), "hi body bye")
        states = [t.state for t in out]
        self.assertEqual(sum(1 for a, b in zip(states, states[1:]) if a != b), 2)

    def test_back_to_back_tool_calls_emit_state_crossings(self):
        r = self._r
        stream = [
            r("<tool_call>", "tool", match=(0,)),
            r("call1_body", "tool"),
            r("</tool_call>", "normal", match=(1,)),
            r("<tool_call>", "tool", match=(0,)),
            r("call2_body", "tool"),
            r("</tool_call>", "normal", match=(1,)),
        ]
        ctx = types.SimpleNamespace(
            sequences={(0,): "<tool_call>", (1,): "</tool_call>"}
        )
        out = list(_process_control_tokens(ctx, iter(stream)))

        self.assertEqual("".join(t.text for t in out), "call1_bodycall2_body")
        states = [t.state for t in out]
        crossings = sum(
            1 for a, b in zip(states, states[1:]) if a == "tool" and b == "normal"
        )
        self.assertEqual(crossings, 2)

    def test_multi_token_match_preserves_order(self):
        r = self._r
        match = (10, 11, 12)
        stream = [
            r("body", "tool"),
            r("</", "tool"),
            r("tool", "tool"),
            r("_call>", "normal", match=match),
            r(" ok", "normal"),
        ]
        ctx = types.SimpleNamespace(sequences={match: "</tool_call>"})
        out = list(_process_control_tokens(ctx, iter(stream)))

        self.assertEqual([t.text for t in out], ["body", "", "", "", " ok"])
        self.assertEqual(
            [t.state for t in out],
            ["tool", "tool", "tool", "normal", "normal"],
        )


class RewindOnlyLayer:
    def __init__(self, offset):
        self.offset = offset
        self.can_rewind_calls = []
        self.rewind_calls = []

    @property
    def nbytes(self):
        return 1

    def is_trimmable(self):
        return False

    def can_rewind(self, n):
        self.can_rewind_calls.append(n)
        return n <= self.offset

    def rewind(self, n):
        self.rewind_calls.append(n)
        if n > self.offset:
            return False
        self.offset -= n
        return True

    def __deepcopy__(self, memo):
        clone = type(self)(self.offset)
        clone.can_rewind_calls = list(self.can_rewind_calls)
        clone.rewind_calls = list(self.rewind_calls)
        return clone


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

    def test_make_state_machine_empty_tool_call_end(self):
        class FakeTokenizer:
            has_thinking = False
            has_tool_calling = True
            tool_call_start = "[TOOL_CALLS]"
            tool_call_end = ""
            tool_call_start_tokens = (100,)
            tool_call_end_tokens = ()
            eos_token_ids = [2]

            def convert_ids_to_tokens(self, t):
                return f"<eos{t}>"

        sm, _ = self.response_generator._make_state_machine(
            ("fake-empty-end", None, None),
            FakeTokenizer(),
            stop_words=[],
        )
        state = sm.make_state()
        state, _, s = sm.match(state, 100)
        self.assertEqual(s, "tool")
        for tok in [42, 43, 44]:
            state, _, s = sm.match(state, tok)
            self.assertEqual(s, "tool")
        state, _, s = sm.match(state, 2)
        self.assertIsNone(s)

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

        # Fetching a cache that is strictly a prefix doesn't remove it from the
        # lru cache
        tokens = tokens + [20] * 5
        c, t = cache.fetch_nearest_cache(model, tokens)
        k, v = c[0].state
        self.assertTrue((k == v).all().item())
        self.assertTrue((k.flatten() == mx.arange(24)).all().item())
        self.assertEqual(t, [20] * 5)
        self.assertEqual(len(cache), 1)

        # Inserting a trimmable cache with shared prefix removes the prefixes
        tokens = tokens + [30] * 3
        c[0].update_and_fetch(*get_kv(8))
        cache.insert_cache(model, tokens, c)
        self.assertEqual(len(cache), 1)

        # Fetching a cache with a shared prefix doesn't remove it either
        tokens = tokens[:26] + [40] * 8
        c, t = cache.fetch_nearest_cache(model, tokens)
        k, v = c[0].state
        self.assertTrue((k == v).all().item())
        self.assertTrue(
            (k.flatten() == mx.concatenate([mx.arange(24), mx.arange(2)])).all().item()
        )
        self.assertEqual(t, [40] * 8)
        self.assertEqual(len(cache), 1)

        # Inserting a diverged cache actually creates another entry
        c[0].update_and_fetch(*get_kv(8))
        cache.insert_cache(model, tokens, c)
        self.assertEqual(len(cache), 2)

    def test_lru(self):
        cache = LRUPromptCache(max_size=2)
        model = ("test", None, None)
        cache.insert_cache(model, [1, 2], [MockCache("test1")])
        cache.insert_cache(model, [2, 3], [MockCache("test2")])

        c, t = cache.fetch_nearest_cache(model, [1, 2])
        self.assertEqual(c, [MockCache("test1")])
        self.assertEqual(t, [])
        c, t = cache.fetch_nearest_cache(model, [1])
        self.assertEqual(c, [MockCache("test1")])
        self.assertEqual(t, [1])
        c, t = cache.fetch_nearest_cache(model, [1, 3, 4])
        self.assertEqual(c, [MockCache("test1")])
        self.assertEqual(t, [3, 4])
        c, t = cache.fetch_nearest_cache(model, [2, 3, 4])
        self.assertEqual(c, [MockCache("test2")])
        self.assertEqual(t, [4])
        c, t = cache.fetch_nearest_cache(model, [2, 4, 5])
        self.assertEqual(c, [MockCache("test2")])
        self.assertEqual(t, [4, 5])

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

        cache.insert_cache(model, [4, 5], [MockCache("test4")], cache_type="user")
        c, t = cache.fetch_nearest_cache(model, [2, 3])
        self.assertEqual(c, None)
        self.assertEqual(t, [2, 3])
        c, t = cache.fetch_nearest_cache(model, [3, 4])
        self.assertEqual(c, [MockCache("test3")])
        self.assertEqual(t, [])
        c, t = cache.fetch_nearest_cache(model, [4, 5])
        self.assertEqual(c, [MockCache("test4")])
        self.assertEqual(t, [])

        cache.insert_cache(model, [5, 6], [MockCache("test5")])
        cache.insert_cache(model, [6, 7], [MockCache("test6")])
        c, t = cache.fetch_nearest_cache(model, [5, 6])
        self.assertEqual(c, None)
        self.assertEqual(t, [5, 6])
        c, t = cache.fetch_nearest_cache(model, [6, 7])
        self.assertEqual(c, [MockCache("test6")])
        self.assertEqual(t, [])
        c, t = cache.fetch_nearest_cache(model, [4, 5])
        self.assertEqual(c, [MockCache("test4")])
        self.assertEqual(t, [])

    def test_insert_trimmable_cache_removes_immediate_prefix(self):
        cache = LRUPromptCache(max_size=10)
        model = ("test", None, None)

        cache.insert_cache(model, [1, 2], [MockCache("ab")])
        self.assertEqual(len(cache), 1)
        self.assertEqual(cache.nbytes, 2)

        cache.insert_cache(model, [1, 2, 3], [MockCache("abc")])
        self.assertEqual(len(cache), 1)
        self.assertEqual(cache.nbytes, 3)

    def test_insert_empty_tokens_does_not_self_destruct(self):
        cache = LRUPromptCache(max_size=10)
        model = ("test", None, None)

        cache.insert_cache(model, [], [MockCache("root")])
        self.assertEqual(len(cache), 1)
        self.assertEqual(cache.nbytes, 4)

        c, t = cache.fetch_nearest_cache(model, [])
        self.assertIsNotNone(c)
        self.assertEqual(t, [])

    def test_fetch_empty_tokens_after_root_eviction(self):
        cache = LRUPromptCache(max_size=10)
        model = ("test", None, None)

        cache.insert_cache(model, [], [MockCache("root")])
        cache.insert_cache(model, [1], [MockCache("a")])

        c, t = cache.fetch_nearest_cache(model, [])
        self.assertIsNone(c)
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

    def test_mixed_cache_longer_hit_reuses_rewindable_non_trimmable_layer(self):
        cache = LRUPromptCache(max_size=10)

        def get_kv(n):
            keys = mx.arange(n).reshape(1, 1, n, 1)
            return keys, keys

        model = ("mixed-rewindable", None, None)
        long_tokens = [1, 2, 3, 4]
        shorter_tokens = [1, 2]

        kv_cache = KVCache()
        kv_cache.update_and_fetch(*get_kv(len(long_tokens)))
        rewind_only = RewindOnlyLayer(offset=len(long_tokens))
        cache.insert_cache(model, long_tokens, [kv_cache, rewind_only])

        reused_cache, remaining = cache.fetch_nearest_cache(model, shorter_tokens)

        self.assertIsNotNone(reused_cache)
        self.assertEqual(remaining, shorter_tokens[-1:])
        self.assertEqual(reused_cache[0].offset, 1)
        self.assertEqual(reused_cache[1].offset, 1)

        exact_cache, exact_remaining = cache.fetch_nearest_cache(model, long_tokens)
        self.assertIsNotNone(exact_cache)
        self.assertEqual(exact_remaining, [])
        self.assertEqual(exact_cache[0].offset, len(long_tokens))
        self.assertEqual(exact_cache[1].offset, len(long_tokens))

    def test_rotating_cache_longer_hit_reuses_chunked_prefill_boundary(self):
        cache = LRUPromptCache(max_size=10)
        model = ("rotating-chunked", None, None)
        long_tokens = list(range(12))
        shorter_tokens = long_tokens[:10]

        kv = mx.arange(12, dtype=mx.float32).reshape(1, 1, 12, 1)
        long_cache = RotatingKVCache(max_size=4)
        long_cache.update_and_fetch(kv[..., :8, :], kv[..., :8, :])
        long_cache.update_and_fetch(kv[..., 8:, :], kv[..., 8:, :])
        cache.insert_cache(model, long_tokens, [long_cache])

        reused_cache, remaining = cache.fetch_nearest_cache(model, shorter_tokens)
        self.assertIsNotNone(reused_cache)
        self.assertEqual(remaining, shorter_tokens[-1:])

        baseline = RotatingKVCache(max_size=4)
        baseline.update_and_fetch(kv[..., :8, :], kv[..., :8, :])
        baseline.update_and_fetch(kv[..., 8:9, :], kv[..., 8:9, :])
        self.assertEqual(reused_cache[0].offset, baseline.offset)
        self.assertTrue(
            mx.array_equal(
                reused_cache[0]._temporal_order(reused_cache[0].keys),
                baseline._temporal_order(baseline.keys),
            )
        )
        self.assertTrue(
            mx.array_equal(
                reused_cache[0]._temporal_order(reused_cache[0].values),
                baseline._temporal_order(baseline.values),
            )
        )

        exact_cache, exact_remaining = cache.fetch_nearest_cache(model, long_tokens)
        self.assertIsNotNone(exact_cache)
        self.assertEqual(exact_remaining, [])
        self.assertEqual(exact_cache[0].offset, len(long_tokens))

    def test_rotating_cache_longer_hit_misses_when_prefix_fell_out_of_window(self):
        cache = LRUPromptCache(max_size=10)
        model = ("rotating-rolling", None, None)
        long_tokens = list(range(6))
        shorter_tokens = long_tokens[:5]

        kv = mx.arange(6, dtype=mx.float32).reshape(1, 1, 6, 1)
        long_cache = RotatingKVCache(max_size=4)
        for i in range(6):
            tok = kv[..., i : i + 1, :]
            long_cache.update_and_fetch(tok, tok)
        cache.insert_cache(model, long_tokens, [long_cache])

        reused_cache, remaining = cache.fetch_nearest_cache(model, shorter_tokens)
        self.assertIsNone(reused_cache)
        self.assertEqual(remaining, shorter_tokens)


class TestPromptBoundaryCapture(unittest.TestCase):
    def setUp(self):
        self.model_provider = DummyModelProvider()
        self.prompt_cache = LRUPromptCache()
        self.response_generator = ResponseGenerator(
            self.model_provider, self.prompt_cache
        )

    def tearDown(self):
        self.response_generator.stop_and_join()

    def _make_args(self):
        return GenerationArguments(
            model=ModelDescription(
                model="default_model",
                draft="default_model",
                adapter=None,
            ),
            sampling=SamplingArguments(
                temperature=0.0,
                top_p=1.0,
                top_k=0,
                min_p=0.0,
                xtc_probability=0.0,
                xtc_threshold=0.0,
            ),
            logits=LogitsProcessorArguments(
                logit_bias=None,
                repetition_penalty=1.0,
                repetition_context_size=20,
            ),
            stop_words=[],
            max_tokens=1,
            num_draft_tokens=1,
            logprobs=False,
            top_logprobs=0,
            seed=None,
            chat_template_kwargs=None,
        )

    def test_serve_single_inserts_prompt_boundary_before_completion_cache(self):
        prompt = [1, 2, 3]
        boundary_cache = ["boundary-cache"]
        live_cache = ["live-cache"]
        rqueue = Queue()
        args = self._make_args()
        request = CompletionRequest(
            request_type="text",
            prompt="hello",
            messages=[],
            tools=None,
            role_mapping=None,
        )

        self.response_generator._tokenize = lambda tokenizer, request, args: prompt
        self.prompt_cache.fetch_nearest_cache = Mock(return_value=(live_cache, prompt))
        self.prompt_cache.insert_cache = Mock()

        def fake_stream_generate(*args, **kwargs):
            kwargs["prompt_cache_capture_callback"](boundary_cache)
            yield SimpleNamespace(
                text="x",
                token=99,
                logprobs=mx.zeros((4,), dtype=mx.float32),
                finish_reason="stop",
            )

        with patch("mlx_lm.server.stream_generate", fake_stream_generate):
            self.response_generator._serve_single((rqueue, request, args))

        insert_calls = self.prompt_cache.insert_cache.call_args_list
        self.assertEqual(len(insert_calls), 2)
        self.assertEqual(insert_calls[0].args[1], prompt)
        self.assertIs(insert_calls[0].args[2], boundary_cache)
        self.assertEqual(insert_calls[1].args[1], prompt + [99])
        self.assertIs(insert_calls[1].args[2], live_cache)

    def test_batch_path_inserts_prompt_boundary_before_completion_cache(self):
        prompt = [1, 2, 3]
        boundary_cache = ["boundary-cache"]
        live_cache = ["live-cache"]
        args = self._make_args()
        request = CompletionRequest(
            request_type="text",
            prompt="hello",
            messages=[],
            tools=None,
            role_mapping=None,
        )

        self.response_generator._tokenize = lambda tokenizer, request, args: prompt
        self.prompt_cache.fetch_nearest_cache = Mock(return_value=(live_cache, prompt))
        self.prompt_cache.insert_cache = Mock()

        class FakeBatchGenerator:
            def __init__(self, *args, prompt_cache_capture_callback=None, **kwargs):
                self.prompt_cache_capture_callback = prompt_cache_capture_callback
                self.prompt_cache_nbytes = 0
                self._done = False

            def insert(
                self,
                prompts,
                max_tokens=None,
                caches=None,
                samplers=None,
                logits_processors=None,
            ):
                return [7]

            def next(self):
                if self._done:
                    return []
                self._done = True
                self.prompt_cache_capture_callback([(7, boundary_cache)])
                return [
                    SimpleNamespace(
                        uid=7,
                        token=99,
                        logprobs=mx.zeros((4,), dtype=mx.float32),
                        finish_reason="length",
                        prompt_cache=live_cache,
                    )
                ]

            def close(self):
                return None

            def remove(self, uids, return_prompt_caches=False):
                return {}

        with patch("mlx_lm.server.BatchGenerator", FakeBatchGenerator):
            ctx, responses = self.response_generator.generate(request, args)
            list(responses)

        insert_calls = self.prompt_cache.insert_cache.call_args_list
        self.assertEqual(len(insert_calls), 2)
        self.assertEqual(insert_calls[0].args[1], prompt)
        self.assertIs(insert_calls[0].args[2], boundary_cache)
        self.assertEqual(insert_calls[1].args[1], prompt + [99])
        self.assertIs(insert_calls[1].args[2], live_cache)


if __name__ == "__main__":
    unittest.main()
