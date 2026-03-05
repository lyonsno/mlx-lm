# Tests Directory Map

This file is a quick navigation guide for running targeted test groups.

## Server API and request flow

- File: `tests/test_server.py`
- Covers:
  - HTTP/API handler behavior.
  - Draft-model and keepalive behavior.
  - `ResponseGenerator` forwarding checks.

## Prompt-cache behavior contracts (black-box)

- File: `tests/test_prompt_cache_server_behavior.py`
- Covers:
  - LRU prompt-cache behavior from external contract perspective.
  - Mixed cache long->short reuse when recoverable.
  - Safe-miss behavior and exact-entry preservation when unrecoverable.

## Prompt-cache rewind internals (white-box, focused)

- File: `tests/test_prompt_cache_server_rewind_internal.py`
- Covers:
  - `CacheList` rewind recursion/partial-trim fail-closed behavior.
  - Rotating rewind failure-path state preservation.
  - Rotating rewind materialization and zero-trim semantics.

## Shared prompt-cache test helpers

- File: `tests/prompt_cache_test_utils.py`
- Contains reusable model/cache builders and stub layers used by the two prompt-cache suites.

## Common commands

Run server + prompt-cache suites:

```bash
PYTHONPATH=. uv run --with pytest python -m pytest \
  tests/test_server.py \
  tests/test_prompt_cache_server_behavior.py \
  tests/test_prompt_cache_server_rewind_internal.py -q
```

Run prompt-cache focused suites:

```bash
PYTHONPATH=. uv run --with pytest python -m pytest \
  tests/test_prompt_cache.py \
  tests/test_prompt_cache_server_behavior.py \
  tests/test_prompt_cache_server_rewind_internal.py -q
```
