# Test Contract Audit

Purpose: catalog dev's test surface for the cache/rewind/server subsystem.
Identify which tests assert behavioral contracts (implementation-agnostic) vs.
which are coupled to dev's current internal structure and will break or become
meaningless when upstream changes (especially `2105aaf`) are reconciled in.

## Test File Inventory

| File | Tests | Scope |
|------|-------|-------|
| `test_prompt_cache.py` | 28 | Cache types (save/load/trim/merge/mask), `_BaseCache` legacy rewind, `BatchRotatingKVCache` memory/mask, `RotatingKVCache` rewind contract |
| `test_prompt_cache_server_behavior.py` | 19 | `LRUPromptCache.fetch_nearest_cache` behavior: rewind, legacy fallback, fail-closed, refcounting, mixed-cache, deepcopy avoidance, CacheList partial-fail, broken-rotating guard |
| `test_generate.py` | 65 | KV quantization, `rewind_prompt_cache()`, speculative decode rewind safety, prefill step validation |
| `test_server.py` | 13 | HTTP server behavior (mostly upstream-original) |
| `test_prefill_step_size_cli.py` | 8 | CLI argument validation |
| `prompt_cache_test_utils.py` | — | Mock layers: `RewindRecorderLayer`, `LegacyTrimLayer`, `UnknownNonTrimmableLayer`, etc. |

## Classification: Behavioral vs. Structural

### Behavioral (implementation-agnostic, should survive reconciliation)

These tests assert *what* the system does, not *how* it's wired internally.

**`test_prompt_cache.py`:**
- `TestPromptCache` (save/load/trim/generate/mask tests) — all behavioral.
  These test the cache types themselves, not the server. Fully portable.
- `TestBatchRotatingKVCacheState` — behavioral. Tests `mx.depends` memory
  safety and mask snapshot correctness. Portable.
- `test_base_cache_legacy_can_rewind_uses_offset_hint` — behavioral contract
  on `_BaseCache.can_rewind()`. Portable if cache-class API is kept.
- `test_base_cache_legacy_can_rewind_fails_closed_on_bad_offset_property` —
  same. Portable.

**`test_prompt_cache_server_behavior.py`:**
- `test_caching` — behavioral: insert cache, fetch prefix, get remaining
  tokens. Fully decoupled from internals (asserts via `fetch_nearest_cache`
  only).
- `test_lru` — behavioral: LRU eviction semantics. Portable.
- `test_lru_bytes` — behavioral: byte-based eviction and `trim_to`. Portable.
- `test_fast_trim_path_fails_closed_on_partial_trim` — behavioral: partial
  rewind → miss, original entry preserved. Uses `RewindRecorderLayer` mock.
  Portable.
- `test_unknown_layer_safe_miss_variants` — behavioral: unknown/non-trimmable
  layers → safe miss. Portable.
- `test_legacy_trimmable_layer_without_rewind_api_still_reuses` — behavioral:
  legacy `is_trimmable()`/`trim()` layer gets reused via longer-prefix path.
  Portable.
- `test_legacy_partial_trim_fails_closed_and_preserves_exact_entry` —
  behavioral: legacy layer partial trim → fail closed, exact entry still
  fetchable. Portable.
- `test_legacy_rewind_only_layer_without_trim_still_reuses` — behavioral:
  `is_trimmable()` + `rewind()` but no `trim()` → still reuses. Portable.
- `test_can_rewind_only_layer_without_rewind_path_safe_miss_skips_deepcopy` —
  behavioral: `can_rewind()` without `rewind()` → skip deepcopy. Portable.
- `test_legacy_offset_insufficient_safe_miss_skips_deepcopy` — behavioral:
  offset-bounded miss avoids deepcopy. Portable.
- `test_composite_partial_trim_safe_miss_keeps_exact_entry_available` —
  behavioral: `CacheList` with partial child → fail closed, exact entry
  preserved. Portable.
- `test_mixed_cache_longer_prefix_reuse_when_rotating_cache_is_full` —
  behavioral: real model, logprob equivalence after rewind. **Strong contract
  test.** Portable.
- `test_mixed_cache_longer_prefix_reuse_after_chunked_prefill` — behavioral:
  same as above with chunked prefill. Portable.
- `test_longer_hit_unrecoverable_rotating_miss_skips_deepcopy` — behavioral:
  unrecoverable RotatingKVCache → skip deepcopy. Portable.
- `test_mixed_cache_longer_prefix_reuse_preserves_refcounted_long_entry` —
  behavioral: refcounting semantics on longer entry after rewind extraction.
  Portable.
- `test_mixed_cache_longer_prefix_reuse_misses_after_decode_rotation` —
  behavioral: post-rotation cache can't be rewound → miss. Portable.
- `test_mixed_cache_decode_rotation_safe_miss_preserves_exact_snapshot` —
  behavioral: post-rotation miss preserves refcounted exact entry's arrays.
  Portable.

**`test_generate.py` (cache/rewind-related subset):**
- `test_rewind_prompt_cache_*` tests — behavioral: assert on the module-level
  `rewind_prompt_cache()` / `can_rewind_prompt_cache()` API in `generate.py`.
  **These functions may not exist in upstream's structure.** The behavioral
  contract is valid but the API surface may need remapping.
- `test_speculative_generate_step_*` tests — behavioral: speculative decode
  rewind safety, fail-closed behavior, snapshot semantics. These test
  `generate_step()` behavior, which is the public API. Portable, but some
  use internal mocks (`_MockRewindableCache`, etc.) that assume the current
  `rewind_prompt_cache` signature.
- `test_maybe_quantize_kv_cache_*` tests — behavioral: KV quantization
  guardrails. Portable.

### Structural (coupled to dev's internals, addressed or triaged)

**`test_prompt_cache_server_behavior.py`:** ~~`test_caching` accesses `_lru`
directly~~ — **Fixed.** Asserts via `fetch_nearest_cache` only.

**`test_prompt_cache_server_rewind_internal.py`:** ~~All 6 tests call private
methods~~ — **Removed.** All assertions lifted:
- 4 RotatingKVCache tests → `test_prompt_cache.py::TestRotatingKVCacheRewind`
- 2 server tests → `test_prompt_cache_server_behavior.py` behavioral tests
  (`test_cachelist_partial_child_failure_is_safe_miss`,
  `test_broken_rotating_cache_fails_closed_even_when_trimmable`)

**`test_generate.py` — triage complete (item 3):**

*Safety-behavioral (must survive reconciliation):*
- `test_rewind_prompt_cache_*` tests (13 tests) — assert critical rewind
  safety invariants: fail-closed, no partial mutation, snapshot-before-mutate,
  rollback on nested/custom/cyclic state, slot cleanup. These are safety
  contracts that any rewind implementation must satisfy. Structural coupling
  is limited to the import path (`rewind_prompt_cache` /
  `can_rewind_prompt_cache` from `mlx_lm.generate`). During reconciliation,
  if functions move or rename, the imports need updating but assertions stay.
- `test_speculative_generate_step_*` tests (22 tests) — test the public
  `speculative_generate_step()` API. Inline mock caches use
  `can_rewind()`/`rewind()` which is the cache-class API. Behavioral safety
  tests for speculative decode rewind failure handling.
- `test_maybe_quantize_kv_cache_*` tests (3 tests) — KV quantization
  guardrails. Fully portable.
- `test_known_fast_rewind_handles_quantized_and_nested_cachelist` — tests
  fast-path rewind with quantized and nested CacheList. Portable.

*Reconciliation action needed:*
- Import remap only. If `rewind_prompt_cache` / `can_rewind_prompt_cache` move
  from `mlx_lm.generate` to another module (or are inlined into the server),
  the test imports need updating. The mock caches' `can_rewind()`/`rewind()`
  methods match the cache-class API that dev adds to `_BaseCache` — if
  upstream doesn't adopt this API, the mocks need adjusting to whatever
  interface upstream uses. This is a trivial mechanical change, not a
  conceptual restructuring.

**`prompt_cache_test_utils.py`:** ~~`RewindRecorderLayer` and
`DeepcopyShouldNotRunLayer` only had cache-owned API~~ — **Fixed.** Both now
have dual-mode support (cache-owned + legacy server-introspection paths).

## Reconciliation Risk Assessment

### Low risk (no changes needed)
- `test_prompt_cache.py` `TestPromptCache` and `TestBatchRotatingKVCacheState`
- All cache type save/load/trim/merge/mask tests
- CLI and prefill step size tests
- HTTP server behavior tests

### Medium risk (import remap only)
- `test_generate.py` rewind tests (13) — import `rewind_prompt_cache` /
  `can_rewind_prompt_cache` from `mlx_lm.generate`. If these functions move
  during reconciliation, update imports. Assertions are implementation-agnostic.
- `test_generate.py` speculative tests (22) — inline mock caches use
  `can_rewind()`/`rewind()`. If upstream doesn't adopt the cache-class API,
  the mock interface needs adjusting. Mechanical change.

### High risk — resolved
- ~~`test_prompt_cache_server_rewind_internal.py`~~ — removed, lifted to
  behavioral level.
- ~~`test_caching` `_lru` access~~ — decoupled.
- ~~`RewindRecorderLayer` / `DeepcopyShouldNotRunLayer` single-mode~~ —
  dual-mode now.

## Action Plan

1. ~~**Decouple `test_caching`** from `_lru` attribute access~~ — **Done.**
   Asserts via `fetch_nearest_cache` only.
2. ~~**Lift `test_prompt_cache_server_rewind_internal.py` assertions**~~ —
   **Done.** 4 RotatingKVCache tests moved to
   `test_prompt_cache.py::TestRotatingKVCacheRewind`. 2 server-level tests
   lifted to behavioral level in `test_prompt_cache_server_behavior.py`.
   Internals file removed.
3. ~~**Identify which `test_generate.py` rewind tests are about speculative
   decode safety vs. API shape**~~ — **Done.** All 38 rewind-related tests
   are safety-behavioral. Structural coupling is limited to import paths and
   mock cache interface (`can_rewind`/`rewind`). Reconciliation requires
   import remap only, not conceptual restructuring. See triage above.
4. ~~**Build a dual-mode `RewindRecorderLayer`**~~ — **Done.**
   `RewindRecorderLayer` and `DeepcopyShouldNotRunLayer` now satisfy both
   the cache-owned API (`can_rewind`/`rewind`) and the legacy
   server-introspection contract (`is_trimmable`/`trim`). Both paths record
   to the same call lists and mutate `offset` identically. The server
   prefers `can_rewind` when present; upstream servers that only know the
   legacy path will use `is_trimmable`/`trim` instead.
5. **Do not modify the behavioral tests.** They are the contract. The
   reconciliation must make the implementation pass them, not the other way
   around.

## Summary

All 5 action items complete. The test surface is reconciliation-ready:
- **Low risk**: 24 cache-type tests, 4 RotatingKVCache rewind tests, 8 CLI
  tests, 13 HTTP server tests — no changes needed.
- **Medium risk (import remap)**: 35 generate.py rewind/speculative tests —
  import paths and mock interface may need updating if functions move.
- **High risk — resolved**: internals file removed, `_lru` decoupled,
  dual-mode mocks in place.
- **Behavioral test suite**: 19 `fetch_nearest_cache`-level tests encode the
  full rewind behavior contract. These are the highest-value reconciliation
  surface.
