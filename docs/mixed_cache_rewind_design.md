# Mixed Cache Rewind/Re-use Design

## Problem
Hybrid attention models (some full-attention layers with `KVCache`, some sliding
layers with `RotatingKVCache`) can lose prompt-cache trim/reuse after sliding
caches fill their fixed window.

Pre-fix behavior (before Phase 1):
- `can_trim_prompt_cache(cache)` requires every cache object to be trimmable.
- `RotatingKVCache.is_trimmable()` returns `False` once offset reaches max window.
- Longer->shorter prompt-cache reuse in `LRUPromptCache.fetch_nearest_cache()`
  then returns `None` for the whole cache list.

Effect:
- Full-attention cache state exists and is correct, but cannot be reused through
  trim/reuse paths because one rotating cache blocks the whole list.

## Scope
Phase 1 (Step-first validation, generic design):
- Restore longer->shorter cache reuse for mixed cache lists without relying on
  rotating private internals from server code.
- Keep behavior/model API stable.
- Keep implementation generic (not Step3.5-specific code paths).

Out of scope for phase 1:
- Re-defining rotating cache trim semantics beyond strict rewind guarantees.
- Replay-based fallback reconstruction.
- CLI/config switches.

## Invariants
- Correctness over speed for fallback path.
- No behavior change for homogeneous trim-trimmable cache lists.
- Exact long entry must remain usable after longer->shorter safe miss.
- Never return partially rewound mixed caches.

## Implemented Algorithm
For longer->shorter reuse in `LRUPromptCache.fetch_nearest_cache()`:
1. Identify longer candidate and compute `num_to_trim`.
2. Run a rewindability precheck across all layers.
3. If precheck fails for any layer, return safe miss (`None, tokens`) with no
   cache mutation.
4. Deep-copy candidate cache only after precheck passes.
5. Rewind copied layers via public cache API (`can_rewind`/`rewind`).
6. Return rewound cache + carry suffix if all layers succeed.
7. Fail closed to miss if any layer rewind fails.

Rotating-specific rewind checks live in cache classes (not server private-field
dispatch), including:
- offset and index bounds,
- materialized backing availability,
- strict no-mutation behavior on rewind failure.

Legacy compatibility behavior:
- Supports legacy `is_trimmable + trim` layers.
- Supports legacy `is_trimmable + rewind` layers.
- Uses offset hints where available to avoid guaranteed-miss deepcopy churn.

## Rollout Plan (Historical, Completed)
1. Added fail-first/regression tests for mixed cache longer->shorter server lookup.
2. Migrated rewind ownership into cache classes and public APIs.
3. Split behavior/integration tests from white-box rewind internals.
4. Validated on Step3.5 Flash and ensured no prompt-cache regressions.
5. Added incremental hardening based on reviewer feedback.

## Risks
- Precheck false positives/negatives in legacy compatibility paths.
- Off-by-one in carry-token boundary for longer->shorter reuse.
- Hidden mutation during fail-closed branches.

## Acceptance Criteria
- Mixed cache longer->shorter reuse returns non-`None` cache where previous code
  returned `None`.
- Returned remainder tokens preserve generation correctness and carry-token
  contract.
- Existing non-mixed cache tests remain green.
- Exact longer entries remain retrievable after safe-miss paths.
