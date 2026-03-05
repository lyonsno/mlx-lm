# PR Split Plan

This document tracks strategically chunked PR slices so each review is small,
testable, and easy to merge.

## Principles

1. Keep each PR scoped to one behavior boundary.
2. Include tests that constrain the changed boundary.
3. Land low-risk API primitives before higher-level call-site migrations.
4. Avoid mixing refactor-only and behavior changes unless tightly coupled.

## Current Workstream: Prompt Cache Rewind Hardening

### Split 1: Cache-Owned Rewind API (In Progress)

- Goal: move rewind invariants into cache classes.
- Scope:
  - Add public `can_rewind(num_to_trim)` and `rewind(num_to_trim)` API on cache types.
  - Implement strict rotating rewind checks inside `RotatingKVCache`.
  - Add/update unit tests for rewind API behavior.
- Expected PR label: `refactor/cache-rewind-api`
- Risk: medium (touches multiple cache classes).
- Merge dependency: none.

### Split 2: Server Migration to Public Rewind API (Planned)

- Goal: remove server coupling to rotating private internals.
- Scope:
  - Update `LRUPromptCache` to call `can_rewind`/`rewind` only.
  - Remove rotating-specific branching/private-field access from server.
  - Keep fail-closed semantics and non-destructive safe-miss behavior.
- Expected PR label: `refactor/server-rewind-dispatch`
- Risk: medium (affects reuse hit/miss paths).
- Merge dependency: Split 1.

### Split 3: Regression + Integration Coverage Pass (Planned)

- Goal: lock down behavior contracts across mixed cache paths.
- Scope:
  - Ensure tests cover:
    - long->short mixed reuse success,
    - decode-rotation unrecoverable safe miss,
    - exact-entry preservation and refcount behavior,
    - no-deepcopy miss path guarantees.
  - Trim brittle internals assertions where possible in favor of behavior checks.
- Expected PR label: `tests/mixed-cache-rewind-contracts`
- Risk: low.
- Merge dependency: Split 2.

## Recently Landed Related Splits

### Landed: Prefill-Step-Size Validation + Forwarding

- Commit: `0c0cbcd`
- Summary:
  - Hardened positive integer parsing and invalid input handling.
  - Added server/benchmark runtime forwarding tests for `prefill_step_size`.

### Landed: KV Bits Coverage

- Commit: `26fb608`
- Summary:
  - Added tests for KV quantization helper threshold behavior.
  - Added `generate_step` KV quantization runtime assertion.
  - Added CLI mismatch guard tests for prompt-cache quantized metadata.

## Notes

- Keep this file updated when scope changes, split names change, or a split is
  merged/cherry-picked.
