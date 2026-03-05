# Mixed Cache Rewind/Re-use Progress

## Goal
Enable robust and safe prompt-cache reuse for mixed cache layouts
(`KVCache + RotatingKVCache`), with fail-closed semantics when rewind is not
provably valid.

## Status (2026-03-05)
Phase 1 implementation is landed and covered by focused tests. The current
workstream has moved from behavior implementation to hardening and reviewer
feedback follow-ups.

## Landed Commits (Most Relevant)
- `ca11989`: mixed-cache rewind safety + core test coverage.
- `a361270`: cache-owned rewind API (`can_rewind`/`rewind`) and server migration.
- `35235be`: restore legacy rewind compatibility path.
- `6b935dc`: split monolithic server prompt-cache tests into focused suites.
- `4517708`: fail-fast legacy miss precheck to skip pointless deepcopy.
- `eb49b74`: restore rewind-only legacy compatibility and offset fail-closed
  behavior after review feedback.

## Checklist
- [x] Confirm root cause for mixed longer->shorter misses in server cache lookup.
- [x] Implement mixed rewind with strict fail-closed behavior.
- [x] Keep exact-hit and shorter-hit behavior unchanged.
- [x] Preserve exact longer entry on safe-miss paths.
- [x] Keep rotating rewind safety checks strict for unrecoverable history.
- [x] Add behavior-level tests for mixed reuse success and decode-rotation miss.
- [x] Add white-box rewind tests for rotating/cachelist edge cases.
- [x] Split tests into behavior vs internal suites with shared helpers.
- [x] Add legacy compatibility guards (legacy trim-only and rewind-only layers).
- [x] Add fail-fast precheck to avoid guaranteed-miss deepcopy churn.
- [ ] Add measured perf numbers for legacy/custom-cache miss-heavy scenarios
  (optional, not expected to move in-tree Step3.5 materially).

## Known Tradeoffs / Boundaries
- Reuse remains fail-closed when rotating history is unrecoverable (for example
  decode-time in-place rotation where `offset` exceeds recoverable backing).
- The precheck perf hardening mainly benefits legacy/custom cache objects; most
  in-tree architectures already use explicit cache `can_rewind` implementations.
