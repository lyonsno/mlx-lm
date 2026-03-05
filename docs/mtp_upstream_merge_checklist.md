# MTP Upstream Merge Checklist

## Purpose
Track merge-readiness for upstreaming MTP-related work with minimal regression
risk and strong reviewer confidence.

Update this file as work lands. Prefer linking a commit hash, test name, or doc
section for every checked item.

## Status
- Last updated: 2026-03-04
- Branch context: `step3p5-mtp-wip-kv`
- Legend:
  - `[x]` complete on branch
  - `[ ]` not complete
  - `[ ] IN PROGRESS` underway, not done

## 1) Scope and PR Strategy
- [ ] Split into reviewable PRs with clear boundaries:
  - model/loading support
  - generation/runtime support
  - CLI/server plumbing
  - docs/benchmarks
- [ ] Keep default behavior unchanged for existing users.
- [ ] Avoid coupling unrelated refactors into MTP PRs.
- [ ] Draft PR descriptions with explicit non-goals.

## 2) Correctness and Safety
- [x] MTP generation entrypoint exists and is callable (`mtp_generate_step`).
- [x] Rejection path parity tests validate all-rejected behavior vs autoregressive baseline.
- [x] Accepted-draft logprob contract aligns with verifier/main model distribution.
- [x] Rewind behavior tested for accept/reject trajectories.
- [x] Stream routing conflict guard exists for `use_mtp=True` + `draft_model`.
- [ ] Add/confirm property-style tests for longer multi-prompt parity sweeps.
- [ ] Add/confirm stress tests around cache exhaustion and long-context boundaries.

## 3) API and Routing Contracts
- [x] `stream_generate` supports explicit MTP routing control (`use_mtp`).
- [x] Non-MTP route behavior is preserved when `use_mtp` is false/absent.
- [x] Route-specific kwarg filtering is tested (no unsupported kwargs forwarded).
- [x] Prompt normalization paths covered for array/list/string inputs on MTP route.
- [ ] Decide and document final policy for `use_mtp` default semantics:
  - explicit opt-in only
  - auto-if-capable
  - tri-state (`None` auto, bool force)
- [ ] Lock final policy in API docs and tests.

## 4) CLI Integration
- [ ] Add CLI flag(s) for MTP mode selection.
- [ ] Validate conflicts and error messages in CLI argument handling.
- [ ] Ensure help text explains interaction with speculative decoding flags.
- [ ] Add CLI-focused tests.

## 5) Server Integration
- [ ] Add request field(s) for MTP routing in server API.
- [ ] Define behavior when both speculative and MTP controls are provided.
- [ ] Ensure streaming/non-streaming responses remain contract-compatible.
- [ ] Add server tests for request validation and route behavior.
- [ ] Update `mlx_lm/SERVER.md`.

## 6) Model Capability and Compatibility
- [ ] Clearly define "MTP-capable" detection contract.
- [ ] Ensure capability errors are user-facing and actionable.
- [ ] Verify behavior on models with MTP weights stripped/absent.
- [ ] Confirm explicit `mtp_cache` path works when cache factory is absent.

## 7) Performance and Benchmark Evidence
- [ ] Benchmark baseline vs MTP-enabled on at least one Step 3.5 Flash setup.
- [ ] Include throughput and memory metrics:
  - prefill tokens/sec
  - decode tokens/sec
  - peak memory
- [ ] Include quality sanity checks for representative prompts.
- [ ] Publish reproducible scripts/commands and environment details.

## 8) Documentation and User Guidance
- [ ] Update `README.md` Python examples for MTP usage.
- [ ] Document default behavior and override controls.
- [ ] Document known limitations and model support matrix.
- [ ] Add migration notes if behavior changes from existing defaults.

## 9) Upstream Readiness Signals
- [ ] Full test suite green on branch.
- [ ] Focused MTP tests green and stable.
- [ ] No unrelated failures introduced by MTP path.
- [ ] PR includes concise risk analysis and rollback plan.

## Evidence Notes (Current)
1. Focused MTP test module was reported green locally:
   - `python -m unittest tests.test_generate_mtp -v`
2. Route and contract tests were expanded significantly under:
   - `tests/test_generate_mtp.py`
3. MTP runtime and routing changes are concentrated in:
   - `mlx_lm/generate.py`

## Update Log
- 2026-03-04: Initialized checklist and marked known branch-complete items for
  generation correctness/routing tests.

