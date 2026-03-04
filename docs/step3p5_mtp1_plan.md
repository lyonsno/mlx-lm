# Step3.5 MTP-1 Plan (Programmatic Path Only)

## Goal
Add a private-branch implementation of Step-3.5-Flash multi-token prediction
(MTP) for a single predictor layer (MTP-1) in `mlx-lm`, without CLI/server
integration.

## Scope
- In scope:
  - Preserve and load Step3.5 MTP weights during conversion/load.
  - Add Step3.5 model-side MTP modules for one predictor layer.
  - Add programmatic generation support for MTP-1 draft-and-verify flow.
  - Keep parameter/module naming quant-friendly for later FP8 work.
- Out of scope:
  - CLI flags and server request/response plumbing.
  - Tensor/pipeline sharding support for MTP modules.
  - Multi-layer MTP scheduling beyond one predictor layer.
  - FP8 implementation in this phase.

## Canonical Naming (for quant-friendly paths)
- `model.mtp.predictor_layers.0.*`
- `model.mtp.hidden_norm.*`
- `model.mtp.emb_norm.*`
- `model.mtp.linear_proj.*`

These names are intentionally stable and shallow so later quantization can
target MTP modules with simple path predicates.

## Weight Mapping Contract
- Step3.5 base decoder layers remain under `model.layers.<i>.*`.
- Extra post-base layers used for MTP are remapped from
  `model.layers.<num_hidden_layers + i>.*` to
  `model.mtp.predictor_layers.<i>.*`.
- Pre-converted MTP keys that already live under `model.mtp.*` are preserved.

## Test-First Acceptance (Phase 1)
- `ModelArgs` accepts and stores `num_nextn_predict_layers`.
- `step3p5.Model.sanitize()`:
  - does not drop `model.mtp.*` keys,
  - remaps extra decoder-layer indices into
    `model.mtp.predictor_layers.<i>.*`.
- `step3p5.Model` exposes parameters under
  `model.mtp.predictor_layers.0.*` when MTP is enabled.

## Risks
- Cache rewind correctness for draft rejection paths.
- Key mapping drift between raw StepFun checkpoints and MLX-converted weights.
- Additional memory overhead before FP8/quant is added for MTP modules.

## Follow-Up (Victory Lap)
- Add FP8/quant support for `model.mtp.*` modules once correctness is stable.
