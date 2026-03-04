# Copyright © 2026 Apple Inc.

import unittest

import mlx.core as mx
from mlx.utils import tree_flatten

from mlx_lm.models import step3p5


def _tiny_step3p5_args(**overrides):
    args = {
        "model_type": "step3p5",
        "hidden_size": 64,
        "num_hidden_layers": 2,
        "vocab_size": 256,
        "num_attention_heads": 4,
        "num_attention_groups": 2,
        "head_dim": 16,
        "intermediate_size": 128,
        "moe_num_experts": 4,
        "moe_top_k": 2,
        "moe_intermediate_size": 32,
        "share_expert_dim": 32,
        "moe_layers_enum": "1",
    }
    args.update(overrides)
    return args


class TestStep3p5MTP(unittest.TestCase):
    def test_model_args_parses_num_nextn_predict_layers(self):
        args = step3p5.ModelArgs.from_dict(
            _tiny_step3p5_args(num_nextn_predict_layers=1)
        )

        self.assertTrue(
            hasattr(args, "num_nextn_predict_layers"),
            "Step3p5 ModelArgs should retain num_nextn_predict_layers from config.",
        )
        self.assertEqual(args.num_nextn_predict_layers, 1)

    def test_sanitize_preserves_and_remaps_mtp_weights(self):
        model = step3p5.Model(step3p5.ModelArgs.from_dict(_tiny_step3p5_args()))
        weights = {
            # This key models a previously-converted MTP tensor.
            "model.mtp.hidden_norm.weight": mx.ones((64,), dtype=mx.float32),
            # This key models extra decoder layers from raw checkpoints.
            "model.layers.2.self_attn.q_proj.weight": mx.zeros(
                (64, 64), dtype=mx.float32
            ),
        }

        converted = model.sanitize(weights)

        self.assertIn(
            "model.mtp.hidden_norm.weight",
            converted,
            "sanitize() should preserve already-canonical model.mtp.* keys.",
        )
        self.assertIn(
            "model.mtp.predictor_layers.0.self_attn.q_proj.weight",
            converted,
            "sanitize() should remap extra decoder layers into canonical MTP paths.",
        )
        self.assertNotIn("model.layers.2.self_attn.q_proj.weight", converted)

    def test_mtp_parameter_paths_exist_when_enabled(self):
        args = step3p5.ModelArgs.from_dict(
            _tiny_step3p5_args(num_nextn_predict_layers=1)
        )
        model = step3p5.Model(args)
        param_keys = {k for k, _ in tree_flatten(model.parameters())}

        self.assertTrue(
            any(k.startswith("model.mtp.predictor_layers.0.") for k in param_keys),
            "Model should expose canonical MTP parameter prefixes for quant-friendly targeting.",
        )


if __name__ == "__main__":
    unittest.main()
