"""Gate 0 self-check for the V2 internal-ablation switches."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import tempfile

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import infer as infer_script  # noqa: E402
import test as test_script  # noqa: E402
from models.damecfusion import DAMECFusionV1  # noqa: E402
from models.mec_module import MECModule  # noqa: E402


ABLATION_OPTIONS = (
    "disable_thermal_prior",
    "disable_sat_uncertainty",
    "disable_smoke_prior",
    "fixed_equal_weights",
)


def assert_tensors_equal(actual: tuple[torch.Tensor, ...], expected: tuple[torch.Tensor, ...]) -> None:
    assert len(actual) == len(expected)
    for actual_tensor, expected_tensor in zip(actual, expected):
        assert torch.equal(actual_tensor, expected_tensor)


def check_default_behavior() -> None:
    torch.manual_seed(20260828)
    reference = MECModule(
        feature_channels=4,
        hidden_channels=4,
        prior_residual_scale=1.0,
        bounded_learned_gap=True,
        equalize_feature_magnitude=True,
    ).eval()
    explicit_defaults = MECModule(
        feature_channels=4,
        hidden_channels=4,
        prior_residual_scale=1.0,
        bounded_learned_gap=True,
        equalize_feature_magnitude=True,
        disable_thermal_prior=False,
        disable_sat_uncertainty=False,
        disable_smoke_prior=False,
        fixed_equal_weights=False,
    ).eval()
    explicit_defaults.load_state_dict(reference.state_dict())
    feat_ir = torch.randn(1, 4, 8, 8)
    feat_vis = torch.randn(1, 4, 8, 8)
    priors = [torch.rand(1, 1, 16, 16) for _ in range(3)]
    with torch.inference_mode():
        assert_tensors_equal(
            explicit_defaults(feat_ir, feat_vis, *priors),
            reference(feat_ir, feat_vis, *priors),
        )


def check_prior_routing() -> None:
    torch.manual_seed(20260828)
    feat_ir = torch.randn(1, 4, 8, 8)
    feat_vis = torch.randn(1, 4, 8, 8)
    priors = [torch.rand(1, 1, 8, 8) for _ in range(3)]
    for prior_index, option in enumerate(ABLATION_OPTIONS[:3]):
        reference = MECModule(4, hidden_channels=4, prior_residual_scale=1.0).eval()
        disabled = MECModule(
            4,
            hidden_channels=4,
            prior_residual_scale=1.0,
            **{option: True},
        ).eval()
        disabled.load_state_dict(reference.state_dict())
        zeroed_priors = priors.copy()
        zeroed_priors[prior_index] = torch.zeros_like(zeroed_priors[prior_index])
        with torch.inference_mode():
            assert_tensors_equal(
                disabled(feat_ir, feat_vis, *priors),
                reference(feat_ir, feat_vis, *zeroed_priors),
            )


def check_fixed_weights() -> None:
    model = MECModule(4, hidden_channels=4, fixed_equal_weights=True).eval()
    feat_ir = torch.randn(1, 4, 8, 8)
    feat_vis = torch.randn(1, 4, 8, 8)
    priors = [torch.rand(1, 1, 8, 8) for _ in range(3)]
    with torch.inference_mode():
        fused, w_ir, w_vis = model(feat_ir, feat_vis, *priors)
    assert torch.equal(w_ir, torch.full_like(w_ir, 0.5))
    assert torch.equal(w_vis, torch.full_like(w_vis, 0.5))
    assert torch.equal(fused, 0.5 * feat_ir + 0.5 * feat_vis)
    assert torch.equal(w_ir + w_vis, torch.ones_like(w_ir))


def checkpoint_namespace(checkpoint: Path) -> argparse.Namespace:
    return argparse.Namespace(
        checkpoint=str(checkpoint),
        feature_channels=4,
        smoke_prior_mode=None,
        prior_residual_scale=None,
        bounded_learned_gap=None,
        equalize_feature_magnitude=None,
    )


def check_checkpoint_restore() -> None:
    model = DAMECFusionV1(
        feature_channels=4,
        smoke_prior_mode="fixed_texture_luma",
        prior_residual_scale=1.0,
        bounded_learned_gap=True,
        equalize_feature_magnitude=True,
        disable_thermal_prior=True,
        disable_sat_uncertainty=True,
        disable_smoke_prior=True,
        fixed_equal_weights=True,
    )
    checkpoint_args = {
        "feature_channels": 4,
        "smoke_prior_mode": "fixed_texture_luma",
        "prior_residual_scale": 1.0,
        "bounded_learned_gap": True,
        "equalize_feature_magnitude": True,
        **{option: True for option in ABLATION_OPTIONS},
    }
    with tempfile.TemporaryDirectory() as temporary_directory:
        checkpoint = Path(temporary_directory) / "ablation.pth"
        torch.save({"model_state_dict": model.state_dict(), "args": checkpoint_args}, checkpoint)
        for loader in (test_script.load_model, infer_script.load_model):
            restored = loader(checkpoint_namespace(checkpoint), torch.device("cpu"))
            for option in ABLATION_OPTIONS:
                assert getattr(restored.mec, option) is True

        old_checkpoint = Path(temporary_directory) / "old.pth"
        old_model = DAMECFusionV1(feature_channels=4)
        torch.save({"model_state_dict": old_model.state_dict(), "args": {"feature_channels": 4}}, old_checkpoint)
        restored_old = test_script.load_model(checkpoint_namespace(old_checkpoint), torch.device("cpu"))
        for option in ABLATION_OPTIONS:
            assert getattr(restored_old.mec, option) is False


def main() -> None:
    check_default_behavior()
    check_prior_routing()
    check_fixed_weights()
    check_checkpoint_restore()
    print("V2 ablation Gate 0 self-check passed.")


if __name__ == "__main__":
    main()
