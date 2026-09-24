"""MEC module for DA-MECFusion V1.

MEC means modal effective contribution. It predicts spatial IR/VIS weights from
two feature maps and three rule-based priors, then fuses the two feature maps.
"""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


class MECModule(nn.Module):
    """Modal effective contribution module with spatial softmax weights."""

    def __init__(
        self,
        feature_channels: int,
        hidden_channels: int = 64,
        use_bn: bool = False,
        prior_residual_scale: float = 0.0,
        bounded_learned_gap: bool = False,
        equalize_feature_magnitude: bool = False,
        disable_thermal_prior: bool = False,
        disable_sat_uncertainty: bool = False,
        disable_smoke_prior: bool = False,
        fixed_equal_weights: bool = False,
    ) -> None:
        super().__init__()
        if feature_channels <= 0:
            raise ValueError("feature_channels must be positive")
        if hidden_channels <= 0:
            raise ValueError("hidden_channels must be positive")
        if not math.isfinite(prior_residual_scale) or prior_residual_scale < 0:
            raise ValueError("prior_residual_scale must be finite and non-negative")

        self.feature_channels = feature_channels
        self.hidden_channels = hidden_channels
        self.use_bn = use_bn
        self.prior_residual_scale = float(prior_residual_scale)
        self.bounded_learned_gap = bool(bounded_learned_gap)
        self.equalize_feature_magnitude = bool(equalize_feature_magnitude)
        self.disable_thermal_prior = bool(disable_thermal_prior)
        self.disable_sat_uncertainty = bool(disable_sat_uncertainty)
        self.disable_smoke_prior = bool(disable_smoke_prior)
        self.fixed_equal_weights = bool(fixed_equal_weights)

        in_channels = 2 * feature_channels + 3
        self.weight_predictor = nn.Sequential(
            *self._conv_block(in_channels, hidden_channels, kernel_size=3, use_bn=use_bn),
            *self._conv_block(hidden_channels, hidden_channels, kernel_size=3, use_bn=use_bn),
        )
        self.logit_conv = nn.Conv2d(hidden_channels, 2, kernel_size=1)
        nn.init.zeros_(self.logit_conv.weight)
        nn.init.zeros_(self.logit_conv.bias)

    @staticmethod
    def _conv_block(
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        use_bn: bool,
    ) -> list[nn.Module]:
        padding = kernel_size // 2
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        ]
        if use_bn:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
        return layers

    @staticmethod
    def _resize_prior(prior: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        if prior.dim() != 4 or prior.size(1) != 1:
            raise ValueError(f"Expected prior with shape [B,1,H,W], got {tuple(prior.shape)}")
        if prior.shape[-2:] == size:
            return prior
        return F.interpolate(prior, size=size, mode="bilinear", align_corners=False)

    @staticmethod
    def _bound_learned_logits(logits: torch.Tensor) -> torch.Tensor:
        if logits.dim() != 4 or logits.size(1) != 2:
            raise ValueError(f"Expected logits with shape [B,2,H,W], got {tuple(logits.shape)}")
        gap = logits[:, 0:1] - logits[:, 1:2]
        bounded_gap = 2.0 * torch.tanh(gap / 2.0)
        return torch.cat((0.5 * bounded_gap, -0.5 * bounded_gap), dim=1)

    def fusion_features(
        self,
        feat_ir: torch.Tensor,
        feat_vis: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the feature pair actually used by the final weighted fusion."""
        if not self.equalize_feature_magnitude:
            return feat_ir, feat_vis
        ir_magnitude = feat_ir.abs().mean(dim=1, keepdim=True)
        vis_magnitude = feat_vis.abs().mean(dim=1, keepdim=True)
        shared_magnitude = 0.5 * (ir_magnitude + vis_magnitude)
        epsilon = 1e-6
        return (
            feat_ir * shared_magnitude / ir_magnitude.clamp_min(epsilon),
            feat_vis * shared_magnitude / vis_magnitude.clamp_min(epsilon),
        )

    def route_priors(
        self,
        thermal_prior: torch.Tensor,
        sat_uncertainty: torch.Tensor,
        smoke_prior: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the prior tensors that are allowed to enter MEC."""
        return (
            torch.zeros_like(thermal_prior) if self.disable_thermal_prior else thermal_prior,
            torch.zeros_like(sat_uncertainty) if self.disable_sat_uncertainty else sat_uncertainty,
            torch.zeros_like(smoke_prior) if self.disable_smoke_prior else smoke_prior,
        )

    def forward(
        self,
        feat_ir: torch.Tensor,
        feat_vis: torch.Tensor,
        thermal_prior: torch.Tensor,
        sat_uncertainty: torch.Tensor,
        smoke_prior: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if feat_ir.dim() != 4:
            raise ValueError(f"Expected feat_ir with shape [B,C,Hf,Wf], got {tuple(feat_ir.shape)}")
        if feat_vis.dim() != 4:
            raise ValueError(f"Expected feat_vis with shape [B,C,Hf,Wf], got {tuple(feat_vis.shape)}")
        if feat_ir.shape != feat_vis.shape:
            raise ValueError(
                f"feat_ir and feat_vis must have the same shape, got "
                f"{tuple(feat_ir.shape)} and {tuple(feat_vis.shape)}"
            )
        if feat_ir.size(1) != self.feature_channels:
            raise ValueError(
                f"Expected feature channel count {self.feature_channels}, got {feat_ir.size(1)}"
            )

        feature_size = feat_ir.shape[-2:]
        thermal_prior = self._resize_prior(thermal_prior, feature_size)
        sat_uncertainty = self._resize_prior(sat_uncertainty, feature_size)
        smoke_prior = self._resize_prior(smoke_prior, feature_size)
        thermal_prior, sat_uncertainty, smoke_prior = self.route_priors(
            thermal_prior,
            sat_uncertainty,
            smoke_prior,
        )

        z = torch.cat(
            [feat_ir, feat_vis, thermal_prior, sat_uncertainty, smoke_prior],
            dim=1,
        )
        logits = self.logit_conv(self.weight_predictor(z))
        if self.bounded_learned_gap:
            logits = self._bound_learned_logits(logits)
        if self.prior_residual_scale > 0:
            centered_priors = [
                prior - prior.mean(dim=(-2, -1), keepdim=True)
                for prior in (thermal_prior, sat_uncertainty, smoke_prior)
            ]
            prior_delta = torch.stack(centered_priors, dim=0).mean(dim=0)
            residual = self.prior_residual_scale * prior_delta
            logits = logits + torch.cat((residual, -residual), dim=1)
        if self.fixed_equal_weights:
            w_ir = torch.full_like(logits[:, 0:1], 0.5)
            w_vis = torch.full_like(logits[:, 1:2], 0.5)
        else:
            weights = torch.softmax(logits, dim=1)
            w_ir = weights[:, 0:1]
            w_vis = weights[:, 1:2]
        fusion_feat_ir, fusion_feat_vis = self.fusion_features(feat_ir, feat_vis)
        fused_feat = w_ir * fusion_feat_ir + w_vis * fusion_feat_vis
        return fused_feat, w_ir, w_vis


def _print_tensor_stats(name: str, x: torch.Tensor) -> None:
    print(
        f"{name}: shape={tuple(x.shape)}, "
        f"min={x.min().item():.6f}, max={x.max().item():.6f}"
    )


if __name__ == "__main__":
    feat_ir = torch.randn(2, 64, 128, 128, requires_grad=True)
    feat_vis = torch.randn(2, 64, 128, 128, requires_grad=True)
    thermal = torch.rand(2, 1, 256, 256)
    sat = torch.rand(2, 1, 256, 256)
    smoke = torch.rand(2, 1, 256, 256)

    model = MECModule(feature_channels=64, hidden_channels=64)
    fused_feat, w_ir, w_vis = model(feat_ir, feat_vis, thermal, sat, smoke)

    _print_tensor_stats("fused_feat", fused_feat)
    _print_tensor_stats("w_ir", w_ir)
    _print_tensor_stats("w_vis", w_vis)
    print(f"w_ir mean: {w_ir.mean().item():.6f}")
    print(f"w_vis mean: {w_vis.mean().item():.6f}")

    assert fused_feat.shape == (2, 64, 128, 128), fused_feat.shape
    assert w_ir.shape == (2, 1, 128, 128), w_ir.shape
    assert w_vis.shape == (2, 1, 128, 128), w_vis.shape
    assert torch.max(torch.abs(w_ir + w_vis - 1.0)).item() < 1e-5
    assert abs(w_ir.mean().item() - 0.5) < 1e-5
    assert abs(w_vis.mean().item() - 0.5) < 1e-5
    assert torch.allclose(fused_feat, 0.5 * feat_ir + 0.5 * feat_vis)

    for prior_index in range(3):
        guided_model = MECModule(
            feature_channels=64,
            hidden_channels=64,
            prior_residual_scale=1.0,
        )
        zero_features = torch.zeros(1, 64, 4, 4)
        priors = [torch.zeros(1, 1, 4, 4) for _ in range(3)]
        priors[prior_index][0, 0, 1, 1] = 1.0
        _, guided_w_ir, guided_w_vis = guided_model(
            zero_features,
            zero_features,
            *priors,
        )
        assert guided_w_ir[0, 0, 1, 1] > guided_w_ir[0, 0, 0, 0], prior_index
        assert guided_w_ir[0, 0, 1, 1] > 0.5, prior_index
        assert torch.max(torch.abs(guided_w_ir + guided_w_vis - 1.0)).item() < 1e-5

    bounded_model = MECModule(feature_channels=64, bounded_learned_gap=True)
    with torch.no_grad():
        bounded_model.logit_conv.bias.copy_(torch.tensor([10.0, -10.0]))
    _, bounded_w_ir, bounded_w_vis = bounded_model(
        torch.zeros(1, 64, 4, 4),
        torch.zeros(1, 64, 4, 4),
        *[torch.zeros(1, 1, 4, 4) for _ in range(3)],
    )
    expected_upper = torch.sigmoid(torch.tensor(2.0)).item()
    assert abs(bounded_w_ir.mean().item() - expected_upper) < 1e-5
    assert torch.max(torch.abs(bounded_w_ir + bounded_w_vis - 1.0)).item() < 1e-5

    equalized_model = MECModule(feature_channels=2, equalize_feature_magnitude=True)
    equalized_ir, equalized_vis = equalized_model.fusion_features(
        torch.full((1, 2, 2, 2), 2.0),
        torch.full((1, 2, 2, 2), 0.5),
    )
    expected_shared_magnitude = torch.full((1, 1, 2, 2), 1.25)
    assert torch.allclose(equalized_ir.abs().mean(dim=1, keepdim=True), expected_shared_magnitude)
    assert torch.allclose(equalized_vis.abs().mean(dim=1, keepdim=True), expected_shared_magnitude)

    for prior_index, option in enumerate(
        ("disable_thermal_prior", "disable_sat_uncertainty", "disable_smoke_prior")
    ):
        reference = MECModule(feature_channels=2, prior_residual_scale=1.0)
        disabled = MECModule(feature_channels=2, prior_residual_scale=1.0, **{option: True})
        disabled.load_state_dict(reference.state_dict())
        small_features = torch.randn(1, 2, 3, 3)
        small_priors = [torch.rand(1, 1, 3, 3) for _ in range(3)]
        zeroed_priors = small_priors.copy()
        zeroed_priors[prior_index] = torch.zeros_like(zeroed_priors[prior_index])
        disabled_outputs = disabled(small_features, small_features, *small_priors)
        reference_outputs = reference(small_features, small_features, *zeroed_priors)
        assert all(
            torch.equal(actual, expected)
            for actual, expected in zip(disabled_outputs, reference_outputs)
        ), option

    fixed_model = MECModule(feature_channels=2, fixed_equal_weights=True)
    fixed_fused, fixed_w_ir, fixed_w_vis = fixed_model(
        torch.full((1, 2, 3, 3), 2.0),
        torch.zeros(1, 2, 3, 3),
        *[torch.rand(1, 1, 3, 3) for _ in range(3)],
    )
    assert torch.equal(fixed_w_ir, torch.full_like(fixed_w_ir, 0.5))
    assert torch.equal(fixed_w_vis, torch.full_like(fixed_w_vis, 0.5))
    assert torch.equal(fixed_fused, torch.ones_like(fixed_fused))

    loss = fused_feat.mean()
    loss.backward()
    print("MECModule self-test passed.")
