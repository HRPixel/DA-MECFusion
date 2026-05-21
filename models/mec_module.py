"""MEC module for DA-MECFusion V1.

MEC means modal effective contribution. It predicts spatial IR/VIS weights from
two feature maps and three rule-based priors, then fuses the two feature maps.
"""

from __future__ import annotations

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
    ) -> None:
        super().__init__()
        if feature_channels <= 0:
            raise ValueError("feature_channels must be positive")
        if hidden_channels <= 0:
            raise ValueError("hidden_channels must be positive")

        self.feature_channels = feature_channels
        self.hidden_channels = hidden_channels
        self.use_bn = use_bn

        in_channels = 2 * feature_channels + 3
        self.weight_predictor = nn.Sequential(
            *self._conv_block(in_channels, hidden_channels, kernel_size=3, use_bn=use_bn),
            *self._conv_block(hidden_channels, hidden_channels, kernel_size=3, use_bn=use_bn),
            nn.Conv2d(hidden_channels, 2, kernel_size=1),
        )

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

        z = torch.cat(
            [feat_ir, feat_vis, thermal_prior, sat_uncertainty, smoke_prior],
            dim=1,
        )
        logits = self.weight_predictor(z)
        weights = torch.softmax(logits, dim=1)
        w_ir = weights[:, 0:1]
        w_vis = weights[:, 1:2]
        fused_feat = w_ir * feat_ir + w_vis * feat_vis
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

    assert fused_feat.shape == (2, 64, 128, 128), fused_feat.shape
    assert w_ir.shape == (2, 1, 128, 128), w_ir.shape
    assert w_vis.shape == (2, 1, 128, 128), w_vis.shape
    assert torch.max(torch.abs(w_ir + w_vis - 1.0)).item() < 1e-5

    loss = fused_feat.mean()
    loss.backward()
    print("MECModule self-test passed.")
