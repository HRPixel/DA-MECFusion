"""Rule-based priors for DA-MECFusion V1.

This module keeps all model-side computation in torch tensors and provides the
three V1 priors used before MEC fusion:
thermal saliency, visible saturation uncertainty, and smoke/low-contrast prior.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def normalize_0_1(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Normalize each batch sample independently to [0, 1]."""
    if x.dim() != 4:
        raise ValueError(f"Expected x with shape [B,C,H,W], got {tuple(x.shape)}")

    x_min = x.amin(dim=(1, 2, 3), keepdim=True)
    x_max = x.amax(dim=(1, 2, 3), keepdim=True)
    return ((x - x_min) / (x_max - x_min + eps)).clamp(0.0, 1.0)


def rgb_to_y(vis: torch.Tensor) -> torch.Tensor:
    """Convert RGB visible image tensor to luminance Y."""
    if vis.dim() != 4 or vis.size(1) != 3:
        raise ValueError(f"Expected vis with shape [B,3,H,W], got {tuple(vis.shape)}")

    r = vis[:, 0:1]
    g = vis[:, 1:2]
    b = vis[:, 2:3]
    return 0.299 * r + 0.587 * g + 0.114 * b


def mean_filter(x: torch.Tensor, kernel_size: int) -> torch.Tensor:
    """Apply a same-size mean filter with avg_pool2d."""
    if x.dim() != 4:
        raise ValueError(f"Expected x with shape [B,C,H,W], got {tuple(x.shape)}")
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd integer")

    return F.avg_pool2d(
        x,
        kernel_size=kernel_size,
        stride=1,
        padding=kernel_size // 2,
    )


def sobel_gradient(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Compute Sobel gradient magnitude for a single-channel image tensor."""
    if x.dim() != 4 or x.size(1) != 1:
        raise ValueError(f"Expected x with shape [B,1,H,W], got {tuple(x.shape)}")

    kx = x.new_tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
    ).view(1, 1, 3, 3)
    ky = x.new_tensor(
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]
    ).view(1, 1, 3, 3)

    gx = F.conv2d(x, kx, padding=1)
    gy = F.conv2d(x, ky, padding=1)
    return torch.sqrt(gx * gx + gy * gy + eps)


def percentile_per_sample(x: torch.Tensor, q: float) -> torch.Tensor:
    """Compute per-sample q percentile for [B,1,H,W] tensors."""
    if x.dim() != 4 or x.size(1) != 1:
        raise ValueError(f"Expected x with shape [B,1,H,W], got {tuple(x.shape)}")
    if q < 0.0 or q > 100.0:
        raise ValueError("q must be in [0, 100]")

    flat = x.flatten(start_dim=1)
    quantile = torch.quantile(flat, q / 100.0, dim=1, keepdim=True)
    return quantile.view(x.size(0), 1, 1, 1)


class ThermalSaliencyPrior(nn.Module):
    """Rule-based infrared thermal saliency prior."""

    def __init__(
        self,
        p_low: float = 1.0,
        p_high: float = 99.0,
        q: float = 85.0,
        gamma_g: float = 0.08,
        gamma_l: float = 0.8,
        alpha: float = 0.7,
        window_size: int = 7,
        eps: float = 1e-6,
        smooth: bool = True,
    ) -> None:
        super().__init__()
        self.p_low = p_low
        self.p_high = p_high
        self.q = q
        self.gamma_g = gamma_g
        self.gamma_l = gamma_l
        self.alpha = alpha
        self.window_size = window_size
        self.eps = eps
        self.smooth = smooth

    def forward(self, ir: torch.Tensor) -> torch.Tensor:
        if ir.dim() != 4 or ir.size(1) != 1:
            raise ValueError(f"Expected ir with shape [B,1,H,W], got {tuple(ir.shape)}")

        p_low = percentile_per_sample(ir, self.p_low)
        p_high = percentile_per_sample(ir, self.p_high)
        ir_clipped = torch.minimum(torch.maximum(ir, p_low), p_high)
        ir_norm = (ir_clipped - p_low) / (p_high - p_low + self.eps)
        ir_norm = ir_norm.clamp(0.0, 1.0)

        tau_g = percentile_per_sample(ir_norm, self.q)
        s_g = torch.sigmoid((ir_norm - tau_g) / self.gamma_g)

        local_mean = mean_filter(ir_norm, self.window_size)
        local_sq_mean = mean_filter(ir_norm * ir_norm, self.window_size)
        local_var = torch.clamp(local_sq_mean - local_mean * local_mean, min=0.0)
        local_std = torch.sqrt(local_var + self.eps)
        c_ir = (ir_norm - local_mean) / (local_std + self.eps)
        s_l = torch.sigmoid(c_ir / self.gamma_l)

        s_thermal = normalize_0_1(self.alpha * s_g + (1.0 - self.alpha) * s_l, self.eps)
        if self.smooth:
            s_thermal = normalize_0_1(mean_filter(s_thermal, 3), self.eps)
        return s_thermal


class VisibleSaturationUncertainty(nn.Module):
    """Visible-light saturation uncertainty prior."""

    def __init__(
        self,
        sat_threshold: float = 0.95,
        gamma_sat: float = 0.05,
        eps: float = 1e-6,
        smooth: bool = True,
        use_white_consistency: bool = False,
    ) -> None:
        super().__init__()
        self.sat_threshold = sat_threshold
        self.gamma_sat = gamma_sat
        self.eps = eps
        self.smooth = smooth
        self.use_white_consistency = use_white_consistency

    def forward(self, vis: torch.Tensor) -> torch.Tensor:
        if vis.dim() != 4 or vis.size(1) != 3:
            raise ValueError(f"Expected vis with shape [B,3,H,W], got {tuple(vis.shape)}")

        max_rgb = vis.amax(dim=1, keepdim=True)
        sat_response = torch.sigmoid((max_rgb - self.sat_threshold) / self.gamma_sat)

        if self.use_white_consistency:
            min_rgb = vis.amin(dim=1, keepdim=True)
            white_consistency = 1.0 - normalize_0_1(max_rgb - min_rgb, self.eps)
            sat_response = sat_response * white_consistency

        vis_y = rgb_to_y(vis)
        grad = sobel_gradient(vis_y, self.eps)
        grad_norm = normalize_0_1(grad, self.eps)
        texture_loss = 1.0 - grad_norm
        u_sat = normalize_0_1(sat_response * texture_loss, self.eps)

        if self.smooth:
            u_sat = normalize_0_1(mean_filter(u_sat, 3), self.eps)
        return u_sat


class SmokeLowContrastPrior(nn.Module):
    """Visible smoke or low-contrast degradation prior."""

    def __init__(
        self,
        window_size: int = 7,
        alpha_c: float = 0.5,
        beta_g: float = 0.5,
        eps: float = 1e-6,
        smooth: bool = True,
        use_sigmoid: bool = False,
        tau_d: float = 0.4,
        gamma_d: float = 0.1,
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.alpha_c = alpha_c
        self.beta_g = beta_g
        self.eps = eps
        self.smooth = smooth
        self.use_sigmoid = use_sigmoid
        self.tau_d = tau_d
        self.gamma_d = gamma_d

    def forward(self, vis: torch.Tensor) -> torch.Tensor:
        if vis.dim() != 4 or vis.size(1) != 3:
            raise ValueError(f"Expected vis with shape [B,3,H,W], got {tuple(vis.shape)}")

        vis_y = rgb_to_y(vis)
        local_mean = mean_filter(vis_y, self.window_size)
        local_mean_sq = mean_filter(vis_y * vis_y, self.window_size)
        local_var = torch.clamp(local_mean_sq - local_mean * local_mean, min=0.0)
        local_std = torch.sqrt(local_var + self.eps)

        contrast_norm = normalize_0_1(local_std, self.eps)
        grad_norm = normalize_0_1(sobel_gradient(vis_y, self.eps), self.eps)
        reliability = self.alpha_c * contrast_norm + self.beta_g * grad_norm

        if self.use_sigmoid:
            d_smoke = torch.sigmoid((self.tau_d - reliability) / self.gamma_d)
        else:
            d_smoke = 1.0 - reliability

        if self.smooth:
            d_smoke = mean_filter(d_smoke, 3)
        return normalize_0_1(d_smoke, self.eps)


class RuleBasedPriorModule(nn.Module):
    """Unified wrapper for the three rule-based DA-MECFusion V1 priors."""

    def __init__(
        self,
        thermal_prior: ThermalSaliencyPrior | None = None,
        saturation_uncertainty: VisibleSaturationUncertainty | None = None,
        smoke_prior: SmokeLowContrastPrior | None = None,
    ) -> None:
        super().__init__()
        self.thermal_prior = thermal_prior or ThermalSaliencyPrior()
        self.saturation_uncertainty = saturation_uncertainty or VisibleSaturationUncertainty()
        self.smoke_prior = smoke_prior or SmokeLowContrastPrior()

    def forward(self, ir: torch.Tensor, vis: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        thermal_prior = self.thermal_prior(ir)
        sat_uncertainty = self.saturation_uncertainty(vis)
        smoke_prior = self.smoke_prior(vis)
        return thermal_prior, sat_uncertainty, smoke_prior


def _print_prior_stats(name: str, x: torch.Tensor) -> None:
    print(
        f"{name}: shape={tuple(x.shape)}, "
        f"min={x.min().item():.6f}, max={x.max().item():.6f}, mean={x.mean().item():.6f}"
    )


if __name__ == "__main__":
    ir = torch.rand(2, 1, 256, 256)
    vis = torch.rand(2, 3, 256, 256)
    module = RuleBasedPriorModule()

    thermal, sat, smoke = module(ir, vis)
    priors = {
        "thermal_prior": thermal,
        "sat_uncertainty": sat,
        "smoke_prior": smoke,
    }

    for prior_name, prior in priors.items():
        _print_prior_stats(prior_name, prior)
        assert prior.shape == (2, 1, 256, 256), prior.shape
        assert not torch.isnan(prior).any(), prior_name
        assert prior.min().item() >= 0.0, prior_name
        assert prior.max().item() <= 1.0, prior_name

    print("RuleBasedPriorModule self-test passed.")
