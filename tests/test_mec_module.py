"""Unit test script for DA-MECFusion V1 MECModule."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.mec_module import MECModule  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test MECModule forward and backward.")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--feature_height", type=int, default=128)
    parser.add_argument("--feature_width", type=int, default=128)
    parser.add_argument("--prior_height", type=int, default=256)
    parser.add_argument("--prior_width", type=int, default=256)
    parser.add_argument("--hidden_channels", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def resolve_device(requested_device: str) -> torch.device:
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        print(f"Requested device '{requested_device}' is unavailable. Falling back to cpu.")
        return torch.device("cpu")
    return torch.device(requested_device)


def print_tensor_stats(name: str, x: torch.Tensor) -> None:
    print(
        f"{name}: shape={tuple(x.shape)}, "
        f"min={x.min().item():.6f}, max={x.max().item():.6f}, "
        f"mean={x.mean().item():.6f}, has_nan={torch.isnan(x).any().item()}"
    )


def assert_weight_valid(name: str, weight: torch.Tensor, expected_shape: tuple[int, ...]) -> None:
    assert tuple(weight.shape) == expected_shape, f"{name} shape mismatch: {tuple(weight.shape)}"
    assert not torch.isnan(weight).any(), f"{name} contains NaN"
    assert weight.min().item() >= -1e-5, f"{name} min is below 0: {weight.min().item()}"
    assert weight.max().item() <= 1.0 + 1e-5, f"{name} max is above 1: {weight.max().item()}"


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    print(f"Using device: {device}")

    feat_shape = (
        args.batch_size,
        args.channels,
        args.feature_height,
        args.feature_width,
    )
    prior_shape = (
        args.batch_size,
        1,
        args.prior_height,
        args.prior_width,
    )
    weight_shape = (
        args.batch_size,
        1,
        args.feature_height,
        args.feature_width,
    )

    feat_ir = torch.randn(*feat_shape, device=device, requires_grad=True)
    feat_vis = torch.randn(*feat_shape, device=device, requires_grad=True)
    thermal = torch.rand(*prior_shape, device=device)
    sat = torch.rand(*prior_shape, device=device)
    smoke = torch.rand(*prior_shape, device=device)

    model = MECModule(
        feature_channels=args.channels,
        hidden_channels=args.hidden_channels,
    ).to(device)

    fused_feat, w_ir, w_vis = model(feat_ir, feat_vis, thermal, sat, smoke)

    print(f"fused_feat shape: {tuple(fused_feat.shape)}")
    print_tensor_stats("w_ir", w_ir)
    print_tensor_stats("w_vis", w_vis)

    weight_sum_error = torch.max(torch.abs(w_ir + w_vis - 1.0)).item()
    print(f"max(abs(w_ir + w_vis - 1)): {weight_sum_error:.8f}")

    assert tuple(fused_feat.shape) == feat_shape, f"fused_feat shape mismatch: {tuple(fused_feat.shape)}"
    assert not torch.isnan(fused_feat).any(), "fused_feat contains NaN"
    assert_weight_valid("w_ir", w_ir, weight_shape)
    assert_weight_valid("w_vis", w_vis, weight_shape)
    assert weight_sum_error < 1e-5, f"Weight sum error is too large: {weight_sum_error}"

    loss = fused_feat.mean()
    loss.backward()

    has_param_grad = any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    assert has_param_grad, "No trainable MECModule parameter received a finite gradient"

    print("All MECModule tests passed.")


if __name__ == "__main__":
    main()
