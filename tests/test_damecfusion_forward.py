"""Forward test script for DA-MECFusion V1."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.damecfusion import DAMECFusionV1  # noqa: E402


REQUIRED_OUTPUT_KEYS = (
    "fused",
    "thermal_prior",
    "sat_uncertainty",
    "smoke_prior",
    "w_ir",
    "w_vis",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test DAMECFusionV1 forward pass.")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--channels", type=int, default=64)
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


def assert_output_valid(name: str, x: torch.Tensor, expected_shape: tuple[int, ...]) -> None:
    assert tuple(x.shape) == expected_shape, f"{name} shape mismatch: {tuple(x.shape)}"
    assert not torch.isnan(x).any(), f"{name} contains NaN"


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    print(f"Using device: {device}")

    expected_shape = (args.batch_size, 1, args.height, args.width)
    ir = torch.rand(*expected_shape, device=device)
    vis = torch.rand(args.batch_size, 3, args.height, args.width, device=device)

    model = DAMECFusionV1(feature_channels=args.channels).to(device)
    outputs = model(ir, vis)

    missing_keys = [key for key in REQUIRED_OUTPUT_KEYS if key not in outputs]
    assert not missing_keys, f"Missing output keys: {missing_keys}"

    for key in REQUIRED_OUTPUT_KEYS:
        print_tensor_stats(key, outputs[key])
        assert_output_valid(key, outputs[key], expected_shape)

    fused = outputs["fused"]
    assert fused.min().item() >= -1e-5, f"fused min is below 0: {fused.min().item()}"
    assert fused.max().item() <= 1.0 + 1e-5, f"fused max is above 1: {fused.max().item()}"

    weight_sum_error = torch.max(torch.abs(outputs["w_ir"] + outputs["w_vis"] - 1.0)).item()
    print(f"max(abs(w_ir + w_vis - 1)): {weight_sum_error:.8f}")
    assert weight_sum_error < 1e-5, f"Weight sum error is too large: {weight_sum_error}"

    loss = outputs["fused"].mean()
    loss.backward()

    has_param_grad = any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    assert has_param_grad, "No trainable DAMECFusionV1 parameter received a finite gradient"

    print("All DAMECFusionV1 forward tests passed.")


if __name__ == "__main__":
    main()
