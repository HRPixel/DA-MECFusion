"""Unit test script for DA-MECFusion V1 rule-based priors."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.priors import RuleBasedPriorModule  # noqa: E402


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("Boolean value expected: true or false")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test RuleBasedPriorModule outputs.")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_debug", type=str2bool, default=False)
    parser.add_argument("--save_dir", type=str, default="debug_outputs/priors")
    return parser.parse_args()


def resolve_device(requested_device: str) -> torch.device:
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        print(f"Requested device '{requested_device}' is unavailable. Falling back to cpu.")
        return torch.device("cpu")
    return torch.device(requested_device)


def print_prior_stats(name: str, prior: torch.Tensor) -> None:
    has_nan = torch.isnan(prior).any().item()
    print(
        f"{name}: shape={tuple(prior.shape)}, "
        f"min={prior.min().item():.6f}, max={prior.max().item():.6f}, "
        f"mean={prior.mean().item():.6f}, has_nan={has_nan}"
    )


def assert_prior_valid(
    name: str,
    prior: torch.Tensor,
    expected_shape: tuple[int, int, int, int],
    tol: float = 1e-5,
) -> None:
    assert tuple(prior.shape) == expected_shape, f"{name} shape mismatch: {tuple(prior.shape)}"
    assert not torch.isnan(prior).any(), f"{name} contains NaN"
    assert prior.min().item() >= -tol, f"{name} min is below 0: {prior.min().item()}"
    assert prior.max().item() <= 1.0 + tol, f"{name} max is above 1: {prior.max().item()}"


def save_debug_images(priors: dict[str, torch.Tensor], save_dir: Path, max_images: int = 4) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    save_dir.mkdir(parents=True, exist_ok=True)
    batch_size = next(iter(priors.values())).shape[0]
    num_images = min(batch_size, max_images)

    for idx in range(num_images):
        fig, axes = plt.subplots(1, len(priors), figsize=(4 * len(priors), 4))
        if len(priors) == 1:
            axes = [axes]

        for axis, (name, prior) in zip(axes, priors.items()):
            image = prior[idx, 0].detach().cpu().numpy()
            axis.imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
            axis.set_title(name)
            axis.axis("off")

            single_path = save_dir / f"{name}_{idx:03d}.png"
            plt.imsave(single_path, image, cmap="gray", vmin=0.0, vmax=1.0)

        fig.tight_layout()
        fig.savefig(save_dir / f"prior_comparison_{idx:03d}.png", dpi=150)
        plt.close(fig)

    print(f"Saved debug prior images to: {save_dir}")


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    print(f"Using device: {device}")

    ir = torch.rand(args.batch_size, 1, args.height, args.width, device=device)
    vis = torch.rand(args.batch_size, 3, args.height, args.width, device=device)

    module = RuleBasedPriorModule().to(device)
    module.eval()

    with torch.no_grad():
        thermal_prior, sat_uncertainty, smoke_prior = module(ir, vis)

    priors = {
        "thermal_prior": thermal_prior,
        "sat_uncertainty": sat_uncertainty,
        "smoke_prior": smoke_prior,
    }
    expected_shape = (args.batch_size, 1, args.height, args.width)

    for name, prior in priors.items():
        print_prior_stats(name, prior)
        assert_prior_valid(name, prior, expected_shape)

    if args.save_debug:
        save_debug_images(priors, Path(args.save_dir))

    print("All prior tests passed.")


if __name__ == "__main__":
    main()
