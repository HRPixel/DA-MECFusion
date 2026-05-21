"""Synthetic semantic tests for DA-MECFusion V1 rule-based priors."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.priors import RuleBasedPriorModule, rgb_to_y  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Synthetic semantic tests for priors.")
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_debug", action="store_true")
    parser.add_argument("--save_dir", type=str, default="debug_outputs/priors_synthetic")
    return parser.parse_args()


def resolve_device(requested_device: str) -> torch.device:
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        print(f"Requested device '{requested_device}' is unavailable. Falling back to cpu.")
        return torch.device("cpu")
    return torch.device(requested_device)


def center_square_mask(height: int, width: int, scale: float, device: torch.device) -> torch.Tensor:
    mask = torch.zeros(1, 1, height, width, dtype=torch.bool, device=device)
    box_h = max(1, int(height * scale))
    box_w = max(1, int(width * scale))
    y0 = (height - box_h) // 2
    x0 = (width - box_w) // 2
    mask[:, :, y0 : y0 + box_h, x0 : x0 + box_w] = True
    return mask


def weak_texture_vis(height: int, width: int, device: torch.device) -> torch.Tensor:
    y = torch.linspace(0.0, 1.0, height, device=device).view(1, 1, height, 1)
    x = torch.linspace(0.0, 1.0, width, device=device).view(1, 1, 1, width)
    texture = 0.35 + 0.05 * torch.sin(12.0 * x) * torch.cos(10.0 * y)
    return texture.repeat(1, 3, 1, 1).clamp(0.0, 1.0)


def checkerboard_vis(height: int, width: int, device: torch.device, block_size: int = 8) -> torch.Tensor:
    yy = torch.arange(height, device=device).view(height, 1)
    xx = torch.arange(width, device=device).view(1, width)
    board = ((yy // block_size + xx // block_size) % 2).float()
    board = 0.2 + 0.6 * board
    return board.view(1, 1, height, width).repeat(1, 3, 1, 1)


def region_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return x[mask].mean()


def print_comparison(name: str, positive_mean: torch.Tensor, reference_mean: torch.Tensor) -> None:
    diff = positive_mean - reference_mean
    print(
        f"{name}: region_mean={positive_mean.item():.6f}, "
        f"reference_mean={reference_mean.item():.6f}, "
        f"difference={diff.item():.6f}"
    )


def save_gray(path: Path, image: torch.Tensor) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    array = image.detach().cpu().squeeze().numpy()
    plt.imsave(path, array, cmap="gray", vmin=0.0, vmax=1.0)


def save_debug_images(
    save_dir: Path,
    ir: torch.Tensor,
    vis: torch.Tensor,
    thermal_prior: torch.Tensor,
    sat_uncertainty: torch.Tensor,
    smoke_prior: torch.Tensor,
) -> None:
    save_gray(save_dir / "synthetic_ir.png", ir[0, 0])
    save_gray(save_dir / "synthetic_vis_y.png", rgb_to_y(vis)[0, 0])
    save_gray(save_dir / "thermal_prior.png", thermal_prior[0, 0])
    save_gray(save_dir / "sat_uncertainty.png", sat_uncertainty[0, 0])
    save_gray(save_dir / "smoke_prior.png", smoke_prior[0, 0])
    print(f"Saved synthetic debug images to: {save_dir}")


def test_thermal_hot_spot(
    module: RuleBasedPriorModule,
    height: int,
    width: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ir = torch.full((1, 1, height, width), 0.1, device=device)
    vis = weak_texture_vis(height, width, device)

    hot_mask = center_square_mask(height, width, scale=0.25, device=device)
    background_mask = ~center_square_mask(height, width, scale=0.45, device=device)
    ir[hot_mask] = 0.9

    thermal_prior, _, _ = module(ir, vis)
    hot_mean = region_mean(thermal_prior, hot_mask)
    background_mean = region_mean(thermal_prior, background_mask)
    print_comparison("thermal_prior hot spot", hot_mean, background_mean)
    assert hot_mean > background_mean, "thermal_prior did not respond more strongly to the hot spot"
    return ir, vis, thermal_prior


def test_visible_saturation(
    module: RuleBasedPriorModule,
    height: int,
    width: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ir = torch.full((1, 1, height, width), 0.1, device=device)
    vis = torch.full((1, 3, height, width), 0.3, device=device)

    sat_mask = center_square_mask(height, width, scale=0.25, device=device)
    background_mask = ~center_square_mask(height, width, scale=0.45, device=device)
    vis = vis.clone()
    vis.expand_as(vis).masked_fill_(sat_mask.expand_as(vis), 1.0)

    _, sat_uncertainty, _ = module(ir, vis)
    sat_mean = region_mean(sat_uncertainty, sat_mask)
    background_mean = region_mean(sat_uncertainty, background_mask)
    print_comparison("sat_uncertainty overexposed block", sat_mean, background_mean)
    assert sat_mean > background_mean, "sat_uncertainty did not respond more strongly to overexposure"
    return ir, vis, sat_uncertainty


def test_low_contrast(
    module: RuleBasedPriorModule,
    height: int,
    width: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ir = torch.full((1, 1, height, width), 0.1, device=device)
    vis = checkerboard_vis(height, width, device)
    vis[:, :, :, width // 2 :] = 0.5

    left_mask = torch.zeros(1, 1, height, width, dtype=torch.bool, device=device)
    right_mask = torch.zeros_like(left_mask)
    margin = max(2, min(height, width) // 32)
    left_mask[:, :, margin : height - margin, margin : width // 2 - margin] = True
    right_mask[:, :, margin : height - margin, width // 2 + margin : width - margin] = True

    _, _, smoke_prior = module(ir, vis)
    low_contrast_mean = region_mean(smoke_prior, right_mask)
    texture_mean = region_mean(smoke_prior, left_mask)
    print_comparison("smoke_prior low contrast", low_contrast_mean, texture_mean)
    assert low_contrast_mean > texture_mean, "smoke_prior did not respond more strongly to low contrast"
    return ir, vis, smoke_prior


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    print(f"Using device: {device}")

    module = RuleBasedPriorModule().to(device)
    module.eval()

    with torch.no_grad():
        thermal_ir, thermal_vis, thermal_prior = test_thermal_hot_spot(
            module, args.height, args.width, device
        )
        sat_ir, sat_vis, sat_uncertainty = test_visible_saturation(
            module, args.height, args.width, device
        )
        smoke_ir, smoke_vis, smoke_prior = test_low_contrast(
            module, args.height, args.width, device
        )

    if args.save_debug:
        save_root = Path(args.save_dir)
        save_debug_images(
            save_root / "thermal_hot_spot",
            thermal_ir,
            thermal_vis,
            thermal_prior,
            module(thermal_ir, thermal_vis)[1],
            module(thermal_ir, thermal_vis)[2],
        )
        save_debug_images(
            save_root / "visible_saturation",
            sat_ir,
            sat_vis,
            module(sat_ir, sat_vis)[0],
            sat_uncertainty,
            module(sat_ir, sat_vis)[2],
        )
        save_debug_images(
            save_root / "low_contrast",
            smoke_ir,
            smoke_vis,
            module(smoke_ir, smoke_vis)[0],
            module(smoke_ir, smoke_vis)[1],
            smoke_prior,
        )

    print("All synthetic prior semantic tests passed.")


if __name__ == "__main__":
    main()
