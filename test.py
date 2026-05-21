"""Batch inference script for DA-MECFusion V1."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.msrs_dataset import MSRSDataset  # noqa: E402
from models.damecfusion import DAMECFusionV1  # noqa: E402
from tools.visualize_priors import visualize_priors  # noqa: E402
from tools.visualize_weights import visualize_mec_weights  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch test DA-MECFusion V1.")
    parser.add_argument("--data_root", type=str, default="data/processed/MSRS")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="results/DA-MECFusion_V1/MSRS")
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--feature_channels", type=int, default=64)
    return parser.parse_args()


def resolve_device(requested_device: str) -> torch.device:
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        print(f"Requested device '{requested_device}' is unavailable. Falling back to cpu.")
        return torch.device("cpu")
    return torch.device(requested_device)


def prepare_output_dirs(save_dir: Path) -> dict[str, Path]:
    dirs = {
        "fused": save_dir / "fused",
        "thermal_prior": save_dir / "thermal_prior",
        "sat_uncertainty": save_dir / "sat_uncertainty",
        "smoke_prior": save_dir / "smoke_prior",
        "w_ir": save_dir / "w_ir",
        "w_vis": save_dir / "w_vis",
    }
    for directory in dirs.values():
        directory.mkdir(parents=True, exist_ok=True)
    return dirs


def sanitize_name(name: Any) -> str:
    name = str(name)
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in name)


def get_names(batch: dict[str, Any], batch_size: int) -> list[str]:
    names = batch.get("name")
    if names is None:
        return [f"{index:03d}" for index in range(batch_size)]
    if isinstance(names, (list, tuple)):
        return [sanitize_name(name) for name in names]
    return [sanitize_name(names)]


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        moved[key] = value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
    return moved


def load_model(args: argparse.Namespace, device: torch.device) -> DAMECFusionV1:
    model = DAMECFusionV1(feature_channels=args.feature_channels).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def save_tensor_image(tensor: torch.Tensor, path: Path) -> None:
    image = tensor.detach().cpu().clamp(0.0, 1.0).squeeze().numpy()
    image_8bit = np.clip(image * 255.0, 0.0, 255.0).round().astype(np.uint8)
    Image.fromarray(image_8bit, mode="L").save(path)


def save_fused_batch(fused: torch.Tensor, names: list[str], save_dir: Path) -> None:
    for index, name in enumerate(names):
        save_tensor_image(fused[index, 0], save_dir / f"{name}.png")


def save_single_maps(batch_tensor: torch.Tensor, names: list[str], save_dir: Path) -> None:
    for index, name in enumerate(names):
        save_tensor_image(batch_tensor[index, 0], save_dir / f"{name}.png")


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    save_dir = Path(args.save_dir)
    output_dirs = prepare_output_dirs(save_dir)

    dataset = MSRSDataset(
        data_root=args.data_root,
        split=args.split,
        height=args.height,
        width=args.width,
        use_label=False,
    )
    if len(dataset) == 0:
        raise RuntimeError(f"Dataset is empty: {args.data_root}/{args.split}")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    model = load_model(args, device)

    with torch.no_grad():
        for batch in tqdm(loader, desc="Testing"):
            batch = move_batch_to_device(batch, device)
            ir = batch["ir"]
            vis = batch["vis"]
            names = get_names(batch, ir.shape[0])

            outputs = model(ir, vis)
            fused = outputs["fused"].clamp(0.0, 1.0)
            thermal_prior = outputs["thermal_prior"].clamp(0.0, 1.0)
            sat_uncertainty = outputs["sat_uncertainty"].clamp(0.0, 1.0)
            smoke_prior = outputs["smoke_prior"].clamp(0.0, 1.0)
            w_ir = outputs["w_ir"].clamp(0.0, 1.0)
            w_vis = outputs["w_vis"].clamp(0.0, 1.0)

            save_fused_batch(fused, names, output_dirs["fused"])
            save_single_maps(thermal_prior, names, output_dirs["thermal_prior"])
            save_single_maps(sat_uncertainty, names, output_dirs["sat_uncertainty"])
            save_single_maps(smoke_prior, names, output_dirs["smoke_prior"])
            save_single_maps(w_ir, names, output_dirs["w_ir"])
            save_single_maps(w_vis, names, output_dirs["w_vis"])

            visualize_priors(
                thermal_prior,
                sat_uncertainty,
                smoke_prior,
                ir=ir,
                vis=vis,
                names=names,
                save_dir=save_dir / "prior_comparison",
                max_images=len(names),
            )
            visualize_mec_weights(
                w_ir,
                w_vis,
                ir=ir,
                vis=vis,
                names=names,
                save_dir=save_dir / "weight_comparison",
                max_images=len(names),
            )

    print(f"Testing finished. Results saved to: {save_dir}")


if __name__ == "__main__":
    main()
