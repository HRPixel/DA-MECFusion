"""Batch inference script for DA-MECFusion V1-minimal on MSRS."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.msrs_dataset import MSRSDataset  # noqa: E402
from models.damecfusion import DAMECFusionV1  # noqa: E402
from utils import (  # noqa: E402
    create_run_dir,
    get_names,
    move_batch_to_device,
    prepare_run_dirs,
    resolve_device,
    save_fused_batch,
    save_single_maps,
    visualize_mec_weights,
    visualize_priors,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch test DA-MECFusion V1-minimal on MSRS.")
    parser.add_argument("--data_root", type=str, default="data/MSRS")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--run_root", type=str, default="runs")
    parser.add_argument("--run_name", type=str, default="test_MSRS")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--feature_channels", type=int, default=64)
    return parser.parse_args()


def load_model(args: argparse.Namespace, device: torch.device) -> DAMECFusionV1:
    model = DAMECFusionV1(feature_channels=args.feature_channels).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    run_dir = create_run_dir(args.run_root, args.run_name)
    dirs = prepare_run_dirs(run_dir)

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

            save_fused_batch(fused, names, dirs["fused"])
            save_single_maps(thermal_prior, names, dirs["thermal"])
            save_single_maps(sat_uncertainty, names, dirs["saturation"])
            save_single_maps(smoke_prior, names, dirs["smoke"])
            save_single_maps(w_ir, names, dirs["w_ir"])
            save_single_maps(w_vis, names, dirs["w_vis"])

            visualize_priors(
                thermal_prior,
                sat_uncertainty,
                smoke_prior,
                ir=ir,
                vis=vis,
                names=names,
                save_dir=None,
                comparison_dir=dirs["prior_comparisons"],
                max_images=len(names),
            )
            visualize_mec_weights(
                w_ir,
                w_vis,
                ir=ir,
                vis=vis,
                names=names,
                save_dir=None,
                comparison_dir=dirs["weight_comparisons"],
                max_images=len(names),
            )

    print("Testing finished.")
    print(f"Run directory: {dirs['root']}")
    print(f"Fused outputs: {dirs['fused']}")


if __name__ == "__main__":
    main()
