"""Single-pair inference script for DA-MECFusion V1-minimal."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch


PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.transforms import preprocess_pair  # noqa: E402
from models.damecfusion import DAMECFusionV1  # noqa: E402
from utils import (  # noqa: E402
    create_run_dir,
    prepare_run_dirs,
    resolve_device,
    save_fused_batch,
    visualize_mec_weights,
    visualize_priors,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run single-pair DA-MECFusion V1-minimal inference.")
    parser.add_argument("--ir_path", type=str, required=True)
    parser.add_argument("--vis_path", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--run_root", type=str, default="runs")
    parser.add_argument("--run_name", type=str, default="infer_MSRS")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=640)
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

    ir, vis, _ = preprocess_pair(
        args.ir_path,
        args.vis_path,
        height=args.height,
        width=args.width,
    )
    ir = ir.unsqueeze(0).to(device)
    vis = vis.unsqueeze(0).to(device)

    model = load_model(args, device)

    with torch.no_grad():
        outputs = model(ir, vis)

    names = ["single"]
    fused = outputs["fused"].clamp(0.0, 1.0)
    thermal_prior = outputs["thermal_prior"].clamp(0.0, 1.0)
    sat_uncertainty = outputs["sat_uncertainty"].clamp(0.0, 1.0)
    smoke_prior = outputs["smoke_prior"].clamp(0.0, 1.0)
    w_ir = outputs["w_ir"].clamp(0.0, 1.0)
    w_vis = outputs["w_vis"].clamp(0.0, 1.0)

    save_fused_batch(fused, names, dirs["fused"])
    visualize_priors(
        thermal_prior,
        sat_uncertainty,
        smoke_prior,
        ir=ir,
        vis=vis,
        names=names,
        save_dir=dirs["priors"],
        comparison_dir=dirs["prior_comparisons"],
        max_images=1,
    )
    visualize_mec_weights(
        w_ir,
        w_vis,
        ir=ir,
        vis=vis,
        names=names,
        save_dir=dirs["weights"],
        comparison_dir=dirs["weight_comparisons"],
        max_images=1,
    )

    print("Inference finished.")
    print(f"Run directory: {dirs['root']}")
    print(f"Fused image: {dirs['fused'] / 'single.png'}")


if __name__ == "__main__":
    main()
