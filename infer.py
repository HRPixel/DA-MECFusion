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
    parser.add_argument(
        "--smoke_prior_mode",
        choices=("base", "fixed_texture", "fixed_texture_luma"),
        default=None,
        help="Override the checkpoint smoke prior mode; old checkpoints default to base.",
    )
    parser.add_argument(
        "--prior_residual_scale",
        type=float,
        default=None,
        help="Override the checkpoint prior residual scale; old checkpoints default to 0.",
    )
    parser.add_argument(
        "--bounded_learned_gap",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override bounded learned-gap mode; old checkpoints default to disabled.",
    )
    parser.add_argument(
        "--equalize_feature_magnitude",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override fusion feature-magnitude equalization; old checkpoints default to disabled.",
    )
    return parser.parse_args()


def load_model(args: argparse.Namespace, device: torch.device) -> DAMECFusionV1:
    checkpoint = torch.load(args.checkpoint, map_location=device)
    checkpoint_args = checkpoint.get("args", {}) if isinstance(checkpoint, dict) else {}
    smoke_prior_mode = args.smoke_prior_mode or checkpoint_args.get("smoke_prior_mode", "base")
    prior_residual_scale = (
        args.prior_residual_scale
        if args.prior_residual_scale is not None
        else float(checkpoint_args.get("prior_residual_scale", 0.0))
    )
    bounded_learned_gap = (
        args.bounded_learned_gap
        if args.bounded_learned_gap is not None
        else bool(checkpoint_args.get("bounded_learned_gap", False))
    )
    equalize_feature_magnitude = (
        args.equalize_feature_magnitude
        if args.equalize_feature_magnitude is not None
        else bool(checkpoint_args.get("equalize_feature_magnitude", False))
    )
    disable_thermal_prior = bool(checkpoint_args.get("disable_thermal_prior", False))
    disable_sat_uncertainty = bool(checkpoint_args.get("disable_sat_uncertainty", False))
    disable_smoke_prior = bool(checkpoint_args.get("disable_smoke_prior", False))
    fixed_equal_weights = bool(checkpoint_args.get("fixed_equal_weights", False))
    model = DAMECFusionV1(
        feature_channels=args.feature_channels,
        smoke_prior_mode=smoke_prior_mode,
        prior_residual_scale=prior_residual_scale,
        bounded_learned_gap=bounded_learned_gap,
        equalize_feature_magnitude=equalize_feature_magnitude,
        disable_thermal_prior=disable_thermal_prior,
        disable_sat_uncertainty=disable_sat_uncertainty,
        disable_smoke_prior=disable_smoke_prior,
        fixed_equal_weights=fixed_equal_weights,
    ).to(device)
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
    print(f"Smoke prior mode: {model.prior_module.smoke_prior.mode}")
    print(f"Prior residual scale: {model.mec.prior_residual_scale}")
    print(f"Bounded learned gap: {model.mec.bounded_learned_gap}")
    print(f"Equalize feature magnitude: {model.mec.equalize_feature_magnitude}")
    print(f"Disable thermal prior to MEC: {model.mec.disable_thermal_prior}")
    print(f"Disable saturation uncertainty to MEC: {model.mec.disable_sat_uncertainty}")
    print(f"Disable smoke prior to MEC: {model.mec.disable_smoke_prior}")
    print(f"Fixed equal weights: {model.mec.fixed_equal_weights}")

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
