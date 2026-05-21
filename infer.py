"""Single-pair inference script for DA-MECFusion V1."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import torch
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.transforms import preprocess_pair  # noqa: E402
from models.damecfusion import DAMECFusionV1  # noqa: E402
from tools.visualize_priors import visualize_priors  # noqa: E402
from tools.visualize_weights import visualize_mec_weights  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run single-pair DA-MECFusion V1 inference.")
    parser.add_argument("--ir_path", type=str, required=True)
    parser.add_argument("--vis_path", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="debug_outputs/infer")
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--feature_channels", type=int, default=64)
    return parser.parse_args()


def resolve_device(requested_device: str) -> torch.device:
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        print(f"Requested device '{requested_device}' is unavailable. Falling back to cpu.")
        return torch.device("cpu")
    return torch.device(requested_device)


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


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

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

    fused = outputs["fused"].clamp(0.0, 1.0)
    thermal_prior = outputs["thermal_prior"].clamp(0.0, 1.0)
    sat_uncertainty = outputs["sat_uncertainty"].clamp(0.0, 1.0)
    smoke_prior = outputs["smoke_prior"].clamp(0.0, 1.0)
    w_ir = outputs["w_ir"].clamp(0.0, 1.0)
    w_vis = outputs["w_vis"].clamp(0.0, 1.0)

    output_paths = {
        "fused": save_dir / "fused.png",
        "thermal_prior": save_dir / "thermal_prior.png",
        "sat_uncertainty": save_dir / "sat_uncertainty.png",
        "smoke_prior": save_dir / "smoke_prior.png",
        "w_ir": save_dir / "w_ir.png",
        "w_vis": save_dir / "w_vis.png",
    }

    save_tensor_image(fused[0, 0], output_paths["fused"])
    save_tensor_image(thermal_prior[0, 0], output_paths["thermal_prior"])
    save_tensor_image(sat_uncertainty[0, 0], output_paths["sat_uncertainty"])
    save_tensor_image(smoke_prior[0, 0], output_paths["smoke_prior"])
    save_tensor_image(w_ir[0, 0], output_paths["w_ir"])
    save_tensor_image(w_vis[0, 0], output_paths["w_vis"])

    visualize_priors(
        thermal_prior,
        sat_uncertainty,
        smoke_prior,
        ir=ir,
        vis=vis,
        names=["single"],
        save_dir=save_dir / "prior_comparison",
        max_images=1,
    )
    visualize_mec_weights(
        w_ir,
        w_vis,
        ir=ir,
        vis=vis,
        names=["single"],
        save_dir=save_dir / "weight_comparison",
        max_images=1,
    )

    print("Inference outputs:")
    for path in output_paths.values():
        print(path)
    print(save_dir / "prior_comparison")
    print(save_dir / "weight_comparison")


if __name__ == "__main__":
    main()
