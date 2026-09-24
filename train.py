"""Training script for DA-MECFusion paired infrared-visible manifests."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.msrs_dataset import MSRSDataset  # noqa: E402
from models.damecfusion import DAMECFusionV1  # noqa: E402
from models.losses import DAMECFusionLoss  # noqa: E402
from utils import (  # noqa: E402
    append_log,
    create_run_dir,
    get_names,
    init_log,
    move_batch_to_device,
    prepare_run_dirs,
    resolve_device,
    save_fused_batch,
    visualize_mec_weights,
    visualize_priors,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DA-MECFusion on a paired IR/VIS manifest.")
    parser.add_argument("--data_root", type=str, default="./data/MSRS")
    parser.add_argument("--train_manifest", type=str, default=None)
    parser.add_argument("--val_manifest", type=str, default=None)
    parser.add_argument("--run_root", type=str, default="./runs")
    parser.add_argument("--run_name", type=str, default="DA-MECFusion_v1_MSRS")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_interval", type=int, default=5)
    parser.add_argument("--debug_samples", type=int, default=8)
    parser.add_argument("--feature_channels", type=int, default=64)
    parser.add_argument("--lambda_int", type=float, default=1.0)
    parser.add_argument("--lambda_grad", type=float, default=10.0)
    parser.add_argument("--lambda_thermal", type=float, default=1.0)
    parser.add_argument("--lambda_sat", type=float, default=0.5)
    parser.add_argument(
        "--gradient_mode",
        choices=("magnitude", "directional"),
        default="magnitude",
        help="Use directional with --lambda_grad 2 for V2-Base; magnitude preserves V1 behavior.",
    )
    parser.add_argument(
        "--intensity_target",
        choices=("max", "mec_weighted"),
        default="max",
        help="Use detached MEC-weighted IR/VIS-Y intensity for V2-Loss-1; max preserves prior behavior.",
    )
    parser.add_argument(
        "--smoke_prior_mode",
        choices=("base", "fixed_texture", "fixed_texture_luma"),
        default="base",
        help="Smoke prior formula; base preserves V1 behavior.",
    )
    parser.add_argument(
        "--prior_residual_scale",
        type=float,
        default=0.0,
        help="Centered prior logit residual scale; 0 preserves V1 behavior.",
    )
    parser.add_argument(
        "--bounded_learned_gap",
        action="store_true",
        help="Bound the learned IR-VIS logit gap to (-2, 2) before adding the prior residual.",
    )
    parser.add_argument(
        "--equalize_feature_magnitude",
        action="store_true",
        help="Equalize per-pixel IR/VIS channel-mean absolute magnitude only for final fusion.",
    )
    parser.add_argument(
        "--disable_thermal_prior",
        action="store_true",
        help="Ablation: stop thermal prior from entering MEC; raw prior and losses stay unchanged.",
    )
    parser.add_argument(
        "--disable_sat_uncertainty",
        action="store_true",
        help="Ablation: stop saturation uncertainty from entering MEC; raw prior and losses stay unchanged.",
    )
    parser.add_argument(
        "--disable_smoke_prior",
        action="store_true",
        help="Ablation: stop smoke prior from entering MEC; raw prior stays available for diagnosis.",
    )
    parser.add_argument(
        "--fixed_equal_weights",
        action="store_true",
        help="Ablation: use fixed w_ir=w_vis=0.5 during training and inference.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overfit_batches", type=int, default=0)
    return parser.parse_args()
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_checkpoint(
    path: Path,
    model: DAMECFusionV1,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    args: argparse.Namespace,
) -> None:
    checkpoint = {
        "epoch": epoch,
        "loss": loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "args": vars(args),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)


def save_debug_outputs(
    model: DAMECFusionV1,
    batch: dict[str, Any],
    device: torch.device,
    epoch: int,
    dirs: dict[str, Path],
    max_images: int,
) -> None:
    model.eval()
    batch = move_batch_to_device(batch, device)
    ir = batch["ir"]
    vis = batch["vis"]
    names = [f"epoch_{epoch:03d}_{name}" for name in get_names(batch, ir.shape[0])]

    with torch.no_grad():
        outputs = model(ir, vis)

    save_fused_batch(outputs["fused"], names, dirs["fused"], max_images=max_images)
    visualize_priors(
        outputs["thermal_prior"],
        outputs["sat_uncertainty"],
        outputs["smoke_prior"],
        ir=ir,
        vis=vis,
        names=names,
        save_dir=dirs["priors"],
        comparison_dir=dirs["prior_comparisons"],
        max_images=max_images,
    )
    visualize_mec_weights(
        outputs["w_ir"],
        outputs["w_vis"],
        ir=ir,
        vis=vis,
        names=names,
        save_dir=dirs["weights"],
        comparison_dir=dirs["weight_comparisons"],
        max_images=max_images,
    )
    model.train()


def train_one_epoch(
    model: DAMECFusionV1,
    criterion: DAMECFusionLoss,
    optimizer: torch.optim.Optimizer,
    loader: DataLoader,
    device: torch.device,
    epoch: int,
    overfit_batches: int,
) -> dict[str, float]:
    model.train()
    totals = {
        "loss_total": 0.0,
        "loss_int": 0.0,
        "loss_grad": 0.0,
        "loss_thermal": 0.0,
        "loss_sat": 0.0,
    }
    num_steps = 0
    progress = tqdm(loader, desc=f"Epoch {epoch}", leave=False)

    for step, batch in enumerate(progress, start=1):
        if overfit_batches > 0 and step > overfit_batches:
            break

        batch = move_batch_to_device(batch, device)
        ir = batch["ir"]
        vis = batch["vis"]

        optimizer.zero_grad(set_to_none=True)
        outputs = model(ir, vis)
        loss_total, loss_dict = criterion(outputs, ir, vis)

        if not torch.isfinite(loss_total):
            raise FloatingPointError(f"Non-finite loss detected at epoch {epoch}, step {step}")

        loss_total.backward()
        optimizer.step()

        for key in totals:
            totals[key] += float(loss_dict[key].detach().cpu().item())
        num_steps += 1

        progress.set_postfix(loss=f"{loss_total.item():.4f}")

    if num_steps == 0:
        raise RuntimeError("No training steps were executed")

    return {key: value / num_steps for key, value in totals.items()}


def validate_one_epoch(
    model: DAMECFusionV1,
    criterion: DAMECFusionLoss,
    loader: DataLoader,
    device: torch.device,
    epoch: int,
) -> dict[str, float]:
    model.eval()
    totals = {
        "loss_total": 0.0,
        "loss_int": 0.0,
        "loss_grad": 0.0,
        "loss_thermal": 0.0,
        "loss_sat": 0.0,
    }
    num_steps = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Validation {epoch}", leave=False):
            batch = move_batch_to_device(batch, device)
            outputs = model(batch["ir"], batch["vis"])
            loss_total, loss_dict = criterion(outputs, batch["ir"], batch["vis"])
            if not torch.isfinite(loss_total):
                raise FloatingPointError(f"Non-finite validation loss at epoch {epoch}")
            for key in totals:
                totals[key] += float(loss_dict[key].cpu().item())
            num_steps += 1

    if num_steps == 0:
        raise RuntimeError("Validation dataset is empty")
    return {key: value / num_steps for key, value in totals.items()}


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    run_dir = create_run_dir(args.run_root, args.run_name)
    dirs = prepare_run_dirs(run_dir)
    log_path = dirs["logs"] / "train_log.csv"
    val_log_path = dirs["logs"] / "val_log.csv"
    init_log(log_path)

    dataset = MSRSDataset(
        data_root=args.data_root,
        split="train",
        height=args.height,
        width=args.width,
        use_label=False,
        manifest_path=args.train_manifest,
    )
    if len(dataset) == 0:
        raise RuntimeError(f"Training dataset is empty: {args.data_root}")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    debug_batch = next(iter(loader))

    val_loader = None
    val_dataset = None
    if args.val_manifest is not None:
        val_dataset = MSRSDataset(
            data_root=args.data_root,
            split="val",
            height=args.height,
            width=args.width,
            use_label=False,
            manifest_path=args.val_manifest,
        )
        if len(val_dataset) == 0:
            raise RuntimeError(f"Validation dataset is empty: {args.val_manifest}")
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            drop_last=False,
        )
        init_log(val_log_path)

    run_metadata = {
        "command": [sys.executable, *sys.argv],
        "args": vars(args),
        "train_samples": len(dataset),
        "val_samples": len(val_dataset) if val_dataset is not None else 0,
    }
    (dirs["root"] / "run_config.json").write_text(
        json.dumps(run_metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    model = DAMECFusionV1(
        feature_channels=args.feature_channels,
        smoke_prior_mode=args.smoke_prior_mode,
        prior_residual_scale=args.prior_residual_scale,
        bounded_learned_gap=args.bounded_learned_gap,
        equalize_feature_magnitude=args.equalize_feature_magnitude,
        disable_thermal_prior=args.disable_thermal_prior,
        disable_sat_uncertainty=args.disable_sat_uncertainty,
        disable_smoke_prior=args.disable_smoke_prior,
        fixed_equal_weights=args.fixed_equal_weights,
    ).to(device)
    criterion = DAMECFusionLoss(
        lambda_int=args.lambda_int,
        lambda_grad=args.lambda_grad,
        lambda_thermal=args.lambda_thermal,
        lambda_sat=args.lambda_sat,
        gradient_mode=args.gradient_mode,
        intensity_target=args.intensity_target,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        averages = train_one_epoch(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            loader=loader,
            device=device,
            epoch=epoch,
            overfit_batches=args.overfit_batches,
        )

        current_lr = optimizer.param_groups[0]["lr"]
        append_log(log_path, epoch, averages, current_lr)
        print(
            f"Epoch {epoch}/{args.epochs} "
            f"loss_total={averages['loss_total']:.6f} "
            f"loss_int={averages['loss_int']:.6f} "
            f"loss_grad={averages['loss_grad']:.6f} "
            f"loss_thermal={averages['loss_thermal']:.6f} "
            f"loss_sat={averages['loss_sat']:.6f}"
        )

        val_averages = None
        if val_loader is not None:
            val_averages = validate_one_epoch(model, criterion, val_loader, device, epoch)
            append_log(val_log_path, epoch, val_averages, current_lr)
            print(f"Validation {epoch}/{args.epochs} loss_total={val_averages['loss_total']:.6f}")

        selection_loss = (
            val_averages["loss_total"] if val_averages is not None else averages["loss_total"]
        )

        latest_path = dirs["checkpoints"] / "latest.pth"
        save_checkpoint(latest_path, model, optimizer, epoch, selection_loss, args)

        if selection_loss < best_loss:
            best_loss = selection_loss
            save_checkpoint(dirs["checkpoints"] / "best.pth", model, optimizer, epoch, best_loss, args)

        if args.save_interval > 0 and epoch % args.save_interval == 0:
            save_checkpoint(
                dirs["checkpoints"] / f"epoch_{epoch:03d}.pth",
                model,
                optimizer,
                epoch,
                selection_loss,
                args,
            )
            save_debug_outputs(
                model=model,
                batch=debug_batch,
                device=device,
                epoch=epoch,
                dirs=dirs,
                max_images=args.debug_samples,
            )

    print("Training finished.")
    print(f"Run directory: {dirs['root']}")
    print(f"Latest checkpoint: {dirs['checkpoints'] / 'latest.pth'}")
    print(f"Best checkpoint: {dirs['checkpoints'] / 'best.pth'}")
    print(f"Training log: {log_path}")
    if val_loader is not None:
        print(f"Validation log: {val_log_path}")


if __name__ == "__main__":
    main()
