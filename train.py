"""Training script for DA-MECFusion V1 on MSRS."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
from tools.visualize_priors import visualize_priors  # noqa: E402
from tools.visualize_weights import visualize_mec_weights  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DA-MECFusion V1 on MSRS.")
    parser.add_argument("--data_root", type=str, default="data/processed/MSRS")
    parser.add_argument("--save_dir", type=str, default="experiments/damecfusion_v1_msrs")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_interval", type=int, default=5)
    parser.add_argument("--debug_samples", type=int, default=8)
    parser.add_argument("--feature_channels", type=int, default=64)
    parser.add_argument("--lambda_int", type=float, default=1.0)
    parser.add_argument("--lambda_grad", type=float, default=10.0)
    parser.add_argument("--lambda_thermal", type=float, default=1.0)
    parser.add_argument("--lambda_sat", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overfit_batches", type=int, default=0)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested_device: str) -> torch.device:
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        print(f"Requested device '{requested_device}' is unavailable. Falling back to cpu.")
        return torch.device("cpu")
    return torch.device(requested_device)


def prepare_dirs(save_dir: Path) -> dict[str, Path]:
    dirs = {
        "root": save_dir,
        "checkpoints": save_dir / "checkpoints",
        "logs": save_dir / "logs",
        "debug_outputs": save_dir / "debug_outputs",
    }
    for directory in dirs.values():
        directory.mkdir(parents=True, exist_ok=True)
    return dirs


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        moved[key] = value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
    return moved


def get_names(batch: dict[str, Any], count: int) -> list[str]:
    names = batch.get("name")
    if names is None:
        return [f"{index:03d}" for index in range(count)]
    if isinstance(names, (list, tuple)):
        return [str(name) for name in names[:count]]
    return [str(names)]


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
    torch.save(checkpoint, path)


def normalize_image(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    return np.clip(image, 0.0, 1.0)


def sanitize_name(name: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in name)


def save_fused_images(
    fused: torch.Tensor,
    names: list[str],
    save_dir: Path,
    max_images: int,
) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    num_images = min(fused.shape[0], max_images)
    fused_cpu = fused.detach().cpu()
    for index in range(num_images):
        name = sanitize_name(names[index] if index < len(names) else f"{index:03d}")
        image = normalize_image(fused_cpu[index, 0].numpy())
        plt.imsave(save_dir / f"fused_{name}.png", image, cmap="gray", vmin=0.0, vmax=1.0)


def save_debug_outputs(
    model: DAMECFusionV1,
    batch: dict[str, Any],
    device: torch.device,
    epoch: int,
    debug_dir: Path,
    max_images: int,
) -> None:
    model.eval()
    batch = move_batch_to_device(batch, device)
    ir = batch["ir"]
    vis = batch["vis"]
    names = get_names(batch, ir.shape[0])
    epoch_dir = debug_dir / f"epoch_{epoch:03d}"

    with torch.no_grad():
        outputs = model(ir, vis)

    save_fused_images(outputs["fused"], names, epoch_dir / "fused", max_images)
    visualize_priors(
        outputs["thermal_prior"],
        outputs["sat_uncertainty"],
        outputs["smoke_prior"],
        ir=ir,
        vis=vis,
        names=names,
        save_dir=epoch_dir / "priors",
        max_images=max_images,
    )
    visualize_mec_weights(
        outputs["w_ir"],
        outputs["w_vis"],
        ir=ir,
        vis=vis,
        names=names,
        save_dir=epoch_dir / "weights",
        max_images=max_images,
    )
    model.train()


def init_log(csv_path: Path) -> None:
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "epoch",
                "loss_total",
                "loss_int",
                "loss_grad",
                "loss_thermal",
                "loss_sat",
                "lr",
            ]
        )


def append_log(csv_path: Path, epoch: int, averages: dict[str, float], lr: float) -> None:
    with csv_path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                epoch,
                averages["loss_total"],
                averages["loss_int"],
                averages["loss_grad"],
                averages["loss_thermal"],
                averages["loss_sat"],
                lr,
            ]
        )


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

        if torch.isnan(loss_total):
            raise FloatingPointError(f"NaN loss detected at epoch {epoch}, step {step}")

        loss_total.backward()
        optimizer.step()

        for key in totals:
            totals[key] += float(loss_dict[key].detach().cpu().item())
        num_steps += 1

        progress.set_postfix(loss=f"{loss_total.item():.4f}")

    if num_steps == 0:
        raise RuntimeError("No training steps were executed")

    return {key: value / num_steps for key, value in totals.items()}


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    dirs = prepare_dirs(Path(args.save_dir))
    log_path = dirs["logs"] / "train_log.csv"
    init_log(log_path)

    dataset = MSRSDataset(
        data_root=args.data_root,
        split="train",
        height=args.height,
        width=args.width,
        use_label=False,
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

    model = DAMECFusionV1(feature_channels=args.feature_channels).to(device)
    criterion = DAMECFusionLoss(
        lambda_int=args.lambda_int,
        lambda_grad=args.lambda_grad,
        lambda_thermal=args.lambda_thermal,
        lambda_sat=args.lambda_sat,
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

        latest_path = dirs["checkpoints"] / "latest.pth"
        save_checkpoint(latest_path, model, optimizer, epoch, averages["loss_total"], args)

        if averages["loss_total"] < best_loss:
            best_loss = averages["loss_total"]
            save_checkpoint(dirs["checkpoints"] / "best.pth", model, optimizer, epoch, best_loss, args)

        if args.save_interval > 0 and epoch % args.save_interval == 0:
            save_checkpoint(
                dirs["checkpoints"] / f"epoch_{epoch:03d}.pth",
                model,
                optimizer,
                epoch,
                averages["loss_total"],
                args,
            )
            save_debug_outputs(
                model=model,
                batch=debug_batch,
                device=device,
                epoch=epoch,
                debug_dir=dirs["debug_outputs"],
                max_images=args.debug_samples,
            )

    print("Training finished.")
    print(f"Latest checkpoint: {dirs['checkpoints'] / 'latest.pth'}")
    print(f"Best checkpoint: {dirs['checkpoints'] / 'best.pth'}")
    print(f"Training log: {log_path}")


if __name__ == "__main__":
    main()
