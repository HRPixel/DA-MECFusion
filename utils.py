"""Shared runtime utilities for DA-MECFusion V1-minimal."""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


def _get_torch() -> Any:
    import torch

    return torch


def _get_pyplot() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def sanitize_name(name: Any) -> str:
    """Return a filesystem-safe sample or run name."""
    name = str(name)
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in name)


def resolve_device(requested_device: str) -> torch.device:
    """Resolve a requested torch device with CPU fallback for unavailable CUDA."""
    torch = _get_torch()
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        print(f"Requested device '{requested_device}' is unavailable. Falling back to cpu.")
        return torch.device("cpu")
    return torch.device(requested_device)


def create_run_dir(
    run_root: str | Path = "runs",
    run_name: str = "DA-MECFusion_v1_MSRS",
    timestamp: str | None = None,
) -> Path:
    """Create a timestamped run directory under run_root."""
    timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(run_root) / f"{sanitize_name(run_name)}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def prepare_run_dirs(run_dir: str | Path) -> dict[str, Path]:
    """Create the minimal shared output tree for one run."""
    root = Path(run_dir)
    dirs = {
        "root": root,
        "checkpoints": root / "checkpoints",
        "outputs": root / "outputs",
        "fused": root / "outputs" / "fused",
        "priors": root / "outputs" / "priors",
        "thermal": root / "outputs" / "priors" / "thermal",
        "saturation": root / "outputs" / "priors" / "saturation",
        "smoke": root / "outputs" / "priors" / "smoke",
        "weights": root / "outputs" / "weights",
        "w_ir": root / "outputs" / "weights" / "w_ir",
        "w_vis": root / "outputs" / "weights" / "w_vis",
        "comparisons": root / "outputs" / "comparisons",
        "prior_comparisons": root / "outputs" / "comparisons" / "priors",
        "weight_comparisons": root / "outputs" / "comparisons" / "weights",
        "logs": root / "logs",
        "metrics": root / "metrics",
    }
    for directory in dirs.values():
        directory.mkdir(parents=True, exist_ok=True)
    return dirs


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """Move tensor values in a dataloader batch to device."""
    torch = _get_torch()
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        moved[key] = value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
    return moved


def get_names(batch: dict[str, Any], count: int) -> list[str]:
    """Get sanitized sample names from a dataloader batch."""
    names = batch.get("name")
    if names is None:
        return [f"{index:03d}" for index in range(count)]
    if hasattr(names, "detach"):
        names = names.detach().cpu().tolist()
    if isinstance(names, np.ndarray):
        names = names.tolist()
    if isinstance(names, (list, tuple)):
        return [sanitize_name(name) for name in names[:count]]
    return [sanitize_name(names)]


def save_tensor_image(tensor: torch.Tensor, path: str | Path) -> None:
    """Save a single-channel tensor image as 8-bit grayscale PNG-compatible data."""
    from PIL import Image

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image = tensor.detach().cpu().clamp(0.0, 1.0).squeeze().numpy()
    image_8bit = np.clip(image * 255.0, 0.0, 255.0).round().astype(np.uint8)
    Image.fromarray(image_8bit, mode="L").save(path)


def save_fused_batch(
    fused: torch.Tensor,
    names: list[str],
    save_dir: str | Path,
    max_images: int | None = None,
) -> None:
    """Save a batch of fused single-channel images."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    limit = fused.shape[0] if max_images is None else min(fused.shape[0], max_images)
    for index in range(limit):
        name = sanitize_name(names[index] if index < len(names) else f"{index:03d}")
        save_tensor_image(fused[index, 0], save_dir / f"{name}.png")


def save_single_maps(
    batch_tensor: torch.Tensor,
    names: list[str],
    save_dir: str | Path,
    max_images: int | None = None,
) -> None:
    """Save a batch of single-channel prior or weight maps."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    limit = batch_tensor.shape[0] if max_images is None else min(batch_tensor.shape[0], max_images)
    for index in range(limit):
        name = sanitize_name(names[index] if index < len(names) else f"{index:03d}")
        save_tensor_image(batch_tensor[index, 0], save_dir / f"{name}.png")


def _to_numpy(x: Any) -> np.ndarray:
    if x is None:
        raise ValueError("Input cannot be None")
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _to_batch_array(x: Any, channels: int | None = None) -> np.ndarray:
    array = _to_numpy(x).astype(np.float32)
    if array.ndim == 2:
        array = array[None, None, :, :]
    elif array.ndim == 3:
        if channels is not None and array.shape[0] == channels:
            array = array[None, :, :, :]
        elif channels is not None and array.shape[-1] == channels:
            array = np.transpose(array, (2, 0, 1))[None, :, :, :]
        else:
            array = array[:, None, :, :]
    elif array.ndim == 4:
        pass
    else:
        raise ValueError(f"Expected input with 2, 3, or 4 dims, got shape {array.shape}")

    if channels is not None and array.shape[1] != channels:
        raise ValueError(f"Expected {channels} channel(s), got shape {array.shape}")
    return array


def _normalize_0_1(image: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    min_value = float(np.nanmin(image))
    max_value = float(np.nanmax(image))
    image = (image - min_value) / (max_value - min_value + eps)
    return np.clip(image, 0.0, 1.0)


def _name_for_index(names: Any, index: int) -> str:
    if names is None:
        return f"{index:03d}"
    if hasattr(names, "detach"):
        names = names.detach().cpu().tolist()
    if isinstance(names, np.ndarray):
        names = names.tolist()
    if isinstance(names, (list, tuple)):
        return sanitize_name(names[index])
    return sanitize_name(names)


def _prepare_gray(batch: np.ndarray, index: int) -> np.ndarray:
    return _normalize_0_1(batch[index, 0])


def _prepare_visible(vis_batch: np.ndarray, index: int) -> np.ndarray:
    image = vis_batch[index]
    if image.shape[0] == 1:
        return _normalize_0_1(image[0])
    if image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))
        return _normalize_0_1(image)
    raise ValueError(f"Visible image must have 1 or 3 channels, got {image.shape[0]}")


def _save_gray(path: Path, image: np.ndarray) -> None:
    plt = _get_pyplot()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(path, _normalize_0_1(image), cmap="gray", vmin=0.0, vmax=1.0)


def _save_prior_comparison(
    path: Path,
    ir_image: np.ndarray,
    vis_image: np.ndarray,
    thermal_image: np.ndarray,
    sat_image: np.ndarray,
    smoke_image: np.ndarray,
) -> None:
    plt = _get_pyplot()
    titles = ["Infrared", "Visible", "Thermal", "Saturation", "Smoke"]
    images = [ir_image, vis_image, thermal_image, sat_image, smoke_image]
    fig, axes = plt.subplots(1, 5, figsize=(15, 3.2))
    for axis, title, image in zip(axes, titles, images):
        if image.ndim == 2:
            axis.imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
        else:
            axis.imshow(image, vmin=0.0, vmax=1.0)
        axis.set_title(title)
        axis.axis("off")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _save_weight_comparison(
    path: Path,
    ir_image: np.ndarray,
    vis_image: np.ndarray,
    w_ir_image: np.ndarray,
    w_vis_image: np.ndarray,
) -> None:
    plt = _get_pyplot()
    titles = ["Infrared", "Visible", "W_ir", "W_vis"]
    images = [ir_image, vis_image, w_ir_image, w_vis_image]
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.2))
    for axis, title, image in zip(axes, titles, images):
        if image.ndim == 2:
            axis.imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
        else:
            axis.imshow(image, vmin=0.0, vmax=1.0)
        axis.set_title(title)
        axis.axis("off")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def visualize_priors(
    thermal_prior: Any,
    sat_uncertainty: Any,
    smoke_prior: Any,
    ir: Any = None,
    vis: Any = None,
    names: Any = None,
    save_dir: str | Path | None = "outputs/priors",
    comparison_dir: str | Path | None = None,
    max_images: int = 8,
) -> None:
    """Save thermal, saturation, smoke prior maps and optional comparison images."""
    thermal_batch = _to_batch_array(thermal_prior, channels=1)
    sat_batch = _to_batch_array(sat_uncertainty, channels=1)
    smoke_batch = _to_batch_array(smoke_prior, channels=1)

    batch_size = thermal_batch.shape[0]
    if sat_batch.shape[0] != batch_size or smoke_batch.shape[0] != batch_size:
        raise ValueError("thermal_prior, sat_uncertainty, and smoke_prior must share batch size")

    ir_batch = _to_batch_array(ir, channels=1) if ir is not None else None
    vis_batch = _to_batch_array(vis, channels=None) if vis is not None else None
    if (ir_batch is None) != (vis_batch is None):
        raise ValueError("ir and vis must be provided together for comparison images")

    save_root = Path(save_dir) if save_dir is not None else None
    comparison_root = Path(comparison_dir) if comparison_dir is not None else None
    num_images = min(batch_size, max_images)
    for index in range(num_images):
        name = _name_for_index(names, index)
        thermal_image = _prepare_gray(thermal_batch, index)
        sat_image = _prepare_gray(sat_batch, index)
        smoke_image = _prepare_gray(smoke_batch, index)

        if save_root is not None:
            _save_gray(save_root / "thermal" / f"{name}.png", thermal_image)
            _save_gray(save_root / "saturation" / f"{name}.png", sat_image)
            _save_gray(save_root / "smoke" / f"{name}.png", smoke_image)

        if comparison_root is not None and ir_batch is not None and vis_batch is not None:
            _save_prior_comparison(
                comparison_root / f"{name}.png",
                _prepare_gray(ir_batch, index),
                _prepare_visible(vis_batch, index),
                thermal_image,
                sat_image,
                smoke_image,
            )


def visualize_mec_weights(
    w_ir: Any,
    w_vis: Any,
    ir: Any = None,
    vis: Any = None,
    names: Any = None,
    save_dir: str | Path | None = "outputs/weights",
    comparison_dir: str | Path | None = None,
    max_images: int = 8,
) -> None:
    """Save MEC weight maps and optional comparison images."""
    w_ir_batch = _to_batch_array(w_ir, channels=1)
    w_vis_batch = _to_batch_array(w_vis, channels=1)
    if w_ir_batch.shape[0] != w_vis_batch.shape[0]:
        raise ValueError("w_ir and w_vis must share batch size")

    ir_batch = _to_batch_array(ir, channels=1) if ir is not None else None
    vis_batch = _to_batch_array(vis, channels=None) if vis is not None else None
    if (ir_batch is None) != (vis_batch is None):
        raise ValueError("ir and vis must be provided together for comparison images")

    save_root = Path(save_dir) if save_dir is not None else None
    comparison_root = Path(comparison_dir) if comparison_dir is not None else None
    num_images = min(w_ir_batch.shape[0], max_images)
    for index in range(num_images):
        name = _name_for_index(names, index)
        w_ir_image = _prepare_gray(w_ir_batch, index)
        w_vis_image = _prepare_gray(w_vis_batch, index)

        if save_root is not None:
            _save_gray(save_root / "w_ir" / f"{name}.png", w_ir_image)
            _save_gray(save_root / "w_vis" / f"{name}.png", w_vis_image)

        if comparison_root is not None and ir_batch is not None and vis_batch is not None:
            _save_weight_comparison(
                comparison_root / f"{name}.png",
                _prepare_gray(ir_batch, index),
                _prepare_visible(vis_batch, index),
                w_ir_image,
                w_vis_image,
            )


def init_log(csv_path: str | Path) -> None:
    """Create a training log CSV with the fixed V1-minimal loss columns."""
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
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


def append_log(csv_path: str | Path, epoch: int, averages: dict[str, float], lr: float) -> None:
    """Append one training epoch row to the fixed V1-minimal training log."""
    csv_path = Path(csv_path)
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
