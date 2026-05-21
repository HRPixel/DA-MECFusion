"""Visualization utility for DA-MECFusion V1 prior maps."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _to_numpy(x: Any) -> np.ndarray:
    if x is None:
        raise ValueError("Input cannot be None")
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    else:
        x = np.asarray(x)
    return x


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


def _sanitize_name(name: Any) -> str:
    name = str(name)
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in name)


def _name_for_index(names: Any, index: int) -> str:
    if names is None:
        return f"{index:03d}"
    if hasattr(names, "detach"):
        names = names.detach().cpu().tolist()
    if isinstance(names, np.ndarray):
        names = names.tolist()
    if isinstance(names, (list, tuple)):
        return _sanitize_name(names[index])
    return _sanitize_name(names)


def _save_gray(path: Path, image: np.ndarray) -> None:
    plt.imsave(path, _normalize_0_1(image), cmap="gray", vmin=0.0, vmax=1.0)


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


def _save_comparison(
    path: Path,
    ir_image: np.ndarray,
    vis_image: np.ndarray,
    thermal_image: np.ndarray,
    sat_image: np.ndarray,
    smoke_image: np.ndarray,
) -> None:
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
    fig.savefig(path, dpi=150)
    plt.close(fig)


def visualize_priors(
    thermal_prior: Any,
    sat_uncertainty: Any,
    smoke_prior: Any,
    ir: Any = None,
    vis: Any = None,
    names: Any = None,
    save_dir: str | Path = "debug_outputs/priors",
    max_images: int = 8,
) -> None:
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

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

    num_images = min(batch_size, max_images)
    for index in range(num_images):
        name = _name_for_index(names, index)
        thermal_image = _prepare_gray(thermal_batch, index)
        sat_image = _prepare_gray(sat_batch, index)
        smoke_image = _prepare_gray(smoke_batch, index)

        _save_gray(save_dir / f"thermal_prior_{name}.png", thermal_image)
        _save_gray(save_dir / f"sat_uncertainty_{name}.png", sat_image)
        _save_gray(save_dir / f"smoke_prior_{name}.png", smoke_image)

        if ir_batch is not None and vis_batch is not None:
            ir_image = _prepare_gray(ir_batch, index)
            vis_image = _prepare_visible(vis_batch, index)
            _save_comparison(
                save_dir / f"prior_comparison_{name}.png",
                ir_image,
                vis_image,
                thermal_image,
                sat_image,
                smoke_image,
            )

    print(f"Saved {num_images} prior visualization sample(s) to: {save_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize DA-MECFusion prior maps.")
    parser.add_argument("--input_pt", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="debug_outputs/priors")
    parser.add_argument("--max_images", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import torch

    data = torch.load(args.input_pt, map_location="cpu")
    visualize_priors(
        thermal_prior=data["thermal_prior"],
        sat_uncertainty=data["sat_uncertainty"],
        smoke_prior=data["smoke_prior"],
        ir=data.get("ir"),
        vis=data.get("vis"),
        names=data.get("names"),
        save_dir=args.save_dir,
        max_images=args.max_images,
    )


if __name__ == "__main__":
    main()
