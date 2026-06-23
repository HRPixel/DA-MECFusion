"""Basic image preprocessing utilities for DA-MECFusion V1.

V1 uses deterministic preprocessing only: image loading, optional paired resize,
tensor conversion, and RGB-to-luminance conversion. Legacy transform code below
is kept commented for reference.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def load_image(path: str | Path, mode: str) -> Image.Image:
    """Load an image with PIL and convert it to mode 'L' or 'RGB'."""
    if mode not in {"L", "RGB"}:
        raise ValueError(f"mode must be 'L' or 'RGB', got {mode!r}")

    image_path = Path(path)
    if not image_path.is_file():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    return Image.open(image_path).convert(mode)


def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    """Convert a PIL image to a float32 tensor in [0, 1]."""
    if not isinstance(img, Image.Image):
        raise TypeError(f"img must be a PIL Image, got {type(img).__name__}")

    array = np.asarray(img, dtype=np.float32) / 255.0
    if array.ndim == 2:
        array = array[None, :, :]
    elif array.ndim == 3 and array.shape[2] == 3:
        array = np.transpose(array, (2, 0, 1))
    else:
        raise ValueError(f"Unsupported PIL image array shape: {array.shape}")

    array = np.ascontiguousarray(np.clip(array, 0.0, 1.0))
    return torch.from_numpy(array).to(dtype=torch.float32)


def resize_pair(
    ir_img: Image.Image,
    vis_img: Image.Image,
    size: tuple[int, int],
) -> tuple[Image.Image, Image.Image]:
    """Resize aligned IR/VIS PIL images to the same (height, width)."""
    if not isinstance(ir_img, Image.Image) or not isinstance(vis_img, Image.Image):
        raise TypeError("ir_img and vis_img must both be PIL Images")
    if len(size) != 2:
        raise ValueError("size must be a (height, width) tuple")

    height, width = int(size[0]), int(size[1])
    if height <= 0 or width <= 0:
        raise ValueError(f"height and width must be positive, got {size}")

    pil_size = (width, height)
    return (
        ir_img.resize(pil_size, resample=Image.BILINEAR),
        vis_img.resize(pil_size, resample=Image.BILINEAR),
    )


def rgb_to_y_tensor(vis: torch.Tensor) -> torch.Tensor:
    """Convert RGB tensor [3,H,W] or [B,3,H,W] to luminance Y."""
    if vis.dim() == 3:
        if vis.size(0) != 3:
            raise ValueError(f"Expected vis with shape [3,H,W], got {tuple(vis.shape)}")
        r = vis[0:1]
        g = vis[1:2]
        b = vis[2:3]
        return 0.299 * r + 0.587 * g + 0.114 * b

    if vis.dim() == 4:
        if vis.size(1) != 3:
            raise ValueError(f"Expected vis with shape [B,3,H,W], got {tuple(vis.shape)}")
        r = vis[:, 0:1]
        g = vis[:, 1:2]
        b = vis[:, 2:3]
        return 0.299 * r + 0.587 * g + 0.114 * b

    raise ValueError(f"Expected vis with 3 or 4 dims, got {tuple(vis.shape)}")


def preprocess_pair(
    ir_path: str | Path,
    vis_path: str | Path,
    height: int | None = None,
    width: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load and preprocess one aligned infrared-visible image pair."""
    ir_img = load_image(ir_path, mode="L")
    vis_img = load_image(vis_path, mode="RGB")

    if (height is None) != (width is None):
        raise ValueError("height and width must be provided together")
    if height is not None and width is not None:
        ir_img, vis_img = resize_pair(ir_img, vis_img, size=(height, width))

    ir_tensor = pil_to_tensor(ir_img)
    vis_tensor = pil_to_tensor(vis_img)
    vis_y_tensor = rgb_to_y_tensor(vis_tensor)
    return ir_tensor, vis_tensor, vis_y_tensor


def _print_tensor_stats(name: str, tensor: torch.Tensor) -> None:
    print(
        f"{name}: shape={tuple(tensor.shape)}, "
        f"min={tensor.min().item():.6f}, max={tensor.max().item():.6f}"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test DA-MECFusion V1 preprocessing.")
    parser.add_argument("--ir_path", type=str, required=True)
    parser.add_argument("--vis_path", type=str, required=True)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    ir, vis, vis_y = preprocess_pair(
        args.ir_path,
        args.vis_path,
        height=args.height,
        width=args.width,
    )
    _print_tensor_stats("ir", ir)
    _print_tensor_stats("vis", vis)
    _print_tensor_stats("vis_y", vis_y)
