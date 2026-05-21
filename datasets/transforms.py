# """Image transform utilities for DA-MECFusion datasets.
#
# The first-stage DA-MECFusion model consumes aligned infrared grayscale images
# and the visible-light Y channel. The visible RGB image is also converted to
# YCbCr so Cb/Cr can be kept for later color reconstruction.
# """
#
# import random
# from numbers import Integral, Real
# from pathlib import Path
# from typing import Dict, Optional, Tuple, Union
#
# import cv2
# import numpy as np
# import torch
#
#
# PathLike = Union[str, Path]
# SizeLike = Union[int, Tuple[int, int]]
#
#
# def _ensure_numeric_array(img: np.ndarray, name: str) -> None:
#     """Validate that an input is a non-empty numeric numpy array."""
#     if not isinstance(img, np.ndarray):
#         raise TypeError(f"{name} must be a numpy.ndarray, got {type(img).__name__}.")
#     if img.size == 0:
#         raise ValueError(f"{name} must not be empty.")
#     if not np.issubdtype(img.dtype, np.number):
#         raise TypeError(f"{name} must have a numeric dtype, got {img.dtype}.")
#     if not np.all(np.isfinite(img)):
#         raise ValueError(f"{name} contains NaN or infinite values.")
#
#
# def _ensure_gray_image(img: np.ndarray, name: str) -> None:
#     """Validate a grayscale image with shape [H, W]."""
#     _ensure_numeric_array(img, name)
#     if img.ndim != 2:
#         raise ValueError(f"{name} must have shape [H, W], got {img.shape}.")
#
#
# def _ensure_rgb_image(img: np.ndarray, name: str) -> None:
#     """Validate an RGB image with shape [H, W, 3]."""
#     _ensure_numeric_array(img, name)
#     if img.ndim != 3 or img.shape[2] != 3:
#         raise ValueError(f"{name} must have shape [H, W, 3], got {img.shape}.")
#
#
# def _ensure_image_2d_or_3d(img: np.ndarray, name: str) -> None:
#     """Validate an image with shape [H, W] or [H, W, C]."""
#     _ensure_numeric_array(img, name)
#     if img.ndim not in (2, 3):
#         raise ValueError(f"{name} must have shape [H, W] or [H, W, C], got {img.shape}.")
#
#
# def _ensure_same_spatial_size(ir: np.ndarray, vis: np.ndarray, context: str) -> None:
#     """Validate that infrared and visible images share the same H and W."""
#     if ir.shape[:2] != vis.shape[:2]:
#         raise ValueError(
#             f"IR and VIS spatial size mismatch {context}: "
#             f"IR={ir.shape[:2]}, VIS={vis.shape[:2]}."
#         )
#
#
# def _parse_size(size: SizeLike, name: str = "size") -> Tuple[int, int]:
#     """Parse an integer or (height, width) size into a validated tuple."""
#     if isinstance(size, bool):
#         raise TypeError(f"{name} must be an int or a (height, width) tuple, got bool.")
#
#     if isinstance(size, Integral):
#         height = width = int(size)
#     elif isinstance(size, (tuple, list)) and len(size) == 2:
#         if not all(isinstance(value, Integral) and not isinstance(value, bool) for value in size):
#             raise TypeError(f"{name} values must be integers, got {size}.")
#         height, width = int(size[0]), int(size[1])
#     else:
#         raise TypeError(f"{name} must be an int or a (height, width) tuple, got {size}.")
#
#     if height <= 0 or width <= 0:
#         raise ValueError(f"{name} values must be positive, got {(height, width)}.")
#     return height, width
#
#
# def _validate_probability(value: float, name: str) -> None:
#     """Validate a probability in [0, 1]."""
#     if isinstance(value, bool) or not isinstance(value, Real):
#         raise TypeError(f"{name} must be a number in [0, 1], got {type(value).__name__}.")
#     if value < 0.0 or value > 1.0:
#         raise ValueError(f"{name} must be in [0, 1], got {value}.")
#
#
# def _as_float255(img: np.ndarray, name: str) -> np.ndarray:
#     """Convert an image/channel to float32 in [0, 255].
#
#     Floating inputs bounded by [0, 1] are treated as normalized images and
#     scaled up to [0, 255]. This keeps reconstruction utilities convenient when
#     called with tensors converted back to numpy arrays.
#     """
#     _ensure_numeric_array(img, name)
#     out = img.astype(np.float32, copy=False)
#     min_value = float(np.min(out))
#     max_value = float(np.max(out))
#
#     if np.issubdtype(img.dtype, np.floating) and min_value >= 0.0 and max_value <= 1.0:
#         out = out * 255.0
#
#     return np.clip(out, 0.0, 255.0).astype(np.float32, copy=False)
#
#
# def read_image(path: PathLike, mode: str = "gray") -> np.ndarray:
#     """Read an image from disk.
#
#     Args:
#         path: Image file path.
#         mode: ``"gray"`` for a grayscale image with shape [H, W], or
#             ``"rgb"`` for an RGB image with shape [H, W, 3].
#
#     Returns:
#         A numpy array read from disk. Grayscale and RGB images are returned as
#         uint8 arrays in OpenCV's decoded value range.
#
#     Raises:
#         FileNotFoundError: If the path does not exist or is not a file.
#         ValueError: If ``mode`` is unsupported or OpenCV fails to decode it.
#     """
#     if not isinstance(mode, str):
#         raise TypeError(f"mode must be a string, got {type(mode).__name__}.")
#
#     image_path = Path(path)
#     mode = mode.lower()
#
#     if not image_path.is_file():
#         raise FileNotFoundError(f"Image file not found: {image_path}")
#
#     if mode == "gray":
#         img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
#         if img is None:
#             raise ValueError(f"Failed to read grayscale image: {image_path}")
#         return img
#
#     if mode == "rgb":
#         img_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
#         if img_bgr is None:
#             raise ValueError(f"Failed to read RGB image: {image_path}")
#         return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
#
#     raise ValueError(f"Unsupported image mode: {mode}. Expected 'gray' or 'rgb'.")
#
#
# def rgb_to_ycbcr(img_rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
#     """Convert an RGB image to Y, Cb, and Cr channels using BT.601.
#
#     Args:
#         img_rgb: RGB image with shape [H, W, 3]. Values may be uint8 [0, 255],
#             float [0, 255], or normalized float [0, 1].
#
#     Returns:
#         Three float32 arrays ``(y, cb, cr)``, each with shape [H, W] and values
#         clipped to [0, 255].
#     """
#     _ensure_rgb_image(img_rgb, "img_rgb")
#     img = _as_float255(img_rgb, "img_rgb")
#
#     r = img[:, :, 0]
#     g = img[:, :, 1]
#     b = img[:, :, 2]
#
#     y = 0.299 * r + 0.587 * g + 0.114 * b
#     cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 128.0
#     cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 128.0
#
#     return (
#         np.clip(y, 0.0, 255.0).astype(np.float32),
#         np.clip(cb, 0.0, 255.0).astype(np.float32),
#         np.clip(cr, 0.0, 255.0).astype(np.float32),
#     )
#
#
# def ycbcr_to_rgb(y: np.ndarray, cb: np.ndarray, cr: np.ndarray) -> np.ndarray:
#     """Reconstruct an RGB image from Y, Cb, and Cr channels.
#
#     Args:
#         y: Luminance channel with shape [H, W].
#         cb: Blue-difference chroma channel with shape [H, W].
#         cr: Red-difference chroma channel with shape [H, W].
#
#     Returns:
#         A uint8 RGB image with shape [H, W, 3].
#
#     Raises:
#         ValueError: If channel shapes are invalid or inconsistent.
#     """
#     _ensure_gray_image(y, "y")
#     _ensure_gray_image(cb, "cb")
#     _ensure_gray_image(cr, "cr")
#
#     if y.shape != cb.shape or y.shape != cr.shape:
#         raise ValueError(f"YCbCr channel shapes must match, got {y.shape}, {cb.shape}, {cr.shape}.")
#
#     y_255 = _as_float255(y, "y")
#     cb_shifted = _as_float255(cb, "cb") - 128.0
#     cr_shifted = _as_float255(cr, "cr") - 128.0
#
#     r = y_255 + 1.402 * cr_shifted
#     g = y_255 - 0.344136 * cb_shifted - 0.714136 * cr_shifted
#     b = y_255 + 1.772 * cb_shifted
#
#     rgb = np.stack([r, g, b], axis=-1)
#     return np.clip(np.rint(rgb), 0.0, 255.0).astype(np.uint8)
#
#
# def normalize_to_01(img: np.ndarray) -> np.ndarray:
#     """Convert a uint8 or float image to float32 values in [0, 1].
#
#     If the maximum value is greater than 1, the image is divided by 255. The
#     final result is always clipped to [0, 1].
#
#     Args:
#         img: Input image array.
#
#     Returns:
#         A float32 numpy array with values in [0, 1].
#     """
#     _ensure_numeric_array(img, "img")
#     out = img.astype(np.float32, copy=False)
#
#     if float(np.max(out)) > 1.0:
#         out = out / 255.0
#
#     return np.clip(out, 0.0, 1.0).astype(np.float32, copy=False)
#
#
# def to_tensor(img: np.ndarray) -> torch.Tensor:
#     """Convert a numpy image to a float32 torch tensor in CHW layout.
#
#     Args:
#         img: Image with shape [H, W] or [H, W, C].
#
#     Returns:
#         ``torch.float32`` tensor. A grayscale input becomes [1, H, W], while a
#         color input becomes [C, H, W]. Values are normalized to [0, 1].
#     """
#     _ensure_image_2d_or_3d(img, "img")
#     img_01 = normalize_to_01(img)
#
#     if img_01.ndim == 2:
#         chw = img_01[None, :, :]
#     else:
#         chw = np.transpose(img_01, (2, 0, 1))
#
#     return torch.from_numpy(np.ascontiguousarray(chw)).to(dtype=torch.float32)
#
#
# def resize_image(img: np.ndarray, size: SizeLike) -> np.ndarray:
#     """Resize one image to ``size``.
#
#     Args:
#         img: Image with shape [H, W] or [H, W, C].
#         size: Target size as ``(height, width)`` or an integer for a square.
#
#     Returns:
#         Resized numpy array.
#     """
#     _ensure_image_2d_or_3d(img, "img")
#     target_h, target_w = _parse_size(size)
#     resized = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
#     return np.ascontiguousarray(resized)
#
#
# def resize_pair(ir: np.ndarray, vis: np.ndarray, size: SizeLike) -> Tuple[np.ndarray, np.ndarray]:
#     """Synchronously resize infrared and visible images.
#
#     Args:
#         ir: Infrared grayscale image with shape [H, W].
#         vis: Visible image with shape [H, W] or [H, W, C].
#         size: Target size as ``(height, width)`` or an integer for a square.
#
#     Returns:
#         ``(ir_resized, vis_resized)`` with identical spatial dimensions.
#     """
#     _ensure_gray_image(ir, "ir")
#     _ensure_image_2d_or_3d(vis, "vis")
#     target_size = _parse_size(size)
#
#     ir_resized = resize_image(ir, target_size)
#     vis_resized = resize_image(vis, target_size)
#     _ensure_same_spatial_size(ir_resized, vis_resized, "after resize")
#     return ir_resized, vis_resized
#
#
# def random_crop_pair(
#     ir: np.ndarray,
#     vis: np.ndarray,
#     crop_size: SizeLike,
# ) -> Tuple[np.ndarray, np.ndarray]:
#     """Synchronously random-crop infrared and visible images.
#
#     The same ``top`` and ``left`` coordinates are used for both arrays, keeping
#     pixel-level alignment intact.
#
#     Args:
#         ir: Infrared grayscale image with shape [H, W].
#         vis: Visible image with shape [H, W] or [H, W, C].
#         crop_size: Crop size as ``(height, width)`` or an integer for a square.
#
#     Returns:
#         Cropped ``(ir, vis)`` arrays.
#
#     Raises:
#         ValueError: If spatial sizes differ or the crop is larger than the
#             current image size.
#     """
#     _ensure_gray_image(ir, "ir")
#     _ensure_image_2d_or_3d(vis, "vis")
#     _ensure_same_spatial_size(ir, vis, "before random crop")
#
#     crop_h, crop_w = _parse_size(crop_size, "crop_size")
#     h, w = ir.shape[:2]
#
#     if crop_h > h or crop_w > w:
#         raise ValueError(
#             f"crop_size {(crop_h, crop_w)} is larger than image size {(h, w)}."
#         )
#
#     if crop_h == h and crop_w == w:
#         return np.ascontiguousarray(ir), np.ascontiguousarray(vis)
#
#     top = random.randint(0, h - crop_h)
#     left = random.randint(0, w - crop_w)
#
#     ir_crop = ir[top : top + crop_h, left : left + crop_w]
#     vis_crop = vis[top : top + crop_h, left : left + crop_w, ...]
#
#     return np.ascontiguousarray(ir_crop), np.ascontiguousarray(vis_crop)
#
#
# def center_crop_pair(
#     ir: np.ndarray,
#     vis: np.ndarray,
#     crop_size: SizeLike,
# ) -> Tuple[np.ndarray, np.ndarray]:
#     """Synchronously center-crop infrared and visible images.
#
#     Args:
#         ir: Infrared grayscale image with shape [H, W].
#         vis: Visible image with shape [H, W] or [H, W, C].
#         crop_size: Crop size as ``(height, width)`` or an integer for a square.
#
#     Returns:
#         Cropped ``(ir, vis)`` arrays using exactly the same crop region.
#
#     Raises:
#         ValueError: If spatial sizes differ or the crop is larger than the
#             current image size.
#     """
#     _ensure_gray_image(ir, "ir")
#     _ensure_image_2d_or_3d(vis, "vis")
#     _ensure_same_spatial_size(ir, vis, "before center crop")
#
#     crop_h, crop_w = _parse_size(crop_size, "crop_size")
#     h, w = ir.shape[:2]
#
#     if crop_h > h or crop_w > w:
#         raise ValueError(
#             f"crop_size {(crop_h, crop_w)} is larger than image size {(h, w)}."
#         )
#
#     top = (h - crop_h) // 2
#     left = (w - crop_w) // 2
#
#     ir_crop = ir[top : top + crop_h, left : left + crop_w]
#     vis_crop = vis[top : top + crop_h, left : left + crop_w, ...]
#
#     return np.ascontiguousarray(ir_crop), np.ascontiguousarray(vis_crop)
#
#
# def random_flip_pair(
#     ir: np.ndarray,
#     vis: np.ndarray,
#     horizontal_prob: float = 0.5,
#     vertical_prob: float = 0.0,
# ) -> Tuple[np.ndarray, np.ndarray]:
#     """Synchronously random-flip infrared and visible images.
#
#     Args:
#         ir: Infrared grayscale image with shape [H, W].
#         vis: Visible image with shape [H, W] or [H, W, C].
#         horizontal_prob: Probability of left-right flipping.
#         vertical_prob: Probability of top-bottom flipping. Defaults to 0.
#
#     Returns:
#         Flipped arrays with contiguous memory layout.
#     """
#     _ensure_gray_image(ir, "ir")
#     _ensure_image_2d_or_3d(vis, "vis")
#     _ensure_same_spatial_size(ir, vis, "before random flip")
#     _validate_probability(horizontal_prob, "horizontal_prob")
#     _validate_probability(vertical_prob, "vertical_prob")
#
#     if random.random() < horizontal_prob:
#         ir = np.flip(ir, axis=1)
#         vis = np.flip(vis, axis=1)
#
#     if random.random() < vertical_prob:
#         ir = np.flip(ir, axis=0)
#         vis = np.flip(vis, axis=0)
#
#     return np.ascontiguousarray(ir), np.ascontiguousarray(vis)
#
#
# def _pack_pair_tensors(ir: np.ndarray, vis_rgb: np.ndarray) -> Dict[str, torch.Tensor]:
#     """Convert aligned IR/RGB arrays into the tensor dictionary used by models."""
#     vis_y, cb, cr = rgb_to_ycbcr(vis_rgb)
#
#     return {
#         "ir": to_tensor(ir),
#         "vis": to_tensor(vis_rgb),
#         "vis_y": to_tensor(vis_y),
#         "cb": to_tensor(cb),
#         "cr": to_tensor(cr),
#     }
#
#
# def prepare_train_pair(
#     ir: np.ndarray,
#     vis_rgb: np.ndarray,
#     img_size: Optional[SizeLike] = 256,
#     use_random_crop: bool = True,
#     use_random_flip: bool = True,
# ) -> Dict[str, torch.Tensor]:
#     """Prepare one aligned infrared-visible pair for training.
#
#     Args:
#         ir: Infrared grayscale image with shape [H, W].
#         vis_rgb: Visible RGB image with shape [H, W, 3].
#         img_size: Target size. If ``None``, the original size is kept. An int
#             means a square target; a tuple means ``(height, width)``.
#         use_random_crop: If true and the image is large enough, use a
#             synchronized random crop. If the image is smaller than ``img_size``,
#             resize both images to ``img_size``.
#         use_random_flip: If true, apply synchronized random flipping.
#
#     Returns:
#         Dictionary with float32 tensors in [0, 1]:
#         ``ir`` [1, H, W], ``vis`` [3, H, W], ``vis_y`` [1, H, W],
#         ``cb`` [1, H, W], and ``cr`` [1, H, W].
#     """
#     _ensure_gray_image(ir, "ir")
#     _ensure_rgb_image(vis_rgb, "vis_rgb")
#     _ensure_same_spatial_size(ir, vis_rgb, "before train preparation")
#
#     if img_size is not None:
#         target_size = _parse_size(img_size, "img_size")
#         target_h, target_w = target_size
#         h, w = ir.shape[:2]
#
#         if use_random_crop:
#             if h >= target_h and w >= target_w:
#                 ir, vis_rgb = random_crop_pair(ir, vis_rgb, target_size)
#             else:
#                 ir, vis_rgb = resize_pair(ir, vis_rgb, target_size)
#         else:
#             ir, vis_rgb = resize_pair(ir, vis_rgb, target_size)
#
#     if use_random_flip:
#         ir, vis_rgb = random_flip_pair(ir, vis_rgb)
#
#     return _pack_pair_tensors(ir, vis_rgb)
#
#
# def prepare_test_pair(
#     ir: np.ndarray,
#     vis_rgb: np.ndarray,
#     img_size: Optional[SizeLike] = None,
# ) -> Dict[str, torch.Tensor]:
#     """Prepare one aligned infrared-visible pair for validation or testing.
#
#     No random crop or random flip is used in this function.
#
#     Args:
#         ir: Infrared grayscale image with shape [H, W].
#         vis_rgb: Visible RGB image with shape [H, W, 3].
#         img_size: Optional target size. If ``None``, the original size is kept.
#
#     Returns:
#         Dictionary with float32 tensors in [0, 1]:
#         ``ir`` [1, H, W], ``vis`` [3, H, W], ``vis_y`` [1, H, W],
#         ``cb`` [1, H, W], and ``cr`` [1, H, W].
#     """
#     _ensure_gray_image(ir, "ir")
#     _ensure_rgb_image(vis_rgb, "vis_rgb")
#     _ensure_same_spatial_size(ir, vis_rgb, "before test preparation")
#
#     if img_size is not None:
#         ir, vis_rgb = resize_pair(ir, vis_rgb, _parse_size(img_size, "img_size"))
#
#     return _pack_pair_tensors(ir, vis_rgb)
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
