"""Basic fusion metrics for DA-MECFusion V1."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import ndimage


def _to_numpy_image(image: Any) -> np.ndarray:
    if hasattr(image, "detach"):
        image = image.detach().cpu().numpy()
    else:
        image = np.asarray(image)
    image = np.squeeze(image).astype(np.float64)
    if image.ndim != 2:
        raise ValueError(f"Expected a grayscale image after squeeze, got shape {image.shape}")
    return np.clip(image, 0.0, 1.0)


def entropy(image: np.ndarray, bins: int = 256, eps: float = 1e-12) -> float:
    hist, _ = np.histogram(image, bins=bins, range=(0.0, 1.0), density=False)
    prob = hist.astype(np.float64)
    prob = prob / (prob.sum() + eps)
    value = -np.sum(prob * np.log2(prob + eps))
    return float(np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0))


def standard_deviation(image: np.ndarray) -> float:
    return float(np.nan_to_num(np.std(image), nan=0.0, posinf=0.0, neginf=0.0))


def average_gradient(image: np.ndarray) -> float:
    dx = np.diff(image, axis=1)
    dy = np.diff(image, axis=0)
    dx_crop = dx[:-1, :]
    dy_crop = dy[:, :-1]
    gradient = np.sqrt((dx_crop * dx_crop + dy_crop * dy_crop) / 2.0)
    value = np.mean(gradient) if gradient.size > 0 else 0.0
    return float(np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0))


def mutual_information(
    image_a: np.ndarray,
    image_b: np.ndarray,
    bins: int = 256,
    eps: float = 1e-12,
) -> float:
    hist_2d, _, _ = np.histogram2d(
        image_a.ravel(),
        image_b.ravel(),
        bins=bins,
        range=((0.0, 1.0), (0.0, 1.0)),
    )
    pxy = hist_2d.astype(np.float64)
    pxy = pxy / (pxy.sum() + eps)
    px = pxy.sum(axis=1, keepdims=True)
    py = pxy.sum(axis=0, keepdims=True)
    value = np.sum(pxy * np.log2((pxy + eps) / (px @ py + eps)))
    return float(np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0))


def sobel_magnitude(image: np.ndarray) -> np.ndarray:
    gx = ndimage.sobel(image, axis=1, mode="reflect")
    gy = ndimage.sobel(image, axis=0, mode="reflect")
    return np.sqrt(gx * gx + gy * gy)


def gradient_correlation(grad_a: np.ndarray, grad_b: np.ndarray, eps: float = 1e-12) -> float:
    a = grad_a.ravel().astype(np.float64)
    b = grad_b.ravel().astype(np.float64)
    a = a - np.mean(a)
    b = b - np.mean(b)
    denominator = np.sqrt(np.sum(a * a) * np.sum(b * b)) + eps
    value = np.sum(a * b) / denominator
    value = (value + 1.0) / 2.0
    return float(np.clip(np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0), 0.0, 1.0))


def qabf_v1_approx(ir: np.ndarray, vis: np.ndarray, fused: np.ndarray) -> float:
    """Approximate V1 edge-preservation score.

    This is not the standard Qabf implementation. V1 uses the average Sobel
    gradient correlation between fused/IR and fused/VIS as a lightweight
    placeholder. It can be replaced by the standard Qabf formula later.
    """
    grad_ir = sobel_magnitude(ir)
    grad_vis = sobel_magnitude(vis)
    grad_fused = sobel_magnitude(fused)
    q_ir = gradient_correlation(grad_fused, grad_ir)
    q_vis = gradient_correlation(grad_fused, grad_vis)
    return float(np.nan_to_num((q_ir + q_vis) / 2.0, nan=0.0, posinf=0.0, neginf=0.0))


def compute_fusion_metrics(ir: Any, vis: Any, fused: Any) -> dict[str, float]:
    ir_np = _to_numpy_image(ir)
    vis_np = _to_numpy_image(vis)
    fused_np = _to_numpy_image(fused)

    if ir_np.shape != vis_np.shape or ir_np.shape != fused_np.shape:
        raise ValueError(
            f"Input shapes must match, got ir={ir_np.shape}, vis={vis_np.shape}, fused={fused_np.shape}"
        )

    metrics = {
        "EN": entropy(fused_np),
        "SD": standard_deviation(fused_np),
        "AG": average_gradient(fused_np),
        "MI": mutual_information(fused_np, ir_np) + mutual_information(fused_np, vis_np),
        "Qabf": qabf_v1_approx(ir_np, vis_np, fused_np),
    }
    return {
        key: float(np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0))
        for key, value in metrics.items()
    }


def compute_all_metrics(ir: Any, vis: Any, fused: Any) -> dict[str, float]:
    """Compatibility wrapper for batch evaluation scripts."""
    return compute_fusion_metrics(ir, vis, fused)


if __name__ == "__main__":
    ir = np.random.rand(256, 256)
    vis = np.random.rand(256, 256)
    fused = np.random.rand(256, 256)
    metric_dict = compute_fusion_metrics(ir, vis, fused)
    print(metric_dict)
    for metric_value in metric_dict.values():
        assert np.isfinite(metric_value)
    print("Fusion metrics self-test passed.")
