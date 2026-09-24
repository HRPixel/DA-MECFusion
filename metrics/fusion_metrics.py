"""Fusion metrics for DA-MECFusion."""

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


def _validate_same_shape(ir: np.ndarray, vis: np.ndarray, fused: np.ndarray) -> None:
    if ir.shape != vis.shape or ir.shape != fused.shape:
        raise ValueError(
            f"Input shapes must match, got ir={ir.shape}, vis={vis.shape}, fused={fused.shape}"
        )


def _sobel_strength_orientation(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    horizontal = np.asarray(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
    )
    vertical = np.asarray(
        [[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]]
    )
    gx = ndimage.convolve(image, horizontal, mode="constant", cval=0.0)
    gy = ndimage.convolve(image, vertical, mode="constant", cval=0.0)
    strength = np.hypot(gx, gy)
    ratio = np.zeros_like(gx)
    np.divide(gy, gx, out=ratio, where=gx != 0.0)
    orientation = np.arctan(ratio)
    orientation[gx == 0.0] = np.pi / 2.0
    return strength, orientation


def _edge_preservation(
    source_strength: np.ndarray,
    source_orientation: np.ndarray,
    fused_strength: np.ndarray,
    fused_orientation: np.ndarray,
) -> np.ndarray:
    maximum = np.maximum(source_strength, fused_strength)
    relative_strength = np.zeros_like(maximum)
    np.divide(
        np.minimum(source_strength, fused_strength),
        maximum,
        out=relative_strength,
        where=maximum > 0.0,
    )
    relative_orientation = 1.0 - (
        np.abs(source_orientation - fused_orientation) / (np.pi / 2.0)
    )
    strength_quality = 0.9994 / (1.0 + np.exp(-15.0 * (relative_strength - 0.5)))
    orientation_quality = 0.9879 / (
        1.0 + np.exp(-22.0 * (relative_orientation - 0.8))
    )
    return strength_quality * orientation_quality


def qabf_standard(ir: np.ndarray, vis: np.ndarray, fused: np.ndarray) -> float:
    """Xydeas-Petrovic edge-information transfer metric with L=1."""
    ir = np.asarray(ir, dtype=np.float64)
    vis = np.asarray(vis, dtype=np.float64)
    fused = np.asarray(fused, dtype=np.float64)
    _validate_same_shape(ir, vis, fused)

    ir_strength, ir_orientation = _sobel_strength_orientation(ir)
    vis_strength, vis_orientation = _sobel_strength_orientation(vis)
    fused_strength, fused_orientation = _sobel_strength_orientation(fused)
    q_ir = _edge_preservation(
        ir_strength, ir_orientation, fused_strength, fused_orientation
    )
    q_vis = _edge_preservation(
        vis_strength, vis_orientation, fused_strength, fused_orientation
    )
    denominator = np.sum(ir_strength + vis_strength)
    if denominator == 0.0:
        return 0.0
    value = np.sum(q_ir * ir_strength + q_vis * vis_strength) / denominator
    return float(np.clip(np.nan_to_num(value, nan=0.0), 0.0, 1.0))


def qabf_standard_region(
    ir: np.ndarray,
    vis: np.ndarray,
    fused: np.ndarray,
    mask: np.ndarray,
) -> float:
    """Xydeas-Petrovic score aggregated only inside a prevalidated mask.

    Gradients are computed on the complete images before masking, preventing an
    artificial edge from being introduced at the region boundary.
    """
    ir = np.asarray(ir, dtype=np.float64)
    vis = np.asarray(vis, dtype=np.float64)
    fused = np.asarray(fused, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    _validate_same_shape(ir, vis, fused)
    if mask.shape != ir.shape or not np.any(mask):
        return float("nan")

    ir_strength, ir_orientation = _sobel_strength_orientation(ir)
    vis_strength, vis_orientation = _sobel_strength_orientation(vis)
    fused_strength, fused_orientation = _sobel_strength_orientation(fused)
    q_ir = _edge_preservation(ir_strength, ir_orientation, fused_strength, fused_orientation)
    q_vis = _edge_preservation(vis_strength, vis_orientation, fused_strength, fused_orientation)
    denominator = np.sum((ir_strength + vis_strength)[mask])
    if denominator == 0.0:
        return float("nan")
    numerator = np.sum((q_ir * ir_strength + q_vis * vis_strength)[mask])
    return float(np.clip(numerator / denominator, 0.0, 1.0))


def _correlation_coefficient(image_a: np.ndarray, image_b: np.ndarray) -> float:
    a = image_a.ravel().astype(np.float64)
    b = image_b.ravel().astype(np.float64)
    a -= np.mean(a)
    b -= np.mean(b)
    denominator = np.sqrt(np.sum(a * a) * np.sum(b * b))
    if denominator == 0.0:
        return 0.0
    return float(np.sum(a * b) / denominator)


def scd(ir: np.ndarray, vis: np.ndarray, fused: np.ndarray) -> float:
    """Sum of correlations of differences (Aslantas-Bendes, 2015)."""
    ir = np.asarray(ir, dtype=np.float64)
    vis = np.asarray(vis, dtype=np.float64)
    fused = np.asarray(fused, dtype=np.float64)
    _validate_same_shape(ir, vis, fused)
    value = _correlation_coefficient(fused - ir, vis) + _correlation_coefficient(
        fused - vis, ir
    )
    return float(np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0))


def _masked_correlation(image_a: np.ndarray, image_b: np.ndarray, mask: np.ndarray) -> float:
    a = image_a[mask].astype(np.float64)
    b = image_b[mask].astype(np.float64)
    if a.size < 2:
        return float("nan")
    a -= np.mean(a)
    b -= np.mean(b)
    denominator = np.sqrt(np.sum(a * a) * np.sum(b * b))
    if denominator == 0.0:
        return float("nan")
    return float(np.sum(a * b) / denominator)


def scd_region(
    ir: np.ndarray,
    vis: np.ndarray,
    fused: np.ndarray,
    mask: np.ndarray,
) -> float:
    """SCD evaluated on valid pixels of one prevalidated region."""
    ir = np.asarray(ir, dtype=np.float64)
    vis = np.asarray(vis, dtype=np.float64)
    fused = np.asarray(fused, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    _validate_same_shape(ir, vis, fused)
    if mask.shape != ir.shape or not np.any(mask):
        return float("nan")
    first = _masked_correlation(fused - ir, vis, mask)
    second = _masked_correlation(fused - vis, ir, mask)
    return first + second if np.isfinite(first) and np.isfinite(second) else float("nan")


def compute_fusion_metrics(
    ir: Any,
    vis: Any,
    fused: Any,
    profile: str = "legacy",
) -> dict[str, float]:
    ir_np = _to_numpy_image(ir)
    vis_np = _to_numpy_image(vis)
    fused_np = _to_numpy_image(fused)

    _validate_same_shape(ir_np, vis_np, fused_np)

    metrics = {
        "EN": entropy(fused_np),
        "SD": standard_deviation(fused_np),
        "AG": average_gradient(fused_np),
        "MI": mutual_information(fused_np, ir_np) + mutual_information(fused_np, vis_np),
    }
    if profile == "legacy":
        metrics["Qabf"] = qabf_v1_approx(ir_np, vis_np, fused_np)
    elif profile == "standard":
        metrics["SCD"] = scd(ir_np, vis_np, fused_np)
        metrics["Qabf_standard"] = qabf_standard(ir_np, vis_np, fused_np)
    else:
        raise ValueError(f"Unknown metric profile: {profile!r}")
    return {
        key: float(np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0))
        for key, value in metrics.items()
    }


def compute_all_metrics(
    ir: Any,
    vis: Any,
    fused: Any,
    profile: str = "legacy",
) -> dict[str, float]:
    """Compatibility wrapper for batch evaluation scripts."""
    return compute_fusion_metrics(ir, vis, fused, profile=profile)


if __name__ == "__main__":
    ir = np.zeros((64, 64), dtype=np.float64)
    vis = np.zeros_like(ir)
    ir[:, 32:] = 1.0
    vis[32:, :] = 1.0
    fused = np.maximum(ir, vis)
    constant = np.full_like(ir, 0.5)

    legacy = compute_fusion_metrics(ir, vis, fused)
    standard = compute_fusion_metrics(ir, vis, fused, profile="standard")
    assert tuple(legacy) == ("EN", "SD", "AG", "MI", "Qabf")
    assert tuple(standard) == ("EN", "SD", "AG", "MI", "SCD", "Qabf_standard")
    assert all(np.isfinite(value) for value in (*legacy.values(), *standard.values()))
    assert np.isfinite(qabf_standard(constant, constant, constant))
    assert np.isfinite(scd(constant, constant, constant))
    assert np.isclose(
        qabf_standard(ir, vis, fused),
        qabf_standard(vis, ir, fused),
    )
    assert np.isclose(scd(ir, vis, fused), scd(vis, ir, fused))
    assert qabf_standard(ir, vis, fused) > qabf_standard(ir, vis, constant)
    assert np.isclose(
        qabf_standard(ir, vis, fused),
        qabf_standard(ir * 255.0, vis * 255.0, fused * 255.0),
    )
    assert np.isclose(
        scd(ir, vis, fused),
        scd(ir * 255.0, vis * 255.0, fused * 255.0),
    )
    region = np.zeros_like(ir, dtype=bool)
    region[8:56, 8:56] = True
    assert np.isfinite(qabf_standard_region(ir, vis, fused, region))
    assert np.isfinite(scd_region(ir, vis, fused, region))
    assert np.isnan(qabf_standard_region(ir, vis, fused, np.zeros_like(region)))
    assert np.isnan(scd_region(constant, constant, constant, region))
    try:
        compute_fusion_metrics(ir, vis, fused, profile="unknown")
    except ValueError:
        pass
    else:
        raise AssertionError("Unknown metric profile must fail.")
    print("Fusion metrics self-test passed.")
