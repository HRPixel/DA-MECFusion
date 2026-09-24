"""Read-only MEC, prior, and smoke-candidate diagnostics for DA-MECFusion."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics
import sys
from tempfile import TemporaryDirectory

from PIL import Image, ImageDraw
import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.msrs_dataset import MSRSDataset  # noqa: E402
from models.damecfusion import DAMECFusionV1  # noqa: E402
from models.priors import (  # noqa: E402
    SmokeLowContrastPrior,
    rgb_to_y,
)
from utils import move_batch_to_device, resolve_device  # noqa: E402


PRIOR_NAMES = ("thermal", "saturation", "smoke")
ALLOWED_LABELS = {"smoke_affected", "overexposure", "ignore"}
SMOKE_CANDIDATES = (
    ("Base", "base"),
    ("FixedTexture", "fixed_texture"),
    ("FixedTextureLuma", "fixed_texture_luma"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose raw MEC weights and prior sensitivity.")
    parser.add_argument("--data_root")
    parser.add_argument("--manifest")
    parser.add_argument("--checkpoint")
    parser.add_argument("--output")
    parser.add_argument(
        "--label_dir",
        help="Optional LabelMe JSON directory; when set, analyze only manifest samples with matching JSON.",
    )
    parser.add_argument(
        "--compare_smoke_candidates",
        action="store_true",
        help="Compare three smoke-prior candidates on matching LabelMe annotations; checkpoint is not required.",
    )
    parser.add_argument("--smoke_tau_contrast", type=float, default=0.025)
    parser.add_argument("--smoke_gamma_contrast", type=float, default=0.010)
    parser.add_argument("--smoke_tau_gradient", type=float, default=0.100)
    parser.add_argument("--smoke_gamma_gradient", type=float, default=0.040)
    parser.add_argument("--smoke_tau_luma", type=float, default=0.450)
    parser.add_argument("--smoke_gamma_luma", type=float, default=0.100)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--equalize_feature_magnitude",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override the checkpoint fusion feature-magnitude mode.",
    )
    parser.add_argument("--self_check", action="store_true")
    return parser.parse_args()


def read_metadata(manifest: Path) -> dict[str, dict[str, str]]:
    metadata: dict[str, dict[str, str]] = {}
    with manifest.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        if "name" not in (reader.fieldnames or []):
            raise ValueError(f"Manifest is missing the name column: {manifest}")
        for row in reader:
            name = row["name"].strip()
            if not name:
                raise ValueError(f"Manifest contains a blank name: {manifest}")
            if name in metadata:
                raise ValueError(f"Manifest contains duplicate name '{name}': {manifest}")
            metadata[name] = {
                "source_class": row.get("source_class", "Unknown").strip() or "Unknown",
                "sequence_id": row.get("sequence_id", "").strip(),
            }
    return metadata


def pearson(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x.float().reshape(-1)
    y = y.float().reshape(-1)
    x = x - x.mean()
    y = y - y.mean()
    denominator = torch.sqrt(torch.sum(x * x) * torch.sum(y * y))
    if denominator.item() <= 1e-12:
        return float("nan")
    return float((torch.sum(x * y) / denominator).item())


def tensor_stats(x: torch.Tensor, prefix: str) -> dict[str, float]:
    return {
        f"{prefix}_min": float(x.min().item()),
        f"{prefix}_max": float(x.max().item()),
        f"{prefix}_mean": float(x.mean().item()),
        f"{prefix}_std": float(x.std(unbiased=False).item()),
    }


def mec_contribution_maps(
    model: DAMECFusionV1,
    feat_ir: torch.Tensor,
    feat_vis: torch.Tensor,
    priors: list[torch.Tensor],
    w_ir: torch.Tensor,
    w_vis: torch.Tensor,
    learned_logits: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Reconstruct MEC logits and measure weighted feature-magnitude proxies."""
    resized_priors = [
        model.mec._resize_prior(prior, feat_ir.shape[-2:])
        for prior in priors
    ]
    routed_priors = list(model.mec.route_priors(*resized_priors))
    raw_learned_logit_gap = learned_logits[:, 0:1] - learned_logits[:, 1:2]
    effective_learned_logits = (
        model.mec._bound_learned_logits(learned_logits)
        if model.mec.bounded_learned_gap
        else learned_logits
    )
    learned_logit_gap = effective_learned_logits[:, 0:1] - effective_learned_logits[:, 1:2]

    if model.mec.fixed_equal_weights:
        learned_logit_gap = torch.zeros_like(learned_logit_gap)
        prior_logit_gap = torch.zeros_like(learned_logit_gap)
        total_logit_gap = torch.zeros_like(learned_logit_gap)
        reconstructed_w_ir = torch.full_like(learned_logit_gap, 0.5)
    else:
        centered_priors = [
            prior - prior.mean(dim=(-2, -1), keepdim=True)
            for prior in routed_priors
        ]
        prior_residual = model.mec.prior_residual_scale * torch.stack(
            centered_priors,
            dim=0,
        ).mean(dim=0)
        prior_logit_gap = 2.0 * prior_residual
        total_logit_gap = learned_logit_gap + prior_logit_gap
        reconstructed_w_ir = torch.sigmoid(total_logit_gap)

    fusion_feat_ir, fusion_feat_vis = model.mec.fusion_features(feat_ir, feat_vis)
    feat_ir_magnitude = feat_ir.abs().mean(dim=1, keepdim=True)
    feat_vis_magnitude = feat_vis.abs().mean(dim=1, keepdim=True)
    fusion_feat_ir_magnitude = fusion_feat_ir.abs().mean(dim=1, keepdim=True)
    fusion_feat_vis_magnitude = fusion_feat_vis.abs().mean(dim=1, keepdim=True)
    weighted_ir_magnitude = (w_ir * fusion_feat_ir).abs().mean(dim=1, keepdim=True)
    weighted_vis_magnitude = (w_vis * fusion_feat_vis).abs().mean(dim=1, keepdim=True)
    contribution_sum = weighted_ir_magnitude + weighted_vis_magnitude
    effective_ir_ratio = torch.where(
        contribution_sum > 1e-12,
        weighted_ir_magnitude / contribution_sum,
        torch.full_like(contribution_sum, 0.5),
    )

    return {
        "feat_ir_magnitude": feat_ir_magnitude,
        "feat_vis_magnitude": feat_vis_magnitude,
        "fusion_feat_ir_magnitude": fusion_feat_ir_magnitude,
        "fusion_feat_vis_magnitude": fusion_feat_vis_magnitude,
        "weighted_ir_magnitude": weighted_ir_magnitude,
        "weighted_vis_magnitude": weighted_vis_magnitude,
        "effective_ir_ratio": effective_ir_ratio,
        "raw_learned_logit_gap": raw_learned_logit_gap,
        "learned_logit_gap": learned_logit_gap,
        "prior_logit_gap": prior_logit_gap,
        "total_logit_gap": total_logit_gap,
        "reconstructed_w_ir": reconstructed_w_ir,
    }


def load_label_masks(label_path: Path, height: int, width: int) -> tuple[torch.Tensor, torch.Tensor]:
    data = json.loads(label_path.read_text(encoding="utf-8"))
    source_width = int(data["imageWidth"])
    source_height = int(data["imageHeight"])
    smoke_image = Image.new("L", (source_width, source_height), 0)
    ignore_image = Image.new("L", (source_width, source_height), 0)
    smoke_draw = ImageDraw.Draw(smoke_image)
    ignore_draw = ImageDraw.Draw(ignore_image)

    for shape in data.get("shapes", []):
        label = str(shape.get("label", ""))
        if label not in ALLOWED_LABELS:
            raise ValueError(f"Unsupported label '{label}' in {label_path}")
        if shape.get("shape_type") != "polygon":
            raise ValueError(f"Only polygon shapes are supported: {label_path}")
        points = [tuple(map(float, point)) for point in shape.get("points", [])]
        if len(points) < 3:
            raise ValueError(f"Polygon has fewer than three points: {label_path}")
        if label == "smoke_affected":
            smoke_draw.polygon(points, fill=1)
        elif label == "ignore":
            ignore_draw.polygon(points, fill=1)

    if (source_width, source_height) != (width, height):
        smoke_image = smoke_image.resize((width, height), Image.Resampling.NEAREST)
        ignore_image = ignore_image.resize((width, height), Image.Resampling.NEAREST)

    smoke = torch.frombuffer(bytearray(smoke_image.tobytes()), dtype=torch.uint8).reshape(1, 1, height, width).bool()
    ignore = torch.frombuffer(bytearray(ignore_image.tobytes()), dtype=torch.uint8).reshape(1, 1, height, width).bool()
    return smoke & ~ignore, ~ignore


def add_region_stats(
    result: dict[str, float],
    priors: list[torch.Tensor],
    w_ir: torch.Tensor,
    smoke_mask: torch.Tensor,
    valid_mask: torch.Tensor,
    diagnostic_maps: dict[str, torch.Tensor] | None = None,
) -> None:
    smoke_mask = smoke_mask.to(w_ir.device)
    valid_mask = valid_mask.to(w_ir.device)
    background_mask = valid_mask & ~smoke_mask
    valid_pixels = int(valid_mask.sum().item())
    if valid_pixels == 0:
        raise ValueError("Annotation contains no valid pixels after applying ignore regions")

    result["region_smoke_fraction"] = float(smoke_mask.sum().item() / valid_pixels)
    region_maps = [*zip(PRIOR_NAMES, priors), ("w_ir", w_ir)]
    if diagnostic_maps is not None:
        region_maps.extend(
            (name, diagnostic_maps[name])
            for name in (
                "effective_ir_ratio",
                "learned_logit_gap",
                "prior_logit_gap",
                "total_logit_gap",
            )
        )
    for name, value in region_maps:
        smoke_mean = float(value[smoke_mask].mean().item()) if smoke_mask.any() else float("nan")
        background_mean = float(value[background_mask].mean().item()) if background_mask.any() else float("nan")
        result[f"region_{name}_smoke_mean"] = smoke_mean
        result[f"region_{name}_background_mean"] = background_mean
        result[f"region_{name}_delta"] = smoke_mean - background_mean


def build_smoke_candidate_priors(
    parameters: dict[str, float],
) -> dict[str, SmokeLowContrastPrior]:
    return {
        key: SmokeLowContrastPrior(mode=key, **parameters)
        for _, key in SMOKE_CANDIDATES
    }


def smoke_candidate_maps(
    vis: torch.Tensor,
    priors: dict[str, SmokeLowContrastPrior],
) -> dict[str, torch.Tensor]:
    if vis.dim() != 4 or vis.size(1) != 3:
        raise ValueError(f"Expected vis with shape [B,3,H,W], got {tuple(vis.shape)}")
    return {key: prior(vis) for key, prior in priors.items()}


def add_smoke_candidate_stats(
    result: dict[str, float],
    candidates: dict[str, torch.Tensor],
    smoke_mask: torch.Tensor,
    valid_mask: torch.Tensor,
) -> None:
    device = next(iter(candidates.values())).device
    smoke_mask = smoke_mask.to(device)
    valid_mask = valid_mask.to(device)
    background_mask = valid_mask & ~smoke_mask
    valid_pixels = int(valid_mask.sum().item())
    if valid_pixels == 0:
        raise ValueError("Annotation contains no valid pixels after applying ignore regions")

    result["region_smoke_fraction"] = float(smoke_mask.sum().item() / valid_pixels)
    for _, key in SMOKE_CANDIDATES:
        value = candidates[key]
        if value.shape != valid_mask.shape:
            raise ValueError(f"Candidate '{key}' shape {tuple(value.shape)} does not match mask {tuple(valid_mask.shape)}")
        smoke_mean = float(value[smoke_mask].mean().item()) if smoke_mask.any() else float("nan")
        background_mean = float(value[background_mask].mean().item()) if background_mask.any() else float("nan")
        result[f"{key}_smoke_mean"] = smoke_mean
        result[f"{key}_background_mean"] = background_mean
        result[f"{key}_delta"] = smoke_mean - background_mean
        result[f"{key}_valid_mean"] = float(value[valid_mask].mean().item())
        result[f"{key}_high_090_fraction"] = float((value[valid_mask] >= 0.9).float().mean().item())


def analyze_batch(
    model: DAMECFusionV1,
    batch: dict[str, object],
    region_masks: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> dict[str, float]:
    ir = batch["ir"]
    vis = batch["vis"]
    if not isinstance(ir, torch.Tensor) or not isinstance(vis, torch.Tensor):
        raise TypeError("Batch must contain tensor ir and vis values")

    feat_ir = model.ir_encoder(ir)
    feat_vis = model.vis_encoder(rgb_to_y(vis))
    priors = list(model.prior_module(ir, vis))
    captured_logits: list[torch.Tensor] = []
    handle = model.mec.logit_conv.register_forward_hook(
        lambda _module, _inputs, output: captured_logits.append(output)
    )
    try:
        _, w_ir, w_vis = model.mec(feat_ir, feat_vis, *priors)
    finally:
        handle.remove()
    if len(captured_logits) != 1:
        raise RuntimeError(f"Expected one MEC learned-logit tensor, got {len(captured_logits)}")
    diagnostic_maps = mec_contribution_maps(
        model,
        feat_ir,
        feat_vis,
        priors,
        w_ir,
        w_vis,
        captured_logits[0],
    )

    result: dict[str, float] = {}
    for name, prior in zip(PRIOR_NAMES, priors):
        result.update(tensor_stats(prior, name))
        result[f"corr_wir_{name}"] = pearson(w_ir, prior)

    result.update(tensor_stats(w_ir, "w_ir"))
    result.update(tensor_stats(w_vis, "w_vis"))
    result["weight_sum_max_error"] = float(torch.max(torch.abs(w_ir + w_vis - 1.0)).item())
    result["fusion_feature_magnitude_max_error"] = float(
        torch.max(
            torch.abs(
                diagnostic_maps["fusion_feat_ir_magnitude"]
                - diagnostic_maps["fusion_feat_vis_magnitude"]
            )
        ).item()
    )
    result["effective_ir_ratio_w_ir_max_error"] = float(
        torch.max(torch.abs(diagnostic_maps["effective_ir_ratio"] - w_ir)).item()
    )
    for name in (
        "feat_ir_magnitude",
        "feat_vis_magnitude",
        "fusion_feat_ir_magnitude",
        "fusion_feat_vis_magnitude",
        "weighted_ir_magnitude",
        "weighted_vis_magnitude",
        "effective_ir_ratio",
        "raw_learned_logit_gap",
        "learned_logit_gap",
        "prior_logit_gap",
        "total_logit_gap",
    ):
        result.update(tensor_stats(diagnostic_maps[name], name))
    for name in (
        "raw_learned_logit_gap",
        "learned_logit_gap",
        "prior_logit_gap",
        "total_logit_gap",
    ):
        result[f"{name}_abs_mean"] = float(diagnostic_maps[name].abs().mean().item())
    result["prior_to_learned_gap_abs_ratio"] = float(
        diagnostic_maps["prior_logit_gap"].abs().mean().item()
        / max(diagnostic_maps["learned_logit_gap"].abs().mean().item(), 1e-12)
    )
    result["reconstructed_w_ir_max_error"] = float(
        torch.max(torch.abs(w_ir - diagnostic_maps["reconstructed_w_ir"])).item()
    )

    for index, name in enumerate(PRIOR_NAMES):
        zeroed = priors.copy()
        zeroed[index] = torch.zeros_like(zeroed[index])
        _, zeroed_w_ir, _ = model.mec(feat_ir, feat_vis, *zeroed)
        result[f"zero_{name}_delta_wir"] = float(torch.mean(torch.abs(w_ir - zeroed_w_ir)).item())
    if region_masks is not None:
        add_region_stats(result, priors, w_ir, *region_masks, diagnostic_maps)
    return result


def finite_mean(rows: list[dict[str, object]], key: str) -> float:
    values = [float(row[key]) for row in rows if math.isfinite(float(row[key]))]
    return statistics.fmean(values) if values else float("nan")


def mean_and_sd(rows: list[dict[str, object]], key: str) -> str:
    values = [float(row[key]) for row in rows if math.isfinite(float(row[key]))]
    if not values:
        return "NA"
    return f"{statistics.fmean(values):.6f} ± {statistics.pstdev(values):.6f}"


def number(value: object) -> str:
    if isinstance(value, str):
        return value
    value = float(value)
    return "NA" if not math.isfinite(value) else f"{value:.6f}"


def group_table(rows: list[dict[str, object]]) -> list[str]:
    groups = [("All", rows)]
    groups.extend(
        (source_class, [row for row in rows if row["source_class"] == source_class])
        for source_class in sorted({str(row["source_class"]) for row in rows})
    )
    lines = [
        "| Group | N | thermal mean | saturation mean | smoke mean | w_ir mean | mean spatial std(w_ir) | max sum error | corr thermal | corr saturation | corr smoke | zero thermal Δw_ir | zero saturation Δw_ir | zero smoke Δw_ir |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for group_name, group_rows in groups:
        lines.append(
            "| "
            + " | ".join(
                [
                    group_name,
                    str(len(group_rows)),
                    mean_and_sd(group_rows, "thermal_mean"),
                    mean_and_sd(group_rows, "saturation_mean"),
                    mean_and_sd(group_rows, "smoke_mean"),
                    mean_and_sd(group_rows, "w_ir_mean"),
                    number(finite_mean(group_rows, "w_ir_std")),
                    number(max(float(row["weight_sum_max_error"]) for row in group_rows)),
                    number(finite_mean(group_rows, "corr_wir_thermal")),
                    number(finite_mean(group_rows, "corr_wir_saturation")),
                    number(finite_mean(group_rows, "corr_wir_smoke")),
                    number(finite_mean(group_rows, "zero_thermal_delta_wir")),
                    number(finite_mean(group_rows, "zero_saturation_delta_wir")),
                    number(finite_mean(group_rows, "zero_smoke_delta_wir")),
                ]
            )
            + " |"
        )
    return lines


def detail_table(rows: list[dict[str, object]]) -> list[str]:
    keys = (
        "thermal_mean",
        "saturation_mean",
        "smoke_mean",
        "w_ir_min",
        "w_ir_max",
        "w_ir_mean",
        "w_ir_std",
        "corr_wir_thermal",
        "corr_wir_saturation",
        "corr_wir_smoke",
        "zero_thermal_delta_wir",
        "zero_saturation_delta_wir",
        "zero_smoke_delta_wir",
    )
    lines = [
        "| Name | Class | Sequence | thermal | saturation | smoke | w_ir min | w_ir max | w_ir mean | w_ir std | corr T | corr Sat | corr Smoke | zero T Δ | zero Sat Δ | zero Smoke Δ |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        values = [str(row["name"]), str(row["source_class"]), str(row["sequence_id"])]
        values.extend(number(row[key]) for key in keys)
        lines.append("| " + " | ".join(values) + " |")
    return lines


def contribution_group_table(rows: list[dict[str, object]]) -> list[str]:
    groups = [("All", rows)]
    groups.extend(
        (source_class, [row for row in rows if row["source_class"] == source_class])
        for source_class in sorted({str(row["source_class"]) for row in rows})
    )
    lines = [
        "| Group | N | encoder abs(IR) | encoder abs(VIS) | fusion abs(IR) | fusion abs(VIS) | weighted abs(IR) | weighted abs(VIS) | effective IR ratio | raw w_ir |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for group_name, group_rows in groups:
        lines.append(
            "| "
            + " | ".join(
                [
                    group_name,
                    str(len(group_rows)),
                    mean_and_sd(group_rows, "feat_ir_magnitude_mean"),
                    mean_and_sd(group_rows, "feat_vis_magnitude_mean"),
                    mean_and_sd(group_rows, "fusion_feat_ir_magnitude_mean"),
                    mean_and_sd(group_rows, "fusion_feat_vis_magnitude_mean"),
                    mean_and_sd(group_rows, "weighted_ir_magnitude_mean"),
                    mean_and_sd(group_rows, "weighted_vis_magnitude_mean"),
                    mean_and_sd(group_rows, "effective_ir_ratio_mean"),
                    mean_and_sd(group_rows, "w_ir_mean"),
                ]
            )
            + " |"
        )
    return lines


def logit_group_table(rows: list[dict[str, object]]) -> list[str]:
    groups = [("All", rows)]
    groups.extend(
        (source_class, [row for row in rows if row["source_class"] == source_class])
        for source_class in sorted({str(row["source_class"]) for row in rows})
    )
    lines = [
        "| Group | N | abs(raw learned gap) | learned gap mean | abs(learned gap) | prior gap mean | abs(prior gap) | total gap mean | abs(total gap) | prior/learned abs ratio | max reconstruction error |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for group_name, group_rows in groups:
        lines.append(
            "| "
            + " | ".join(
                [
                    group_name,
                    str(len(group_rows)),
                    mean_and_sd(group_rows, "raw_learned_logit_gap_abs_mean"),
                    mean_and_sd(group_rows, "learned_logit_gap_mean"),
                    mean_and_sd(group_rows, "learned_logit_gap_abs_mean"),
                    mean_and_sd(group_rows, "prior_logit_gap_mean"),
                    mean_and_sd(group_rows, "prior_logit_gap_abs_mean"),
                    mean_and_sd(group_rows, "total_logit_gap_mean"),
                    mean_and_sd(group_rows, "total_logit_gap_abs_mean"),
                    mean_and_sd(group_rows, "prior_to_learned_gap_abs_ratio"),
                    number(max(float(row["reconstructed_w_ir_max_error"]) for row in group_rows)),
                ]
            )
            + " |"
        )
    return lines


def contribution_detail_table(rows: list[dict[str, object]]) -> list[str]:
    keys = (
        "feat_ir_magnitude_mean",
        "feat_vis_magnitude_mean",
        "weighted_ir_magnitude_mean",
        "weighted_vis_magnitude_mean",
        "effective_ir_ratio_mean",
        "raw_learned_logit_gap_abs_mean",
        "learned_logit_gap_mean",
        "learned_logit_gap_abs_mean",
        "prior_logit_gap_mean",
        "prior_logit_gap_abs_mean",
        "total_logit_gap_mean",
        "prior_to_learned_gap_abs_ratio",
    )
    lines = [
        "| Name | Class | Sequence | encoder abs(IR) | encoder abs(VIS) | weighted abs(IR) | weighted abs(VIS) | effective IR ratio | abs(raw learned gap) | learned gap | abs(learned gap) | prior gap | abs(prior gap) | total gap | prior/learned |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        values = [str(row["name"]), str(row["source_class"]), str(row["sequence_id"])]
        values.extend(number(row[key]) for key in keys)
        lines.append("| " + " | ".join(values) + " |")
    return lines


def region_group_table(rows: list[dict[str, object]]) -> list[str]:
    annotated = [row for row in rows if "region_smoke_fraction" in row]
    groups = [
        ("Annotated", annotated),
        ("Smoke-positive", [row for row in annotated if float(row["region_smoke_fraction"]) > 0]),
        ("Empty-negative", [row for row in annotated if float(row["region_smoke_fraction"]) == 0]),
    ]
    lines = [
        "| Group | N | smoke area fraction | smoke prior: smoke | smoke prior: background | smoke prior Δ | thermal prior Δ | saturation prior Δ | w_ir Δ | effective IR ratio Δ | learned gap Δ | prior gap Δ | total gap Δ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for group_name, group_rows in groups:
        if not group_rows:
            continue
        lines.append(
            "| "
            + " | ".join(
                [
                    group_name,
                    str(len(group_rows)),
                    mean_and_sd(group_rows, "region_smoke_fraction"),
                    mean_and_sd(group_rows, "region_smoke_smoke_mean"),
                    mean_and_sd(group_rows, "region_smoke_background_mean"),
                    mean_and_sd(group_rows, "region_smoke_delta"),
                    mean_and_sd(group_rows, "region_thermal_delta"),
                    mean_and_sd(group_rows, "region_saturation_delta"),
                    mean_and_sd(group_rows, "region_w_ir_delta"),
                    mean_and_sd(group_rows, "region_effective_ir_ratio_delta"),
                    mean_and_sd(group_rows, "region_learned_logit_gap_delta"),
                    mean_and_sd(group_rows, "region_prior_logit_gap_delta"),
                    mean_and_sd(group_rows, "region_total_logit_gap_delta"),
                ]
            )
            + " |"
        )
    return lines


def region_detail_table(rows: list[dict[str, object]]) -> list[str]:
    annotated = [row for row in rows if "region_smoke_fraction" in row]
    keys = (
        "region_smoke_fraction",
        "region_smoke_smoke_mean",
        "region_smoke_background_mean",
        "region_smoke_delta",
        "region_thermal_delta",
        "region_saturation_delta",
        "region_w_ir_delta",
        "region_effective_ir_ratio_delta",
        "region_learned_logit_gap_delta",
        "region_prior_logit_gap_delta",
        "region_total_logit_gap_delta",
    )
    lines = [
        "| Name | Class | Sequence | smoke fraction | smoke prior: smoke | smoke prior: background | smoke prior Δ | thermal Δ | saturation Δ | w_ir Δ | effective IR ratio Δ | learned gap Δ | prior gap Δ | total gap Δ |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in annotated:
        values = [str(row["name"]), str(row["source_class"]), str(row["sequence_id"])]
        values.extend(number(row[key]) for key in keys)
        lines.append("| " + " | ".join(values) + " |")
    return lines


def smoke_candidate_summary_table(rows: list[dict[str, object]]) -> list[str]:
    positives = [row for row in rows if float(row["region_smoke_fraction"]) > 0]
    negatives = [row for row in rows if float(row["region_smoke_fraction"]) == 0]
    lines = [
        "| Candidate | smoke region | Fire background | Fire Δ | positive Δ | No Fire mean | smoke−No Fire | No Fire ≥0.9 | SEQ_07 Δ | SEQ_10 Δ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for display_name, key in SMOKE_CANDIDATES:
        smoke_mean = finite_mean(positives, f"{key}_smoke_mean")
        nofire_mean = finite_mean(negatives, f"{key}_background_mean")
        seq_07 = [row for row in positives if row["sequence_id"] == "SEQ_07"]
        seq_10 = [row for row in positives if row["sequence_id"] == "SEQ_10"]
        positive_count = sum(float(row[f"{key}_delta"]) > 0 for row in positives)
        lines.append(
            "| "
            + " | ".join(
                [
                    display_name,
                    mean_and_sd(positives, f"{key}_smoke_mean"),
                    mean_and_sd(positives, f"{key}_background_mean"),
                    mean_and_sd(positives, f"{key}_delta"),
                    f"{positive_count}/{len(positives)}",
                    mean_and_sd(negatives, f"{key}_background_mean"),
                    number(smoke_mean - nofire_mean),
                    mean_and_sd(negatives, f"{key}_high_090_fraction"),
                    mean_and_sd(seq_07, f"{key}_delta"),
                    mean_and_sd(seq_10, f"{key}_delta"),
                ]
            )
            + " |"
        )
    return lines


def smoke_candidate_detail_table(rows: list[dict[str, object]]) -> list[str]:
    lines = [
        "| Name | Class | Sequence | smoke fraction | Candidate | smoke region | background | Δ | valid mean | ≥0.9 fraction |",
        "|---|---|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        for display_name, key in SMOKE_CANDIDATES:
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row["name"]),
                        str(row["source_class"]),
                        str(row["sequence_id"]),
                        number(row["region_smoke_fraction"]),
                        display_name,
                        number(row[f"{key}_smoke_mean"]),
                        number(row[f"{key}_background_mean"]),
                        number(row[f"{key}_delta"]),
                        number(row[f"{key}_valid_mean"]),
                        number(row[f"{key}_high_090_fraction"]),
                    ]
                )
                + " |"
            )
    return lines


def write_smoke_candidate_report(
    output: Path,
    manifest: Path,
    label_dir: Path,
    device: torch.device,
    rows: list[dict[str, object]],
    parameters: dict[str, float],
) -> None:
    positives = [row for row in rows if float(row["region_smoke_fraction"]) > 0]
    negatives = [row for row in rows if float(row["region_smoke_fraction"]) == 0]
    if not positives or not negatives:
        raise ValueError("Candidate comparison requires both smoke-positive and empty-negative annotations")
    gaps = []
    for display_name, key in SMOKE_CANDIDATES:
        gap = finite_mean(positives, f"{key}_smoke_mean") - finite_mean(
            negatives, f"{key}_background_mean"
        )
        gaps.append((display_name, gap))
    ranking = " > ".join(name for name, _ in sorted(gaps, key=lambda item: item[1], reverse=True))

    lines = [
        "# DA-MECFusion smoke prior candidate comparison",
        "",
        f"- Manifest: `{manifest.resolve()}`",
        f"- Label directory: `{label_dir.resolve()}`",
        f"- Device: `{device}`",
        f"- Annotated samples: `{len(rows)}` ({len(positives)} smoke-positive, {len(negatives)} empty-negative)",
        "- This is calibration-only mechanism screening; it is not an independent performance evaluation.",
        "- `smoke_affected` is positive, `ignore` is excluded, and `overexposure` is not treated as smoke.",
        "",
        "## Fixed candidate parameters",
        "",
        "| Parameter | Value |",
        "|---|---:|",
        *[f"| {name} | {value:.6f} |" for name, value in parameters.items()],
        "",
        "## Candidate summary",
        "",
        *smoke_candidate_summary_table(rows),
        "",
        f"- Diagnostic ranking by mean smoke-region minus No Fire mean: `{ranking}`.",
        "- Ranking alone does not select the final prior; per-sample Δ and No Fire high-response ratio must also pass.",
        "",
        "## Per-sample details",
        "",
        *smoke_candidate_detail_table(rows),
        "",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")


def write_report(
    output: Path,
    checkpoint: Path,
    manifest: Path,
    device: torch.device,
    rows: list[dict[str, object]],
    smoke_prior_mode: str,
    prior_residual_scale: float,
    bounded_learned_gap: bool,
    equalize_feature_magnitude: bool,
    disable_thermal_prior: bool,
    disable_sat_uncertainty: bool,
    disable_smoke_prior: bool,
    fixed_equal_weights: bool,
    label_dir: Path | None = None,
) -> None:
    lines = [
        "# DA-MECFusion MEC/prior diagnosis",
        "",
        f"- Checkpoint: `{checkpoint.resolve()}`",
        f"- Manifest: `{manifest.resolve()}`",
        f"- Device: `{device}`",
        f"- Smoke prior mode: `{smoke_prior_mode}`",
        f"- Prior residual scale: `{prior_residual_scale:.6f}`",
        f"- Bounded learned gap: `{bounded_learned_gap}`",
        f"- Equalize feature magnitude: `{equalize_feature_magnitude}`",
        f"- Disable thermal prior to MEC: `{disable_thermal_prior}`",
        f"- Disable saturation uncertainty to MEC: `{disable_sat_uncertainty}`",
        f"- Disable smoke prior to MEC: `{disable_smoke_prior}`",
        f"- Fixed equal weights: `{fixed_equal_weights}`",
        f"- Samples: `{len(rows)}`",
        "- Zero-prior sensitivity is the raw mean absolute change in `w_ir` after setting one prior to zero.",
        "- Effective IR ratio uses the feature pair actually entering final weighted fusion; when enabled, that pair is magnitude-equalized while the weight predictor still uses the original encoder features.",
        "- Effective IR ratio is a pre-decoder feature-magnitude proxy, not causal attribution or a supervised modality label.",
        f"- Max fusion feature magnitude error: `{max(float(row['fusion_feature_magnitude_max_error']) for row in rows):.8f}`",
        f"- Max effective IR ratio vs w_ir error: `{max(float(row['effective_ir_ratio_w_ir_max_error']) for row in rows):.8f}`",
        "- Raw learned gap is the weight predictor's `IR logit - VIS logit`; learned gap is after the optional `2*tanh(gap/2)` bound.",
        "- The explicit prior term contributes twice the centered residual to total logit gap.",
        "- With fixed equal weights, learned/prior/total applied gaps are reported as zero; the raw predictor gap is diagnostic-only.",
        "- Fire/No Fire is used only for grouped diagnosis, not as model supervision.",
        "",
        "## Group summary",
        "",
        *group_table(rows),
        "",
        "## Weighted feature contribution proxy",
        "",
        *contribution_group_table(rows),
        "",
        "## MEC logit decomposition",
        "",
        *logit_group_table(rows),
        "",
    ]
    annotated = [row for row in rows if "region_smoke_fraction" in row]
    if label_dir is not None:
        lines.extend(
            [
                "## Annotated smoke-region diagnosis",
                "",
                f"- Label directory: `{label_dir.resolve()}`",
                f"- Annotated samples found: `{len(annotated)}`",
                "- `smoke_affected` is the positive region; `ignore` is excluded; `overexposure` is not treated as smoke.",
                "- Δ is the per-image mean in the smoke region minus the valid background mean. Group values are unweighted per-image mean ± population SD.",
                "",
                *region_group_table(rows),
                "",
                *region_detail_table(rows),
                "",
            ]
        )
    lines.extend(
        [
            "## Per-sample prior and raw-weight details",
            "",
            *detail_table(rows),
            "",
            "## Per-sample contribution and logit details",
            "",
            *contribution_detail_table(rows),
            "",
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")


def self_check() -> None:
    x = torch.tensor([0.0, 1.0, 2.0])
    assert abs(pearson(x, 2.0 * x) - 1.0) < 1e-6
    assert math.isnan(pearson(torch.ones(3), x))
    clipped = tensor_stats(torch.tensor([0.25, 0.75]), "x")
    assert clipped["x_mean"] == 0.5
    with TemporaryDirectory() as directory:
        label_path = Path(directory) / "sample.json"
        label_path.write_text(
            json.dumps(
                {
                    "imageWidth": 4,
                    "imageHeight": 2,
                    "shapes": [
                        {
                            "label": "smoke_affected",
                            "shape_type": "polygon",
                            "points": [[0, 0], [1, 0], [1, 1], [0, 1]],
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        smoke_mask, valid_mask = load_label_masks(label_path, height=2, width=4)
        region_result: dict[str, float] = {}
        maps = [
            torch.tensor([[[[0.75, 0.75, 0.25, 0.25], [0.75, 0.75, 0.25, 0.25]]]])
            for _ in PRIOR_NAMES
        ]
        add_region_stats(region_result, maps, maps[0], smoke_mask, valid_mask)
        assert region_result["region_smoke_fraction"] == 0.5
        assert region_result["region_smoke_delta"] == 0.5
    bright_flat = torch.full((1, 3, 8, 8), 0.7)
    dark_flat = torch.full((1, 3, 8, 8), 0.1)
    torch.manual_seed(0)
    textured = torch.rand(1, 3, 8, 8)
    parameters = {
        "tau_contrast": 0.025,
        "gamma_contrast": 0.010,
        "tau_gradient": 0.100,
        "gamma_gradient": 0.040,
        "tau_luma": 0.450,
        "gamma_luma": 0.100,
    }
    candidate_priors = build_smoke_candidate_priors(parameters)
    bright_maps = smoke_candidate_maps(bright_flat, candidate_priors)
    dark_maps = smoke_candidate_maps(dark_flat, candidate_priors)
    textured_maps = smoke_candidate_maps(textured, candidate_priors)
    for candidate in bright_maps.values():
        assert candidate.shape == (1, 1, 8, 8)
        assert candidate.min().item() >= 0.0
        assert candidate.max().item() <= 1.0
    assert bright_maps["fixed_texture"].mean() > textured_maps["fixed_texture"].mean()
    assert bright_maps["fixed_texture_luma"].mean() > dark_maps["fixed_texture_luma"].mean()
    torch.manual_seed(1)
    model = DAMECFusionV1(
        feature_channels=4,
        hidden_channels=4,
        encoder_blocks=0,
        decoder_blocks=0,
        prior_residual_scale=1.0,
        bounded_learned_gap=True,
        equalize_feature_magnitude=True,
    ).eval()
    with torch.no_grad():
        model.mec.logit_conv.bias.copy_(torch.tensor([10.0, -10.0]))
    diagnosis = analyze_batch(
        model,
        {
            "ir": torch.rand(1, 1, 8, 8),
            "vis": torch.rand(1, 3, 8, 8),
        },
    )
    assert 0.0 <= diagnosis["effective_ir_ratio_mean"] <= 1.0
    assert diagnosis["weighted_ir_magnitude_mean"] >= 0.0
    assert diagnosis["weighted_vis_magnitude_mean"] >= 0.0
    assert diagnosis["raw_learned_logit_gap_abs_mean"] > 19.0
    assert diagnosis["learned_logit_gap_abs_mean"] <= 2.0
    assert diagnosis["prior_logit_gap_abs_mean"] > 0.0
    assert diagnosis["reconstructed_w_ir_max_error"] < 1e-6
    assert abs(
        diagnosis["fusion_feat_ir_magnitude_mean"]
        - diagnosis["fusion_feat_vis_magnitude_mean"]
    ) < 1e-6
    assert abs(diagnosis["effective_ir_ratio_mean"] - diagnosis["w_ir_mean"]) < 1e-6
    assert diagnosis["fusion_feature_magnitude_max_error"] < 1e-6
    assert diagnosis["effective_ir_ratio_w_ir_max_error"] < 1e-6
    print("MEC/prior diagnosis self-check passed.")


def main() -> None:
    args = parse_args()
    if args.self_check:
        self_check()
        return

    required = ["data_root", "manifest", "output"]
    if not args.compare_smoke_candidates:
        required.append("checkpoint")
    missing = [name for name in required if not getattr(args, name)]
    if missing:
        raise ValueError(f"Missing required arguments: {', '.join(missing)}")
    if args.compare_smoke_candidates and not args.label_dir:
        raise ValueError("--compare_smoke_candidates requires --label_dir")

    data_root = Path(args.data_root)
    manifest = Path(args.manifest)
    output = Path(args.output)
    label_dir = Path(args.label_dir) if args.label_dir else None
    metadata = read_metadata(manifest)
    device = resolve_device(args.device)

    dataset = MSRSDataset(
        data_root=str(data_root),
        split="diagnostic",
        height=args.height,
        width=args.width,
        use_label=False,
        manifest_path=str(manifest),
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    if args.compare_smoke_candidates:
        assert label_dir is not None
        parameters = {
            "tau_contrast": args.smoke_tau_contrast,
            "gamma_contrast": args.smoke_gamma_contrast,
            "tau_gradient": args.smoke_tau_gradient,
            "gamma_gradient": args.smoke_gamma_gradient,
            "tau_luma": args.smoke_tau_luma,
            "gamma_luma": args.smoke_gamma_luma,
        }
        candidate_priors = {
            key: prior.to(device).eval()
            for key, prior in build_smoke_candidate_priors(parameters).items()
        }
        candidate_rows: list[dict[str, object]] = []
        with torch.inference_mode():
            for batch in loader:
                name = str(batch["name"][0])
                if name not in metadata:
                    raise KeyError(f"Missing metadata for sample '{name}'")
                label_path = label_dir / f"{name}.json"
                if not label_path.exists():
                    continue
                vis = batch["vis"]
                if not isinstance(vis, torch.Tensor):
                    raise TypeError("Batch must contain tensor vis values")
                candidates = smoke_candidate_maps(vis.to(device), candidate_priors)
                row: dict[str, object] = {"name": name, **metadata[name]}
                add_smoke_candidate_stats(
                    row,
                    candidates,
                    *load_label_masks(label_path, args.height, args.width),
                )
                candidate_rows.append(row)
        if not candidate_rows:
            raise ValueError(f"No matching LabelMe JSON files found in: {label_dir}")
        print(f"Analyzed {len(candidate_rows)} annotated samples")
        write_smoke_candidate_report(
            output,
            manifest,
            label_dir,
            device,
            candidate_rows,
            parameters,
        )
        print(f"Wrote report: {output}")
        return

    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    checkpoint_args = checkpoint.get("args", {}) if isinstance(checkpoint, dict) else {}
    feature_channels = int(checkpoint_args.get("feature_channels", 64))
    smoke_prior_mode = str(checkpoint_args.get("smoke_prior_mode", "base"))
    prior_residual_scale = float(checkpoint_args.get("prior_residual_scale", 0.0))
    bounded_learned_gap = bool(checkpoint_args.get("bounded_learned_gap", False))
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
        feature_channels=feature_channels,
        smoke_prior_mode=smoke_prior_mode,
        prior_residual_scale=prior_residual_scale,
        bounded_learned_gap=bounded_learned_gap,
        equalize_feature_magnitude=equalize_feature_magnitude,
        disable_thermal_prior=disable_thermal_prior,
        disable_sat_uncertainty=disable_sat_uncertainty,
        disable_smoke_prior=disable_smoke_prior,
        fixed_equal_weights=fixed_equal_weights,
    ).to(device)
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
    model.eval()

    rows: list[dict[str, object]] = []
    with torch.inference_mode():
        for batch in loader:
            name = str(batch["name"][0])
            if name not in metadata:
                raise KeyError(f"Missing metadata for sample '{name}'")
            region_masks = None
            if label_dir is not None:
                label_path = label_dir / f"{name}.json"
                if not label_path.exists():
                    continue
                region_masks = load_label_masks(label_path, args.height, args.width)
            batch = move_batch_to_device(batch, device)
            row: dict[str, object] = {"name": name, **metadata[name]}
            row.update(analyze_batch(model, batch, region_masks))
            rows.append(row)
            if len(rows) % 10 == 0:
                print(f"Analyzed {len(rows)} samples")

    if label_dir is not None and not any("region_smoke_fraction" in row for row in rows):
        raise ValueError(f"No matching LabelMe JSON files found in: {label_dir}")
    print(f"Analyzed {len(rows)} samples")
    write_report(
        output,
        checkpoint_path,
        manifest,
        device,
        rows,
        smoke_prior_mode,
        prior_residual_scale,
        bounded_learned_gap,
        equalize_feature_magnitude,
        disable_thermal_prior,
        disable_sat_uncertainty,
        disable_smoke_prior,
        fixed_equal_weights,
        label_dir,
    )
    print(f"Wrote report: {output}")


if __name__ == "__main__":
    main()
