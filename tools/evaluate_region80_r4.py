#!/usr/bin/env python3
"""Run the frozen P-002 R4 regional evaluation for A0-A3."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import math
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image
from scipy.stats import rankdata
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.sample_index import read_manifest  # noqa: E402
from datasets.transforms import preprocess_pair  # noqa: E402
from eval import read_gray_01  # noqa: E402
from metrics.fusion_metrics import qabf_standard_region, scd_region  # noqa: E402
from test import load_model  # noqa: E402
from tools.diagnose_mec_priors import analyze_batch  # noqa: E402
from tools.prepare_region80_r4 import load_frozen_evaluation_masks  # noqa: E402
from utils import resolve_device  # noqa: E402


MODE_IDS = ("A0", "A1", "A2", "A3")
COMPARISONS = (("A0", "A1"), ("A0", "A2"), ("A0", "A3"), ("A1", "A3"))
SUMMARY_METRICS = (
    "region_qabf_standard",
    "region_scd",
    "region_range_excess_255",
    "region_w_ir_delta",
    "region_effective_ir_ratio_delta",
    "region_smoke_prior_delta",
    "region_thermal_prior_delta",
    "region_saturation_prior_delta",
    "full_range_excess_255",
    "smoke_prior_global_mean",
    "smoke_prior_high_090_fraction",
    "w_ir_global_mean",
    "effective_ir_ratio_global_mean",
    "thermal_celsius_spearman",
    "thermal_top10_delta",
)
PAIR_METRICS = {
    "region_qabf_standard": "higher_is_better",
    "region_scd": "higher_is_better",
    "region_range_excess_255": "lower_is_better",
    "region_w_ir_delta": "higher_means_more_relative_ir_routing",
    "region_effective_ir_ratio_delta": "higher_means_more_relative_ir_contribution",
    "full_range_excess_255": "lower_is_better",
    "w_ir_global_mean": "descriptive",
    "effective_ir_ratio_global_mean": "descriptive",
}
ANALYSIS_REGION_KEYS = {
    "region_thermal_delta": "region_thermal_prior_delta",
    "region_saturation_delta": "region_saturation_prior_delta",
    "region_smoke_delta": "region_smoke_prior_delta",
    "region_w_ir_delta": "region_w_ir_delta",
    "region_effective_ir_ratio_delta": "region_effective_ir_ratio_delta",
    "region_learned_logit_gap_delta": "region_learned_logit_gap_delta",
    "region_prior_logit_gap_delta": "region_prior_logit_gap_delta",
    "region_total_logit_gap_delta": "region_total_logit_gap_delta",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [{key: (value or "").strip() for key, value in row.items()} for row in csv.DictReader(handle)]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"yes", "true", "1"}:
        return True
    if normalized in {"no", "false", "0"}:
        return False
    raise ValueError(f"Invalid boolean value: {value!r}")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    fields = list(rows[0])
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: "" if isinstance(value, float) and not math.isfinite(value) else value for key, value in row.items()})


def brightness(ir_mean: float, vis_mean: float, fused_mean: float) -> tuple[int, int, int, float]:
    low, high = sorted((ir_mean, vis_mean))
    if fused_mean > high:
        return 1, 1, 0, (fused_mean - high) * 255.0
    if fused_mean < low:
        return 1, 0, 1, (low - fused_mean) * 255.0
    return 0, 0, 0, 0.0


def quantize_01(tensor: torch.Tensor) -> np.ndarray:
    array = tensor.detach().cpu().clamp(0.0, 1.0).squeeze().numpy()
    return np.rint(array * 255.0).astype(np.uint8).astype(np.float64) / 255.0


def masked_mean(image: np.ndarray, mask: np.ndarray) -> float:
    return float(np.mean(image[mask])) if np.any(mask) else float("nan")


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2 or b.size != a.size:
        return float("nan")
    rank_a = rankdata(a)
    rank_b = rankdata(b)
    rank_a -= np.mean(rank_a)
    rank_b -= np.mean(rank_b)
    denominator = np.sqrt(np.sum(rank_a * rank_a) * np.sum(rank_b * rank_b))
    return float(np.sum(rank_a * rank_b) / denominator) if denominator > 0.0 else float("nan")


def load_region_masks(mask_path: Path) -> dict[str, np.ndarray]:
    masks = {
        name: np.asarray(image, dtype=np.uint8) > 0
        for name, image in load_frozen_evaluation_masks(mask_path).items()
    }
    if (
        np.any(masks["smoke_eroded"] & ~masks["smoke"])
        or np.any(masks["smoke"] & masks["background"])
        or np.any((masks["smoke"] | masks["background"]) & ~masks["valid"])
    ):
        raise ValueError(f"Invalid frozen mask relationships: {mask_path}")
    return masks


def read_mask_asset_hashes(mask_assets_dir: Path) -> dict[str, str]:
    path = mask_assets_dir / "checksums.sha256"
    hashes: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split("  *", 1)
        if len(parts) != 2 or len(parts[0]) != 64 or not parts[1].startswith("masks/"):
            raise ValueError(f"Invalid mask asset checksum line: {line}")
        if parts[1] in hashes:
            raise ValueError(f"Duplicate mask asset checksum path: {parts[1]}")
        hashes[parts[1]] = parts[0].lower()
    return hashes


def load_celsius(path: Path, expected_shape: tuple[int, int]) -> np.ndarray:
    with Image.open(path) as image:
        array = np.asarray(image, dtype=np.float64)
    array = np.squeeze(array)
    if array.shape != expected_shape:
        raise ValueError(f"Celsius TIFF shape mismatch: {path}, got {array.shape}, expected {expected_shape}")
    return array


def validate_inputs(
    data_root: Path,
    calibration_manifest: Path,
    mask_manifest: Path,
    mask_assets_dir: Path,
    run_registry: Path,
) -> tuple[list[dict[str, str]], dict[str, object], dict[str, dict[str, str]], list[dict[str, str]]]:
    mask_rows = read_csv(mask_manifest)
    if len(mask_rows) != 80 or len({row["name"] for row in mask_rows}) != 80:
        raise ValueError("Mask manifest must contain 80 unique rows")
    expected_statuses = {"usable_smoke_mask": 69, "full_frame_unsegmentable": 3, "empty_negative": 8}
    actual_statuses = {status: sum(row["mask_status"] == status for row in mask_rows) for status in expected_statuses}
    if actual_statuses != expected_statuses:
        raise ValueError(f"Mask status counts changed: {actual_statuses}")
    if sum(row["mechanism_region_eligible"] == "yes" for row in mask_rows) != 69:
        raise ValueError("Mechanism-region denominator changed")
    if sum(row["edge_region_eligible"] == "yes" for row in mask_rows) != 69:
        raise ValueError("Edge-region denominator changed")

    asset_hashes = read_mask_asset_hashes(mask_assets_dir)
    expected_assets = {f"masks/{row['name']}.png" for row in mask_rows}
    if set(asset_hashes) != expected_assets:
        raise ValueError("Frozen mask asset set must match the 80-row mask manifest")

    for row in mask_rows:
        label_path = (mask_manifest.parent / row["label_json"]).resolve()
        if not label_path.is_file() or sha256(label_path).lower() != row["label_json_sha256"].lower():
            raise ValueError(f"Label JSON missing or changed: {row['name']}")
        relative_mask = f"masks/{row['name']}.png"
        mask_path = mask_assets_dir / relative_mask
        if not mask_path.is_file() or sha256(mask_path).lower() != asset_hashes[relative_mask]:
            raise ValueError(f"Frozen mask asset missing or changed: {row['name']}")
        masks = load_region_masks(mask_path)
        observed = {
            "smoke_pixels_raw": int(masks["smoke"].sum()),
            "smoke_pixels_eroded_1px": int(masks["smoke_eroded"].sum()),
            "background_pixels": int(masks["background"].sum()),
        }
        if any(observed[key] != int(row[key]) for key in observed):
            expected = {key: int(row[key]) for key in observed}
            raise ValueError(f"Frozen mask asset pixel count changed: {row['name']}, observed={observed}, expected={expected}")
        row["mask_resolved"] = str(mask_path.resolve())

    calibration_rows = read_csv(calibration_manifest)
    calibration_meta = {row["name"]: row for row in calibration_rows}
    if len(calibration_meta) != len(calibration_rows) or not {row["name"] for row in mask_rows} <= set(calibration_meta):
        raise ValueError("Calibration manifest does not contain all unique region80 names")
    records = {record.name: record for record in read_manifest(calibration_manifest, data_root)}
    for row in mask_rows:
        name = row["name"]
        meta = calibration_meta[name]
        if meta.get("source_class") != row["source_class"] or meta.get("sequence_id") != row["sequence_id"]:
            raise ValueError(f"Calibration metadata differs from frozen region80: {name}")
        record = records[name]
        tiff = data_root / meta["tiff"]
        if not record.ir.is_file() or not record.vis.is_file() or not tiff.is_file():
            raise FileNotFoundError(f"Missing IR/VIS/Celsius input for {name}")

    registry_rows = read_csv(run_registry)
    if tuple(row["id"] for row in registry_rows) != MODE_IDS:
        raise ValueError("Run registry must contain A0-A3 in order")
    for row in registry_rows:
        checkpoint = resolve_project_path(row["checkpoint"])
        actual_hash = sha256(checkpoint).upper() if checkpoint.is_file() else ""
        if actual_hash != row["checkpoint_sha256"].upper():
            raise ValueError(f"Checkpoint missing or hash mismatch: {row['id']}")
        row["checkpoint_resolved"] = str(checkpoint.resolve())
        row["checkpoint_sha256_actual"] = actual_hash

    return mask_rows, records, calibration_meta, registry_rows


def verify_model_identity(model: object, spec: dict[str, str]) -> None:
    checks = {
        "feature_channels": int(spec["feature_channels"]) == model.mec.feature_channels,
        "smoke_prior_mode": spec["smoke_prior_mode"] == model.prior_module.smoke_prior.mode,
        "prior_residual_scale": math.isclose(float(spec["prior_residual_scale"]), model.mec.prior_residual_scale),
        "bounded_learned_gap": parse_bool(spec["bounded_learned_gap"]) == model.mec.bounded_learned_gap,
        "equalize_feature_magnitude": parse_bool(spec["equalize_feature_magnitude"]) == model.mec.equalize_feature_magnitude,
        "disable_thermal_prior": parse_bool(spec["disable_thermal_prior"]) == model.mec.disable_thermal_prior,
        "disable_sat_uncertainty": parse_bool(spec["disable_sat_uncertainty"]) == model.mec.disable_sat_uncertainty,
        "disable_smoke_prior": parse_bool(spec["disable_smoke_prior"]) == model.mec.disable_smoke_prior,
        "fixed_equal_weights": parse_bool(spec["fixed_equal_weights"]) == model.mec.fixed_equal_weights,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"Checkpoint configuration mismatch for {spec['id']}: {failed}")


def model_args(spec: dict[str, str]) -> argparse.Namespace:
    return argparse.Namespace(
        checkpoint=spec["checkpoint_resolved"],
        feature_channels=int(spec["feature_channels"]),
        smoke_prior_mode=None,
        prior_residual_scale=None,
        bounded_learned_gap=None,
        equalize_feature_magnitude=None,
    )


def thermal_stats(thermal_prior: np.ndarray, celsius: np.ndarray, valid: np.ndarray) -> tuple[int, float, float, float]:
    usable = valid & np.isfinite(celsius)
    count = int(usable.sum())
    if count < 256:
        return count, float("nan"), float("nan"), float("nan")
    values = celsius[usable]
    prior_values = thermal_prior[usable]
    threshold = float(np.quantile(values, 0.9))
    high = usable & (celsius >= threshold)
    rest = usable & ~high
    delta = masked_mean(thermal_prior, high) - masked_mean(thermal_prior, rest) if np.any(rest) else float("nan")
    return count, threshold, spearman(values, prior_values), delta


def analyze_one(
    mode: str,
    tag: str,
    model: object,
    record: object,
    meta: dict[str, str],
    ledger: dict[str, str],
    mask_path: Path,
    celsius_path: Path,
    device: torch.device,
    height: int,
    width: int,
) -> dict[str, object]:
    ir_tensor, vis_tensor, _ = preprocess_pair(record.ir, record.vis, height=height, width=width)
    batch = {"ir": ir_tensor.unsqueeze(0).to(device), "vis": vis_tensor.unsqueeze(0).to(device), "name": [record.name]}
    masks = load_region_masks(mask_path)
    if masks["valid"].shape != (height, width):
        raise ValueError(f"Mask shape mismatch: {record.name}")
    region_masks = None
    if ledger["mask_status"] == "usable_smoke_mask":
        region_masks = (
            torch.from_numpy(masks["smoke"]).reshape(1, 1, height, width),
            torch.from_numpy(masks["valid"]).reshape(1, 1, height, width),
        )

    with torch.inference_mode():
        diagnostics = analyze_batch(model, batch, region_masks)
        outputs = model(batch["ir"], batch["vis"])

    ir = read_gray_01(record.ir)
    vis = read_gray_01(record.vis)
    if ir.shape != (height, width) or vis.shape != (height, width):
        raise ValueError(f"Input image shape mismatch: {record.name}")
    fused = quantize_01(outputs["fused"])
    thermal_prior = outputs["thermal_prior"].detach().cpu().squeeze().numpy().astype(np.float64)
    smoke_prior = outputs["smoke_prior"].detach().cpu().squeeze().numpy().astype(np.float64)
    celsius = load_celsius(celsius_path, (height, width))
    thermal_count, thermal_threshold, thermal_corr, thermal_delta = thermal_stats(thermal_prior, celsius, masks["valid"])

    ir_mean, vis_mean, fused_mean = float(ir.mean()), float(vis.mean()), float(fused.mean())
    full_outside, full_above, full_below, full_excess = brightness(ir_mean, vis_mean, fused_mean)
    row: dict[str, object] = {
        "mode": mode,
        "tag": tag,
        "name": record.name,
        "sequence_id": meta["sequence_id"],
        "source_class": meta["source_class"],
        "mask_status": ledger["mask_status"],
        "smoke_pixels_raw": int(ledger["smoke_pixels_raw"]),
        "smoke_pixels_eroded_1px": int(ledger["smoke_pixels_eroded_1px"]),
        "background_pixels": int(ledger["background_pixels"]),
        "full_ir_mean_255": ir_mean * 255.0,
        "full_vis_mean_255": vis_mean * 255.0,
        "full_fused_mean_255": fused_mean * 255.0,
        "full_outside_input_mean_range": full_outside,
        "full_above_input_mean_range": full_above,
        "full_below_input_mean_range": full_below,
        "full_range_excess_255": full_excess,
        "thermal_prior_global_mean": float(thermal_prior.mean()),
        "saturation_prior_global_mean": float(outputs["sat_uncertainty"].mean().item()),
        "smoke_prior_global_mean": float(smoke_prior.mean()),
        "smoke_prior_high_090_fraction": float(np.mean(smoke_prior >= 0.9)),
        "w_ir_global_mean": float(diagnostics["w_ir_mean"]),
        "effective_ir_ratio_global_mean": float(diagnostics["effective_ir_ratio_mean"]),
        "zero_smoke_prior_delta_wir": float(diagnostics["zero_smoke_delta_wir"]),
        "thermal_valid_pixels": thermal_count,
        "thermal_top10_threshold_celsius": thermal_threshold,
        "thermal_celsius_spearman": thermal_corr,
        "thermal_top10_delta": thermal_delta,
    }

    for source_key, output_key in ANALYSIS_REGION_KEYS.items():
        row[output_key] = float(diagnostics.get(source_key, float("nan")))

    if ledger["mask_status"] == "usable_smoke_mask":
        smoke = masks["smoke"]
        ir_region = masked_mean(ir, smoke)
        vis_region = masked_mean(vis, smoke)
        fused_region = masked_mean(fused, smoke)
        outside, above, below, excess = brightness(ir_region, vis_region, fused_region)
        row.update({
            "region_ir_mean_255": ir_region * 255.0,
            "region_vis_mean_255": vis_region * 255.0,
            "region_fused_mean_255": fused_region * 255.0,
            "region_outside_input_mean_range": outside,
            "region_above_input_mean_range": above,
            "region_below_input_mean_range": below,
            "region_range_excess_255": excess,
            "region_qabf_standard": qabf_standard_region(ir, vis, fused, masks["smoke_eroded"]),
            "region_scd": scd_region(ir, vis, fused, smoke),
        })
    else:
        for key in (
            "region_ir_mean_255", "region_vis_mean_255", "region_fused_mean_255",
            "region_outside_input_mean_range", "region_above_input_mean_range",
            "region_below_input_mean_range", "region_range_excess_255",
            "region_qabf_standard", "region_scd",
        ):
            row[key] = float("nan")
    return row


def groups(rows: list[dict[str, object]]) -> list[tuple[str, list[dict[str, object]]]]:
    return [
        ("Fire_All", [row for row in rows if row["source_class"] == "Fire"]),
        ("Fire_SEQ_07", [row for row in rows if row["source_class"] == "Fire" and row["sequence_id"] == "SEQ_07"]),
        ("Fire_SEQ_10", [row for row in rows if row["source_class"] == "Fire" and row["sequence_id"] == "SEQ_10"]),
        ("NoFire", [row for row in rows if row["source_class"] == "No Fire"]),
    ]


def summarize(per_image: list[dict[str, object]]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for mode in MODE_IDS:
        mode_rows = [row for row in per_image if row["mode"] == mode]
        for group, group_rows in groups(mode_rows):
            for metric in SUMMARY_METRICS:
                values = np.asarray([float(row[metric]) for row in group_rows], dtype=np.float64)
                values = values[np.isfinite(values)]
                if not values.size:
                    continue
                output.append({
                    "mode": mode,
                    "group": group,
                    "metric": metric,
                    "n": int(values.size),
                    "mean": float(np.mean(values)),
                    "sd_population": float(np.std(values)),
                    "median": float(np.median(values)),
                    "q1": float(np.quantile(values, 0.25)),
                    "q3": float(np.quantile(values, 0.75)),
                })
    return output


def paired_differences(per_image: list[dict[str, object]]) -> list[dict[str, object]]:
    index = {(str(row["mode"]), str(row["name"])): row for row in per_image}
    output: list[dict[str, object]] = []
    for left, right in COMPARISONS:
        left_rows = [row for row in per_image if row["mode"] == left]
        for group, group_rows in groups(left_rows):
            for metric, direction in PAIR_METRICS.items():
                deltas = []
                for row in group_rows:
                    other = index[(right, str(row["name"]))]
                    left_value, right_value = float(row[metric]), float(other[metric])
                    if math.isfinite(left_value) and math.isfinite(right_value):
                        deltas.append(left_value - right_value)
                if not deltas:
                    continue
                values = np.asarray(deltas, dtype=np.float64)
                output.append({
                    "comparison": f"{left}-{right}",
                    "group": group,
                    "metric": metric,
                    "direction": direction,
                    "n": int(values.size),
                    "mean_delta": float(np.mean(values)),
                    "sd_population": float(np.std(values)),
                    "median_delta": float(np.median(values)),
                    "q1_delta": float(np.quantile(values, 0.25)),
                    "q3_delta": float(np.quantile(values, 0.75)),
                    "positive_count": int(np.sum(values > 0.0)),
                    "negative_count": int(np.sum(values < 0.0)),
                    "zero_count": int(np.sum(values == 0.0)),
                    "positive_fraction": float(np.mean(values > 0.0)),
                })
    return output


def write_report(path: Path, registry: list[dict[str, str]]) -> None:
    identity_lines = ["| ID | 变体 | checkpoint SHA256 |", "|---|---|---|"]
    identity_lines.extend(f"| {row['id']} | {row['tag']} | `{row['checkpoint_sha256_actual']}` |" for row in registry)
    lines = [
        "# P-002 R4：区域评价原始结果报告",
        "",
        "> 状态：`RAW_RESULTS_PENDING_PEER_REVIEW`  ",
        "> 自动汇总不等于论文结论；必须审查逐图、分序列方向与失败案例。",
        "",
        "## 1. 冻结身份",
        "",
        *identity_lines,
        "",
        "## 2. 输出文件",
        "",
        "- `per_image_metrics.csv`：A0–A3 × 80张的逐图区域、机制、亮度和thermal代理指标。",
        "- `group_summary.csv`：按Fire All、Fire SEQ_07、Fire SEQ_10、No Fire进行描述统计。",
        "- `paired_differences.csv`：A0-A1、A0-A2、A0-A3、A1-A3逐图配对差值；不包含p值。",
        "- `run_identity.csv`：实际checkpoint路径、哈希与配置。",
        "",
        "## 3. 必须采用的审查顺序",
        "",
        "1. 先核对A0-A2在SEQ_07与SEQ_10的`region_w_ir_delta`和`region_effective_ir_ratio_delta`方向。",
        "2. 再核对A0-A2区域QAB/F的均值、median和正向比例；两序列冲突时只能报告场景依赖。",
        "3. 联合检查区域SCD与亮度越界，不能由单个指标宣布质量提升。",
        "4. A0-A1只支持三类prior整体作用；无人工overexposure阳性，不能单独归因saturation质量收益。",
        "5. A0-A3和A1-A3分别用于完整prior引导动态MEC及feature-only动态MEC相对固定等权的边界。",
        "",
        "## 4. 停止规则",
        "",
        "A0-A2只有在两条Fire序列的路由方向一致、区域QAB/F均值和median方向一致且亮度/代表性伪影未恶化时，才可写受限区域质量改善。若只有路由响应而质量混合，则只保留机制结论；若方向冲突或质量为负，不调参、不换样，原样报告。",
        "",
        "同一序列帧不是独立重复，本报告不输出逐帧显著性检验。三个`full_frame_unsegmentable`区域指标固定NA，八个No Fire只作阴性压力测试。",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Output already exists; refusing to overwrite: {output_dir}")
    data_root = Path(args.data_root)
    mask_manifest = Path(args.mask_manifest)
    mask_assets_dir = Path(args.mask_assets_dir)
    mask_rows, records, calibration_meta, registry = validate_inputs(
        data_root,
        Path(args.calibration_manifest),
        mask_manifest,
        mask_assets_dir,
        Path(args.run_registry),
    )
    device = resolve_device(args.device)
    if args.check_inputs_only:
        for spec in registry:
            model = load_model(model_args(spec), device)
            verify_model_identity(model, spec)
            print(f"{spec['id']}_MODEL_IDENTITY_PASSED")
            del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print("P002_R4_INPUT_GATE_PASSED")
        for row in registry:
            print(f"{row['id']}={row['checkpoint_sha256_actual']}")
        return

    ledger = {row["name"]: row for row in mask_rows}
    per_image: list[dict[str, object]] = []
    for spec in registry:
        model = load_model(model_args(spec), device)
        verify_model_identity(model, spec)
        print(f"Running {spec['id']} ({spec['tag']})")
        for index, item in enumerate(mask_rows, 1):
            name = item["name"]
            meta = calibration_meta[name]
            per_image.append(analyze_one(
                spec["id"], spec["tag"], model, records[name], meta, ledger[name],
                Path(item["mask_resolved"]), data_root / meta["tiff"], device, args.height, args.width,
            ))
            if index % 10 == 0 or index == len(mask_rows):
                print(f"{spec['id']}: {index}/{len(mask_rows)}")
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if len(per_image) != 320:
        raise RuntimeError(f"Expected 320 per-image rows, got {len(per_image)}")
    summary = summarize(per_image)
    paired = paired_differences(per_image)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f".{output_dir.name}_", dir=output_dir.parent) as temporary:
        target = Path(temporary)
        write_csv(target / "per_image_metrics.csv", per_image)
        write_csv(target / "group_summary.csv", summary)
        write_csv(target / "paired_differences.csv", paired)
        write_csv(target / "run_identity.csv", registry)
        write_report(target / "R4_REGION_EVALUATION_RAW_REPORT.md", registry)
        files = sorted(path for path in target.iterdir() if path.is_file())
        (target / "checksums.sha256").write_text(
            "".join(f"{sha256(path)}  *{path.name}\n" for path in files),
            encoding="utf-8",
        )
        target.replace(output_dir)
    print("P002_R4_EVALUATION_COMPLETED_PENDING_REVIEW")
    print(f"output_dir={output_dir.resolve()}")


def self_check() -> None:
    outside = brightness(10 / 255, 20 / 255, 25 / 255)
    assert outside[:3] == (1, 1, 0) and math.isclose(outside[3], 5.0)
    assert brightness(10 / 255, 20 / 255, 15 / 255) == (0, 0, 0, 0.0)
    tensor = torch.tensor([[[[0.0, 0.5, 1.0]]]])
    assert np.array_equal(quantize_01(tensor), np.asarray([0.0, 128 / 255, 1.0]))
    assert math.isclose(spearman(np.arange(5.0), np.arange(5.0)), 1.0)
    synthetic = []
    for mode, value in (("A0", 2.0), ("A1", 1.0), ("A2", 1.5), ("A3", 0.5)):
        row = {"mode": mode, "name": "F_TEST", "source_class": "Fire", "sequence_id": "SEQ_10"}
        row.update({metric: value for metric in SUMMARY_METRICS})
        synthetic.append(row)
    assert summarize(synthetic)
    paired = paired_differences(synthetic)
    assert any(row["comparison"] == "A0-A2" and row["mean_delta"] == 0.5 for row in paired)
    print("evaluate_region80_r4 self-check passed")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate frozen P-002 region80 masks with A0-A3 checkpoints.")
    parser.add_argument("--data_root")
    parser.add_argument("--calibration_manifest")
    parser.add_argument("--mask_manifest")
    parser.add_argument("--mask_assets_dir")
    parser.add_argument("--run_registry")
    parser.add_argument("--output_dir")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--check_inputs_only", action="store_true")
    parser.add_argument("--self_check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        self_check()
        return
    required = (
        args.data_root,
        args.calibration_manifest,
        args.mask_manifest,
        args.mask_assets_dir,
        args.run_registry,
        args.output_dir,
    )
    if not all(required):
        raise ValueError("All path arguments are required")
    run(args)


if __name__ == "__main__":
    main()
