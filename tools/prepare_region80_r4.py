#!/usr/bin/env python3
"""Freeze the P-002 region80 mask ledger before regional evaluation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageOps


ALLOWED_LABELS = {"smoke_affected", "overexposure", "ignore"}
EXPECTED_BATCHES = Counter({"batch20": 20, "batch60": 60})
EVALUATION_MASK_BITS = {"valid": 1, "smoke": 2, "smoke_eroded": 4, "background": 8}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [{key: (value or "").strip() for key, value in row.items()} for row in csv.DictReader(handle)]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def count_pixels(mask: Image.Image) -> int:
    return mask.histogram()[255]


def rasterize_label_masks(path: Path) -> tuple[dict[str, int], dict[str, Image.Image]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    width, height = int(data.get("imageWidth", 0)), int(data.get("imageHeight", 0))
    if (width, height) != (640, 512):
        raise ValueError(f"Unexpected image size in {path}: {(width, height)}")

    masks = {label: Image.new("L", (width, height), 0) for label in ALLOWED_LABELS}
    counts = Counter()
    for index, shape in enumerate(data.get("shapes", []), 1):
        label = str(shape.get("label", ""))
        if label not in ALLOWED_LABELS or shape.get("shape_type") != "polygon":
            raise ValueError(f"Unsupported label or shape at {path}:{index}")
        # Freeze LabelMe subpixel vertices to integer pixels before Pillow sees them.
        # int() preserves the truncation rule used to create the original R4 ledger.
        points = [tuple(int(float(coordinate)) for coordinate in point) for point in shape.get("points", [])]
        if len(points) < 3:
            raise ValueError(f"Polygon has fewer than three points at {path}:{index}")
        ImageDraw.Draw(masks[label]).polygon(points, fill=255)
        counts[label] += 1

    return dict(counts), masks


def build_evaluation_masks(path: Path) -> tuple[dict[str, int], dict[str, Image.Image]]:
    counts, masks = rasterize_label_masks(path)
    smoke = masks["smoke_affected"]
    overexposure = masks["overexposure"]
    ignore = masks["ignore"]
    valid = ImageOps.invert(ignore)
    smoke = ImageChops.multiply(smoke, valid)
    overexposure = ImageChops.multiply(overexposure, valid)
    overlap = ImageChops.multiply(smoke, overexposure)
    smoke_only = ImageChops.multiply(smoke, ImageOps.invert(overexposure))
    overexposure_only = ImageChops.multiply(overexposure, ImageOps.invert(smoke))
    background = ImageChops.multiply(valid, ImageOps.invert(ImageChops.lighter(smoke, overexposure)))
    eroded_smoke = smoke.filter(ImageFilter.MinFilter(3))
    return counts, {
        "valid": valid,
        "smoke": smoke,
        "smoke_eroded": eroded_smoke,
        "background": background,
        "smoke_only": smoke_only,
        "overexposure_only": overexposure_only,
        "overlap": overlap,
        "ignore": ignore,
    }


def load_masks(path: Path) -> tuple[dict[str, int], dict[str, int]]:
    counts, masks = build_evaluation_masks(path)

    pixels = {
        "valid_pixels": count_pixels(masks["valid"]),
        "smoke_pixels_raw": count_pixels(masks["smoke"]),
        "smoke_pixels_eroded_1px": count_pixels(masks["smoke_eroded"]),
        "smoke_only_pixels": count_pixels(masks["smoke_only"]),
        "overexposure_only_pixels": count_pixels(masks["overexposure_only"]),
        "overlap_pixels": count_pixels(masks["overlap"]),
        "ignore_pixels": count_pixels(masks["ignore"]),
        "background_pixels": count_pixels(masks["background"]),
    }
    return counts, pixels


def encode_evaluation_masks(masks: dict[str, Image.Image]) -> Image.Image:
    encoded = Image.new("L", masks["valid"].size, 0)
    for name, bit in EVALUATION_MASK_BITS.items():
        encoded = ImageChops.add(encoded, masks[name].point(lambda value, flag=bit: flag if value else 0))
    return encoded


def load_frozen_evaluation_masks(path: Path) -> dict[str, Image.Image]:
    with Image.open(path) as image:
        encoded = image.convert("L")
    if encoded.size != (640, 512) or encoded.getextrema()[1] > 15:
        raise ValueError(f"Invalid frozen evaluation mask: {path}")
    return {
        name: encoded.point([255 if value & bit else 0 for value in range(256)])
        for name, bit in EVALUATION_MASK_BITS.items()
    }


def freeze_mask_assets(mask_manifest: Path, output_dir: Path) -> None:
    if output_dir.exists():
        raise FileExistsError(f"Output already exists; refusing to overwrite: {output_dir}")
    rows = read_csv(mask_manifest)
    if len(rows) != 80 or len({row["name"] for row in rows}) != 80:
        raise ValueError("Mask manifest must contain 80 unique rows")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f".{output_dir.name}_", dir=output_dir.parent) as temporary:
        target = Path(temporary)
        masks_dir = target / "masks"
        masks_dir.mkdir()
        for row in rows:
            label_path = (mask_manifest.parent / row["label_json"]).resolve()
            if not label_path.is_file() or sha256(label_path).lower() != row["label_json_sha256"].lower():
                raise ValueError(f"Label JSON missing or changed: {row['name']}")
            _, masks = build_evaluation_masks(label_path)
            observed = {
                "smoke_pixels_raw": count_pixels(masks["smoke"]),
                "smoke_pixels_eroded_1px": count_pixels(masks["smoke_eroded"]),
                "background_pixels": count_pixels(masks["background"]),
            }
            if any(observed[key] != int(row[key]) for key in observed):
                raise ValueError(f"Frozen mask pixel count changed before asset creation: {row['name']}")
            encode_evaluation_masks(masks).save(masks_dir / f"{row['name']}.png", format="PNG")

        mask_files = sorted(masks_dir.glob("*.png"))
        if len(mask_files) != 80:
            raise RuntimeError(f"Expected 80 frozen mask assets, got {len(mask_files)}")
        (target / "checksums.sha256").write_text(
            "".join(f"{sha256(path)}  *masks/{path.name}\n" for path in mask_files),
            encoding="utf-8",
        )
        (target / "README.md").write_text(
            "# R4 frozen evaluation masks\n\n"
            "Each 640x512 grayscale PNG stores bit flags: valid=1, smoke=2, "
            "smoke_eroded=4, background=8. The checksum file freezes all 80 PNG assets.\n",
            encoding="utf-8",
        )
        target.replace(output_dir)
    print("P002_R4_MASK_ASSETS_FROZEN")
    print(f"output_dir={output_dir.resolve()}")


def classify(
    row: dict[str, str],
    shape_counts: dict[str, int],
    pixels: dict[str, int],
    unsegmentable: set[str],
    min_region_pixels: int,
    min_background_pixels: int,
) -> dict[str, str]:
    name, source_class = row["name"], row["source_class"]
    smoke_polygons = shape_counts.get("smoke_affected", 0)
    total_shapes = sum(shape_counts.values())

    if name in unsegmentable:
        if source_class != "Fire" or total_shapes != 0:
            raise ValueError(f"Approved unsegmentable sample must be an empty Fire JSON: {name}")
        status = "full_frame_unsegmentable"
    elif source_class == "Fire" and smoke_polygons > 0:
        status = "usable_smoke_mask"
    elif source_class == "No Fire" and total_shapes == 0:
        status = "empty_negative"
    else:
        raise ValueError(f"Annotation status is inconsistent with frozen class: {name}")

    mechanism_eligible = (
        status == "usable_smoke_mask"
        and pixels["smoke_pixels_raw"] >= min_region_pixels
        and pixels["background_pixels"] >= min_background_pixels
    )
    edge_eligible = (
        status == "usable_smoke_mask"
        and pixels["smoke_pixels_eroded_1px"] >= min_region_pixels
        and pixels["background_pixels"] >= min_background_pixels
    )
    negative_eligible = status == "empty_negative" and pixels["valid_pixels"] >= min_background_pixels

    if status == "full_frame_unsegmentable":
        reason = "full_frame_unsegmentable"
    elif status == "empty_negative":
        reason = "not_applicable_empty_negative"
    elif pixels["background_pixels"] < min_background_pixels:
        reason = "background_too_small"
    elif pixels["smoke_pixels_raw"] < min_region_pixels:
        reason = "smoke_region_too_small"
    elif pixels["smoke_pixels_eroded_1px"] < min_region_pixels:
        reason = "eroded_smoke_region_too_small"
    else:
        reason = ""

    valid_pixels = pixels["valid_pixels"]
    return {
        "selection_index": row["selection_index"],
        "name": name,
        "sequence_id": row["sequence_id"],
        "source_class": source_class,
        "annotation_batch": row["annotation_batch"],
        "mask_status": status,
        "smoke_polygons": str(smoke_polygons),
        "overexposure_polygons": str(shape_counts.get("overexposure", 0)),
        "ignore_polygons": str(shape_counts.get("ignore", 0)),
        **{key: str(value) for key, value in pixels.items()},
        "smoke_fraction_raw": f"{pixels['smoke_pixels_raw'] / valid_pixels:.8f}" if valid_pixels else "",
        "mechanism_region_eligible": "yes" if mechanism_eligible else "no",
        "edge_region_eligible": "yes" if edge_eligible else "no",
        "negative_control_eligible": "yes" if negative_eligible else "no",
        "exclusion_reason": reason,
    }


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build(
    region_manifest: Path,
    batch20_dir: Path,
    batch60_dir: Path,
    output_dir: Path,
    unsegmentable: set[str],
    min_region_pixels: int,
    min_background_pixels: int,
) -> None:
    if output_dir.exists():
        raise FileExistsError(f"Output already exists; refusing to overwrite: {output_dir}")
    if min_region_pixels <= 0 or min_background_pixels <= 0:
        raise ValueError("Pixel thresholds must be positive")

    rows = read_csv(region_manifest)
    if len(rows) != 80 or len({row["name"] for row in rows}) != 80:
        raise ValueError("Frozen region manifest must contain 80 unique rows")
    if Counter(row["annotation_batch"] for row in rows) != EXPECTED_BATCHES:
        raise ValueError("Frozen region manifest must contain batch20=20 and batch60=60")
    fire_names = {row["name"] for row in rows if row["source_class"] == "Fire"}
    if not unsegmentable or not unsegmentable <= fire_names:
        raise ValueError("Unsegmentable names must be a non-empty subset of frozen Fire samples")

    records: list[dict[str, str]] = []
    for row in sorted(rows, key=lambda item: int(item["selection_index"])):
        annotation_dir = batch20_dir if row["annotation_batch"] == "batch20" else batch60_dir
        label_path = annotation_dir / "labels_json" / f"{row['name']}.json"
        if not label_path.is_file():
            raise FileNotFoundError(label_path)
        shape_counts, pixels = load_masks(label_path)
        record = classify(row, shape_counts, pixels, unsegmentable, min_region_pixels, min_background_pixels)
        record["label_json"] = f"../{annotation_dir.name}/labels_json/{row['name']}.json"
        record["label_json_sha256"] = sha256(label_path)
        records.append(record)

    statuses = Counter(row["mask_status"] for row in records)
    if statuses != Counter({"usable_smoke_mask": 69, "full_frame_unsegmentable": 3, "empty_negative": 8}):
        raise ValueError(f"Unexpected frozen mask statuses: {dict(statuses)}")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f".{output_dir.name}_", dir=output_dir.parent) as temporary:
        target = Path(temporary)
        ledger = target / "region80_mask_manifest.csv"
        write_csv(ledger, records)

        mechanism_count = sum(row["mechanism_region_eligible"] == "yes" for row in records)
        edge_count = sum(row["edge_region_eligible"] == "yes" for row in records)
        negative_count = sum(row["negative_control_eligible"] == "yes" for row in records)
        usable = [row for row in records if row["mask_status"] == "usable_smoke_mask"]
        min_raw_smoke = min(int(row["smoke_pixels_raw"]) for row in usable)
        min_eroded_smoke = min(int(row["smoke_pixels_eroded_1px"]) for row in usable)
        min_background = min(int(row["background_pixels"]) for row in usable)
        max_smoke_fraction = max(float(row["smoke_fraction_raw"]) for row in usable)
        sequence_counts = Counter((row["sequence_id"], row["mask_status"]) for row in records)
        sequence_lines = ["| 序列 | mask状态 | 数量 |", "|---|---|---:|"]
        sequence_lines.extend(
            f"| {sequence} | {status} | {count} |"
            for (sequence, status), count in sorted(sequence_counts.items())
        )
        report = "\n".join([
            "# P-002 R4：region80区域评价前冻结报告",
            "",
            "> 状态：`R4_PREFLIGHT_PASSED / R4_EVALUATION_PROTOCOL_PENDING`  ",
            "> 本报告不包含A0–A3区域性能结果。",
            "",
            "## 1. 具体目标",
            "",
            "在读取区域性能结果前，合并R2/R3已验收标签，冻结每张图的mask语义、像素有效性和统计排除原因。",
            "",
            "## 2. 冻结规则",
            "",
            f"- 图像为统计单位；输入固定为80张，不重新选样。",
            f"- MEC/prior区域均值使用原始人工mask；烟雾区至少{min_region_pixels}像素，有效背景至少{min_background_pixels}像素。",
            f"- QAB/F等边缘指标使用向内腐蚀1像素的烟雾mask；腐蚀后仍须至少{min_region_pixels}像素，有效背景至少{min_background_pixels}像素。",
            "- 背景为排除smoke、overexposure与ignore后的有效像素；不把腐蚀掉的人工边界环重新计入背景。",
            "- `full_frame_unsegmentable`的区域指标固定记NA；`empty_negative`只进入No Fire假响应/压力测试，不进入smoke区域减背景统计。",
            "",
            "## 3. 输入审计结果",
            "",
            "- 冻结样本：80；batch20=20，batch60=60。",
            f"- mask状态：usable_smoke_mask={statuses['usable_smoke_mask']}，full_frame_unsegmentable={statuses['full_frame_unsegmentable']}，empty_negative={statuses['empty_negative']}。",
            f"- MEC/prior区域统计当前可用：{mechanism_count}张Fire。",
            f"- 腐蚀后边缘区域统计当前可用：{edge_count}张Fire。",
            f"- No Fire阴性压力测试可用：{negative_count}张。",
            f"- 可用Fire的最小原始烟雾区={min_raw_smoke}像素，最小腐蚀后烟雾区={min_eroded_smoke}像素，最小有效背景={min_background}像素，最大烟雾占比={max_smoke_fraction:.8f}。",
            "",
            *sequence_lines,
            "",
            "## 4. 输出与边界",
            "",
            "- `region80_mask_manifest.csv`逐图记录label路径及哈希、mask状态、区域/背景像素数、两类可用性和排除原因。",
            "- `checksums.sha256`冻结本次预检输出；后续评价不得就地改写清单。",
            "- 人工overexposure阳性仍为0，因此不能开展或声称saturation人工区域校准。",
            "- 本阶段只证明区域评价输入可用，不证明smoke prior、MEC或融合质量有效。",
            "- 69张可用Fire来自SEQ_07的13张和SEQ_10的56张，序列明显不均衡；后续须先分序列报告，合并结果只作描述性汇总。",
            "- 同一序列的相邻帧不是独立重复；不以逐帧p值声称统计显著，也不报告单标注者数据的标注者间一致性。",
            "",
            "## 5. 下一门禁",
            "",
            "先审查本清单的实际可用分母与排除原因；批准后再编写并静态检查R4区域评价命令，之后才可读取A0–A3输出。无需重训。",
            "",
            "## 6. 转移与回传",
            "",
            "本预检在本机完成，无需转移或回传文件。后续若在工作站运行区域评价，只转移获批的评价脚本、命令文档和本冻结目录；回传终端Markdown及完整R4输出目录，不回传模型训练run。",
            "",
        ])
        (target / "R4_PREFLIGHT_REPORT.md").write_text(report, encoding="utf-8")
        checksum_paths = sorted(path for path in target.iterdir() if path.is_file())
        checksum_text = "".join(f"{sha256(path)}  *{path.name}\n" for path in checksum_paths)
        (target / "checksums.sha256").write_text(checksum_text, encoding="utf-8")
        target.replace(output_dir)

    print("P002_R4_PREFLIGHT_PASSED")
    print(f"output_dir={output_dir.resolve()}")
    print(f"mechanism_region_eligible={mechanism_count}")
    print(f"edge_region_eligible={edge_count}")
    print(f"negative_control_eligible={negative_count}")


def self_check() -> None:
    pixels = {
        "valid_pixels": 400,
        "smoke_pixels_raw": 100,
        "smoke_pixels_eroded_1px": 64,
        "smoke_only_pixels": 100,
        "overexposure_only_pixels": 0,
        "overlap_pixels": 0,
        "ignore_pixels": 0,
        "background_pixels": 300,
    }
    row = {"selection_index": "1", "name": "F_TEST", "sequence_id": "SEQ_10", "source_class": "Fire", "annotation_batch": "batch20"}
    usable = classify(row, {"smoke_affected": 1}, pixels, set(), 64, 64)
    assert usable["mask_status"] == "usable_smoke_mask" and usable["edge_region_eligible"] == "yes"
    empty = classify({**row, "name": "F_EMPTY"}, {}, pixels | {"smoke_pixels_raw": 0, "smoke_pixels_eroded_1px": 0}, {"F_EMPTY"}, 64, 64)
    assert empty["mask_status"] == "full_frame_unsegmentable" and empty["mechanism_region_eligible"] == "no"
    negative = classify({**row, "name": "N_TEST", "source_class": "No Fire"}, {}, pixels | {"smoke_pixels_raw": 0, "smoke_pixels_eroded_1px": 0}, set(), 64, 64)
    assert negative["mask_status"] == "empty_negative" and negative["negative_control_eligible"] == "yes"
    with TemporaryDirectory() as temporary:
        label = Path(temporary) / "fractional.json"
        label.write_text(json.dumps({
            "imageWidth": 640,
            "imageHeight": 512,
            "shapes": [{
                "label": "smoke_affected",
                "shape_type": "polygon",
                "points": [[0.9, 0.9], [5.9, 0.9], [0.9, 5.9]],
            }],
        }), encoding="utf-8")
        _, masks = rasterize_label_masks(label)
        assert masks["smoke_affected"].getpixel((0, 0)) == 255
        assert masks["smoke_affected"].getpixel((6, 1)) == 0
        _, evaluation_masks = build_evaluation_masks(label)
        frozen = Path(temporary) / "mask.png"
        encode_evaluation_masks(evaluation_masks).save(frozen, format="PNG")
        restored = load_frozen_evaluation_masks(frozen)
        for name in EVALUATION_MASK_BITS:
            assert restored[name].tobytes() == evaluation_masks[name].tobytes()
    print("prepare_region80_r4 self-check passed")


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze P-002 region80 mask statuses and valid regional-evaluation denominators.")
    parser.add_argument("--region_manifest")
    parser.add_argument("--batch20_dir")
    parser.add_argument("--batch60_dir")
    parser.add_argument("--output_dir")
    parser.add_argument("--unsegmentable", default="")
    parser.add_argument("--min_region_pixels", type=int, default=256)
    parser.add_argument("--min_background_pixels", type=int, default=256)
    parser.add_argument("--freeze_from_mask_manifest")
    parser.add_argument("--mask_assets_dir")
    parser.add_argument("--self_check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        self_check()
        return
    if args.freeze_from_mask_manifest or args.mask_assets_dir:
        if not args.freeze_from_mask_manifest or not args.mask_assets_dir:
            raise ValueError("--freeze_from_mask_manifest and --mask_assets_dir are required together")
        freeze_mask_assets(Path(args.freeze_from_mask_manifest), Path(args.mask_assets_dir))
        return
    required = (args.region_manifest, args.batch20_dir, args.batch60_dir, args.output_dir)
    if not all(required):
        raise ValueError("--region_manifest, --batch20_dir, --batch60_dir and --output_dir are required")
    build(
        Path(args.region_manifest),
        Path(args.batch20_dir),
        Path(args.batch60_dir),
        Path(args.output_dir),
        {name.strip() for name in args.unsegmentable.split(",") if name.strip()},
        args.min_region_pixels,
        args.min_background_pixels,
    )


if __name__ == "__main__":
    main()
