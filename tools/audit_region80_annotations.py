#!/usr/bin/env python3
"""Validate a returned P-002 LabelMe annotation package and render review sheets."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
from collections import Counter
from pathlib import Path, PurePosixPath, PureWindowsPath
from tempfile import TemporaryDirectory

from PIL import Image, ImageDraw


ALLOWED_LABELS = {"smoke_affected", "overexposure", "ignore"}
COLORS = {
    "smoke_affected": (255, 165, 0, 100),
    "overexposure": (255, 0, 255, 100),
    "ignore": (0, 255, 255, 100),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as file:
        return [{key: (value or "").strip() for key, value in row.items()} for row in csv.DictReader(file)]


def write_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def polygon_area(points: list[tuple[float, float]]) -> float:
    return abs(sum(x1 * y2 - x2 * y1 for (x1, y1), (x2, y2) in zip(points, points[1:] + points[:1])) / 2.0)


def relative_files(root: Path) -> dict[str, Path]:
    return {path.relative_to(root).as_posix(): path for path in root.rglob("*") if path.is_file()}


def inspect_label(path: Path, row: dict[str, str], allowed_empty_fire: set[str]) -> tuple[dict[str, object], list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    record: dict[str, object] = {
        "name": row["name"],
        "source_class": row["source_class"],
        "sequence_id": row["sequence_id"],
        "pilot8": row.get("pilot8", "no"),
        "json_sha256": sha256(path),
        "image_path": "",
        "shape_count": 0,
        "smoke_polygons": 0,
        "overexposure_polygons": 0,
        "ignore_polygons": 0,
        "min_polygon_area": "",
        "max_polygon_area": "",
    }
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        errors.append(f"json_parse:{error}")
        record["issues"] = "; ".join(errors)
        return record, errors, warnings

    image_path = str(data.get("imagePath", ""))
    record["image_path"] = image_path
    expected_image = f"{row['name']}.JPG"
    normalized_image_path = image_path.replace("\\", "/")
    if not image_path or PurePosixPath(normalized_image_path).name != expected_image:
        errors.append("imagePath_name_mismatch")
    if PureWindowsPath(image_path).is_absolute() or PurePosixPath(image_path).is_absolute():
        errors.append("imagePath_not_relative")
    if data.get("imageData") is not None:
        warnings.append("imageData_not_null")
    if (data.get("imageWidth"), data.get("imageHeight")) != (640, 512):
        errors.append("image_size_not_640x512")

    shapes = data.get("shapes")
    if not isinstance(shapes, list):
        errors.append("shapes_not_list")
        shapes = []
    areas: list[float] = []
    labels = Counter()
    for index, shape in enumerate(shapes, 1):
        if not isinstance(shape, dict):
            errors.append(f"shape_{index}_not_object")
            continue
        label = str(shape.get("label", ""))
        labels[label] += 1
        if label not in ALLOWED_LABELS:
            errors.append(f"shape_{index}_unsupported_label:{label}")
        if shape.get("shape_type") != "polygon":
            errors.append(f"shape_{index}_not_polygon")
        raw_points = shape.get("points")
        if not isinstance(raw_points, list) or len(raw_points) < 3:
            errors.append(f"shape_{index}_fewer_than_3_points")
            continue
        points: list[tuple[float, float]] = []
        for point in raw_points:
            try:
                x, y = float(point[0]), float(point[1])
            except (IndexError, TypeError, ValueError):
                errors.append(f"shape_{index}_invalid_point")
                points = []
                break
            if not math.isfinite(x) or not math.isfinite(y) or not (-1e-6 <= x <= 640.0 + 1e-6 and -1e-6 <= y <= 512.0 + 1e-6):
                errors.append(f"shape_{index}_point_out_of_bounds")
            points.append((min(640.0, max(0.0, x)), min(512.0, max(0.0, y))))
        if points:
            area = polygon_area(points)
            if area <= 0.0:
                errors.append(f"shape_{index}_zero_area")
            else:
                areas.append(area)

    record.update({
        "shape_count": len(shapes),
        "smoke_polygons": labels["smoke_affected"],
        "overexposure_polygons": labels["overexposure"],
        "ignore_polygons": labels["ignore"],
        "min_polygon_area": f"{min(areas):.2f}" if areas else "",
        "max_polygon_area": f"{max(areas):.2f}" if areas else "",
    })
    if row["source_class"] == "Fire" and labels["smoke_affected"] == 0:
        if row["name"] in allowed_empty_fire:
            warnings.append("approved_full_frame_unsegmentable")
        else:
            errors.append("screened_smoke_positive_without_smoke_polygon")
    if row["source_class"] == "No Fire" and shapes:
        errors.append("clean_negative_not_empty")
    if labels["overexposure"]:
        errors.append("R0_overexposure_conflict")
    if labels["ignore"]:
        warnings.append("ignore_added_requires_R0_review")
    record["issues"] = "; ".join([*errors, *warnings])
    return record, errors, warnings


def render_sheets(rows: list[dict[str, str]], annotated: Path, output: Path) -> list[str]:
    output.mkdir(parents=True, exist_ok=True)
    sheets: list[str] = []
    for page, start in enumerate(range(0, len(rows), 4), 1):
        canvas = Image.new("RGB", (1280, 1024), "black")
        draw = ImageDraw.Draw(canvas)
        for cell, row in enumerate(rows[start:start + 4]):
            x0, y0 = (cell % 2) * 640, (cell // 2) * 512
            name = row["name"]
            with Image.open(annotated / "visible" / f"{name}.JPG") as image:
                image = image.convert("RGBA")
            data = json.loads((annotated / "labels_json" / f"{name}.json").read_text(encoding="utf-8"))
            layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
            layer_draw = ImageDraw.Draw(layer)
            for shape in data.get("shapes", []):
                color = COLORS.get(str(shape.get("label", "")))
                points = shape.get("points", [])
                if color and shape.get("shape_type") == "rectangle" and isinstance(points, list) and len(points) == 2:
                    (x1, y1), (x2, y2) = [(float(point[0]), float(point[1])) for point in points]
                    layer_draw.rectangle((min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)), fill=color, outline=color[:3], width=3)
                elif color and isinstance(points, list) and len(points) >= 3:
                    points_xy = [(float(point[0]), float(point[1])) for point in points]
                    layer_draw.polygon(points_xy, fill=color, outline=color[:3], width=3)
            canvas.paste(Image.alpha_composite(image, layer).convert("RGB"), (x0, y0))
            draw.rectangle((x0, y0, x0 + 150, y0 + 22), fill="black")
            draw.text((x0 + 5, y0 + 4), name, fill="white")
        sheet = output / f"overlay_{page:02d}.png"
        canvas.save(sheet)
        sheets.append(sheet.name)
    return sheets


def audit(package: Path, annotated: Path, output: Path, expected_count: int, allowed_empty_fire: set[str] | None = None) -> tuple[str, list[str]]:
    allowed_empty_fire = allowed_empty_fire or set()
    stage = "R2" if expected_count == 20 else "R3"
    manifest = package / ("batch20_manifest.csv" if expected_count == 2 else f"batch{expected_count}_manifest.csv")
    if not manifest.is_file() or not annotated.is_dir() or output.exists():
        raise FileNotFoundError("Missing package/annotated directory, manifest, or output already exists")
    rows = read_csv(manifest)
    if len(rows) != expected_count or len({row["name"] for row in rows}) != expected_count:
        raise ValueError(f"Expected {expected_count} unique manifest rows")
    expected_names = {row["name"] for row in rows}
    invalid_exceptions = allowed_empty_fire - {row["name"] for row in rows if row["source_class"] == "Fire"}
    if invalid_exceptions:
        raise ValueError(f"Approved empty-mask names are not Fire samples in this manifest: {sorted(invalid_exceptions)}")
    annotated_files = relative_files(annotated)
    package_files = relative_files(package)
    errors: list[str] = []
    warnings: list[str] = []
    changed_fixed: list[str] = []
    missing_fixed: list[str] = []
    for relative, original in package_files.items():
        returned = annotated / relative
        if not returned.is_file():
            missing_fixed.append(relative)
        elif sha256(original) != sha256(returned):
            changed_fixed.append(relative)
    if missing_fixed:
        errors.extend(f"missing_fixed:{item}" for item in missing_fixed)
    if changed_fixed:
        errors.extend(f"changed_fixed:{item}" for item in changed_fixed)

    expected_new = {f"labels_json/{row['name']}.json" for row in rows if row.get("pilot8", "no") == "no"}
    allowed_extras = {"annotation_notes.md", "annoation_notes.md"}
    extras = sorted(set(annotated_files) - set(package_files) - expected_new - allowed_extras)
    if extras:
        warnings.extend(f"unexpected_extra:{item}" for item in extras)
    json_paths = sorted((annotated / "labels_json").glob("*.json")) if (annotated / "labels_json").is_dir() else []
    actual_names = {path.stem for path in json_paths}
    missing_json = sorted(expected_names - actual_names)
    extra_json = sorted(actual_names - expected_names)
    if missing_json:
        errors.extend(f"missing_json:{name}" for name in missing_json)
    if extra_json:
        errors.extend(f"extra_json:{name}" for name in extra_json)
    for folder in ("visible", "ir_reference"):
        actual = {path.stem for path in (annotated / folder).glob("*.JPG")} if (annotated / folder).is_dir() else set()
        if actual != expected_names:
            errors.append(f"{folder}_name_mismatch")

    records: list[dict[str, object]] = []
    for row in rows:
        label_path = annotated / "labels_json" / f"{row['name']}.json"
        if label_path.is_file():
            record, item_errors, item_warnings = inspect_label(label_path, row, allowed_empty_fire)
            records.append(record)
            errors.extend(f"{row['name']}:{item}" for item in item_errors)
            warnings.extend(f"{row['name']}:{item}" for item in item_warnings)

    status = "HOLD" if errors else "AUTOMATED_PASS_PENDING_VISUAL"
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f".{output.name}_", dir=output.parent) as temporary:
        staging = Path(temporary)
        summary_fields = ("name", "source_class", "sequence_id", "pilot8", "json_sha256", "image_path", "shape_count", "smoke_polygons", "overexposure_polygons", "ignore_polygons", "min_polygon_area", "max_polygon_area", "issues")
        write_csv(staging / "annotation_summary.csv", summary_fields, records)
        sheets = render_sheets(rows, annotated, staging / "overlays")
        report = [
            f"# P-002 {stage}：{expected_count}张LabelMe回传验收报告",
            "",
            "## 1. 具体目标",
            "",
            f"核验当前{expected_count}张回传是否满足文件冻结、JSON结构、polygon几何、标签规则和目视审查前提，决定能否进入下一门禁。",
            "",
            "## 2. 需要解决的问题",
            "",
            "确认manifest内JSON齐全可用、全部原始包资产未被办公电脑改写，并只接受经过批准的空mask例外。",
            "",
            "## 3. 自动验收流程",
            "",
            f"1. 以原始 `batch{expected_count}_package` 为冻结参照，逐文件SHA256比较回传包内已有文件。",
            f"2. 以 `batch{expected_count}_manifest.csv` 核对{expected_count}个名称、Visible、IR参考和JSON的一一对应。",
            "3. 检查JSON可解析性、相对imagePath、640×512尺寸、允许标签、polygon类型、至少3点、有限坐标、边界和非零面积。",
            "4. 对Fire检查存在 `smoke_affected`，对No Fire检查空标注；出现`overexposure`或`ignore`时记录R0一致性问题。",
            "5. 生成叠加轮廓图，供人工逐图复核。",
            "",
            "## 4. 当前自动结果",
            "",
            f"- 状态：`{status}`",
            f"- manifest样本：{len(rows)}；回传JSON：{len(json_paths)}；需提交JSON：{len(expected_names)}。",
            f"- 冻结文件：原始{len(package_files)}项，缺失{len(missing_fixed)}项，发生哈希变化{len(changed_fixed)}项。",
            f"- JSON名称缺失：{len(missing_json)}；额外JSON：{len(extra_json)}。",
            f"- 合法标签总数：smoke={sum(int(item['smoke_polygons']) for item in records)}，overexposure={sum(int(item['overexposure_polygons']) for item in records)}，ignore={sum(int(item['ignore_polygons']) for item in records)}。",
            "",
            "### 阻塞项",
            "",
        ]
        report += [f"- `{item}`" for item in errors] if errors else ["- 无自动阻塞项。"]
        report += [
            "",
            "### 需人工复核项",
            "",
        ]
        report += [f"- `{item}`" for item in warnings] if warnings else ["- 无自动警告项。"]
        report += [
            "",
            "## 5. 目视复核材料",
            "",
            "叠加图以橙色表示`smoke_affected`、洋红表示`overexposure`、青色表示`ignore`。自动通过不等于边界质量通过；需要逐图确认烟雾区域没有明显扩张到天空、云、一般亮区或无关背景。",
            "",
            *[f"- `overlays/{name}`" for name in sheets],
            "",
            "## 6. 通过/失败规则",
            "",
            f"只有冻结资产零变化、{expected_count}/{expected_count} JSON齐全、所有结构/几何检查通过且目视边界通过，{stage}才可标记为通过。任一标签非法、polygon越界/退化、No Fire非空或未经批准的Fire空mask出现时，保持HOLD。",
            "",
            "## 7. 本轮转移与回传",
            "",
            f"本轮为本机只读验收，无需再转移源码、模型、数据、权重或环境。若需人工修复，必须再次回传完整 `batch{expected_count}_annotated/`；无需回传终端文档。除经批准修复的JSON或备注外，图像、manifest、README和校验清单必须保持不变。",
        ]
        (staging / f"{stage}_ANNOTATION_AUDIT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
        staging.replace(output)
    return status, errors


def self_check() -> None:
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        package = root / "package"
        for folder in (package / "visible", package / "ir_reference", package / "labels_json"):
            folder.mkdir(parents=True)
        rows = [
            {"annotation_index": "1", "name": "F_00001", "sequence_id": "SEQ_00", "source_class": "Fire", "pilot8": "no"},
            {"annotation_index": "2", "name": "N_00001", "sequence_id": "SEQ_00", "source_class": "No Fire", "pilot8": "yes"},
        ]
        write_csv(package / "batch20_manifest.csv", tuple(rows[0]), rows)
        (package / "README.md").write_text("test", encoding="utf-8")
        for row in rows:
            for folder in ("visible", "ir_reference"):
                Image.new("RGB", (640, 512), "gray").save(package / folder / f"{row['name']}.JPG")
            shapes = [] if row["source_class"] == "No Fire" else [{"label": "smoke_affected", "points": [[1, 1], [50, 1], [1, 50]], "shape_type": "polygon"}]
            (package / "labels_json" / f"{row['name']}.json").write_text(json.dumps({"imagePath": f"..\\visible\\{row['name']}.JPG", "imageData": None, "imageWidth": 640, "imageHeight": 512, "shapes": shapes}), encoding="utf-8")
        annotated = root / "annotated"
        shutil.copytree(package, annotated)
        output = root / "audit"
        status, errors = audit(package, annotated, output, expected_count=2)
        assert status == "AUTOMATED_PASS_PENDING_VISUAL" and not errors and (output / "annotation_summary.csv").is_file(), (status, errors)
    print("audit_region80_annotations self-check passed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit a returned P-002 LabelMe annotation package.")
    parser.add_argument("--package_dir")
    parser.add_argument("--annotated_dir")
    parser.add_argument("--output_dir")
    parser.add_argument("--expected_count", type=int, default=20)
    parser.add_argument("--allowed_empty_fire", default="", help="Comma-separated approved Fire names whose region masks are unavailable.")
    parser.add_argument("--self_check", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_check:
        self_check()
        return
    required = ("package_dir", "annotated_dir", "output_dir")
    missing = [name for name in required if not getattr(args, name)]
    if missing:
        raise ValueError(f"Missing required arguments: {', '.join('--' + name for name in missing)}")
    allowed_empty_fire = {name.strip() for name in args.allowed_empty_fire.split(",") if name.strip()}
    status, _ = audit(Path(args.package_dir), Path(args.annotated_dir), Path(args.output_dir), args.expected_count, allowed_empty_fire)
    stage = "R2" if args.expected_count == 20 else "R3"
    print(f"P-002 {stage} audit status: {status}")
    print(f"P-002 {stage} audit output: {Path(args.output_dir).resolve()}")
    if status == "HOLD":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
