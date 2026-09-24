#!/usr/bin/env python3
"""Freeze P-002 region80 and build the blinded first-20 LabelMe package."""

from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory


PILOT_NAMES = {"F_00243", "F_00267", "F_00484", "F_00620", "N_00015", "N_00064", "N_00065", "N_00093"}
KEEP_TARGETS = {("Fire", "SEQ_07"): 13, ("Fire", "SEQ_10"): 59, ("No Fire", "SEQ_07"): 8}
NEW_BATCH_TARGETS = {("Fire", "SEQ_07"): 2, ("Fire", "SEQ_10"): 10}
AXES = ("temporal", "thermal", "saturation", "smoke")
STRATA = ("low", "mid", "high")


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


def parse_priors(report: Path) -> dict[str, dict[str, float]]:
    priors: dict[str, dict[str, float]] = {}
    for line in report.read_text(encoding="utf-8-sig").splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) >= 6 and cells[0].startswith(("F_", "N_")) and cells[0] not in priors:
            try:
                priors[cells[0]] = {
                    "thermal": float(cells[3]),
                    "saturation": float(cells[4]),
                    "smoke": float(cells[5]),
                }
            except ValueError:
                continue
    return priors


def rank_strata(rows: list[dict[str, object]], value_key: str, output_key: str) -> None:
    ordered = sorted(rows, key=lambda row: (float(row[value_key]), str(row["name"])))
    count = len(ordered)
    for rank, row in enumerate(ordered):
        row[output_key] = STRATA[min(2, rank * 3 // count)]


def evenly_spaced(items: list[dict[str, object]], count: int) -> list[dict[str, object]]:
    if count == 0:
        return []
    return [items[min(len(items) - 1, int((index + 0.5) * len(items) / count))] for index in range(count)]


def freeze_region80(rows: list[dict[str, object]]) -> None:
    groups: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in rows:
        groups.setdefault((str(row["source_class"]), str(row["sequence_id"])), []).append(row)
    if Counter({key: len(value) for key, value in groups.items()}) != Counter({("Fire", "SEQ_10"): 70, ("Fire", "SEQ_07"): 16, ("No Fire", "SEQ_07"): 8}):
        raise ValueError("R0 group counts changed; expected Fire SEQ_10=70, Fire SEQ_07=16, No Fire SEQ_07=8")

    for key, group in groups.items():
        group.sort(key=lambda row: (int(row["frame_number"]), str(row["name"])))
        for rank, row in enumerate(group):
            row["temporal_stratum"] = STRATA[min(2, rank * 3 // len(group))]
        for axis in ("thermal", "saturation", "smoke"):
            rank_strata(group, f"{axis}_prior_mean", f"{axis}_stratum")

        target = KEEP_TARGETS[key]
        forced = [row for row in group if str(row["name"]) in PILOT_NAMES or key[0] == "No Fire"]
        eligible = [row for row in group if row not in forced]
        # ponytail: deterministic thinning is sufficient for this fixed 94-frame pool; revisit only if independent sequences expand.
        excluded = evenly_spaced(eligible, len(group) - target)
        excluded_names = {str(row["name"]) for row in excluded}
        for row in group:
            row["selected_region80"] = str(row["name"]) not in excluded_names
            row["exclusion_reason"] = "" if row["selected_region80"] else "proportional_temporal_thinning"

    selected = [row for row in rows if row["selected_region80"]]
    if len(selected) != 80 or not PILOT_NAMES <= {str(row["name"]) for row in selected}:
        raise AssertionError("region80 selection count or pilot retention failed")
    for key in (("Fire", "SEQ_07"), ("Fire", "SEQ_10")):
        group = [row for row in selected if (row["source_class"], row["sequence_id"]) == key]
        for axis in AXES:
            if {str(row[f"{axis}_stratum"]) for row in group} != set(STRATA):
                raise ValueError(f"Selected {key} does not cover all {axis} strata")


def choose_batch20(rows: list[dict[str, object]]) -> None:
    for row in rows:
        row["annotation_batch"] = "batch20" if str(row["name"]) in PILOT_NAMES else ("batch60" if row["selected_region80"] else "excluded")

    chosen = [row for row in rows if str(row["name"]) in PILOT_NAMES and row["source_class"] == "Fire"]
    remaining = dict(NEW_BATCH_TARGETS)
    while sum(remaining.values()):
        candidates = [
            row for row in rows
            if row["selected_region80"]
            and row["annotation_batch"] == "batch60"
            and remaining.get((str(row["source_class"]), str(row["sequence_id"])), 0) > 0
        ]
        covered = {(axis, str(row[f"{axis}_stratum"])) for row in chosen for axis in AXES}

        def score(row: dict[str, object]) -> tuple[int, float, str]:
            missing = sum((axis, str(row[f"{axis}_stratum"])) not in covered for axis in AXES)
            same_sequence = [item for item in chosen if item["sequence_id"] == row["sequence_id"]]
            distance = min((abs(int(row["frame_number"]) - int(item["frame_number"])) for item in same_sequence), default=10**9)
            return missing, float(distance), str(row["name"])

        pick = max(candidates, key=score)
        pick["annotation_batch"] = "batch20"
        chosen.append(pick)
        remaining[(str(pick["source_class"]), str(pick["sequence_id"]))] -= 1

    batch = [row for row in rows if row["annotation_batch"] == "batch20"]
    if len(batch) != 20 or not PILOT_NAMES <= {str(row["name"]) for row in batch}:
        raise AssertionError("batch20 count or pilot retention failed")
    fire_batch = [row for row in batch if row["source_class"] == "Fire"]
    for axis in AXES:
        if {str(row[f"{axis}_stratum"]) for row in fire_batch} != set(STRATA):
            raise ValueError(f"batch20 Fire samples do not cover all {axis} strata")


def validate_inputs(screening_rows: list[dict[str, str]], package_rows: list[dict[str, str]], priors: dict[str, dict[str, float]], pilot_dir: Path) -> None:
    required = {"name", "sequence_id", "source_class", "visible_file", "ir_reference_file", "smoke_presence", "overexposure_presence", "ignore_burden", "scene_group", "confidence"}
    if len(screening_rows) != 94 or any(required - set(row) for row in screening_rows):
        raise ValueError("screening94.csv must contain the expected columns and 94 rows")
    names = [row["name"] for row in screening_rows]
    if len(set(names)) != 94 or set(names) != {row["name"] for row in package_rows} or set(names) != set(priors):
        raise ValueError("screening, package manifest, and A0 prior names do not match 94/94")
    if {path.stem for path in pilot_dir.glob("*.json")} != PILOT_NAMES:
        raise ValueError("pilot label directory must contain the frozen pilot8 JSON files")
    for row in screening_rows:
        if row["smoke_presence"] not in {"yes", "no"} or row["overexposure_presence"] != "no" or row["ignore_burden"] != "none" or row["confidence"] != "high":
            raise ValueError(f"Unexpected R0 decision for {row['name']}; R1 requires the audited screened CSV")
        expected_scene = "smoke" if row["smoke_presence"] == "yes" else "clean_hard_negative"
        if row["scene_group"] != expected_scene:
            raise ValueError(f"R0 scene mapping mismatch for {row['name']}")


def build(screening: Path, package_root: Path, a0_report: Path, pilot_dir: Path, output_dir: Path) -> None:
    for path in (screening, package_root / "package_manifest.csv", a0_report):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not pilot_dir.is_dir():
        raise FileNotFoundError(pilot_dir)
    if output_dir.exists():
        raise FileExistsError(f"Output already exists; refusing to overwrite: {output_dir}")

    screening_rows = read_csv(screening)
    package_rows = read_csv(package_root / "package_manifest.csv")
    package_by_name = {row["name"]: row for row in package_rows}
    priors = parse_priors(a0_report)
    validate_inputs(screening_rows, package_rows, priors, pilot_dir)

    rows: list[dict[str, object]] = []
    for human in screening_rows:
        name = human["name"]
        rows.append({
            **human,
            "frame_number": int(name.split("_")[-1]),
            "thermal_prior_mean": priors[name]["thermal"],
            "saturation_prior_mean": priors[name]["saturation"],
            "smoke_prior_mean": priors[name]["smoke"],
            "pilot8": name in PILOT_NAMES,
        })
    freeze_region80(rows)
    choose_batch20(rows)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f".{output_dir.name}_", dir=output_dir.parent) as temporary:
        root = Path(temporary)
        package = root / "batch20_package"
        for folder in (package / "visible", package / "ir_reference", package / "labels_json"):
            folder.mkdir(parents=True, exist_ok=True)

        selected = sorted((row for row in rows if row["selected_region80"]), key=lambda row: (row["annotation_batch"], row["source_class"], row["sequence_id"], row["frame_number"]))
        region_fields = ("selection_index", "name", "sequence_id", "source_class", "scene_group", "smoke_presence", "overexposure_presence", "ignore_burden", "confidence", "pilot8", "annotation_batch", "source_visible_file", "source_ir_reference_file")
        region_rows = []
        for index, row in enumerate(selected, 1):
            region_rows.append({
                "selection_index": index,
                "name": row["name"],
                "sequence_id": row["sequence_id"],
                "source_class": row["source_class"],
                "scene_group": row["scene_group"],
                "smoke_presence": row["smoke_presence"],
                "overexposure_presence": row["overexposure_presence"],
                "ignore_burden": row["ignore_burden"],
                "confidence": row["confidence"],
                "pilot8": "yes" if row["pilot8"] else "no",
                "annotation_batch": row["annotation_batch"],
                "source_visible_file": f"../annotation_region80_screened/{row['visible_file']}",
                "source_ir_reference_file": f"../annotation_region80_screened/{row['ir_reference_file']}",
            })
        write_csv(root / "region80_manifest.csv", region_fields, region_rows)

        audit_fields = ("name", "sequence_id", "source_class", "frame_number", "scene_group", "thermal_prior_mean", "thermal_stratum", "saturation_prior_mean", "saturation_stratum", "smoke_prior_mean", "smoke_stratum", "temporal_stratum", "pilot8", "selected_region80", "annotation_batch", "exclusion_reason")
        audit_rows = []
        for row in sorted(rows, key=lambda item: (item["source_class"], item["sequence_id"], item["frame_number"])):
            audit_rows.append({field: ("yes" if row[field] else "no") if field in {"pilot8", "selected_region80"} else row[field] for field in audit_fields})
        write_csv(root / "selection_audit.csv", audit_fields, audit_rows)

        batch = sorted((row for row in rows if row["annotation_batch"] == "batch20"), key=lambda row: (not row["pilot8"], row["source_class"], row["sequence_id"], row["frame_number"]))
        batch_fields = ("annotation_index", "name", "sequence_id", "source_class", "scene_group", "smoke_presence", "overexposure_presence", "ignore_burden", "confidence", "pilot8", "visible_file", "ir_reference_file", "label_json")
        batch_rows = []
        for index, row in enumerate(batch, 1):
            name = str(row["name"])
            asset = package_by_name[name]
            source_vis = package_root / asset["visible_file"]
            source_ir = package_root / asset["ir_reference_file"]
            if sha256(source_vis) != asset["visible_sha256"] or sha256(source_ir) != asset["ir_sha256"]:
                raise ValueError(f"R0 package asset hash mismatch: {name}")
            shutil.copy2(source_vis, package / "visible" / source_vis.name)
            shutil.copy2(source_ir, package / "ir_reference" / source_ir.name)
            if row["pilot8"]:
                shutil.copy2(pilot_dir / f"{name}.json", package / "labels_json" / f"{name}.json")
            batch_rows.append({
                "annotation_index": index,
                "name": name,
                "sequence_id": row["sequence_id"],
                "source_class": row["source_class"],
                "scene_group": row["scene_group"],
                "smoke_presence": row["smoke_presence"],
                "overexposure_presence": row["overexposure_presence"],
                "ignore_burden": row["ignore_burden"],
                "confidence": row["confidence"],
                "pilot8": "yes" if row["pilot8"] else "no",
                "visible_file": f"visible/{source_vis.name}",
                "ir_reference_file": f"ir_reference/{source_ir.name}",
                "label_json": f"labels_json/{name}.json",
            })
        write_csv(package / "batch20_manifest.csv", batch_fields, batch_rows)

        package_readme = (
            "# P-002 R1 首批20张LabelMe标注包\n\n"
            "## 具体目标与要解决的问题\n\n"
            "本批用于先验证区域边界规范能否稳定执行，再决定是否继续剩余60张。需要解决的是：烟雾边界是否可重复、空阴性是否正确保存、标签/图像/文件名是否一一对应。标注不用于训练或语义分割。\n\n"
            "## 办公电脑具体操作\n\n"
            "1. 完整复制本文件夹；不要只复制 `visible/`。\n"
            "2. 用LabelMe打开 `visible/`，把输出目录设为 `labels_json/`。\n"
            "3. 8个pilot JSON已存在，只用于边界示例，不要重画或另存。\n"
            "4. 对其余12张逐张标注；只允许 `smoke_affected`、`overexposure`、`ignore`，shape只用polygon。\n"
            "5. 烟雾和过曝光只依据Visible；IR只作配准/目标位置参考。普通云、天空、一般发白或亮点不得充当烟雾/过曝光。\n"
            "6. R0未发现过曝光阳性；若细标时发现明确曝光截断，先在单独文本中记录样本名并暂停该图，不静默改变R0结论。\n"
            "7. 无区域的图像也必须保存JSON且 `shapes=[]`。完成后应有20个同名JSON，其中原8个pilot保持字节不变。\n"
            "8. 回传整个本文件夹，建议复制后重命名为 `batch20_annotated`；无需回传终端文档。\n\n"
            "## 通过/失败条件\n\n"
            "通过：20/20 JSON齐全、仅含允许标签、均为有效polygon或空JSON、图像名/尺寸匹配、原8个pilot未变化。任一文件缺失、非法标签、越界、pilot被改写或对烟雾/过曝光定义有系统性歧义即暂停在R2，不进入剩余60张。\n"
        )
        (package / "README.md").write_text(package_readme, encoding="utf-8")

        operation = (
            "# P-002 R1：80张冻结与首批20张标注操作流程\n\n"
            "## 1. 具体目标\n\n"
            "把已验收的94张人工盘点冻结为80张区域评价样本，并生成首批20张盲化LabelMe包。\n\n"
            "## 2. 需要解决的问题\n\n"
            "- 在真实样本池没有过曝光阳性的条件下，不伪造配额；\n"
            "- 同时保留pilot8、全部No Fire、两个Fire序列及帧段/prior低中高覆盖；\n"
            "- 不按A0–A3融合质量挑图，避免结果导向抽样；\n"
            "- 先用20张验证标注规范，避免错误扩散到剩余60张。\n\n"
            "## 3. 冻结规则\n\n"
            "80张由No Fire SEQ_07全部8张、Fire SEQ_07 13/16张和Fire SEQ_10 59/70张组成；pilot8全部保留。Fire未入选帧采用按帧序均匀稀疏规则，随后只检查每个Fire序列的temporal/thermal/saturation/smoke低中高层是否均覆盖。A0报告只提供prior分层，不使用任何A0–A3融合质量指标。\n\n"
            "首批20张为pilot8加12张新Fire；新增量为SEQ_07 2张、SEQ_10 10张。候选只在冻结80内按缺失分层与帧距确定。`batch20_package`不包含prior数值或融合质量结果。\n\n"
            "## 4. 产物与用途\n\n"
            "- `region80_manifest.csv`：正式80张名单；\n"
            "- `selection_audit.csv`：94张选择/排除与prior分层审计，仅内部留存，不交给标注者；\n"
            "- `batch20_package/`：办公电脑唯一需要接收的标注包；\n"
            "- `checksums.sha256`：R1初始产物冻结哈希。\n\n"
            "## 5. 执行前转移清单\n\n"
            "必需：只转移完整 `batch20_package/` 至办公电脑。可选：无。无需重复转移：脚本、A0报告、模型权重、FLAME3原始数据、A0–A3运行目录、Python环境。\n\n"
            "## 6. 执行后回传清单\n\n"
            "必需：回传完整 `batch20_annotated/`，其中包含README、batch20_manifest、20张Visible、20张IR参考、20个labels_json和input_checksums。无需回传终端文档。原8个pilot JSON及所有已有文件必须保持不变；只允许新增12个JSON和可选的异常记录文本。\n\n"
            "## 7. 当前研究边界\n\n"
            "本节点只推进smoke区域证据与clean阴性对照；sat_uncertainty的人工作曝光区域校准继续受阻，thermal仍只允许使用Celsius高温代理。任何标签均不进入融合训练监督，不构成语义分割任务。\n"
        )
        (root / "R1_OBJECTIVE_AND_PROCEDURE.md").write_text(operation, encoding="utf-8")

        package_targets = sorted(path for path in package.rglob("*") if path.is_file())
        (package / "input_checksums.sha256").write_text("".join(f"{sha256(path)}  *{path.relative_to(package).as_posix()}\n" for path in package_targets), encoding="utf-8")
        root_targets = sorted(path for path in root.rglob("*") if path.is_file())
        (root / "checksums.sha256").write_text("".join(f"{sha256(path)}  *{path.relative_to(root).as_posix()}\n" for path in root_targets), encoding="utf-8")
        root.replace(output_dir)


def self_check() -> None:
    rows: list[dict[str, object]] = []
    specifications = (("Fire", "SEQ_07", 16), ("Fire", "SEQ_10", 70), ("No Fire", "SEQ_07", 8))
    pilot_order = iter(sorted(PILOT_NAMES))
    pilot_by_group = {("Fire", "SEQ_07"): [], ("Fire", "SEQ_10"): [], ("No Fire", "SEQ_07"): []}
    for name in pilot_order:
        key = ("No Fire", "SEQ_07") if name.startswith("N_") else (("Fire", "SEQ_07") if name in {"F_00243", "F_00484"} else ("Fire", "SEQ_10"))
        pilot_by_group[key].append(name)
    for source, sequence, count in specifications:
        names = pilot_by_group[(source, sequence)] + [f"{'N' if source == 'No Fire' else 'F'}_{sequence[-2:]}{index:03d}" for index in range(count - len(pilot_by_group[(source, sequence)]))]
        for index, name in enumerate(names):
            rows.append({"name": name, "source_class": source, "sequence_id": sequence, "frame_number": index, "thermal_prior_mean": index, "saturation_prior_mean": count - index, "smoke_prior_mean": (index * 7) % count, "pilot8": name in PILOT_NAMES})
    freeze_region80(rows)
    choose_batch20(rows)
    assert Counter((row["source_class"], row["sequence_id"]) for row in rows if row["selected_region80"]) == Counter(KEEP_TARGETS)
    assert Counter((row["source_class"], row["sequence_id"]) for row in rows if row["annotation_batch"] == "batch20") == Counter({("Fire", "SEQ_07"): 4, ("Fire", "SEQ_10"): 12, ("No Fire", "SEQ_07"): 4})
    print("prepare_region80_r1 self-check passed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze P-002 region80 and create its blinded first-20 LabelMe package.")
    parser.add_argument("--screening")
    parser.add_argument("--package_root")
    parser.add_argument("--a0_report")
    parser.add_argument("--pilot_label_dir")
    parser.add_argument("--output_dir")
    parser.add_argument("--self_check", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_check:
        self_check()
        return
    required = ("screening", "package_root", "a0_report", "pilot_label_dir", "output_dir")
    missing = [name for name in required if not getattr(args, name)]
    if missing:
        raise ValueError(f"Missing required arguments: {', '.join('--' + name for name in missing)}")
    build(Path(args.screening), Path(args.package_root), Path(args.a0_report), Path(args.pilot_label_dir), Path(args.output_dir))
    print(f"P-002 R1 package created: {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
