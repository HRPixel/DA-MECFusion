from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory


EXPECTED_BATCHES = Counter({"batch20": 20, "batch60": 60})
EXPECTED_BATCH60_GROUPS = Counter({("Fire", "SEQ_07"): 9, ("Fire", "SEQ_10"): 47, ("No Fire", "SEQ_07"): 4})


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_batch60(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    required = {
        "selection_index", "name", "sequence_id", "source_class", "scene_group",
        "smoke_presence", "overexposure_presence", "ignore_burden", "confidence",
        "pilot8", "annotation_batch", "source_visible_file", "source_ir_reference_file",
    }
    if len(rows) != 80 or any(required - set(row) for row in rows):
        raise ValueError("region80_manifest.csv must contain the frozen 80 rows and expected columns")
    names = [row["name"] for row in rows]
    if len(set(names)) != 80:
        raise ValueError("region80_manifest.csv contains duplicate names")
    if Counter(row["annotation_batch"] for row in rows) != EXPECTED_BATCHES:
        raise ValueError("region80_manifest.csv must contain exactly 20 batch20 and 60 batch60 rows")

    batch60 = [row for row in rows if row["annotation_batch"] == "batch60"]
    groups = Counter((row["source_class"], row["sequence_id"]) for row in batch60)
    if groups != EXPECTED_BATCH60_GROUPS or any(row["pilot8"] != "no" for row in batch60):
        raise ValueError("batch60 group counts or pilot exclusion differ from the frozen R1 protocol")
    return sorted(batch60, key=lambda row: int(row["selection_index"]))


def resolve_source(manifest: Path, relative: str, screened_root: Path) -> Path:
    source = (manifest.parent / Path(relative)).resolve()
    if not source.is_relative_to(screened_root.resolve()) or not source.is_file():
        raise ValueError(f"Invalid or missing screened source: {relative}")
    return source


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fields = (
        "annotation_index", "name", "sequence_id", "source_class", "scene_group",
        "smoke_presence", "overexposure_presence", "ignore_burden", "confidence",
        "visible_file", "ir_reference_file", "label_json", "visible_sha256", "ir_sha256",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build(manifest: Path, output_dir: Path) -> None:
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    if output_dir.exists():
        raise FileExistsError(f"Output already exists; refusing to overwrite: {output_dir}")

    batch60 = select_batch60(read_csv(manifest))
    screened_root = manifest.parent.parent / "annotation_region80_screened"
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    with TemporaryDirectory(prefix=f".{output_dir.name}_", dir=output_dir.parent) as temporary:
        package = Path(temporary)
        for folder in (package / "visible", package / "ir_reference", package / "labels_json"):
            folder.mkdir(parents=True, exist_ok=True)

        package_rows: list[dict[str, str]] = []
        for index, row in enumerate(batch60, 1):
            visible = resolve_source(manifest, row["source_visible_file"], screened_root)
            infrared = resolve_source(manifest, row["source_ir_reference_file"], screened_root)
            target_visible = package / "visible" / visible.name
            target_ir = package / "ir_reference" / infrared.name
            shutil.copy2(visible, target_visible)
            shutil.copy2(infrared, target_ir)
            package_rows.append({
                "annotation_index": str(index),
                "name": row["name"],
                "sequence_id": row["sequence_id"],
                "source_class": row["source_class"],
                "scene_group": row["scene_group"],
                "smoke_presence": row["smoke_presence"],
                "overexposure_presence": row["overexposure_presence"],
                "ignore_burden": row["ignore_burden"],
                "confidence": row["confidence"],
                "visible_file": f"visible/{visible.name}",
                "ir_reference_file": f"ir_reference/{infrared.name}",
                "label_json": f"labels_json/{row['name']}.json",
                "visible_sha256": sha256(target_visible),
                "ir_sha256": sha256(target_ir),
            })

        write_csv(package / "batch60_manifest.csv", package_rows)
        readme = (
            "# P-002 R3：剩余60张LabelMe标注包\n\n"
            "## 具体目标与需要解决的问题\n\n"
            "完成冻结region80中尚未标注的60张区域标注，使其能与R2已通过的20张合并为80张。需要保持标签定义、polygon规范和No Fire空阴性一致；本批不重新抽样，不用于训练或语义分割。\n\n"
            "## 办公电脑操作\n\n"
            "1. 完整复制本文件夹，并在副本上操作；不要只复制`visible/`。\n"
            "2. 用LabelMe逐张打开`visible/`，将JSON保存到`labels_json/`，文件名必须与图像同名。\n"
            "3. 只允许`smoke_affected`、`overexposure`、`ignore`；所有非空shape只用polygon。\n"
            "4. 烟雾和过曝光仅依据Visible判断；IR只作目标位置和配准参考。云、天空、一般发白、亮点或高温目标不能代替烟雾/过曝光。\n"
            "5. 56张Fire按可见烟雾主体画粗边界，不追求逐像素轮廓；4张No Fire逐图确认后保存`shapes=[]`空JSON。\n"
            "6. R0未发现明确过曝光阳性；若发现疑似曝光截断，不直接标注，先在`annotation_notes.md`记录样本名和原因并暂停该图。\n"
            "7. 完成后必须有60个JSON；不要修改Visible、IR参考、manifest、README或input_checksums。\n"
            "8. 将副本文件夹重命名为`batch60_annotated`并完整回传；人工任务无需终端文档。\n\n"
            "## 通过/失败条件\n\n"
            "通过前提：60/60 JSON齐全、图像名和640×512尺寸一致、只含允许标签、非空shape均为有效polygon、4张No Fire为空JSON，且非JSON冻结资产哈希不变。发现漏文件、非法标签、rectangle、越界、普通天空/云误标或标签定义出现系统性歧义时，R3保持HOLD。\n"
        )
        (package / "README.md").write_text(readme, encoding="utf-8")

        targets = sorted(path for path in package.rglob("*") if path.is_file())
        checksum_text = "".join(f"{sha256(path)}  *{path.relative_to(package).as_posix()}\n" for path in targets)
        (package / "input_checksums.sha256").write_text(checksum_text, encoding="utf-8")
        package.replace(output_dir)


def self_check() -> None:
    rows = []
    groups = (("Fire", "SEQ_07", 13), ("Fire", "SEQ_10", 59), ("No Fire", "SEQ_07", 8))
    index = 0
    for source_class, sequence_id, count in groups:
        batch20_count = 4 if sequence_id == "SEQ_07" else 12
        for group_index in range(count):
            index += 1
            rows.append({
                "selection_index": str(index), "name": f"X_{index:05d}",
                "sequence_id": sequence_id, "source_class": source_class,
                "scene_group": "smoke" if source_class == "Fire" else "clean_hard_negative",
                "smoke_presence": "yes" if source_class == "Fire" else "no",
                "overexposure_presence": "no", "ignore_burden": "none", "confidence": "high",
                "pilot8": "no", "annotation_batch": "batch20" if group_index < batch20_count else "batch60",
                "source_visible_file": "visible/x.jpg", "source_ir_reference_file": "ir_reference/x.jpg",
            })
    selected = select_batch60(rows)
    assert len(selected) == 60
    assert Counter((row["source_class"], row["sequence_id"]) for row in selected) == EXPECTED_BATCH60_GROUPS
    print("prepare_region80_r3 self-check passed")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create the frozen P-002 R3 remaining-60 LabelMe package.")
    parser.add_argument("--region_manifest")
    parser.add_argument("--output_dir")
    parser.add_argument("--self_check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        self_check()
        return
    if not args.region_manifest or not args.output_dir:
        raise ValueError("--region_manifest and --output_dir are required")
    build(Path(args.region_manifest), Path(args.output_dir))
    print(f"P-002 R3 package created: {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
