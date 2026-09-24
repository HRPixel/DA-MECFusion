#!/usr/bin/env python3
"""Build the blind 94-image R0 screening package for P-002."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image


SCREENING_FIELDS = (
    "screening_index",
    "name",
    "sequence_id",
    "source_class",
    "visible_file",
    "ir_reference_file",
    "smoke_presence",
    "overexposure_presence",
    "ignore_burden",
    "scene_group",
    "confidence",
    "note",
)
PACKAGE_FIELDS = (
    "name",
    "sequence_id",
    "source_class",
    "source_ir",
    "source_vis",
    "ir_reference_file",
    "visible_file",
    "ir_bytes",
    "ir_sha256",
    "visible_bytes",
    "visible_sha256",
)
EXPECTED_GROUPS = Counter({("Fire", "SEQ_10"): 70, ("Fire", "SEQ_07"): 16, ("No Fire", "SEQ_07"): 8})


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_rows(manifest: Path, data_root: Path) -> list[dict[str, str | Path]]:
    with manifest.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        required = {"name", "ir", "vis", "sequence_id", "source_class"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Manifest is missing columns: {sorted(missing)}")
        rows = []
        seen = set()
        for raw in reader:
            values = {key: (raw.get(key) or "").strip() for key in required}
            if any(not value for value in values.values()):
                raise ValueError(f"Manifest has a blank required field: {raw}")
            if values["name"] in seen:
                raise ValueError(f"Duplicate sample name: {values['name']}")
            seen.add(values["name"])
            rows.append(
                {
                    **values,
                    "ir_path": data_root / values["ir"],
                    "vis_path": data_root / values["vis"],
                }
            )
    return rows


def verify_image(path: Path, expected_size: tuple[int, int]) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")
    with Image.open(path) as image:
        if image.size != expected_size:
            raise ValueError(f"Unexpected image size for {path}: {image.size}, expected {expected_size}")
        image.verify()


def write_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build_package(
    data_root: Path,
    manifest: Path,
    pilot_label_dir: Path,
    output_dir: Path,
    *,
    expected_count: int = 94,
    expected_pilot_count: int = 8,
    expected_groups: Counter[tuple[str, str]] | None = EXPECTED_GROUPS,
    expected_size: tuple[int, int] = (640, 512),
) -> None:
    for path, label in ((data_root, "data root"), (pilot_label_dir, "pilot label directory")):
        if not path.is_dir():
            raise FileNotFoundError(f"Missing {label}: {path}")
    if not manifest.is_file():
        raise FileNotFoundError(f"Missing manifest: {manifest}")
    if output_dir.exists():
        raise FileExistsError(f"Output already exists; refusing to overwrite: {output_dir}")

    rows = read_rows(manifest, data_root)
    if len(rows) != expected_count:
        raise ValueError(f"Expected {expected_count} manifest rows, found {len(rows)}")
    groups = Counter((str(row["source_class"]), str(row["sequence_id"])) for row in rows)
    if expected_groups is not None and groups != expected_groups:
        raise ValueError(f"Calibration group counts changed: {dict(groups)}")

    names = {str(row["name"]) for row in rows}
    pilot_files = sorted(pilot_label_dir.glob("*.json"))
    if len(pilot_files) != expected_pilot_count:
        raise ValueError(f"Expected {expected_pilot_count} pilot JSON files, found {len(pilot_files)}")
    unknown_pilot = sorted(path.stem for path in pilot_files if path.stem not in names)
    if unknown_pilot:
        raise ValueError(f"Pilot labels are not in calibration manifest: {unknown_pilot}")

    for row in rows:
        verify_image(Path(row["ir_path"]), expected_size)
        verify_image(Path(row["vis_path"]), expected_size)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f".{output_dir.name}_", dir=output_dir.parent) as temporary:
        staging = Path(temporary)
        visible_dir = staging / "visible"
        ir_dir = staging / "ir_reference"
        visible_dir.mkdir()
        ir_dir.mkdir()
        screening_rows: list[dict[str, object]] = []
        package_rows: list[dict[str, object]] = []

        for index, row in enumerate(rows, start=1):
            name = str(row["name"])
            source_ir = Path(row["ir_path"])
            source_vis = Path(row["vis_path"])
            copied_ir = ir_dir / f"{name}{source_ir.suffix.upper()}"
            copied_vis = visible_dir / f"{name}{source_vis.suffix.upper()}"
            shutil.copy2(source_ir, copied_ir)
            shutil.copy2(source_vis, copied_vis)
            ir_rel = copied_ir.relative_to(staging).as_posix()
            vis_rel = copied_vis.relative_to(staging).as_posix()
            screening_rows.append(
                {
                    "screening_index": index,
                    "name": name,
                    "sequence_id": row["sequence_id"],
                    "source_class": row["source_class"],
                    "visible_file": vis_rel,
                    "ir_reference_file": ir_rel,
                    "smoke_presence": "",
                    "overexposure_presence": "",
                    "ignore_burden": "",
                    "scene_group": "",
                    "confidence": "",
                    "note": "",
                }
            )
            package_rows.append(
                {
                    "name": name,
                    "sequence_id": row["sequence_id"],
                    "source_class": row["source_class"],
                    "source_ir": row["ir"],
                    "source_vis": row["vis"],
                    "ir_reference_file": ir_rel,
                    "visible_file": vis_rel,
                    "ir_bytes": copied_ir.stat().st_size,
                    "ir_sha256": sha256(copied_ir),
                    "visible_bytes": copied_vis.stat().st_size,
                    "visible_sha256": sha256(copied_vis),
                }
            )

        write_csv(staging / "screening94.csv", SCREENING_FIELDS, screening_rows)
        write_csv(staging / "package_manifest.csv", PACKAGE_FIELDS, package_rows)
        group_text = ", ".join(f"{source}/{sequence}={count}" for (source, sequence), count in sorted(groups.items()))
        (staging / "README.md").write_text(
            "# P-002 R0：FLAME3 calibration-94 图像级盘点包\n\n"
            f"- 样本数：{len(rows)}（{group_text}）\n"
            f"- calibration manifest SHA256：`{sha256(manifest)}`\n"
            f"- 已核对pilot JSON：{len(pilot_files)}个，且全部属于本清单。\n"
            f"- 图像完整性：IR/VIS均可解码且尺寸为{expected_size[0]}×{expected_size[1]}。\n\n"
            "## 操作\n\n"
            "1. 逐行同时查看 `visible/` 与 `ir_reference/` 中的同名图像；烟雾和过曝光边界只依据Visible判断。\n"
            "2. 只填写 `screening94.csv` 最后六列，不修改名称、序列、来源类别或文件路径。\n"
            "3. `smoke_presence`、`overexposure_presence`：`yes/no/uncertain`。\n"
            "4. `ignore_burden`：`none/low/high`。\n"
            "5. `scene_group`：`smoke/overexposure/mixed/clean_hard_negative`。\n"
            "6. `confidence`：`high/medium/low`；`note`只写主要歧义，可留空。\n"
            "7. No Fire仍须逐张判断，不得因目录名称直接填为阴性。完成后以UTF-8 CSV保存并回传整个文件夹。\n\n"
            "本包不包含任何A0–A3逐图指标、prior数值或模型筛选结果；R0不画多边形，也不自动冻结80张。\n",
            encoding="utf-8",
        )

        checksum_targets = sorted(path for path in staging.rglob("*") if path.is_file())
        (staging / "checksums.sha256").write_text(
            "".join(f"{sha256(path)}  *{path.relative_to(staging).as_posix()}\n" for path in checksum_targets),
            encoding="utf-8",
        )
        staging.replace(output_dir)


def self_check() -> None:
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        data = root / "data"
        labels = root / "labels"
        labels.mkdir()
        rows = []
        for index in range(3):
            name = f"X_{index:05d}"
            ir = data / "ir" / f"{index:05d}.JPG"
            vis = data / "vis" / f"{index:05d}.JPG"
            ir.parent.mkdir(parents=True, exist_ok=True)
            vis.parent.mkdir(parents=True, exist_ok=True)
            Image.new("L", (640, 512), index * 20).save(ir)
            Image.new("RGB", (640, 512), (index * 20, 0, 0)).save(vis)
            rows.append(
                {
                    "name": name,
                    "ir": ir.relative_to(data).as_posix(),
                    "vis": vis.relative_to(data).as_posix(),
                    "sequence_id": "SEQ_00",
                    "source_class": "Test",
                }
            )
        manifest = root / "calibration.csv"
        write_csv(manifest, ("name", "ir", "vis", "sequence_id", "source_class"), rows)
        (labels / "X_00000.json").write_text(json.dumps({"shapes": []}), encoding="utf-8")
        output = root / "package"
        build_package(
            data,
            manifest,
            labels,
            output,
            expected_count=3,
            expected_pilot_count=1,
            expected_groups=None,
        )
        assert len(list((output / "visible").glob("*.JPG"))) == 3
        with (output / "screening94.csv").open(encoding="utf-8-sig", newline="") as file:
            screening = csv.DictReader(file)
            assert tuple(screening.fieldnames or ()) == SCREENING_FIELDS
            assert len(list(screening)) == 3
        assert (output / "checksums.sha256").is_file()
    print("prepare_region80_screening self-check passed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the P-002 blind calibration-94 R0 screening package.")
    parser.add_argument("--data_root")
    parser.add_argument("--manifest")
    parser.add_argument("--pilot_label_dir")
    parser.add_argument("--output_dir")
    parser.add_argument("--self_check", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_check:
        self_check()
        return
    missing = [name for name in ("data_root", "manifest", "pilot_label_dir", "output_dir") if not getattr(args, name)]
    if missing:
        raise ValueError(f"Missing required arguments: {', '.join('--' + name for name in missing)}")
    build_package(Path(args.data_root), Path(args.manifest), Path(args.pilot_label_dir), Path(args.output_dir))
    print(f"P-002 R0 screening package created: {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
