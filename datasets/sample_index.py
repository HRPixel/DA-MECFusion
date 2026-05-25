"""Generic paired image sample index utilities.

The manifest format is intentionally simple CSV with paths relative to the
dataset root:

name,ir,vis,label
00001D,test/ir/00001D.png,test/vis/00001D.png,test/label/00001D.png
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class SampleRecord:
    name: str
    ir: Path
    vis: Path
    label: Path | None = None


def scan_images(
    directory: str | Path,
    recursive: bool = False,
    name_fn: Callable[[Path], str] | None = None,
) -> dict[str, Path]:
    """Scan image files and index them by a stable sample name."""
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"Image directory not found: {directory}")

    iterator = directory.rglob("*") if recursive else directory.iterdir()
    images: dict[str, Path] = {}
    for path in sorted(iterator):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        name = name_fn(path) if name_fn is not None else path.stem
        if name in images:
            raise ValueError(
                f"Duplicate sample name '{name}' in {directory}. "
                "Use a dataset-specific name_fn or rename files to make keys unique."
            )
        images[name] = path
    return images


def pair_image_maps(
    ir_files: dict[str, Path],
    vis_files: dict[str, Path],
    label_files: dict[str, Path] | None = None,
) -> tuple[list[SampleRecord], list[str], list[str], list[str]]:
    """Pair IR/VIS maps by sample name and optionally attach labels."""
    ir_names = set(ir_files)
    vis_names = set(vis_files)
    common_names = sorted(ir_names & vis_names)
    missing_ir = sorted(vis_names - ir_names)
    missing_vis = sorted(ir_names - vis_names)
    missing_label: list[str] = []

    records: list[SampleRecord] = []
    for name in common_names:
        label_path = None
        if label_files is not None:
            if name in label_files:
                label_path = label_files[name]
            else:
                missing_label.append(name)
        records.append(SampleRecord(name=name, ir=ir_files[name], vis=vis_files[name], label=label_path))

    return records, missing_ir, missing_vis, missing_label


def resolve_manifest_path(data_root: str | Path, split: str) -> Path:
    """Return the default manifest path for a dataset split."""
    return Path(data_root) / "splits" / f"{split}.csv"


def _relative_or_absolute(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path)


def write_manifest(records: list[SampleRecord], manifest_path: str | Path, data_root: str | Path) -> None:
    """Write sample records as a CSV manifest."""
    manifest_path = Path(manifest_path)
    data_root = Path(data_root)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    with manifest_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=("name", "ir", "vis", "label"))
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "name": record.name,
                    "ir": _relative_or_absolute(record.ir, data_root),
                    "vis": _relative_or_absolute(record.vis, data_root),
                    "label": _relative_or_absolute(record.label, data_root) if record.label else "",
                }
            )


def _resolve_record_path(raw_path: str, data_root: Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return data_root / path


def read_manifest(manifest_path: str | Path, data_root: str | Path) -> list[SampleRecord]:
    """Read a CSV manifest and resolve paths against data_root."""
    manifest_path = Path(manifest_path)
    data_root = Path(data_root)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    records: list[SampleRecord] = []
    with manifest_path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        required = {"name", "ir", "vis"}
        missing_columns = required - set(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(f"Manifest {manifest_path} is missing columns: {sorted(missing_columns)}")

        for row in reader:
            label_value = (row.get("label") or "").strip()
            records.append(
                SampleRecord(
                    name=row["name"].strip(),
                    ir=_resolve_record_path(row["ir"].strip(), data_root),
                    vis=_resolve_record_path(row["vis"].strip(), data_root),
                    label=_resolve_record_path(label_value, data_root) if label_value else None,
                )
            )
    return records


def build_split_manifest(
    data_root: str | Path,
    split: str,
    ir_subdir: str = "ir",
    vis_subdir: str = "vis",
    label_subdir: str = "label",
    recursive: bool = False,
) -> tuple[list[SampleRecord], list[str], list[str], list[str]]:
    """Build records for a processed dataset split without modifying images."""
    split_root = Path(data_root) / split
    ir_files = scan_images(split_root / ir_subdir, recursive=recursive)
    vis_files = scan_images(split_root / vis_subdir, recursive=recursive)
    label_dir = split_root / label_subdir
    label_files = scan_images(label_dir, recursive=recursive) if label_dir.is_dir() else None
    return pair_image_maps(ir_files, vis_files, label_files)
