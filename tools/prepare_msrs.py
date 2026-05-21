"""Prepare MSRS data into the DA-MECFusion V1 processed layout."""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
IR_DIR_NAMES = ("ir", "infrared", "Infrared")
VIS_DIR_NAMES = ("vis", "visible", "Visible", "vi", "VI")
LABEL_DIR_NAMES = ("label", "labels", "Label", "Labels", "Segmentation_labels")


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("Boolean value expected: true or false")


def find_named_dir(root: Path, candidates: tuple[str, ...], required: bool = True) -> Path | None:
    for name in candidates:
        direct = root / name
        if direct.is_dir():
            return direct

    lower_candidates = {name.lower() for name in candidates}
    for path in root.iterdir() if root.is_dir() else []:
        if path.is_dir() and path.name.lower() in lower_candidates:
            return path

    if required:
        raise FileNotFoundError(
            f"Could not find any of {candidates} under {root}. "
            "Please manually arrange the raw MSRS directory or adjust the script."
        )
    return None


def list_images(directory: Path) -> dict[str, Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Image directory not found: {directory}")

    images: dict[str, Path] = {}
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        if path.stem in images:
            raise ValueError(f"Duplicate image stem '{path.stem}' in {directory}")
        images[path.stem] = path
    return images


def discover_flat_layout(raw_root: Path) -> tuple[Path, Path, Path | None]:
    ir_dir = find_named_dir(raw_root, IR_DIR_NAMES, required=True)
    vis_dir = find_named_dir(raw_root, VIS_DIR_NAMES, required=True)
    label_dir = find_named_dir(raw_root, LABEL_DIR_NAMES, required=False)
    assert ir_dir is not None and vis_dir is not None
    return ir_dir, vis_dir, label_dir


def pair_files(
    ir_files: dict[str, Path],
    vis_files: dict[str, Path],
    label_files: dict[str, Path] | None,
) -> tuple[list[dict[str, Path | str]], list[str], list[str], list[str]]:
    ir_stems = set(ir_files)
    vis_stems = set(vis_files)
    common_stems = sorted(ir_stems & vis_stems)
    missing_ir = sorted(vis_stems - ir_stems)
    missing_vis = sorted(ir_stems - vis_stems)
    missing_label: list[str] = []

    samples: list[dict[str, Path | str]] = []
    for stem in common_stems:
        sample: dict[str, Path | str] = {
            "name": stem,
            "ir": ir_files[stem],
            "vis": vis_files[stem],
        }
        if label_files is not None:
            if stem in label_files:
                sample["label"] = label_files[stem]
            else:
                missing_label.append(stem)
        samples.append(sample)

    if not samples:
        raise RuntimeError(
            "No paired IR/VIS samples found. Please check that filenames share the same stems."
        )
    return samples, missing_ir, missing_vis, missing_label


def ensure_output_dirs(out_root: Path) -> None:
    for split in ("train", "test"):
        for subdir in ("ir", "vis", "label"):
            (out_root / split / subdir).mkdir(parents=True, exist_ok=True)


def place_file(src: Path, dst: Path, copy_files: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()

    if copy_files:
        shutil.copy2(src, dst)
    else:
        try:
            dst.symlink_to(src.resolve())
        except OSError:
            shutil.copy2(src, dst)


def write_split(
    samples: list[dict[str, Path | str]],
    out_root: Path,
    split: str,
    copy_files: bool,
) -> int:
    copied_labels = 0
    for sample in samples:
        name = str(sample["name"])
        ir_src = Path(sample["ir"])
        vis_src = Path(sample["vis"])
        place_file(ir_src, out_root / split / "ir" / f"{name}{ir_src.suffix.lower()}", copy_files)
        place_file(vis_src, out_root / split / "vis" / f"{name}{vis_src.suffix.lower()}", copy_files)

        if "label" in sample:
            label_src = Path(sample["label"])
            place_file(
                label_src,
                out_root / split / "label" / f"{name}{label_src.suffix.lower()}",
                copy_files,
            )
            copied_labels += 1
    return copied_labels


def save_split_info(
    out_root: Path,
    total_count: int,
    train_count: int,
    test_count: int,
    missing_ir: list[str],
    missing_vis: list[str],
    missing_label: list[str],
    copied_train_labels: int,
    copied_test_labels: int,
) -> None:
    lines = [
        "MSRS split information",
        f"total_samples: {total_count}",
        f"train_count: {train_count}",
        f"test_count: {test_count}",
        f"train_labels_copied: {copied_train_labels}",
        f"test_labels_copied: {copied_test_labels}",
        f"missing_ir_count: {len(missing_ir)}",
        f"missing_vis_count: {len(missing_vis)}",
        f"missing_label_count: {len(missing_label)}",
        "",
        "missing_ir:",
        *missing_ir,
        "",
        "missing_vis:",
        *missing_vis,
        "",
        "missing_label:",
        *missing_label,
        "",
    ]
    (out_root / "split_info.txt").write_text("\n".join(lines), encoding="utf-8")


def prepare_msrs(
    raw_root: str | Path,
    out_root: str | Path,
    train_ratio: float = 0.8,
    copy_files: bool = True,
    seed: int = 42,
) -> None:
    raw_root = Path(raw_root)
    out_root = Path(out_root)

    if not raw_root.is_dir():
        raise FileNotFoundError(f"Raw MSRS root not found: {raw_root}")
    if not 0.0 < train_ratio < 1.0:
        raise ValueError(f"train_ratio must be in (0, 1), got {train_ratio}")

    ir_dir, vis_dir, label_dir = discover_flat_layout(raw_root)
    ir_files = list_images(ir_dir)
    vis_files = list_images(vis_dir)
    label_files = list_images(label_dir) if label_dir is not None else None

    samples, missing_ir, missing_vis, missing_label = pair_files(ir_files, vis_files, label_files)
    rng = random.Random(seed)
    rng.shuffle(samples)

    train_count = int(len(samples) * train_ratio)
    train_samples = samples[:train_count]
    test_samples = samples[train_count:]

    ensure_output_dirs(out_root)
    copied_train_labels = write_split(train_samples, out_root, "train", copy_files)
    copied_test_labels = write_split(test_samples, out_root, "test", copy_files)
    save_split_info(
        out_root=out_root,
        total_count=len(samples),
        train_count=len(train_samples),
        test_count=len(test_samples),
        missing_ir=missing_ir,
        missing_vis=missing_vis,
        missing_label=missing_label,
        copied_train_labels=copied_train_labels,
        copied_test_labels=copied_test_labels,
    )

    print("MSRS preparation finished.")
    print(f"Raw root: {raw_root}")
    print(f"Output root: {out_root}")
    print(f"Total paired samples: {len(samples)}")
    print(f"Train samples: {len(train_samples)}")
    print(f"Test samples: {len(test_samples)}")
    print(f"Missing IR files: {len(missing_ir)}")
    print(f"Missing VIS files: {len(missing_vis)}")
    print(f"Missing labels: {len(missing_label)}")
    print(f"Split info: {out_root / 'split_info.txt'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare MSRS dataset for DA-MECFusion V1.")
    parser.add_argument("--raw_root", type=str, default="data/raw/MSRS")
    parser.add_argument("--out_root", type=str, default="data/processed/MSRS")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--copy_files", type=str2bool, default=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    prepare_msrs(
        raw_root=args.raw_root,
        out_root=args.out_root,
        train_ratio=args.train_ratio,
        copy_files=args.copy_files,
        seed=args.seed,
    )
