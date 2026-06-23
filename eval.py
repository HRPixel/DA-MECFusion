"""Batch evaluation script for DA-MECFusion V1-minimal MSRS results."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
import warnings

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.sample_index import read_manifest  # noqa: E402
from metrics.fusion_metrics import compute_all_metrics  # noqa: E402
from utils import create_run_dir, prepare_run_dirs  # noqa: E402


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
METRIC_KEYS = ("EN", "SD", "AG", "MI", "Qabf")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate DA-MECFusion V1-minimal MSRS results.")
    parser.add_argument("--ir_dir", type=str, default=None)
    parser.add_argument("--vis_dir", type=str, default=None)
    parser.add_argument("--fused_dir", type=str, required=True)
    parser.add_argument("--manifest", type=str, default=None)
    parser.add_argument("--data_root", type=str, default="data/MSRS")
    parser.add_argument("--run_root", type=str, default="runs")
    parser.add_argument("--run_name", type=str, default="eval_MSRS")
    parser.add_argument("--save_csv", type=str, default=None)
    parser.add_argument("--summary_txt", type=str, default=None)
    parser.add_argument("--recursive", action="store_true")
    return parser.parse_args()


def scan_images(directory: Path, recursive: bool = False) -> dict[str, Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Image directory not found: {directory}")

    iterator = directory.rglob("*") if recursive else directory.iterdir()
    images: dict[str, Path] = {}
    for path in sorted(iterator):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        stem = path.stem
        if stem in images:
            raise ValueError(
                f"Duplicate image stem '{stem}' in {directory}. "
                "Use unique filenames for stem-based pairing."
            )
        images[stem] = path
    return images


def read_gray_01(path: Path) -> np.ndarray:
    image = Image.open(path).convert("L")
    array = np.asarray(image, dtype=np.float64) / 255.0
    return np.clip(array, 0.0, 1.0)


def write_csv(rows: list[dict[str, float | str]], save_csv: Path) -> None:
    save_csv.parent.mkdir(parents=True, exist_ok=True)
    with save_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=("name", *METRIC_KEYS))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def compute_summary(rows: list[dict[str, float | str]]) -> dict[str, tuple[float, float]]:
    summary: dict[str, tuple[float, float]] = {}
    for key in METRIC_KEYS:
        values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        summary[key] = (float(np.mean(values)), float(np.std(values)))
    return summary


def write_summary(summary: dict[str, tuple[float, float]], summary_txt: Path, num_images: int) -> None:
    summary_txt.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"evaluated_images: {num_images}", ""]
    for key in METRIC_KEYS:
        mean_value, std_value = summary[key]
        lines.append(f"{key}: mean={mean_value:.6f}, std={std_value:.6f}")
    summary_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_summary(summary: dict[str, tuple[float, float]], num_images: int) -> None:
    print(f"Evaluated images: {num_images}")
    for key in METRIC_KEYS:
        mean_value, std_value = summary[key]
        print(f"{key}: mean={mean_value:.6f}, std={std_value:.6f}")


def infer_data_root_from_manifest(manifest_path: Path) -> Path:
    if manifest_path.parent.name == "splits":
        return manifest_path.parent.parent
    return manifest_path.parent


def evaluate_with_manifest(
    manifest_path: Path,
    data_root: Path | None,
    fused_dir: Path,
    save_csv: Path,
    summary_txt: Path,
    recursive: bool = False,
) -> None:
    data_root = data_root or infer_data_root_from_manifest(manifest_path)
    records = read_manifest(manifest_path, data_root)
    fused_files = scan_images(fused_dir, recursive=recursive)

    rows: list[dict[str, float | str]] = []
    for record in records:
        name = record.name
        if name not in fused_files:
            warnings.warn(f"Fused image missing for '{name}', skipping.", stacklevel=2)
            continue
        if not record.ir.is_file():
            warnings.warn(f"IR image missing for '{name}': {record.ir}; skipping.", stacklevel=2)
            continue
        if not record.vis.is_file():
            warnings.warn(f"VIS image missing for '{name}': {record.vis}; skipping.", stacklevel=2)
            continue

        ir = read_gray_01(record.ir)
        vis = read_gray_01(record.vis)
        fused = read_gray_01(fused_files[name])

        if ir.shape != vis.shape or ir.shape != fused.shape:
            warnings.warn(
                f"Shape mismatch for '{name}': ir={ir.shape}, vis={vis.shape}, fused={fused.shape}; skipping.",
                stacklevel=2,
            )
            continue

        metrics = compute_all_metrics(ir, vis, fused)
        row: dict[str, float | str] = {"name": name}
        row.update({key: float(metrics[key]) for key in METRIC_KEYS})
        rows.append(row)

    finish_evaluation(rows, save_csv, summary_txt)


def evaluate(
    ir_dir: Path,
    vis_dir: Path,
    fused_dir: Path,
    save_csv: Path,
    summary_txt: Path,
    recursive: bool = False,
) -> None:
    ir_files = scan_images(ir_dir, recursive=recursive)
    vis_files = scan_images(vis_dir, recursive=recursive)
    fused_files = scan_images(fused_dir, recursive=recursive)

    rows: list[dict[str, float | str]] = []
    for name in sorted(ir_files):
        if name not in vis_files:
            warnings.warn(f"VIS image missing for '{name}', skipping.", stacklevel=2)
            continue
        if name not in fused_files:
            warnings.warn(f"Fused image missing for '{name}', skipping.", stacklevel=2)
            continue

        ir = read_gray_01(ir_files[name])
        vis = read_gray_01(vis_files[name])
        fused = read_gray_01(fused_files[name])

        if ir.shape != vis.shape or ir.shape != fused.shape:
            warnings.warn(
                f"Shape mismatch for '{name}': ir={ir.shape}, vis={vis.shape}, fused={fused.shape}; skipping.",
                stacklevel=2,
            )
            continue

        metrics = compute_all_metrics(ir, vis, fused)
        row: dict[str, float | str] = {"name": name}
        row.update({key: float(metrics[key]) for key in METRIC_KEYS})
        rows.append(row)

    finish_evaluation(rows, save_csv, summary_txt)


def finish_evaluation(
    rows: list[dict[str, float | str]],
    save_csv: Path,
    summary_txt: Path,
) -> None:
    if not rows:
        raise RuntimeError("No valid IR/VIS/fused image triplets were evaluated.")
    write_csv(rows, save_csv)
    summary = compute_summary(rows)
    write_summary(summary, summary_txt, len(rows))
    print_summary(summary, len(rows))
    print(f"Metrics CSV saved to: {save_csv}")
    print(f"Summary saved to: {summary_txt}")


def main() -> None:
    args = parse_args()
    run_dir = create_run_dir(args.run_root, args.run_name)
    dirs = prepare_run_dirs(run_dir)
    save_csv = Path(args.save_csv) if args.save_csv else dirs["metrics"] / "metrics.csv"
    summary_txt = Path(args.summary_txt) if args.summary_txt else dirs["metrics"] / "summary.txt"

    if args.manifest:
        evaluate_with_manifest(
            manifest_path=Path(args.manifest),
            data_root=Path(args.data_root) if args.data_root else None,
            fused_dir=Path(args.fused_dir),
            save_csv=save_csv,
            summary_txt=summary_txt,
            recursive=args.recursive,
        )
    else:
        if args.ir_dir is None or args.vis_dir is None:
            raise ValueError("Either --manifest or both --ir_dir and --vis_dir must be provided.")
        evaluate(
            ir_dir=Path(args.ir_dir),
            vis_dir=Path(args.vis_dir),
            fused_dir=Path(args.fused_dir),
            save_csv=save_csv,
            summary_txt=summary_txt,
            recursive=args.recursive,
        )

    print(f"Run directory: {dirs['root']}")


if __name__ == "__main__":
    main()
