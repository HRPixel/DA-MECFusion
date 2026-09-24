"""Summarize Gate 3 ablation metrics, brightness stability, and paired deltas."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image


MODES = {"A0": "a0", "A1": "a1", "A2": "a2", "A3": "a3"}
SPLITS = {"fire": ("test_fire.csv", 123), "nofire": ("test_no_fire.csv", 108)}
METRICS = ("EN", "SD", "AG", "MI", "SCD", "Qabf_standard")
CONTRASTS = (("A0", "A1"), ("A0", "A2"), ("A1", "A3"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root")
    parser.add_argument("--split_root")
    parser.add_argument("--run_root")
    parser.add_argument("--output_dir")
    parser.add_argument("--gate3_state", default=None)
    parser.add_argument("--self_check", action="store_true")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def local_run(run_root: Path, recorded: str) -> Path:
    return run_root / Path(recorded).name


def discover_one(run_root: Path, prefix: str) -> Path:
    matches = sorted(path for path in run_root.glob(f"{prefix}_*") if path.is_dir())
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one run for {prefix}, found {len(matches)}")
    return matches[0]


def resolve_runs(run_root: Path, state_path: Path | None) -> dict[str, dict[str, dict[str, Path]]]:
    resolved: dict[str, dict[str, dict[str, Path]]] = {}
    state = {row["Id"]: row for row in read_csv(state_path)} if state_path else {}
    if state_path and set(state) != set(MODES):
        raise ValueError(f"Gate 3 state must contain {sorted(MODES)}: {state_path}")
    for mode, short in MODES.items():
        resolved[mode] = {}
        for split in SPLITS:
            if state:
                test_key = "FireTestDirectory" if split == "fire" else "NoFireTestDirectory"
                eval_key = "FireEvalDirectory" if split == "fire" else "NoFireEvalDirectory"
                test_run = local_run(run_root, state[mode][test_key])
                eval_run = local_run(run_root, state[mode][eval_key])
            else:
                test_run = discover_one(run_root, f"test_abl_{short}_{split}")
                eval_run = discover_one(run_root, f"eval_abl_{short}_{split}")
            if not test_run.is_dir() or not eval_run.is_dir():
                raise FileNotFoundError(f"Missing run for {mode}/{split}: {test_run}, {eval_run}")
            resolved[mode][split] = {"test": test_run, "eval": eval_run}
    return resolved


def image_gray(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    return np.asarray(Image.open(path).convert("L"), dtype=np.float64)


def mean_std(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    return float(array.mean()), float(array.std())


def brightness(ir_mean: float, vis_mean: float, fused_mean: float) -> tuple[int, int, int, float]:
    low, high = sorted((ir_mean, vis_mean))
    if fused_mean > high:
        return 1, 1, 0, fused_mean - high
    if fused_mean < low:
        return 1, 0, 1, low - fused_mean
    return 0, 0, 0, 0.0


def load_manifests(split_root: Path) -> dict[str, dict[str, dict[str, str]]]:
    manifests: dict[str, dict[str, dict[str, str]]] = {}
    for split, (filename, expected) in SPLITS.items():
        rows = read_csv(split_root / filename)
        if len(rows) != expected or len({row["name"] for row in rows}) != expected:
            raise ValueError(f"Manifest count/name mismatch: {filename}")
        manifests[split] = {row["name"]: row for row in rows}
    return manifests


def build_per_image(
    data_root: Path,
    manifests: dict[str, dict[str, dict[str, str]]],
    runs: dict[str, dict[str, dict[str, Path]]],
) -> list[dict[str, object]]:
    input_stats: dict[tuple[str, str], tuple[float, float]] = {}
    for split, rows in manifests.items():
        for name, row in rows.items():
            ir = image_gray(data_root / row["ir"])
            vis = image_gray(data_root / row["vis"])
            if ir.shape != vis.shape:
                raise ValueError(f"Input shape mismatch: {name}")
            input_stats[(split, name)] = (float(ir.mean()), float(vis.mean()))

    output: list[dict[str, object]] = []
    for mode in MODES:
        for split, manifest in manifests.items():
            metric_rows = {row["name"]: row for row in read_csv(runs[mode][split]["eval"] / "metrics/metrics.csv")}
            if set(metric_rows) != set(manifest):
                raise ValueError(f"Metric names differ from manifest: {mode}/{split}")
            fused_dir = runs[mode][split]["test"] / "outputs/fused"
            for name, meta in manifest.items():
                fused_mean = float(image_gray(fused_dir / f"{name}.png").mean())
                ir_mean, vis_mean = input_stats[(split, name)]
                outside, above, below, excess = brightness(ir_mean, vis_mean, fused_mean)
                row: dict[str, object] = {
                    "mode": mode,
                    "split": split,
                    "name": name,
                    "sequence_id": meta["sequence_id"],
                    "source_class": meta["source_class"],
                }
                row.update({key: float(metric_rows[name][key]) for key in METRICS})
                row.update(
                    {
                        "ir_mean_255": ir_mean,
                        "vis_mean_255": vis_mean,
                        "fused_mean_255": fused_mean,
                        "outside_input_mean_range": outside,
                        "above_input_mean_range": above,
                        "below_input_mean_range": below,
                        "range_excess_255": excess,
                    }
                )
                output.append(row)
    return output


def summarize(per_image: list[dict[str, object]]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for mode in MODES:
        for split in SPLITS:
            rows = [row for row in per_image if row["mode"] == mode and row["split"] == split]
            result: dict[str, object] = {"mode": mode, "split": split, "n": len(rows)}
            for metric in METRICS:
                mean, std = mean_std([float(row[metric]) for row in rows])
                result[f"{metric}_mean"] = mean
                result[f"{metric}_std"] = std
            outside = [row for row in rows if int(row["outside_input_mean_range"])]
            result.update(
                {
                    "outside_count": len(outside),
                    "outside_ratio": len(outside) / len(rows),
                    "above_count": sum(int(row["above_input_mean_range"]) for row in rows),
                    "below_count": sum(int(row["below_input_mean_range"]) for row in rows),
                    "conditional_excess_255": float(np.mean([float(row["range_excess_255"]) for row in outside])) if outside else 0.0,
                    "all_sample_excess_255": float(np.mean([float(row["range_excess_255"]) for row in rows])),
                }
            )
            output.append(result)
    return output


def paired(per_image: list[dict[str, object]]) -> list[dict[str, object]]:
    index = {(str(row["mode"]), str(row["split"]), str(row["name"])): row for row in per_image}
    output: list[dict[str, object]] = []
    for left, right in CONTRASTS:
        for split in SPLITS:
            names = sorted(name for mode, current_split, name in index if mode == left and current_split == split)
            sequences = ["All"]
            if split == "fire":
                sequences += sorted({str(index[(left, split, name)]["sequence_id"]) for name in names})
            for sequence in sequences:
                selected = names if sequence == "All" else [name for name in names if index[(left, split, name)]["sequence_id"] == sequence]
                for metric in METRICS:
                    values = np.asarray(
                        [float(index[(left, split, name)][metric]) - float(index[(right, split, name)][metric]) for name in selected],
                        dtype=np.float64,
                    )
                    output.append(
                        {
                            "contrast": f"{left}-{right}",
                            "split": split,
                            "sequence": sequence,
                            "metric": metric,
                            "n": len(values),
                            "mean_delta": float(values.mean()),
                            "std_delta": float(values.std()),
                            "median_delta": float(np.median(values)),
                            "positive_count": int((values > 0).sum()),
                            "negative_count": int((values < 0).sum()),
                            "zero_count": int((values == 0).sum()),
                        }
                    )
    return output


def sequence_summary(per_image: list[dict[str, object]]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    sequences = sorted({str(row["sequence_id"]) for row in per_image if row["split"] == "fire"})
    for mode in MODES:
        for sequence in sequences:
            rows = [row for row in per_image if row["mode"] == mode and row["split"] == "fire" and row["sequence_id"] == sequence]
            result: dict[str, object] = {"mode": mode, "sequence": sequence, "n": len(rows)}
            for metric in METRICS:
                mean, std = mean_std([float(row[metric]) for row in rows])
                result[f"{metric}_mean"] = mean
                result[f"{metric}_std"] = std
            outside = [row for row in rows if int(row["outside_input_mean_range"])]
            result["outside_count"] = len(outside)
            result["outside_ratio"] = len(outside) / len(rows)
            result["conditional_excess_255"] = float(np.mean([float(row["range_excess_255"]) for row in outside])) if outside else 0.0
            output.append(result)
    return output


def markdown(summary: list[dict[str, object]], paired_rows: list[dict[str, object]]) -> str:
    lines = [
        "# V2 Gate 3消融定量汇总",
        "",
        "本报告覆盖standard指标和图像均值亮度稳定性；机制与人工标注区域结论仍需结合Gate 3独立诊断报告。",
        "",
        "## 全局指标",
        "",
        "| Split | Mode | EN | SD | AG | MI | SCD | Qabf_standard |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['split']} | {row['mode']} | {row['EN_mean']:.6f} | {row['SD_mean']:.6f} | {row['AG_mean']:.6f} | "
            f"{row['MI_mean']:.6f} | {row['SCD_mean']:.6f} | {row['Qabf_standard_mean']:.6f} |"
        )
    lines += [
        "",
        "## 亮度稳定性",
        "",
        "| Split | Mode | 越出两输入均值范围 | 高于上界 | 低于下界 | 越界样本平均幅度 /255 | 全样本平均幅度 /255 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['split']} | {row['mode']} | {row['outside_count']}/{row['n']} ({row['outside_ratio']:.2%}) | "
            f"{row['above_count']} | {row['below_count']} | {row['conditional_excess_255']:.6f} | {row['all_sample_excess_255']:.6f} |"
        )
    lines += [
        "",
        "## 冻结的主要对比",
        "",
        "正差值表示左侧模型的指标值更大。相邻帧不是独立实验重复，因此帧数只作配对描述。",
        "",
        "| Contrast | Split | Metric | Mean delta | Positive frames |",
        "|---|---|---|---:|---:|",
    ]
    for row in paired_rows:
        if row["sequence"] == "All" and row["metric"] in {"SCD", "Qabf_standard"}:
            lines.append(
                f"| {row['contrast']} | {row['split']} | {row['metric']} | {row['mean_delta']:+.6f} | "
                f"{row['positive_count']}/{row['n']} |"
            )
    lines += [
        "",
        "## 证据边界",
        "",
        "- Fire仅包含两个序列，No Fire仅包含一个序列；逐帧差值是配对描述证据，不是独立重复显著性检验。",
        "- EN、SD、AG和MI为辅助指标；必须联合解释Qabf_standard/SCD、亮度稳定性及机制/区域证据。",
        "- 本文件本身不能建立prior或MEC的因果贡献结论。",
        "",
    ]
    return "\n".join(lines)


def self_check() -> None:
    assert brightness(10.0, 20.0, 25.0) == (1, 1, 0, 5.0)
    assert brightness(10.0, 20.0, 5.0) == (1, 0, 1, 5.0)
    assert brightness(10.0, 20.0, 15.0) == (0, 0, 0, 0.0)
    assert mean_std([1.0, 3.0]) == (2.0, 1.0)
    print("analyze_v2_ablation_gate3 self-check passed")


def main() -> None:
    args = parse_args()
    if args.self_check:
        self_check()
        return
    missing = [name for name in ("data_root", "split_root", "run_root", "output_dir") if not getattr(args, name)]
    if missing:
        raise ValueError(f"Missing required arguments: {', '.join(missing)}")
    data_root = Path(args.data_root)
    split_root = Path(args.split_root)
    run_root = Path(args.run_root)
    output_dir = Path(args.output_dir)
    state_path = Path(args.gate3_state) if args.gate3_state else None
    for path in (data_root, split_root, run_root):
        if not path.is_dir():
            raise FileNotFoundError(path)
    if state_path and not state_path.is_file():
        raise FileNotFoundError(state_path)

    manifests = load_manifests(split_root)
    runs = resolve_runs(run_root, state_path)
    per_image = build_per_image(data_root, manifests, runs)
    summary_rows = summarize(per_image)
    paired_rows = paired(per_image)
    sequence_rows = sequence_summary(per_image)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "per_image.csv", per_image)
    write_csv(output_dir / "summary.csv", summary_rows)
    write_csv(output_dir / "paired_differences.csv", paired_rows)
    write_csv(output_dir / "fire_sequence_summary.csv", sequence_rows)
    (output_dir / "summary.md").write_text(markdown(summary_rows, paired_rows), encoding="utf-8")
    print(f"Gate 3 quantitative analysis passed: {output_dir}")


if __name__ == "__main__":
    main()
