#!/usr/bin/env python3
"""Create explicit sequence-level FLAME3 manifests from inventory.csv."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory


MANIFEST_FIELDS = ("name", "ir", "vis", "label", "tiff", "sequence_id", "source_class")
REQUIRED_INVENTORY_FIELDS = {
    "sample_id",
    "ir_path",
    "vis_path",
    "tiff_path",
    "sequence_id",
    "source_class",
    "status",
}


def read_inventory(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Inventory not found: {path}")
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_INVENTORY_FIELDS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Inventory is missing columns: {sorted(missing)}")
        rows = [row for row in reader if row["status"].strip() == "pass"]
    names = [row["sample_id"].strip() for row in rows]
    if len(names) != len(set(names)):
        raise ValueError("Inventory contains duplicate sample_id values")
    if any(not row["sequence_id"].strip() for row in rows):
        raise ValueError("Every passing inventory row must have a sequence_id")
    return rows


def make_splits(
    rows: list[dict[str, str]],
    train_sequences: set[str] | None,
    val_sequences: set[str],
    test_sequences: set[str],
    calibration_sequences: set[str],
) -> dict[str, list[dict[str, str]]]:
    named = {
        "validation": val_sequences,
        "test": test_sequences,
        "calibration": calibration_sequences,
    }
    requested = [(name, sequence) for name, values in named.items() for sequence in values]
    requested += [("train", sequence) for sequence in (train_sequences or set())]
    owners: dict[str, str] = {}
    for split, sequence in requested:
        if sequence in owners:
            raise ValueError(f"Sequence {sequence} appears in both {owners[sequence]} and {split}")
        owners[sequence] = split

    available = {row["sequence_id"].strip() for row in rows}
    unknown = set(owners) - available
    if unknown:
        raise ValueError(f"Unknown sequence IDs: {sorted(unknown)}")
    if train_sequences is None:
        train_sequences = available - set(owners)
    else:
        unassigned = available - set(owners)
        if unassigned:
            raise ValueError(f"Explicit split leaves sequences unassigned: {sorted(unassigned)}")
    named["train"] = train_sequences

    splits: dict[str, list[dict[str, str]]] = {}
    for split, sequences in named.items():
        selected = [row for row in rows if row["sequence_id"].strip() in sequences]
        splits[split] = sorted(selected, key=lambda row: row["sample_id"])
    for split in ("train", "validation", "test"):
        if not splits[split]:
            raise ValueError(f"{split} split is empty")
    return splits


def write_outputs(splits: dict[str, list[dict[str, str]]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = ["# FLAME3 sequence split summary", ""]
    for split, rows in splits.items():
        path = output_dir / f"{split}.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        "name": row["sample_id"],
                        "ir": row["ir_path"],
                        "vis": row["vis_path"],
                        "label": "",
                        "tiff": row["tiff_path"],
                        "sequence_id": row["sequence_id"],
                        "source_class": row["source_class"],
                    }
                )
        classes = Counter(row["source_class"] for row in rows)
        sequences = sorted({row["sequence_id"] for row in rows})
        summary.extend(
            [
                f"## {split}",
                "",
                f"- Samples: {len(rows)}",
                f"- Fire: {classes['Fire']}",
                f"- No Fire: {classes['No Fire']}",
                f"- Sequences: {', '.join(sequences) or '(none)'}",
                "",
            ]
        )
    (output_dir / "split_summary.md").write_text("\n".join(summary), encoding="utf-8")


def self_check() -> None:
    rows = [
        {
            "sample_id": f"S_{index}",
            "ir_path": f"ir/{index}.jpg",
            "vis_path": f"vis/{index}.jpg",
            "tiff_path": f"tiff/{index}.tiff",
            "sequence_id": f"SEQ_{index}",
            "source_class": "Fire" if index < 4 else "No Fire",
            "status": "pass",
        }
        for index in range(1, 5)
    ]
    splits = make_splits(rows, None, {"SEQ_3"}, {"SEQ_4"}, {"SEQ_2"})
    assert [row["sequence_id"] for row in splits["train"]] == ["SEQ_1"]
    with TemporaryDirectory() as directory:
        output_dir = Path(directory)
        write_outputs(splits, output_dir)
        assert (output_dir / "train.csv").is_file()
        assert (output_dir / "split_summary.md").is_file()
    print("flame3_make_splits self-check passed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path)
    parser.add_argument("--output_dir", type=Path)
    parser.add_argument("--train_sequences", nargs="+", default=None)
    parser.add_argument("--val_sequences", nargs="+")
    parser.add_argument("--test_sequences", nargs="+")
    parser.add_argument("--calibration_sequences", nargs="*", default=[])
    parser.add_argument("--self_check", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_check:
        self_check()
        return
    if args.inventory is None or args.output_dir is None:
        raise ValueError("--inventory and --output_dir are required")
    if not args.val_sequences or not args.test_sequences:
        raise ValueError("--val_sequences and --test_sequences are required")
    rows = read_inventory(args.inventory)
    splits = make_splits(
        rows,
        set(args.train_sequences) if args.train_sequences else None,
        set(args.val_sequences),
        set(args.test_sequences),
        set(args.calibration_sequences),
    )
    write_outputs(splits, args.output_dir)
    for split, split_rows in splits.items():
        print(f"{split}: {len(split_rows)}")
    print(f"Wrote manifests to: {args.output_dir}")


if __name__ == "__main__":
    main()
