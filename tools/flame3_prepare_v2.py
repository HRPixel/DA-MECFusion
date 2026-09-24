#!/usr/bin/env python3
"""Audit the FLAME3 CV subset before DA-MECFusion V2 experiments."""

from __future__ import annotations

import argparse
import csv
import hashlib
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image


LAYOUT = {
    "Fire": {
        "prefix": "F",
        "vis": Path("Fire/RGB/Corrected FOV"),
        "ir": Path("Fire/Thermal/Raw JPG"),
        "tiff": Path("Fire/Thermal/Celsius TIFF"),
    },
    "No Fire": {
        "prefix": "N",
        "vis": Path("No Fire/RGB/Corrected FOV"),
        "ir": Path("No Fire/Thermal/Raw JPG"),
        "tiff": Path("No Fire/Thermal/Celsius TIFF"),
    },
}
EXPECTED_SIZE = (640, 512)
INVENTORY_FIELDS = (
    "sample_id",
    "source_class",
    "original_stem",
    "sequence_id",
    "vis_path",
    "ir_path",
    "tiff_path",
    "vis_size",
    "ir_size",
    "tiff_size",
    "vis_mode",
    "ir_mode",
    "tiff_mode",
    "tiff_dtype",
    "tiff_min_c",
    "tiff_max_c",
    "tiff_nonfinite_pixels",
    "tiff_low_clip_pixels",
    "tiff_ge_500_pixels",
    "tiff_ge_600_pixels",
    "vis_datetime",
    "ir_datetime",
    "pair_time_delta_seconds",
    "gps_lat",
    "gps_lon",
    "gps_alt_m",
    "vis_sha256",
    "ir_sha256",
    "tiff_sha256",
    "status",
    "reason",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_map(folder: Path) -> tuple[dict[str, Path], list[str]]:
    if not folder.is_dir():
        return {}, [f"directory_missing:{folder}"]
    result: dict[str, Path] = {}
    errors: list[str] = []
    for path in sorted(folder.iterdir()):
        if not path.is_file():
            continue
        key = path.stem.casefold()
        if key in result:
            errors.append(f"duplicate_stem:{path.stem}")
        else:
            result[key] = path
    return result, errors


def parse_datetime(value: object) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.strptime(str(value), "%Y:%m:%d %H:%M:%S")
    except ValueError:
        return None


def dms_to_decimal(value: object) -> float:
    degrees, minutes, seconds = value  # type: ignore[misc]
    return float(degrees) + float(minutes) / 60.0 + float(seconds) / 3600.0


def read_jpg(path: Path) -> dict[str, object]:
    with Image.open(path) as image:
        image.load()
        exif = image.getexif()
        captured = parse_datetime(exif.get(36867) or exif.get(36868) or exif.get(306))
        gps_ifd = exif.get_ifd(34853) if exif.get(34853) is not None else {}
        lat = lon = alt = ""
        if gps_ifd.get(2) and gps_ifd.get(4):
            lat_value = dms_to_decimal(gps_ifd[2])
            lon_value = dms_to_decimal(gps_ifd[4])
            if gps_ifd.get(1) == "S":
                lat_value = -lat_value
            if gps_ifd.get(3) == "W":
                lon_value = -lon_value
            lat, lon = lat_value, lon_value
            if gps_ifd.get(6) is not None:
                alt_value = float(gps_ifd[6])
                alt = -alt_value if gps_ifd.get(5) == 1 else alt_value
        return {
            "size": f"{image.width}x{image.height}",
            "mode": image.mode,
            "datetime": captured,
            "lat": lat,
            "lon": lon,
            "alt": alt,
        }


def read_tiff(path: Path) -> dict[str, object]:
    with Image.open(path) as image:
        values = np.asarray(image)
        finite = np.isfinite(values)
        finite_values = values[finite]
        if finite_values.size == 0:
            minimum = maximum = ""
        else:
            minimum = float(finite_values.min())
            maximum = float(finite_values.max())
        return {
            "size": f"{image.width}x{image.height}",
            "mode": image.mode,
            "dtype": str(values.dtype),
            "min": minimum,
            "max": maximum,
            "nonfinite": int((~finite).sum()),
            "low_clip": int((values <= -40.5).sum()),
            "ge_500": int((values >= 500.0).sum()),
            "ge_600": int((values >= 600.0).sum()),
        }


def relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def blank_row(sample_id: str, source_class: str, stem: str) -> dict[str, object]:
    return {field: "" for field in INVENTORY_FIELDS} | {
        "sample_id": sample_id,
        "source_class": source_class,
        "original_stem": stem,
        "status": "pass",
    }


def write_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def assign_sequences(rows: list[dict[str, object]], gap_seconds: float) -> list[dict[str, object]]:
    valid = [row for row in rows if row["status"] == "pass" and row.get("vis_datetime")]
    valid.sort(key=lambda row: (str(row["vis_datetime"]), str(row["sample_id"])))
    if not valid:
        return []

    # ponytail: time gaps are the smallest auditable grouping rule; review boundaries before splitting.
    groups: list[list[dict[str, object]]] = [[valid[0]]]
    for row in valid[1:]:
        previous = datetime.fromisoformat(str(groups[-1][-1]["vis_datetime"]))
        current = datetime.fromisoformat(str(row["vis_datetime"]))
        if (current - previous).total_seconds() > gap_seconds:
            groups.append([row])
        else:
            groups[-1].append(row)

    summaries: list[dict[str, object]] = []
    for index, group in enumerate(groups, start=1):
        sequence_id = f"SEQ_{index:02d}"
        for row in group:
            row["sequence_id"] = sequence_id
        start = datetime.fromisoformat(str(group[0]["vis_datetime"]))
        end = datetime.fromisoformat(str(group[-1]["vis_datetime"]))
        classes = Counter(str(row["source_class"]) for row in group)
        summaries.append(
            {
                "sequence_id": sequence_id,
                "samples": len(group),
                "fire": classes["Fire"],
                "no_fire": classes["No Fire"],
                "start": start.isoformat(sep=" "),
                "end": end.isoformat(sep=" "),
                "duration_seconds": int((end - start).total_seconds()),
            }
        )
    return summaries


def audit(data_root: Path, output_dir: Path, sequence_gap_seconds: float) -> None:
    if not data_root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {data_root}")
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    structural_errors: list[dict[str, object]] = []
    hashes: defaultdict[str, list[str]] = defaultdict(list)

    for source_class, layout in LAYOUT.items():
        maps: dict[str, dict[str, Path]] = {}
        for modality in ("vis", "ir", "tiff"):
            mapping, errors = file_map(data_root / layout[modality])  # type: ignore[index]
            maps[modality] = mapping
            for error in errors:
                structural_errors.append(
                    {"sample_id": "", "source_class": source_class, "reason": f"{modality}:{error}"}
                )

        stems = sorted(set().union(*(set(mapping) for mapping in maps.values())))
        for stem in stems:
            sample_id = f"{layout['prefix']}_{stem.upper()}"
            row = blank_row(sample_id, source_class, stem)
            paths = {modality: maps[modality].get(stem) for modality in ("vis", "ir", "tiff")}
            missing = [modality for modality, path in paths.items() if path is None]
            if missing:
                row["status"] = "exclude"
                row["reason"] = "missing:" + ",".join(missing)
                rows.append(row)
                continue

            vis_path, ir_path, tiff_path = paths["vis"], paths["ir"], paths["tiff"]
            assert vis_path is not None and ir_path is not None and tiff_path is not None
            row.update(
                {
                    "vis_path": relative(vis_path, data_root),
                    "ir_path": relative(ir_path, data_root),
                    "tiff_path": relative(tiff_path, data_root),
                }
            )

            reasons: list[str] = []
            try:
                vis = read_jpg(vis_path)
                ir = read_jpg(ir_path)
                tiff = read_tiff(tiff_path)
                row.update(
                    {
                        "vis_size": vis["size"],
                        "ir_size": ir["size"],
                        "tiff_size": tiff["size"],
                        "vis_mode": vis["mode"],
                        "ir_mode": ir["mode"],
                        "tiff_mode": tiff["mode"],
                        "tiff_dtype": tiff["dtype"],
                        "tiff_min_c": tiff["min"],
                        "tiff_max_c": tiff["max"],
                        "tiff_nonfinite_pixels": tiff["nonfinite"],
                        "tiff_low_clip_pixels": tiff["low_clip"],
                        "tiff_ge_500_pixels": tiff["ge_500"],
                        "tiff_ge_600_pixels": tiff["ge_600"],
                        "vis_datetime": vis["datetime"].isoformat() if vis["datetime"] else "",
                        "ir_datetime": ir["datetime"].isoformat() if ir["datetime"] else "",
                        "gps_lat": vis["lat"],
                        "gps_lon": vis["lon"],
                        "gps_alt_m": vis["alt"],
                    }
                )
                if vis["datetime"] and ir["datetime"]:
                    row["pair_time_delta_seconds"] = abs(
                        (vis["datetime"] - ir["datetime"]).total_seconds()  # type: ignore[operator]
                    )
                if vis["size"] != "640x512" or ir["size"] != "640x512" or tiff["size"] != "640x512":
                    reasons.append("unexpected_size")
                if int(tiff["nonfinite"]) > 0:
                    reasons.append("tiff_nonfinite")
            except Exception as error:  # record the damaged sample instead of aborting the full audit
                reasons.append(f"read_error:{type(error).__name__}:{error}")

            for modality, path in (("vis", vis_path), ("ir", ir_path), ("tiff", tiff_path)):
                try:
                    digest = sha256_file(path)
                    row[f"{modality}_sha256"] = digest
                    hashes[digest].append(f"{sample_id}:{modality}:{relative(path, data_root)}")
                except OSError as error:
                    reasons.append(f"hash_error:{modality}:{error}")

            if reasons:
                row["status"] = "exclude"
                row["reason"] = ";".join(reasons)
            rows.append(row)

    sequences = assign_sequences(rows, sequence_gap_seconds)
    excluded = [
        {"sample_id": row["sample_id"], "source_class": row["source_class"], "reason": row["reason"]}
        for row in rows
        if row["status"] != "pass"
    ] + structural_errors
    duplicates = [
        {"sha256": digest, "occurrences": len(files), "files": ";".join(files)}
        for digest, files in sorted(hashes.items())
        if len(files) > 1
    ]

    write_csv(output_dir / "inventory.csv", INVENTORY_FIELDS, rows)
    write_csv(output_dir / "excluded.csv", ("sample_id", "source_class", "reason"), excluded)
    write_csv(
        output_dir / "duplicate_candidates.csv",
        ("sha256", "occurrences", "files"),
        duplicates,
    )
    write_csv(
        output_dir / "sequence_candidates.csv",
        ("sequence_id", "samples", "fire", "no_fire", "start", "end", "duration_seconds"),
        sequences,
    )

    passed = [row for row in rows if row["status"] == "pass"]
    class_counts = Counter(str(row["source_class"]) for row in passed)
    low_clip_images = sum(int(row["tiff_low_clip_pixels"] or 0) > 0 for row in passed)
    ge500_images = sum(int(row["tiff_ge_500_pixels"] or 0) > 0 for row in passed)
    ge600_images = sum(int(row["tiff_ge_600_pixels"] or 0) > 0 for row in passed)
    pseudo_labels = list((data_root / "Fire/Label/train").glob("*.png"))

    summary = f"""# FLAME3 V2 data audit summary

- Dataset root: `{data_root}`
- Audit script SHA-256: `{sha256_file(Path(__file__))}`
- Sequence gap candidate: `{sequence_gap_seconds:g}` seconds
- Valid paired samples: `{len(passed)}`
- Fire: `{class_counts['Fire']}`
- No Fire: `{class_counts['No Fire']}`
- Excluded/structural errors: `{len(excluded)}`
- Exact duplicate hash groups: `{len(duplicates)}`
- Candidate flight sequences: `{len(sequences)}`
- TIFF images containing values <= -40.5 C: `{low_clip_images}`
- TIFF images containing values >= 500 C: `{ge500_images}`
- TIFF images containing values >= 600 C: `{ge600_images}`
- Fire pseudo-label PNG files: `{len(pseudo_labels)}` (not treated as ground truth)

## Decision

`excluded.csv` and `duplicate_candidates.csv` must be reviewed before split generation.
Sequence candidates are provisional and must be checked at their time boundaries.
"""
    (output_dir / "audit_summary.md").write_text(summary, encoding="utf-8")

    evidence_files = sorted(
        path for path in output_dir.iterdir() if path.is_file() and path.name != "output_hashes.sha256"
    )
    hash_lines = [f"{sha256_file(path)}  {path.name}" for path in evidence_files]
    (output_dir / "output_hashes.sha256").write_text("\n".join(hash_lines) + "\n", encoding="utf-8")

    print(summary)
    print(f"Audit evidence written to: {output_dir}")


def self_check() -> None:
    assert parse_datetime("2022:10:25 16:53:40") == datetime(2022, 10, 25, 16, 53, 40)
    assert parse_datetime("bad") is None
    assert abs(dms_to_decimal((42, 30, 0)) - 42.5) < 1e-9
    print("FLAME3 audit self-check passed.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    audit_parser = subparsers.add_parser("audit", help="Audit paired FLAME3 Fire/No Fire data")
    audit_parser.add_argument("--data_root", type=Path, required=True)
    audit_parser.add_argument("--output_dir", type=Path, required=True)
    audit_parser.add_argument("--sequence_gap_seconds", type=float, default=30.0)
    subparsers.add_parser("self-check", help="Run the small built-in logic check")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "self-check":
        self_check()
    else:
        audit(args.data_root, args.output_dir, args.sequence_gap_seconds)


if __name__ == "__main__":
    main()
