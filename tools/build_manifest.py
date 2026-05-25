"""Build split manifests for processed paired-image datasets.

This script does not move, rename, copy, or edit image files. It only scans the
existing processed layout and writes CSV manifests under data_root/splits/.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.sample_index import build_split_manifest, resolve_manifest_path, write_manifest  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build dataset split CSV manifests.")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "test"])
    parser.add_argument("--ir_subdir", type=str, default="ir")
    parser.add_argument("--vis_subdir", type=str, default="vis")
    parser.add_argument("--label_subdir", type=str, default="label")
    parser.add_argument("--recursive", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    for split in args.splits:
        records, missing_ir, missing_vis, missing_label = build_split_manifest(
            data_root=data_root,
            split=split,
            ir_subdir=args.ir_subdir,
            vis_subdir=args.vis_subdir,
            label_subdir=args.label_subdir,
            recursive=args.recursive,
        )
        manifest_path = resolve_manifest_path(data_root, split)
        write_manifest(records, manifest_path, data_root)

        print(f"[{split}] manifest: {manifest_path}")
        print(f"[{split}] paired samples: {len(records)}")
        print(f"[{split}] missing IR: {len(missing_ir)}")
        print(f"[{split}] missing VIS: {len(missing_vis)}")
        print(f"[{split}] missing labels: {len(missing_label)}")
        if missing_ir:
            print(f"[{split}] first missing IR names: {missing_ir[:10]}")
        if missing_vis:
            print(f"[{split}] first missing VIS names: {missing_vis[:10]}")


if __name__ == "__main__":
    main()
