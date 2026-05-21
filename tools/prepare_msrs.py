# tools/prepare_msrs.py

'''
python tools/prepare_msrs.py \
  --src data/raw/MSRS \
  --dst data/processed/MSRS \
  --overwrite
'''

import argparse
import shutil
from pathlib import Path


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def list_images(folder: Path):
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")

    files = [
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMG_EXTS
    ]

    return sorted(files, key=lambda x: x.stem)


def clear_and_make_dir(folder: Path, overwrite: bool = False):
    if folder.exists() and overwrite:
        shutil.rmtree(folder)
    folder.mkdir(parents=True, exist_ok=True)


def copy_split(src_split: Path, dst_split: Path, overwrite: bool = False):
    src_ir = src_split / "ir"
    src_vis = src_split / "vi"
    src_label = src_split / "Segmentation_labels"

    dst_ir = dst_split / "ir"
    dst_vis = dst_split / "vis"
    dst_label = dst_split / "labels"

    clear_and_make_dir(dst_ir, overwrite=overwrite)
    clear_and_make_dir(dst_vis, overwrite=overwrite)
    clear_and_make_dir(dst_label, overwrite=overwrite)

    ir_files = list_images(src_ir)
    vis_files = list_images(src_vis)

    ir_map = {p.stem: p for p in ir_files}
    vis_map = {p.stem: p for p in vis_files}

    common_names = sorted(set(ir_map.keys()) & set(vis_map.keys()))

    missing_ir = sorted(set(vis_map.keys()) - set(ir_map.keys()))
    missing_vis = sorted(set(ir_map.keys()) - set(vis_map.keys()))

    print(f"\n[{src_split.name}]")
    print(f"IR images: {len(ir_files)}")
    print(f"VIS images: {len(vis_files)}")
    print(f"Paired images: {len(common_names)}")

    if missing_ir:
        print(f"[Warning] Missing IR for {len(missing_ir)} VIS images.")
    if missing_vis:
        print(f"[Warning] Missing VIS for {len(missing_vis)} IR images.")

    if len(common_names) == 0:
        raise RuntimeError(f"No paired images found in {src_split}")

    for name in common_names:
        ir_src = ir_map[name]
        vis_src = vis_map[name]

        ir_dst = dst_ir / f"{name}{ir_src.suffix.lower()}"
        vis_dst = dst_vis / f"{name}{vis_src.suffix.lower()}"

        shutil.copy2(ir_src, ir_dst)
        shutil.copy2(vis_src, vis_dst)

    copied_labels = 0

    if src_label.exists():
        label_files = list_images(src_label)
        label_map = {p.stem: p for p in label_files}

        for name in common_names:
            if name in label_map:
                label_src = label_map[name]
                label_dst = dst_label / f"{name}{label_src.suffix.lower()}"
                shutil.copy2(label_src, label_dst)
                copied_labels += 1

        print(f"Labels copied: {copied_labels}")
    else:
        print("No Segmentation_labels folder found.")

    return {
        "split": src_split.name,
        "ir": len(ir_files),
        "vis": len(vis_files),
        "pairs": len(common_names),
        "labels": copied_labels,
        "missing_ir": len(missing_ir),
        "missing_vis": len(missing_vis),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Prepare MSRS dataset for DA-MECFusion."
    )

    parser.add_argument(
        "--src",
        type=str,
        default="data/raw/MSRS",
        help="Path to raw MSRS dataset."
    )

    parser.add_argument(
        "--dst",
        type=str,
        default="data/processed/MSRS",
        help="Path to processed MSRS dataset."
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing processed folders."
    )

    args = parser.parse_args()

    src_root = Path(args.src)
    dst_root = Path(args.dst)

    if not src_root.exists():
        raise FileNotFoundError(f"Raw MSRS path not found: {src_root}")

    all_stats = []

    for split in ["train", "test"]:
        src_split = src_root / split
        dst_split = dst_root / split

        if not src_split.exists():
            raise FileNotFoundError(f"Split folder not found: {src_split}")

        stats = copy_split(
            src_split=src_split,
            dst_split=dst_split,
            overwrite=args.overwrite
        )
        all_stats.append(stats)

    print("\nPreparation finished.")
    print(f"Processed dataset saved to: {dst_root}")

    total_pairs = sum(item["pairs"] for item in all_stats)
    total_labels = sum(item["labels"] for item in all_stats)

    print(f"Total paired images: {total_pairs}")
    print(f"Total labels copied: {total_labels}")


if __name__ == "__main__":
    main()