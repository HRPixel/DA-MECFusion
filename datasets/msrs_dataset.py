"""MSRS dataset reader for DA-MECFusion V1."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import warnings

from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.transforms import pil_to_tensor, preprocess_pair  # noqa: E402


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _scan_images(directory: Path) -> dict[str, Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Image directory not found: {directory}")

    images: dict[str, Path] = {}
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        stem = path.stem
        if stem in images:
            raise ValueError(f"Duplicate image stem '{stem}' in {directory}")
        images[stem] = path
    return images


class MSRSDataset(Dataset):
    """Aligned MSRS infrared-visible dataset."""

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        height: int | None = None,
        width: int | None = None,
        use_label: bool = False,
    ) -> None:
        super().__init__()
        self.data_root = Path(data_root)
        self.split = split
        self.height = height
        self.width = width
        self.use_label = use_label

        if (height is None) != (width is None):
            raise ValueError("height and width must be provided together")

        split_root = self.data_root / split
        self.ir_dir = split_root / "ir"
        self.vis_dir = split_root / "vis"
        self.label_dir = split_root / "label"

        ir_files = _scan_images(self.ir_dir)
        vis_files = _scan_images(self.vis_dir)

        ir_stems = set(ir_files)
        vis_stems = set(vis_files)
        missing_vis = sorted(ir_stems - vis_stems)
        missing_ir = sorted(vis_stems - ir_stems)
        if missing_vis or missing_ir:
            message_parts = []
            if missing_vis:
                message_parts.append(f"IR files without matching VIS: {missing_vis[:10]}")
            if missing_ir:
                message_parts.append(f"VIS files without matching IR: {missing_ir[:10]}")
            raise ValueError("MSRS filename pairing failed. " + " ".join(message_parts))

        self.samples = [
            {
                "name": stem,
                "ir_path": ir_files[stem],
                "vis_path": vis_files[stem],
            }
            for stem in sorted(ir_stems)
        ]

        self.label_files: dict[str, Path] = {}
        if self.use_label:
            if self.label_dir.is_dir():
                self.label_files = _scan_images(self.label_dir)
                missing_labels = [sample["name"] for sample in self.samples if sample["name"] not in self.label_files]
                if missing_labels:
                    warnings.warn(
                        f"{len(missing_labels)} sample(s) do not have matching labels. "
                        "Those samples will return no label.",
                        stacklevel=2,
                    )
            else:
                warnings.warn(
                    f"Label directory not found: {self.label_dir}. Labels will be omitted.",
                    stacklevel=2,
                )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        sample = self.samples[index]
        ir, vis, vis_y = preprocess_pair(
            sample["ir_path"],
            sample["vis_path"],
            height=self.height,
            width=self.width,
        )

        item: dict[str, torch.Tensor | str] = {
            "ir": ir,
            "vis": vis,
            "vis_y": vis_y,
            "name": sample["name"],
        }

        if self.use_label and sample["name"] in self.label_files:
            label_img = Image.open(self.label_files[sample["name"]]).convert("L")
            if self.height is not None and self.width is not None:
                label_img = label_img.resize((self.width, self.height), resample=Image.NEAREST)
            item["label"] = pil_to_tensor(label_img)

        if ir.dim() != 3 or ir.size(0) != 1:
            raise ValueError(f"Invalid ir tensor shape for {sample['name']}: {tuple(ir.shape)}")
        if vis.dim() != 3 or vis.size(0) != 3:
            raise ValueError(f"Invalid vis tensor shape for {sample['name']}: {tuple(vis.shape)}")
        if vis_y.dim() != 3 or vis_y.size(0) != 1:
            raise ValueError(f"Invalid vis_y tensor shape for {sample['name']}: {tuple(vis_y.shape)}")

        return item


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test MSRSDataset.")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--use_label", action="store_true")
    return parser.parse_args()


def _print_sample_shapes(sample: dict[str, torch.Tensor | str]) -> None:
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            print(f"{key}: shape={tuple(value.shape)}, min={value.min().item():.6f}, max={value.max().item():.6f}")
        else:
            print(f"{key}: {value}")


if __name__ == "__main__":
    args = _parse_args()
    dataset = MSRSDataset(
        data_root=args.data_root,
        split=args.split,
        height=args.height,
        width=args.width,
        use_label=args.use_label,
    )
    print(f"dataset length: {len(dataset)}")

    if len(dataset) == 0:
        raise RuntimeError("Dataset is empty")

    first_sample = dataset[0]
    print("first sample:")
    _print_sample_shapes(first_sample)

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    batch = next(iter(loader))
    print("batch:")
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            print(f"{key}: shape={tuple(value.shape)}")
        else:
            print(f"{key}: {value}")
