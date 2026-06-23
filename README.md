# DA-MECFusion v1-minimal

DA-MECFusion v1-minimal is a minimal research version for degradation-aware
infrared-visible image fusion on the MSRS dataset.

This version keeps one complete dataset pipeline only: MSRS. It includes
training, single-pair inference, batch testing, metric evaluation, and the
necessary visualizations for fused images, rule-based priors, MEC weights, and
comparison figures.

## Scope

The V1-minimal forward pipeline is:

```text
IR/VIS input
-> rule-based priors
-> CNN encoders
-> MEC spatial dynamic weights
-> fusion decoder
-> fused output
```

Tensor layout is `[B, C, H, W]`, and image tensors are expected in `[0, 1]`.

## Included

- `datasets/msrs_dataset.py`: MSRS dataset reader.
- `datasets/transforms.py`: deterministic image preprocessing.
- `datasets/sample_index.py`: CSV manifest utilities.
- `models/priors.py`: thermal saliency, visible saturation uncertainty, and
  smoke/low-contrast priors.
- `models/encoders.py`: same-resolution CNN encoders.
- `models/mec_module.py`: MEC spatial weighting module.
- `models/decoder.py`: single-channel fusion decoder.
- `models/damecfusion.py`: assembled DA-MECFusion V1 model.
- `models/losses.py`: V1 fusion loss.
- `metrics/fusion_metrics.py`: fusion metrics used by `eval.py`.
- `utils.py`: shared runtime utilities for run directories, image saving,
  visualizations, and logs.

## Not Included In V1-Minimal

This version does not connect TNO, M3FD, FLAME3, external baselines, or
ablation experiments. The `data/FLAME3/` folder is only reserved for future data
extension work and is not loaded by the V1-minimal training, testing, inference,
or evaluation flow.

## Expected Data Layout

```text
data/
├── MSRS/
│   ├── train/
│   │   ├── ir/
│   │   ├── vis/
│   │   └── label/
│   ├── test/
│   │   ├── ir/
│   │   ├── vis/
│   │   └── label/
│   └── splits/
│       ├── train.csv
│       └── test.csv
└── FLAME3/
    └── README.md
```

CSV manifests use this format:

```csv
name,ir,vis,label
00001D,train/ir/00001D.png,train/vis/00001D.png,train/label/00001D.png
```

## Run Directory Layout

`train.py`, `infer.py`, `test.py`, and `eval.py` create timestamped run
directories under `runs/`:

```text
runs/<run_name>_<YYYYMMDD_HHMMSS>/
├── checkpoints/
├── outputs/
│   ├── fused/
│   ├── priors/
│   │   ├── thermal/
│   │   ├── saturation/
│   │   └── smoke/
│   ├── weights/
│   │   ├── w_ir/
│   │   └── w_vis/
│   └── comparisons/
│       ├── priors/
│       └── weights/
├── logs/
└── metrics/
```

## Example Commands

Train:

```bash
python train.py --data_root data/MSRS --run_root runs --run_name DA-MECFusion_v1_MSRS
```

Single-pair inference:

```bash
python infer.py --ir_path data/MSRS/test/ir/00001.png --vis_path data/MSRS/test/vis/00001.png --checkpoint runs/DA-MECFusion_v1_MSRS_YYYYMMDD_HHMMSS/checkpoints/best.pth
```

Batch test:

```bash
python test.py --data_root data/MSRS --checkpoint runs/DA-MECFusion_v1_MSRS_YYYYMMDD_HHMMSS/checkpoints/best.pth
```

Evaluate:

```bash
python eval.py --manifest data/MSRS/splits/test.csv --fused_dir runs/test_MSRS_YYYYMMDD_HHMMSS/outputs/fused
```
