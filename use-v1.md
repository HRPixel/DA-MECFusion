# DA-MECFusion v1-minimal usage

Current v1-minimal only supports the MSRS pipeline.

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
