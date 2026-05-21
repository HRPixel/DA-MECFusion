# DA-MECFusion

PyTorch implementation workspace for DA-MECFusion V1, a minimal infrared-visible
image fusion pipeline for complex degradation conditions.

## V1 Scope

The current V1 target is the smallest runnable forward loop:

```text
IR/VIS input
-> rule-based priors
-> CNN encoders
-> MEC spatial dynamic weights
-> fusion decoder
-> fused output
```

V1 intentionally avoids Transformer modules, task-specific losses, and extra
dataset expansion. Tensor layout is `[B, C, H, W]`, and image tensors are
expected in `[0, 1]`.

## Implemented So Far

- `models/priors.py`: thermal saliency, visible saturation uncertainty, and
  smoke/low-contrast priors.
- `models/mec_module.py`: MEC spatial weighting module.
- `models/encoders.py`: simple same-resolution CNN encoder.
- `models/decoder.py`: single-channel fusion decoder.
- `models/damecfusion.py`: assembled DA-MECFusion V1 forward model.
- `tests/`: focused forward/unit test scripts for the modules above.

## Example Checks

```bash
python models/damecfusion.py
python tests/test_damecfusion_forward.py --device cuda
```
