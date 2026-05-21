# Developer Notes

## Environment Dependencies

1. The current `requirements.txt` is a basic dependency checklist.
2. For offline workstation deployment, pin concrete versions according to the
   target Python version, CUDA version, and available PyTorch wheelhouse.
3. For model forward tests only, the minimal dependencies are `torch`, `numpy`,
   `Pillow`, and `matplotlib`.
4. For training and evaluation, `tqdm`, `opencv-python`, `scipy`, and `pandas`
   are also needed.
