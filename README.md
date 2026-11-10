# CMSCL-Net

Core implementation for single-tree species classification from terrestrial laser scanning (TLS) point clouds. The model combines point-cloud, depth-image and morphological features.

## Code

Entry point: `percentage_train.py`. Data loading and projection are implemented in `dataset.py` and `util.py`; network modules are in `model/`.

The eight Python source files are unchanged copies of the supplied research project. Their SHA-256 checksums are listed in `SOURCE_MANIFEST.json`. This repository contains the core model, without the full baseline and figure-generation workflows.

## Data

Public dataset: [Owen, Flynn and Lines, Zenodo, 2022](https://doi.org/10.5281/zenodo.6962717). Related paper: [Owen et al., 2021](https://doi.org/10.1111/1365-2745.13670).

Download the tree clouds and `tree-meta.csv` from Zenodo. Data and trained weights are not included here. The driver expects:

```text
data/
  meta/tree-meta.csv
  train/<tree-id>.txt
  val/<tree-id>.txt
  test/<tree-id>.txt
```

Tree files must contain numeric XYZ columns and have basenames matching the metadata `id` field. Keep only tree TXT files in the split directories. The `sp` label mapping is PINNIG=0, PINPIN=1, PINSYL=2, QUEFAG=3 and QUEILE=4. Use the original experimental partitions when available; downloading the source dataset alone does not recover those partitions.

## Dependencies and driver usage

The point branch requires CUDA. Install mutually compatible PyTorch and TorchVision builds; `requirements.in` lists the direct dependencies, without a verified version lock.

The driver accepts:

```bash
python percentage_train.py --train_ratios 1.0 --epoch 1600 --supcon_epochs 800 --batchsize 32 --log_dir logs/full_pool
```

`1.0` selects the entire existing training partition. This is a driver-usage example, not a verified command for reproducing the paper's reported accuracy.

## Verification status

Source checksums and Python syntax have been checked. End-to-end reproduction has not been verified. The original partition manifests and exact experimental environment are not included. The current source has the following limitations:

- `--resume` does not restore a complete training run; do not use it.
- CUDA token parameter registration and checkpoint compatibility require verification.
- Evaluation resamples points on repeated visits, and the driver saves a checkpoint selected by the maximum test accuracy across epochs. Check the evaluation protocol before interpreting this as a held-out final result.

## Attribution

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and `licenses/` for upstream attributions and license texts. No project-wide license has been selected for the authors' original contributions.
