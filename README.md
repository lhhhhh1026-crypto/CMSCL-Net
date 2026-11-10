# CMSCL-Net

**Publication preparation candidate. This is not a verified reproduction release.**

This candidate contains a small source snapshot for the CMSCL-Net three-branch model: point-cloud features, depth-image features, and morphological features. The research source files are copied without modification from the supplied project. No dataset, trained checkpoint, experiment log, or manuscript is included.

## Status and selected entry point

`percentage_train.py` is the proposed preparation entry point because it uses the current dataset's four-item DataLoader interface and the fusion module's three-item return interface. It also contains command-line options, validation, training, and checkpoint functions.

This selection is an engineering assessment. It does **not** establish that this script produced the manuscript's reported results. The old `train.py` and the experimental notebooks are excluded from this small candidate because their interfaces disagree with the current `dataset.py` and fusion model. All originals remain in the supplied project.

Read [REPRODUCIBILITY_STATUS.md](REPRODUCIBILITY_STATUS.md) before attempting training or describing this repository in a paper. Important open items include the original data partitions, experiment provenance, CUDA device handling, dependency versions, and a broken resume path.

## Files

| File | Role |
| --- | --- |
| `percentage_train.py` | Proposed training and evaluation entry point; currently a preparation candidate |
| `dataset.py` | Tree-cloud loading, label encoding, sampling, augmentation, and feature preparation |
| `util.py` | Projection, depth-image and other utility functions |
| `model/mlp_former.py` | Point-cloud feature branch |
| `model/simpleview.py` | Depth-image feature branch wrapper |
| `model/resnet.py`, `model/resnet50_101.py` | Image branch support code |
| `model/attention_fusion.py` | Morphological branch, feature fusion, projection and classifier |

This snapshot does not include the comparison-model suite, full figure-generation workflow, or all experiments in the manuscript.

## Dataset

The public source dataset is **Individual TLS tree clouds collected from both Alto Tajo and Cuellar in Spain**, by Owen, Flynn, and Lines, Zenodo, 2022:

https://doi.org/10.5281/zenodo.6962717

The associated Owen et al. research paper was published in 2021:

https://doi.org/10.1111/1365-2745.13670

Download the point clouds and `tree-meta.csv` from the original repository. Follow the dataset's own reuse terms and cite its creators. Large source data are intentionally not duplicated here. **The download does not supply or establish the exact partitions used for the CMSCL-Net manuscript.**

The selected driver currently expects this local layout:

```text
data/
  meta/
    tree-meta.csv
  train/
    <tree-id>.txt
  val/
    <tree-id>.txt
  test/
    <tree-id>.txt
```

Each tree file must contain numeric XYZ columns, with a basename matching `id` in the metadata. The active loader reads the first three columns, recentres points, and normalizes them. Only tree TXT files should be put inside the three split directories because the current loader enumerates directory entries directly.

The current label mapping is:

| Metadata code | Label |
| --- | --- |
| PINNIG | 0 |
| PINPIN | 1 |
| PINSYL | 2 |
| QUEFAG | 3 |
| QUEILE | 4 |

Restore the original partition manifests and document how the validation set relates to the manuscript's train/test counts. Do not invent a new random split and call it the original split. Store small partition ID lists in a separate tracked directory such as `splits/`; large clouds can remain outside GitHub.

## Environment

The original experiment's Python/PyTorch/CUDA and package versions have not been recovered. `requirements.in` lists direct packages observed in this candidate and is **not** a tested lock file. Install compatible PyTorch and TorchVision for the target CUDA environment, then install the remaining listed packages in an isolated environment. Consult https://pytorch.org/get-started/locally/ for the matching installer.

The existing point branch explicitly allocates on CUDA, so its advertised CPU fallback is not functional. The preparation check used a separate CPU-only Python environment and could not validate the point branch. No full training, real-data inference, or reproduction of the reported accuracy has been performed.

After the known issues are resolved and the original partitions and environment are restored, the existing parser supports this full-training-pool invocation:

```bash
python percentage_train.py --train_ratios 1.0 --epoch 1600 --supcon_epochs 800 --batchsize 32 --log_dir logs/full_pool
```

This is a parser-compatible example, **not a verified manuscript reproduction command**. `1.0` selects all files already assigned to `data/train/`, not all trees in the source dataset. The seed, partitioning, scheduler and evaluation protocol must be checked against the actual experiment records. Do not use `--resume` until its documented defects have been corrected and verified.

## License and attribution

No project-wide license has been selected for the authors' original contributions. Do not describe this candidate as a fully licensed open-source release yet. Preserve all existing author/copyright headers.

[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) records the visible upstream attributions. Copies of the DGCNN MIT and TorchVision BSD-3-Clause license texts are included under `licenses/`. Their inclusion does not establish the full provenance of every local modification. Remaining attribution and project-license decisions must be completed before a formal release.

## Citation and release

The manuscript is being prepared for submission. No publication DOI, final software version, or archival DOI is asserted here. After the repository has been corrected and validated, publish a versioned release, record the commit and environment, and consider archiving that release with Zenodo. A GitHub source URL and a code archive DOI are different identifiers.
