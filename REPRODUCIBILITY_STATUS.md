# Reproducibility status

Status on 2026-11-10: **preparation only; not verified against manuscript results**.

## Checks completed

- The source project contains 901 files, 8,479,590,348 bytes (about 7.90 GiB), excluding no files from the inventory.
- All 43 Python files and 15 code cells from five notebooks passed a syntax-only check in the preparation environment. Syntax success does not establish import or runtime compatibility.
- Text pattern scanning found no matches for the credential patterns used. This does not cover binary model files, all secret formats, or guarantee the absence of sensitive material.
- A small random-input check of `CanAttention` returned tensors of shape `[2, 512]`, `[2, 2048]`, and `[2, 5]`.
- The point model could not initialize in the available `torch 2.8.0+cpu` test environment because its constructor explicitly moves a tensor to CUDA.
- Importing the image branch was blocked by missing `torchvision` in that test environment. A successful image-branch forward pass is not claimed.

## Required work before a formal reproduction release

1. **Recover the actual experiment provenance.** `percentage_train.py` is the best available preparation base, but its use for the reported main result is not established. Keep this distinction in release notes.
2. **Restore and identify partitions.** The selected driver needs train, validation, and test folders. No original ID manifests or local metadata file were supplied in the data-free copy. The available ratio-experiment logs show a different training-pool organization from an undivided 2105-tree train split; do not silently replace the paper's partitions.
3. **Check the point token registration.** `model/mlp_former.py:249` uses `nn.Parameter(...).to('cuda')`. A device-copy operation may return an ordinary Tensor instead of a registered Parameter. In the actual CUDA environment check `dict(model.named_parameters())`, `state_dict()`, optimizer membership, gradients, and checkpoint compatibility. This has not been runtime-confirmed for the original experiment. Changing registration can change training and must be reported, not silently called the original implementation.
4. **Resolve device handling.** `model/mlp_former.py:59` and `:249` use explicit CUDA allocations. Multi-device and CPU execution are unverified.
5. **Fix or remove the advertised resume interface in a documented revision.** `percentage_train.py:472` uses the local `logger` before assignment. The code that restores model/optimizer/scheduler state and start epoch in `main()` is commented out. Reading a checkpoint header is not resuming training. Do not use or advertise this option meanwhile.
6. **Separate validation selection from test reporting.** The driver writes both best-validation and best-test checkpoints and reports the maximum test score over epochs. Establish which checkpoint and evaluation protocol produced the manuscript result. The maximum over repeatedly examined test scores does not establish a held-out final estimate. Do not retroactively relabel it or edit the manuscript's result from this code audit.
7. **Record sampling and small-batch behavior.** The evaluation dataset's RNG state advances on each visit. A seed does not freeze a tree's sampled points across epochs. Check reproducibility of evaluation and any final batch of size one with the model's BatchNorm layers; do not silently change sampling, batch dropping, or scientific results.
8. **Recover the original software environment.** Direct dependency names are supplied, but original versions, CUDA libraries and hardware are unknown. A working current environment cannot by itself prove original-result reproduction.
9. **Complete source attribution and license selection.** Preserve upstream licenses and identify any additional sources. An authors' project license must not overwrite upstream license obligations.
10. **Perform real-data checks.** After restoring the original data, run a dataset-loading check, one training and evaluation batch, checkpoint save/load comparison, and the required full experiment. Record commands, split IDs, versions, checkpoint identity, and results.

## Why the old files were excluded

- `train.py:61–62` passes `seg_root` and `balanced`, which the active dataset constructor does not accept.
- `train.py:157` calls nonexistent `shuffle_indices()`.
- `train.py:176` and `:270` expect five sliced/batched results, while the active dataset indexes one item and returns four results (`dataset.py:133–166`).
- `train.py:199` and `:286` expect two outputs from the fusion model, which returns three (`model/attention_fusion.py:249`).
- `train.py:138` supplies an image-summary input with five dimensions after batching, while `SimpleView.forward` unpacks a four-dimensional tensor.
- `mutimodal.ipynb` contains duplicate alternative training cells, incompatible dataset calls, and inconsistent two/three-output fusion unpacking.
- `dataset.ipynb` imports missing `model.render_point` and passes an unsupported `balanced` argument.
- The comparison-model and visualization files are outside the scope of this small three-branch source candidate. Their absence means the repository cannot be described as containing every manuscript experiment.

All research source files included here are byte-for-byte copies. No algorithm, metric, training strategy, partition, model weight, or paper text was changed during preparation.
