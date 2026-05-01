# IRIS-SurgWMBench

This repository adapts the official IRIS codebase, **Transformers are Sample-Efficient World Models**, toward a PyTorch 2.x baseline for:

**SurgWMBench: A Dataset and World-Model Benchmark for Surgical Instrument Motion Planning**

The original IRIS implementation remains under `src/` for reference. New SurgWMBench-specific code lives in the isolated `iris_surgwmbench/` package so the upstream Atari-oriented code stays readable.

## Current Status

Implemented first-pass data foundation:

- final-layout `SurgWMBenchClipDataset`
- `SurgWMBenchFrameDataset` for tokenizer pretraining frames
- `SurgWMBenchRawVideoDataset` with OpenCV and extracted-frame fallback
- sparse, dense, window, and frame-tokenizer collators
- sparse/dense trajectory metrics
- toy SurgWMBench generator
- read-only dataset validator
- pytest coverage for loaders, collators, metrics, and validator smoke tests

Not implemented yet:

- discrete visual tokenizer training for SurgWMBench
- action tokenizer module
- autoregressive Transformer world model adaptation
- coordinate head, policy/action prior, rollout API
- model training and evaluation commands

## Repository Layout

```text
src/                         Original IRIS codebase
config/                      Original Hydra configs
iris_surgwmbench/data/       SurgWMBench loaders, transforms, collators
iris_surgwmbench/evaluation/ Metrics
tools/                       Toy dataset and validation commands
tests/                       Pytest coverage for SurgWMBench data foundation
assets/                      Original IRIS media
results/                     Original IRIS result artifacts
```

## Dataset Location

The local canonical dataset root used in this workspace is:

```text
/mnt/hdd1/neurips2026_dataset_track/SurgWMBench
```

The loader follows `SurgWMBench/README.md` as the dataset contract. It uses official manifests only:

```text
manifests/train.jsonl
manifests/val.jsonl
manifests/test.jsonl
manifests/all.jsonl
```

Do not create random train/val/test splits. Sparse 20-anchor human labels are the primary benchmark target. Dense interpolation coordinates are auxiliary pseudo labels and are kept separate from human labels.

## Setup

Install PyTorch 2.x for your CUDA/CPU environment first, then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

If you need a specific CUDA wheel, install PyTorch from the official selector before running the command above.

## Validate the Real Dataset

Run a read-only sanity check on local SurgWMBench files:

```bash
python -m tools.validate_surgwmbench_loader \
  --dataset-root /mnt/hdd1/neurips2026_dataset_track/SurgWMBench \
  --manifest manifests/train.jsonl \
  --interpolation-method linear \
  --check-files \
  --num-samples 32
```

Expected output:

```text
SurgWMBench validation passed.
```

## DataLoader Smoke Test

Sparse human-anchor batch:

```bash
python - <<'PY'
from torch.utils.data import DataLoader
from iris_surgwmbench.data import SurgWMBenchClipDataset, collate_sparse_anchors

root = "/mnt/hdd1/neurips2026_dataset_track/SurgWMBench"
dataset = SurgWMBenchClipDataset(root, "manifests/train.jsonl", image_size=64, frame_sampling="sparse_anchors")
batch = next(iter(DataLoader(dataset, batch_size=4, num_workers=2, collate_fn=collate_sparse_anchors)))
print(batch["frames"].shape, batch["coords_norm"].shape, batch["actions_delta_dt"].shape)
PY
```

Dense pseudo-coordinate batch:

```bash
python - <<'PY'
from torch.utils.data import DataLoader
from iris_surgwmbench.data import SurgWMBenchClipDataset, collate_dense_variable_length

root = "/mnt/hdd1/neurips2026_dataset_track/SurgWMBench"
dataset = SurgWMBenchClipDataset(root, "manifests/train.jsonl", image_size=32, frame_sampling="dense")
batch = next(iter(DataLoader(dataset, batch_size=2, num_workers=2, collate_fn=collate_dense_variable_length)))
print(batch["frames"].shape, batch["frame_mask"].sum(dim=1).tolist(), batch["actions_delta_dt"].shape)
PY
```

## Toy Dataset

Create a synthetic dataset for CPU-only tests:

```bash
python -m tools.make_toy_surgwmbench --output-root /tmp/SurgWMBench_toy --num-clips 2
```

Validate it:

```bash
python -m tools.validate_surgwmbench_loader \
  --dataset-root /tmp/SurgWMBench_toy \
  --manifest manifests/train.jsonl \
  --interpolation-method linear \
  --check-files
```

## Tests

Run the full local test suite:

```bash
pytest
```

Current expected result:

```text
21 passed
```

## Original IRIS

IRIS is a data-efficient world-model agent composed of:

1. a discrete autoencoder/tokenizer,
2. an autoregressive Transformer world model,
3. an agent trained in imagined rollouts.

The original Atari training entrypoint is still available:

```bash
python src/main.py env.train.id=BreakoutNoFrameskip-v4 wandb.mode=disabled
```

This path is legacy and depends on Atari/gym tooling. The SurgWMBench adaptation should use the new `iris_surgwmbench/` package.

## Citation

If you use the original IRIS method, cite:

```bibtex
@inproceedings{
  iris2023,
  title={Transformers are Sample-Efficient World Models},
  author={Vincent Micheli and Eloi Alonso and François Fleuret},
  booktitle={The Eleventh International Conference on Learning Representations},
  year={2023},
  url={https://openreview.net/forum?id=vhFu1Acb0xb}
}
```
