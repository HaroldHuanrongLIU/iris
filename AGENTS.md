# Repository Guidelines

## Project Structure & Module Organization

This repository is the official IRIS codebase with an added SurgWMBench extension. Keep upstream IRIS files readable and avoid changing them unless required.

- `src/`: original IRIS implementation, including tokenizer, Transformer world model, Atari envs, agent, and trainer.
- `config/`: Hydra configs for the original IRIS training stack.
- `iris_surgwmbench/`: PyTorch 2.x SurgWMBench extension package. Current scope is data loading, collators, raw-video access, and metrics.
- `tools/`: local command modules such as `make_toy_surgwmbench` and `validate_surgwmbench_loader`.
- `tests/`: pytest coverage for SurgWMBench loaders, collators, metrics, and validator smoke tests.
- `assets/` and `results/`: original IRIS media and result artifacts.

## Build, Test, and Development Commands

Use Python from the active local environment.

```bash
pytest
```
Runs the local SurgWMBench test suite.

```bash
python -m tools.validate_surgwmbench_loader \
  --dataset-root /mnt/hdd1/neurips2026_dataset_track/SurgWMBench \
  --manifest manifests/train.jsonl \
  --interpolation-method linear \
  --check-files --num-samples 32
```
Validates real dataset files without modifying annotations.

```bash
python -m tools.make_toy_surgwmbench --output-root /tmp/SurgWMBench_toy
```
Creates a synthetic final-layout dataset for smoke testing.

Original IRIS training still uses:

```bash
python src/main.py env.train.id=BreakoutNoFrameskip-v4 wandb.mode=disabled
```

## Coding Style & Naming Conventions

Use Python 3.10+ style with type hints, `pathlib.Path`, dataclasses where helpful, and explicit errors for schema mismatches. Use 4-space indentation and snake_case for functions, variables, modules, and test files. Keep new SurgWMBench work under `iris_surgwmbench/`, `tools/`, or `tests/`; do not mix it into `src/` unless adapting original IRIS intentionally.

## Testing Guidelines

Tests use `pytest`. Name tests `test_*.py` and prefer synthetic data under `tmp_path`. Coverage should protect dataset invariants: official manifests only, exactly 20 sparse human anchors, variable dense clip lengths, separate human vs interpolated coordinate sources, and no random split creation. Run `pytest` before committing.

## Commit & Pull Request Guidelines

Recent history uses short imperative or descriptive messages, for example `Add SurgWMBench data foundation` and `Upgrade tqdm`. Keep commits focused and avoid bundling unrelated refactors. PRs should include a concise summary, test commands run, dataset path assumptions, and any known limitations. Include screenshots only for UI or visualization changes.

## SurgWMBench-Specific Instructions

The local canonical dataset root is `/mnt/hdd1/neurips2026_dataset_track/SurgWMBench`. Treat `SurgWMBench/README.md` as the dataset contract. Do not modify dataset annotations, manifests, interpolation files, or source videos. Dense interpolation coordinates are auxiliary pseudo labels; sparse 20-anchor human labels are the primary benchmark target.
