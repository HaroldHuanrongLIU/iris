"""Train the IRIS SurgWMBench adapter."""

from __future__ import annotations

import argparse
import json

from iris_surgwmbench.adapter import train_adapter


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--manifest", default="manifests/train.jsonl")
    parser.add_argument("--train-manifest", default=None)
    parser.add_argument("--val-manifest", default=None)
    parser.add_argument("--target", choices=["sparse_20_anchor", "dense_pseudo"], default="sparse_20_anchor")
    parser.add_argument("--interpolation-method", default="linear")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--token-dim", type=int, default=64)
    parser.add_argument("--num-tokens", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--recon-weight", type=float, default=0.1)
    parser.add_argument("--entropy-weight", type=float, default=1e-4)
    parser.add_argument("--max-clips", type=int, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> int:
    result = train_adapter(build_parser().parse_args())
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
