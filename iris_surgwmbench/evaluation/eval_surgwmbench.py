"""Evaluate the IRIS SurgWMBench adapter."""

from __future__ import annotations

import argparse
import json

from iris_surgwmbench.adapter import eval_adapter


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--manifest", default="manifests/test.jsonl")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--target", choices=["sparse_20_anchor", "dense_pseudo"], default="sparse_20_anchor")
    parser.add_argument("--interpolation-method", default="linear")
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-clips", type=int, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    return parser


def main() -> int:
    result = eval_adapter(build_parser().parse_args())
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
