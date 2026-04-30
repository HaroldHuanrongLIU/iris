from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from tools.make_toy_surgwmbench import make_toy_surgwmbench
from tools.validate_surgwmbench_loader import validate_surgwmbench


def test_loader_validator_function_passes_toy_dataset(tmp_path):
    root = make_toy_surgwmbench(tmp_path / "SurgWMBench")

    errors = validate_surgwmbench(
        dataset_root=root,
        manifest="manifests/train.jsonl",
        interpolation_method="linear",
        check_files=True,
        num_samples=2,
    )

    assert errors == []


def test_loader_validator_cli_passes_toy_dataset(tmp_path):
    root = make_toy_surgwmbench(tmp_path / "SurgWMBench")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "tools.validate_surgwmbench_loader",
            "--dataset-root",
            str(root),
            "--manifest",
            "manifests/train.jsonl",
            "--interpolation-method",
            "linear",
            "--check-files",
            "--num-samples",
            "2",
        ],
        cwd=str(Path(__file__).resolve().parents[1]),
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "SurgWMBench validation passed." in result.stdout
