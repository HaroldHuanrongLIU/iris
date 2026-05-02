from __future__ import annotations

from types import SimpleNamespace

from iris_surgwmbench.adapter import eval_adapter, train_adapter
from tools.make_toy_surgwmbench import make_toy_surgwmbench


def test_iris_adapter_train_eval_smoke(tmp_path):
    root = make_toy_surgwmbench(tmp_path / "SurgWMBench", num_clips=1)
    output_dir = tmp_path / "run"
    train_result = train_adapter(
        SimpleNamespace(
            dataset_root=str(root),
            manifest="manifests/train.jsonl",
            train_manifest="manifests/train.jsonl",
            val_manifest="manifests/val.jsonl",
            target="sparse_20_anchor",
            interpolation_method="linear",
            output_dir=str(output_dir),
            epochs=1,
            batch_size=1,
            learning_rate=1e-3,
            image_size=32,
            token_dim=8,
            num_tokens=16,
            hidden_dim=16,
            num_layers=1,
            num_heads=2,
            recon_weight=0.01,
            entropy_weight=1e-5,
            max_clips=1,
            max_frames=None,
            num_workers=0,
            device="cpu",
            seed=7,
        )
    )
    result = eval_adapter(
        SimpleNamespace(
            dataset_root=str(root),
            manifest="manifests/test.jsonl",
            checkpoint=train_result["checkpoint"],
            target="sparse_20_anchor",
            interpolation_method="linear",
            output=str(output_dir / "metrics.json"),
            batch_size=1,
            max_clips=1,
            max_frames=None,
            num_workers=0,
            device="cpu",
        )
    )
    assert result["baseline"] == "iris"
    assert result["experiment_target"] == "sparse_20_anchor"
    assert "ade" in result["metrics_overall"]
