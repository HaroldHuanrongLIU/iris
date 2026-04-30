from __future__ import annotations

import torch

from iris_surgwmbench.data import (
    SurgWMBenchClipDataset,
    SurgWMBenchFrameDataset,
    collate_dense_variable_length,
    collate_frame_tokenizer,
    collate_sparse_anchors,
    collate_window_sequences,
)
from tools.make_toy_surgwmbench import make_toy_surgwmbench


def test_collate_sparse_returns_shapes_and_actions(tmp_path):
    root = make_toy_surgwmbench(tmp_path / "SurgWMBench")
    dataset = SurgWMBenchClipDataset(root, "manifests/train.jsonl", image_size=32, frame_sampling="sparse_anchors")

    batch = collate_sparse_anchors([dataset[0], dataset[1]])

    assert batch["frames"].shape == (2, 20, 3, 32, 32)
    assert batch["coords_norm"].shape == (2, 20, 2)
    assert batch["coords_px"].shape == (2, 20, 2)
    assert batch["sampled_indices"].shape == (2, 20)
    assert batch["human_anchor_mask"].shape == (2, 20)
    assert batch["anchor_dt"].shape == (2, 19)
    assert batch["actions_delta"].shape == (2, 19, 2)
    assert batch["actions_delta_dt"].shape == (2, 19, 3)
    assert batch["direction_classes"].shape == (2, 19)
    assert batch["magnitudes"].shape == (2, 19)
    assert batch["human_anchor_mask"].all()

    expected_delta = batch["coords_norm"][:, 1:] - batch["coords_norm"][:, :-1]
    expected_dt = (batch["sampled_indices"][:, 1:] - batch["sampled_indices"][:, :-1]).float()
    expected_dt = expected_dt / torch.clamp(batch["num_frames"].float() - 1.0, min=1.0).unsqueeze(1)
    assert torch.allclose(batch["actions_delta"], expected_delta)
    assert torch.allclose(batch["actions_delta_dt"][..., :2], expected_delta)
    assert torch.allclose(batch["actions_delta_dt"][..., 2], expected_dt)
    assert torch.all(batch["direction_classes"] >= 0)
    assert torch.all(batch["direction_classes"] <= 8)
    assert torch.allclose(batch["magnitudes"], torch.linalg.norm(expected_delta, dim=-1))


def test_collate_dense_pads_variable_length_clips_and_masks(tmp_path):
    root = make_toy_surgwmbench(tmp_path / "SurgWMBench")
    dataset = SurgWMBenchClipDataset(root, "manifests/train.jsonl", image_size=32, frame_sampling="dense")

    batch = collate_dense_variable_length([dataset[0], dataset[1]])

    assert batch["frames"].shape == (2, 31, 3, 32, 32)
    assert batch["coords_norm"].shape == (2, 31, 2)
    assert batch["coords_px"].shape == (2, 31, 2)
    assert batch["frame_mask"].shape == (2, 31)
    assert batch["action_mask"].shape == (2, 30)
    assert batch["direction_classes"].shape == (2, 30)
    assert batch["magnitudes"].shape == (2, 30)
    assert int(batch["frame_mask"][0].sum()) == 25
    assert int(batch["frame_mask"][1].sum()) == 31
    assert int(batch["action_mask"][0].sum()) == 24
    assert int(batch["action_mask"][1].sum()) == 30
    assert not batch["frame_mask"][0, 25:].any()
    assert torch.all(batch["frame_indices"][0, 25:] == -1)
    assert torch.all(batch["coord_source"][0, 25:] == 0)
    assert torch.all(batch["label_weight"][0, 25:] == 0)
    assert torch.all(batch["confidence"][0, 25:] == 0)

    expected_delta = batch["coords_norm"][0, 1:25] - batch["coords_norm"][0, :24]
    assert torch.allclose(batch["actions_delta"][0, :24], expected_delta)
    assert torch.all(batch["actions_delta"][0, 24:] == 0)


def test_collate_window_sequences_aliases_dense_padding(tmp_path):
    root = make_toy_surgwmbench(tmp_path / "SurgWMBench")
    dataset = SurgWMBenchClipDataset(root, "manifests/train.jsonl", image_size=16, frame_sampling="window", max_frames=10)

    batch = collate_window_sequences([dataset[0], dataset[1]])

    assert batch["frames"].shape == (2, 10, 3, 16, 16)
    assert batch["frame_mask"].all()
    assert batch["action_mask"].shape == (2, 9)


def test_collate_frame_tokenizer_returns_images_and_metadata(tmp_path):
    root = make_toy_surgwmbench(tmp_path / "SurgWMBench")
    dataset = SurgWMBenchFrameDataset(root, "manifests/train.jsonl", image_size=20)

    batch = collate_frame_tokenizer([dataset[0], dataset[1]])

    assert batch["images"].shape == (2, 3, 20, 20)
    assert batch["frames"].shape == (2, 3, 20, 20)
    assert len(batch["metadata"]) == 2
    assert batch["metadata"][0]["local_frame_idx"] == 0
