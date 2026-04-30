from __future__ import annotations

import pytest
import torch

from iris_surgwmbench.data import (
    SOURCE_TO_CODE,
    SurgWMBenchClipDataset,
    SurgWMBenchFrameDataset,
    SurgWMBenchRawVideoDataset,
)
from tools.make_toy_surgwmbench import make_toy_surgwmbench


def test_sparse_dataset_loads_exactly_20_anchors(tmp_path):
    root = make_toy_surgwmbench(tmp_path / "SurgWMBench")
    dataset = SurgWMBenchClipDataset(
        dataset_root=root,
        manifest="manifests/train.jsonl",
        image_size=32,
        frame_sampling="sparse_anchors",
    )

    sample = dataset[0]

    assert sample["frames"].shape == (20, 3, 32, 32)
    assert sample["sampled_indices"].shape == (20,)
    assert sample["human_anchor_coords_px"].shape == (20, 2)
    assert sample["human_anchor_coords_norm"].shape == (20, 2)
    assert sample["selected_coords_norm"].shape == (20, 2)
    assert torch.equal(sample["frame_indices"], sample["human_anchor_local_indices"])
    assert torch.all(sample["selected_coord_sources"] == SOURCE_TO_CODE["human"])
    assert sample["dense_coords_norm"] is None


def test_dense_dataset_loads_variable_length_coordinates_and_sources(tmp_path):
    root = make_toy_surgwmbench(tmp_path / "SurgWMBench")
    dataset = SurgWMBenchClipDataset(
        dataset_root=root,
        manifest="manifests/train.jsonl",
        image_size=32,
        frame_sampling="dense",
        interpolation_method="linear",
    )

    sample0 = dataset[0]
    sample1 = dataset[1]

    assert sample0["frames"].shape == (25, 3, 32, 32)
    assert sample1["frames"].shape == (31, 3, 32, 32)
    assert sample0["dense_coords_norm"].shape == (25, 2)
    assert sample0["selected_coords_px"].shape == (25, 2)
    assert int((sample0["selected_coord_sources"] == SOURCE_TO_CODE["human"]).sum()) == 20
    assert int((sample0["selected_coord_sources"] == SOURCE_TO_CODE["interpolated"]).sum()) == 5
    assert torch.all(sample0["selected_label_weights"][sample0["selected_coord_sources"] == SOURCE_TO_CODE["human"]] == 1.0)
    assert torch.all(
        sample0["selected_label_weights"][sample0["selected_coord_sources"] == SOURCE_TO_CODE["interpolated"]] == 0.5
    )


def test_window_dataset_uses_deterministic_prefix_window(tmp_path):
    root = make_toy_surgwmbench(tmp_path / "SurgWMBench")
    dataset = SurgWMBenchClipDataset(
        dataset_root=root,
        manifest="manifests/train.jsonl",
        image_size=16,
        frame_sampling="window",
        max_frames=12,
    )

    sample = dataset[0]

    assert sample["frames"].shape == (12, 3, 16, 16)
    assert sample["frame_indices"].tolist() == list(range(12))
    assert sample["selected_coords_norm"].shape == (12, 2)


def test_interpolation_method_switching_loads_selected_file(tmp_path):
    root = make_toy_surgwmbench(tmp_path / "SurgWMBench")
    linear = SurgWMBenchClipDataset(
        dataset_root=root,
        manifest="manifests/train.jsonl",
        image_size=32,
        frame_sampling="dense",
        interpolation_method="linear",
    )[0]
    pchip = SurgWMBenchClipDataset(
        dataset_root=root,
        manifest="manifests/train.jsonl",
        image_size=32,
        frame_sampling="dense",
        interpolation_method="pchip",
    )[0]

    non_anchor = torch.nonzero(linear["selected_coord_sources"] == SOURCE_TO_CODE["interpolated"])[0].item()
    assert linear["interpolation_method"] == "linear"
    assert pchip["interpolation_method"] == "pchip"
    assert not torch.allclose(linear["selected_coords_px"][non_anchor], pchip["selected_coords_px"][non_anchor])


def test_private_use_path_alias_resolves_star_directories(tmp_path):
    root = make_toy_surgwmbench(tmp_path / "SurgWMBench", private_use_path_alias=True)
    dataset = SurgWMBenchClipDataset(
        dataset_root=root,
        manifest="manifests/train.jsonl",
        image_size=16,
        frame_sampling="dense",
        interpolation_method="linear",
    )

    sample = dataset[0]

    assert sample["trajectory_id"].endswith("\uf021")
    assert sample["frames"].shape == (25, 3, 16, 16)
    assert "*" in sample["annotation_path"]


def test_strict_loader_rejects_missing_interpolation_file(tmp_path):
    root = make_toy_surgwmbench(tmp_path / "SurgWMBench", missing_interpolation_method="pchip")
    dataset = SurgWMBenchClipDataset(
        dataset_root=root,
        manifest="manifests/train.jsonl",
        image_size=32,
        frame_sampling="dense",
        interpolation_method="pchip",
        strict=True,
    )

    with pytest.raises(FileNotFoundError, match="Interpolation file not found"):
        _ = dataset[0]


def test_loader_rejects_wrong_dataset_version_unless_allowed(tmp_path):
    root = make_toy_surgwmbench(tmp_path / "SurgWMBench", bad_version=True)

    with pytest.raises(ValueError, match="dataset_version"):
        SurgWMBenchClipDataset(
            dataset_root=root,
            manifest="manifests/train.jsonl",
            image_size=32,
            frame_sampling="sparse_anchors",
            strict=True,
        )

    dataset = SurgWMBenchClipDataset(
        dataset_root=root,
        manifest="manifests/train.jsonl",
        image_size=32,
        frame_sampling="sparse_anchors",
        strict=True,
        allow_legacy_version=True,
    )
    assert dataset[0]["frames"].shape[0] == 20


def test_dataset_loader_does_not_create_random_splits(tmp_path):
    root = make_toy_surgwmbench(tmp_path / "SurgWMBench")
    manifests_dir = root / "manifests"
    before = sorted(path.name for path in manifests_dir.iterdir())

    dataset = SurgWMBenchClipDataset(
        dataset_root=root,
        manifest="manifests/train.jsonl",
        image_size=32,
        frame_sampling="sparse_anchors",
    )
    _ = dataset[0]

    after = sorted(path.name for path in manifests_dir.iterdir())
    assert after == before == ["all.jsonl", "test.jsonl", "train.jsonl", "val.jsonl"]


def test_frame_dataset_returns_image_and_metadata(tmp_path):
    root = make_toy_surgwmbench(tmp_path / "SurgWMBench")
    dataset = SurgWMBenchFrameDataset(root, "manifests/train.jsonl", image_size=24)

    image, metadata = dataset[0]

    assert image.shape == (3, 24, 24)
    assert metadata["patient_id"] == "video_01"
    assert metadata["trajectory_id"] == "traj_001"
    assert metadata["local_frame_idx"] == 0
    assert metadata["frame_path"].endswith("000000.jpg")


def test_raw_video_dataset_supports_extracted_frame_backend(tmp_path):
    root = make_toy_surgwmbench(tmp_path / "SurgWMBench")
    dataset = SurgWMBenchRawVideoDataset(
        root,
        split="train",
        clip_length=4,
        stride=2,
        image_size=16,
        backend="frames",
        max_clips_per_video=1,
    )

    sample = dataset[0]

    assert sample["frames"].shape == (4, 3, 16, 16)
    assert sample["source_video_id"] == "video_01"
    assert sample["start_frame"] == 0
    assert sample["frame_indices"].tolist() == [0, 1, 2, 3]
