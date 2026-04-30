"""Collators for SurgWMBench sequence and frame datasets."""

from __future__ import annotations

from typing import Any

import torch


def _metadata(batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = (
        "patient_id",
        "source_video_id",
        "source_video_path",
        "trajectory_id",
        "difficulty",
        "num_frames",
        "image_size_original",
        "annotation_path",
        "interpolation_path",
        "interpolation_method",
    )
    return [{key: item.get(key) for key in keys} for item in batch]


def _direction_classes(dxdy: torch.Tensor, stay_threshold: float = 1e-8) -> torch.Tensor:
    """Map normalized deltas to 0=stay and 1..8 compass classes.

    Classes rotate counter-clockwise in screen coordinates converted to
    mathematical orientation: E, NE, N, NW, W, SW, S, SE.
    """

    magnitudes = torch.linalg.norm(dxdy, dim=-1)
    classes = torch.zeros(dxdy.shape[:-1], dtype=torch.long, device=dxdy.device)
    moving = magnitudes > stay_threshold
    if moving.any():
        angles = torch.atan2(-dxdy[..., 1], dxdy[..., 0])
        sector = torch.round(angles / (torch.pi / 4.0)).long().remainder(8)
        classes = torch.where(moving, sector + 1, classes)
    return classes


def _actions_from_coords(
    coords_norm: torch.Tensor,
    frame_indices: torch.Tensor,
    num_frames: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size, timesteps = coords_norm.shape[:2]
    action_t = max(timesteps - 1, 0)
    actions_delta = torch.zeros(batch_size, action_t, 2, dtype=torch.float32)
    actions_delta_dt = torch.zeros(batch_size, action_t, 3, dtype=torch.float32)
    direction_classes = torch.zeros(batch_size, action_t, dtype=torch.long)
    magnitudes = torch.zeros(batch_size, action_t, dtype=torch.float32)
    if action_t == 0:
        return actions_delta, actions_delta_dt, direction_classes, magnitudes

    actions_delta = coords_norm[:, 1:] - coords_norm[:, :-1]
    denom = torch.clamp(num_frames.to(torch.float32) - 1.0, min=1.0).unsqueeze(1)
    dt = (frame_indices[:, 1:] - frame_indices[:, :-1]).to(torch.float32) / denom
    actions_delta_dt = torch.cat([actions_delta, dt.unsqueeze(-1)], dim=-1)
    direction_classes = _direction_classes(actions_delta)
    magnitudes = torch.linalg.norm(actions_delta, dim=-1)

    if mask is not None:
        action_mask = mask[:, 1:] & mask[:, :-1]
        actions_delta = actions_delta.masked_fill(~action_mask.unsqueeze(-1), 0.0)
        actions_delta_dt = actions_delta_dt.masked_fill(~action_mask.unsqueeze(-1), 0.0)
        direction_classes = direction_classes.masked_fill(~action_mask, 0)
        magnitudes = magnitudes.masked_fill(~action_mask, 0.0)
    return actions_delta, actions_delta_dt, direction_classes, magnitudes


def collate_sparse_anchors(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate samples loaded with ``frame_sampling='sparse_anchors'``."""

    if not batch:
        raise ValueError("Cannot collate an empty batch.")
    if any(item["frames"] is None for item in batch):
        raise ValueError("collate_sparse_anchors requires return_images=True samples.")

    frames = torch.stack([item["frames"] for item in batch], dim=0)
    coords_norm = torch.stack([item["selected_coords_norm"] for item in batch], dim=0)
    coords_px = torch.stack([item["selected_coords_px"] for item in batch], dim=0)
    sampled_indices = torch.stack([item["sampled_indices"] for item in batch], dim=0)
    frame_indices = torch.stack([item["frame_indices"] for item in batch], dim=0)
    coord_source = torch.stack([item["selected_coord_sources"] for item in batch], dim=0)
    label_weight = torch.stack([item["selected_label_weights"] for item in batch], dim=0)
    confidence = torch.stack([item["selected_confidence"] for item in batch], dim=0)
    num_frames = torch.as_tensor([int(item["num_frames"]) for item in batch], dtype=torch.long)

    if frames.shape[1] != 20 or coords_norm.shape[1] != 20 or sampled_indices.shape[1] != 20:
        raise ValueError("Sparse SurgWMBench batches must contain exactly 20 human anchors.")

    human_anchor_mask = torch.ones(coords_norm.shape[:2], dtype=torch.bool)
    actions_delta, actions_delta_dt, direction_classes, magnitudes = _actions_from_coords(
        coords_norm, sampled_indices, num_frames, mask=human_anchor_mask
    )
    anchor_dt = actions_delta_dt[..., 2]

    return {
        "frames": frames,
        "coords_norm": coords_norm,
        "coords_px": coords_px,
        "sampled_indices": sampled_indices,
        "frame_indices": frame_indices,
        "human_anchor_mask": human_anchor_mask,
        "frame_mask": human_anchor_mask,
        "num_frames": num_frames,
        "anchor_dt": anchor_dt,
        "actions_delta": actions_delta,
        "actions_delta_dt": actions_delta_dt,
        "direction_classes": direction_classes,
        "magnitudes": magnitudes,
        "coord_source": coord_source,
        "label_weight": label_weight,
        "confidence": confidence,
        "difficulty": [item["difficulty"] for item in batch],
        "metadata": _metadata(batch),
    }


def collate_dense_variable_length(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate variable-length dense SurgWMBench samples with padding."""

    if not batch:
        raise ValueError("Cannot collate an empty batch.")
    if any(item["frames"] is None for item in batch):
        raise ValueError("collate_dense_variable_length requires return_images=True samples.")

    batch_size = len(batch)
    max_t = max(int(item["frames"].shape[0]) for item in batch)
    channels, height, width = batch[0]["frames"].shape[1:]

    frames = torch.zeros(batch_size, max_t, channels, height, width, dtype=torch.float32)
    coords_norm = torch.zeros(batch_size, max_t, 2, dtype=torch.float32)
    coords_px = torch.zeros(batch_size, max_t, 2, dtype=torch.float32)
    frame_mask = torch.zeros(batch_size, max_t, dtype=torch.bool)
    coord_source = torch.zeros(batch_size, max_t, dtype=torch.long)
    label_weight = torch.zeros(batch_size, max_t, dtype=torch.float32)
    confidence = torch.zeros(batch_size, max_t, dtype=torch.float32)
    frame_indices = torch.full((batch_size, max_t), -1, dtype=torch.long)
    num_frames = torch.as_tensor([int(item["num_frames"]) for item in batch], dtype=torch.long)

    for row, item in enumerate(batch):
        t = int(item["frames"].shape[0])
        frames[row, :t] = item["frames"]
        coords_norm[row, :t] = item["selected_coords_norm"]
        coords_px[row, :t] = item["selected_coords_px"]
        frame_mask[row, :t] = True
        coord_source[row, :t] = item["selected_coord_sources"]
        label_weight[row, :t] = item["selected_label_weights"]
        confidence[row, :t] = item["selected_confidence"]
        frame_indices[row, :t] = item["frame_indices"]

    actions_delta, actions_delta_dt, direction_classes, magnitudes = _actions_from_coords(
        coords_norm, frame_indices, num_frames, mask=frame_mask
    )
    action_mask = frame_mask[:, 1:] & frame_mask[:, :-1] if max_t > 1 else torch.zeros(batch_size, 0, dtype=torch.bool)

    return {
        "frames": frames,
        "coords_norm": coords_norm,
        "coords_px": coords_px,
        "frame_mask": frame_mask,
        "coord_source": coord_source,
        "label_weight": label_weight,
        "confidence": confidence,
        "frame_indices": frame_indices,
        "num_frames": num_frames,
        "actions_delta": actions_delta,
        "actions_delta_dt": actions_delta_dt,
        "action_mask": action_mask,
        "direction_classes": direction_classes,
        "magnitudes": magnitudes,
        "difficulty": [item["difficulty"] for item in batch],
        "metadata": _metadata(batch),
    }


def collate_window_sequences(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate window samples; windows are padded like dense sequences."""

    return collate_dense_variable_length(batch)


def collate_frame_tokenizer(batch: list[tuple[torch.Tensor, dict[str, Any]]]) -> dict[str, Any]:
    """Collate frame-level tokenizer pretraining samples."""

    if not batch:
        raise ValueError("Cannot collate an empty batch.")
    images, metadata = zip(*batch)
    return {
        "images": torch.stack(list(images), dim=0),
        "frames": torch.stack(list(images), dim=0),
        "metadata": list(metadata),
    }


__all__ = [
    "collate_dense_variable_length",
    "collate_frame_tokenizer",
    "collate_sparse_anchors",
    "collate_window_sequences",
]
