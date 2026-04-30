"""Create a small synthetic SurgWMBench dataset for tests and smoke runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

INTERPOLATION_METHODS = ("linear", "pchip", "akima", "cubic_spline")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_image(path: Path, value: int, image_size_hw: tuple[int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = image_size_hw
    y = np.linspace(0, 1, height, dtype=np.float32)[:, None]
    x = np.linspace(0, 1, width, dtype=np.float32)[None, :]
    array = np.zeros((height, width, 3), dtype=np.uint8)
    array[..., 0] = np.clip(value + 60 * x, 0, 255).astype(np.uint8)
    array[..., 1] = np.clip(value + 80 * y, 0, 255).astype(np.uint8)
    array[..., 2] = np.clip(value, 0, 255)
    Image.fromarray(array).save(path)


def _write_tiny_video(path: Path, image_size_hw: tuple[int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import cv2  # type: ignore
    except ImportError:
        path.write_bytes(b"synthetic")
        return

    height, width = image_size_hw
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 5, (width, height))
    if not writer.isOpened():
        path.write_bytes(b"synthetic")
        return
    for idx in range(8):
        frame = np.full((height, width, 3), idx * 20, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def _anchor_indices(num_frames: int) -> list[int]:
    return [round(i * (num_frames - 1) / 19) for i in range(20)]


def _coord_px(local_frame_idx: int, clip_idx: int, offset: float = 0.0) -> list[float]:
    return [
        8.0 + clip_idx * 2.0 + local_frame_idx * 1.3 + offset,
        5.0 + clip_idx * 1.5 + local_frame_idx * 0.8 + offset,
    ]


def _coord_norm(coord_px: list[float], image_size_hw: tuple[int, int]) -> list[float]:
    height, width = image_size_hw
    return [float(coord_px[0] / width), float(coord_px[1] / height)]


def make_toy_surgwmbench(
    output_root: str | Path,
    *,
    num_clips: int = 2,
    image_size_hw: tuple[int, int] = (48, 64),
    lengths: tuple[int, ...] = (25, 31, 23, 35),
    bad_version: bool = False,
    missing_interpolation_method: str | None = None,
    private_use_path_alias: bool = False,
) -> Path:
    """Create and return a synthetic SurgWMBench root."""

    root = Path(output_root).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    version = "SurgWMBench-v0" if bad_version else "SurgWMBench"

    (root / "metadata").mkdir(parents=True, exist_ok=True)
    _write_json(
        root / "metadata" / "dataset_stats.json",
        {"clips": int(num_clips), "human_anchors": int(num_clips * 20), "dataset_version": version},
    )
    _write_json(root / "metadata" / "difficulty_rubric.json", {})
    _write_json(root / "metadata" / "interpolation_config.json", {"default_interpolation_method": "linear"})
    _write_json(root / "metadata" / "source_videos.json", {"source_videos": {}})
    _write_json(root / "metadata" / "validation_report.json", {"errors": 0, "warnings": 0})
    (root / "README.md").write_text("# SurgWMBench Toy Dataset\n", encoding="utf-8")

    entries: list[dict[str, Any]] = []
    difficulties: list[str | None] = ["low", "medium", "high", None]
    for clip_idx in range(num_clips):
        patient_id = f"video_{clip_idx + 1:02d}"
        source_video_id = patient_id
        trajectory_id = f"traj_{clip_idx + 1:03d}"
        stored_trajectory_id = f"{trajectory_id}\uf021" if private_use_path_alias and clip_idx == 0 else trajectory_id
        disk_trajectory_id = f"{trajectory_id}*" if private_use_path_alias and clip_idx == 0 else trajectory_id
        num_frames = lengths[clip_idx % len(lengths)]
        difficulty = difficulties[clip_idx % len(difficulties)]
        sampled_indices = _anchor_indices(num_frames)
        sampled_set = set(sampled_indices)

        source_video_rel = f"videos/{source_video_id}/video_left.avi"
        _write_tiny_video(root / source_video_rel, image_size_hw)

        frames_dir_rel = f"clips/{patient_id}/{stored_trajectory_id}/frames"
        disk_frames_dir_rel = f"clips/{patient_id}/{disk_trajectory_id}/frames"
        annotation_rel = f"clips/{patient_id}/{stored_trajectory_id}/annotation.json"
        disk_annotation_rel = f"clips/{patient_id}/{disk_trajectory_id}/annotation.json"
        interpolation_files = {
            method: f"interpolations/{patient_id}/{stored_trajectory_id}.{method}.json" for method in INTERPOLATION_METHODS
        }
        disk_interpolation_files = {
            method: f"interpolations/{patient_id}/{disk_trajectory_id}.{method}.json" for method in INTERPOLATION_METHODS
        }

        frames: list[dict[str, Any]] = []
        for local_idx in range(num_frames):
            frame_rel = f"{frames_dir_rel}/{local_idx:06d}.jpg"
            disk_frame_rel = f"{disk_frames_dir_rel}/{local_idx:06d}.jpg"
            _write_image(root / disk_frame_rel, value=(clip_idx * 30 + local_idx) % 255, image_size_hw=image_size_hw)
            is_anchor = local_idx in sampled_set
            anchor_idx = sampled_indices.index(local_idx) if is_anchor else None
            coord_px = _coord_px(local_idx, clip_idx) if is_anchor else None
            frames.append(
                {
                    "local_frame_idx": local_idx,
                    "source_frame_idx": 1000 + local_idx,
                    "frame_path": frame_rel,
                    "is_human_labeled": is_anchor,
                    "anchor_idx": anchor_idx,
                    "human_coord_px": coord_px,
                    "human_coord_norm": _coord_norm(coord_px, image_size_hw) if coord_px is not None else None,
                    "coord_source": "human" if is_anchor else "unlabeled",
                }
            )

        human_anchors: list[dict[str, Any]] = []
        for anchor_idx, local_idx in enumerate(sampled_indices):
            coord_px = _coord_px(local_idx, clip_idx)
            human_anchors.append(
                {
                    "anchor_idx": anchor_idx,
                    "old_frame_idx": anchor_idx,
                    "local_frame_idx": local_idx,
                    "source_frame_idx": 1000 + local_idx,
                    "label_name": "instrument_tip",
                    "value": coord_px,
                    "coord_px": coord_px,
                    "coord_norm": _coord_norm(coord_px, image_size_hw),
                }
            )

        for method_idx, method in enumerate(INTERPOLATION_METHODS):
            coordinates: list[dict[str, Any]] = []
            for local_idx in range(num_frames):
                is_anchor = local_idx in sampled_set
                offset = 0.0 if is_anchor else method_idx * 0.25
                coord_px = _coord_px(local_idx, clip_idx, offset=offset)
                coordinates.append(
                    {
                        "local_frame_idx": local_idx,
                        "coord_px": coord_px,
                        "coord_norm": _coord_norm(coord_px, image_size_hw),
                        "source": "human" if is_anchor else "interpolated",
                        "anchor_idx": sampled_indices.index(local_idx) if is_anchor else None,
                        "confidence": 1.0 if is_anchor else 0.6,
                        "label_weight": 1.0 if is_anchor else 0.5,
                        "is_out_of_bounds": False,
                    }
                )
            _write_json(
                root / disk_interpolation_files[method],
                {
                    "dataset_version": version,
                    "patient_id": patient_id,
                    "trajectory_id": stored_trajectory_id,
                    "interpolation_method": method,
                    "num_frames": num_frames,
                    "image_size": {"width": image_size_hw[1], "height": image_size_hw[0]},
                    "coordinates": coordinates,
                },
            )

        if missing_interpolation_method is not None:
            missing_path = root / disk_interpolation_files[missing_interpolation_method]
            if missing_path.exists():
                missing_path.unlink()

        annotation = {
            "dataset_version": version,
            "patient_id": patient_id,
            "source_video_id": source_video_id,
            "source_video_path": source_video_rel,
            "trajectory_id": stored_trajectory_id,
            "difficulty": difficulty,
            "num_frames": num_frames,
            "image_size": {"width": image_size_hw[1], "height": image_size_hw[0]},
            "coordinate_format": "pixel_xy",
            "coordinate_origin": "top_left",
            "num_human_anchors": 20,
            "sampled_indices": sampled_indices,
            "available_interpolation_methods": list(INTERPOLATION_METHODS),
            "default_interpolation_method": "linear",
            "frames": frames,
            "human_anchors": human_anchors,
            "interpolation_files": interpolation_files,
        }
        _write_json(root / disk_annotation_rel, annotation)

        entries.append(
            {
                "dataset_version": version,
                "patient_id": patient_id,
                "source_video_id": source_video_id,
                "source_video_path": source_video_rel,
                "trajectory_id": stored_trajectory_id,
                "difficulty": difficulty,
                "num_frames": num_frames,
                "annotation_path": annotation_rel,
                "frames_dir": frames_dir_rel,
                "interpolation_files": interpolation_files,
                "default_interpolation_method": "linear",
                "num_human_anchors": 20,
                "sampled_indices": sampled_indices,
            }
        )

    manifest_text = "\n".join(json.dumps(entry) for entry in entries) + "\n"
    (root / "manifests").mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test", "all"):
        (root / "manifests" / f"{split}.jsonl").write_text(manifest_text, encoding="utf-8")
    return root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", "--output", required=True, type=Path)
    parser.add_argument("--num-clips", type=int, default=2)
    parser.add_argument("--height", type=int, default=48)
    parser.add_argument("--width", type=int, default=64)
    args = parser.parse_args()

    root = make_toy_surgwmbench(
        args.output_root,
        num_clips=args.num_clips,
        image_size_hw=(args.height, args.width),
    )
    print(f"Wrote toy SurgWMBench dataset to {root}")


if __name__ == "__main__":
    main()
