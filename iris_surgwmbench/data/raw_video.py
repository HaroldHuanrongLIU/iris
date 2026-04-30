"""Raw-video and extracted-frame sequence datasets for SurgWMBench."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from torch.utils.data import Dataset

from .surgwmbench import (
    DATASET_VERSION,
    _frame_local_index,
    _frame_path_value,
    load_json,
    read_jsonl_manifest,
    resolve_dataset_path,
)
from .transforms import image_size_to_hw, load_rgb_frame

VideoBackend = Literal["opencv", "frames", "opencv_or_frames"]


def _require_cv2() -> Any:
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "SurgWMBenchRawVideoDataset backend='opencv' requires opencv-python. "
            "Install OpenCV or use backend='frames'."
        ) from exc
    return cv2


@dataclass(frozen=True)
class _VideoClip:
    kind: Literal["video", "frames"]
    source_video_id: str
    source_video_path: str
    start_frame: int
    frame_indices: tuple[int, ...]
    video_path: Path | None = None
    frame_paths: tuple[Path, ...] = ()


def _frame_to_tensor(frame_bgr: np.ndarray, image_size: int | tuple[int, int] | None) -> torch.Tensor:
    cv2 = _require_cv2()
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    target_hw = image_size_to_hw(image_size)
    if target_hw is not None:
        target_size_wh = (int(target_hw[1]), int(target_hw[0]))
        frame_rgb = cv2.resize(frame_rgb, target_size_wh, interpolation=cv2.INTER_LINEAR)
    array = frame_rgb.astype(np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def _video_frame_count(path: Path) -> int | None:
    cv2 = _require_cv2()
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            return None
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        return count if count > 0 else None
    finally:
        capture.release()


def _read_video_window(path: Path, frame_indices: tuple[int, ...], image_size: int | tuple[int, int] | None) -> torch.Tensor:
    cv2 = _require_cv2()
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    try:
        frames: list[torch.Tensor] = []
        for frame_index in frame_indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
            ok, frame = capture.read()
            if not ok or frame is None:
                raise RuntimeError(f"Could not read frame {frame_index} from {path}")
            frames.append(_frame_to_tensor(frame, image_size))
        return torch.stack(frames, dim=0)
    finally:
        capture.release()


class SurgWMBenchRawVideoDataset(Dataset):
    """Load unlabeled source-video windows for tokenizer/world-model pretraining."""

    def __init__(
        self,
        dataset_root: str | Path,
        split: str = "train",
        source_video_manifest: str | Path | None = None,
        clip_length: int = 16,
        stride: int = 4,
        image_size: int | tuple[int, int] | None = 128,
        backend: VideoBackend = "opencv",
        max_videos: int | None = None,
        max_clips_per_video: int | None = None,
    ) -> None:
        if clip_length <= 0:
            raise ValueError("clip_length must be positive.")
        if stride <= 0:
            raise ValueError("stride must be positive.")
        if backend not in ("opencv", "frames", "opencv_or_frames"):
            raise ValueError(f"Unsupported backend={backend!r}")

        self.dataset_root = Path(dataset_root).expanduser()
        self.split = split
        self.source_video_manifest = source_video_manifest
        self.clip_length = int(clip_length)
        self.stride = int(stride)
        self.image_size = image_size
        self.backend = backend
        self.manifest_path = self.dataset_root / "manifests" / f"{split}.jsonl"
        self.entries = read_jsonl_manifest(self.manifest_path)
        self._annotations = [self._load_annotation(entry) for entry in self.entries]
        self.clips = self._build_index(max_videos=max_videos, max_clips_per_video=max_clips_per_video)
        if not self.clips:
            raise ValueError("No raw-video or frame-fallback clips could be indexed from SurgWMBench data.")

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, index: int) -> dict[str, Any]:
        clip = self.clips[index]
        if clip.kind == "video":
            if clip.video_path is None:
                raise RuntimeError("Video clip is missing video_path.")
            frames = _read_video_window(clip.video_path, clip.frame_indices, self.image_size)
        else:
            frames = torch.stack([load_rgb_frame(path, self.image_size)[0] for path in clip.frame_paths], dim=0)
        return {
            "frames": frames,
            "source_video_id": clip.source_video_id,
            "source_video_path": clip.source_video_path,
            "start_frame": int(clip.start_frame),
            "frame_indices": torch.as_tensor(clip.frame_indices, dtype=torch.long),
        }

    def _load_annotation(self, entry: dict[str, Any]) -> dict[str, Any]:
        annotation_path = resolve_dataset_path(self.dataset_root, entry.get("annotation_path"))
        if annotation_path is None or not annotation_path.exists():
            raise FileNotFoundError(f"Annotation not found for manifest entry: {entry}")
        annotation = load_json(annotation_path)
        if not isinstance(annotation, dict):
            raise ValueError(f"Annotation must be a JSON object: {annotation_path}")
        version = annotation.get("dataset_version", entry.get("dataset_version"))
        if version != DATASET_VERSION:
            raise ValueError(f"{annotation_path}: dataset_version={version!r}, expected {DATASET_VERSION!r}")
        return annotation

    def _source_video_entries(self) -> list[dict[str, str]]:
        manifest_path = (
            resolve_dataset_path(self.dataset_root, self.source_video_manifest)
            if self.source_video_manifest is not None
            else None
        )
        raw_entries: Any = self.entries
        if manifest_path is not None and manifest_path.exists():
            payload = load_json(manifest_path)
            if isinstance(payload, dict):
                raw_entries = payload.get("videos", payload.get("source_videos", payload))
                if isinstance(raw_entries, dict):
                    raw_entries = [
                        {"source_video_id": key, **value} if isinstance(value, dict) else {"source_video_id": key}
                        for key, value in raw_entries.items()
                    ]
            elif isinstance(payload, list):
                raw_entries = payload
            else:
                raw_entries = []

        seen: set[tuple[str, str]] = set()
        result: list[dict[str, str]] = []
        for entry in raw_entries:
            if not isinstance(entry, dict):
                continue
            source_video_id = str(entry.get("source_video_id") or entry.get("id") or entry.get("video_id") or "")
            source_video_path = str(entry.get("source_video_path") or entry.get("path") or entry.get("video_path") or "")
            if not source_video_id or not source_video_path:
                continue
            source_split = entry.get("source_dataset_split")
            if source_split is not None and self.split != "all" and str(source_split) != self.split:
                continue
            key = (source_video_id, source_video_path)
            if key in seen:
                continue
            seen.add(key)
            result.append({"source_video_id": source_video_id, "source_video_path": source_video_path})
        return result

    def _build_index(self, max_videos: int | None, max_clips_per_video: int | None) -> list[_VideoClip]:
        clips: list[_VideoClip] = []
        videos = self._source_video_entries()
        if max_videos is not None and max_videos > 0:
            videos = videos[:max_videos]

        if self.backend in ("opencv", "opencv_or_frames"):
            _require_cv2()
            for video in videos:
                video_path = resolve_dataset_path(self.dataset_root, video["source_video_path"])
                if video_path is None or not video_path.exists():
                    if self.backend == "opencv":
                        raise FileNotFoundError(f"Source video not found: {video['source_video_path']}")
                    continue
                frame_count = _video_frame_count(video_path)
                if frame_count is None or frame_count < self.clip_length:
                    if self.backend == "opencv":
                        raise RuntimeError(f"Source video is not decodable or too short: {video_path}")
                    continue
                per_video = 0
                for start in range(0, frame_count - self.clip_length + 1, self.stride):
                    frame_indices = tuple(range(start, start + self.clip_length))
                    clips.append(
                        _VideoClip(
                            kind="video",
                            source_video_id=video["source_video_id"],
                            source_video_path=video["source_video_path"],
                            start_frame=start,
                            frame_indices=frame_indices,
                            video_path=video_path,
                        )
                    )
                    per_video += 1
                    if max_clips_per_video is not None and per_video >= max_clips_per_video:
                        break

        if clips or self.backend == "opencv":
            return clips
        return self._frame_fallback_index(max_videos=max_videos, max_clips_per_video=max_clips_per_video)

    def _frame_fallback_index(self, max_videos: int | None, max_clips_per_video: int | None) -> list[_VideoClip]:
        clips: list[_VideoClip] = []
        per_video_counts: dict[str, int] = {}
        allowed_videos = None
        if max_videos is not None and max_videos > 0:
            allowed_videos = {entry["source_video_id"] for entry in self._source_video_entries()[:max_videos]}

        for entry, annotation in zip(self.entries, self._annotations):
            source_video_id = str(entry.get("source_video_id", annotation.get("source_video_id", "")))
            if allowed_videos is not None and source_video_id not in allowed_videos:
                continue
            frames = annotation.get("frames", [])
            if not isinstance(frames, list) or len(frames) < self.clip_length:
                continue
            frame_paths: list[Path] = []
            frame_indices: list[int] = []
            for fallback_idx, frame in enumerate(frames):
                path_value = _frame_path_value(frame)
                local_idx = _frame_local_index(frame, fallback_idx)
                frame_path = resolve_dataset_path(self.dataset_root, path_value) if path_value else None
                if frame_path is None:
                    frames_dir = resolve_dataset_path(self.dataset_root, entry.get("frames_dir"))
                    if frames_dir is None:
                        continue
                    frame_path = frames_dir / f"{local_idx:06d}.jpg"
                if not frame_path.exists():
                    continue
                frame_paths.append(frame_path)
                frame_indices.append(local_idx)

            for start in range(0, len(frame_paths) - self.clip_length + 1, self.stride):
                if max_clips_per_video is not None and per_video_counts.get(source_video_id, 0) >= max_clips_per_video:
                    break
                selected_paths = tuple(frame_paths[start : start + self.clip_length])
                selected_indices = tuple(frame_indices[start : start + self.clip_length])
                clips.append(
                    _VideoClip(
                        kind="frames",
                        source_video_id=source_video_id,
                        source_video_path=str(entry.get("source_video_path", annotation.get("source_video_path", ""))),
                        start_frame=int(selected_indices[0]),
                        frame_indices=selected_indices,
                        frame_paths=selected_paths,
                    )
                )
                per_video_counts[source_video_id] = per_video_counts.get(source_video_id, 0) + 1
        return clips


__all__ = ["SurgWMBenchRawVideoDataset"]
