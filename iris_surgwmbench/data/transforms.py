"""Image transforms for SurgWMBench loaders."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image


def image_size_to_hw(image_size: int | tuple[int, int] | list[int] | None) -> tuple[int, int] | None:
    """Normalize an optional image size argument to ``(height, width)``."""

    if image_size is None or image_size is False:
        return None
    if isinstance(image_size, int):
        return int(image_size), int(image_size)
    if isinstance(image_size, (tuple, list)) and len(image_size) == 2:
        return int(image_size[0]), int(image_size[1])
    raise ValueError(f"Unsupported image_size={image_size!r}; expected int, (height, width), or None.")


def load_rgb_frame(
    path: str | Path,
    image_size: int | tuple[int, int] | list[int] | None = 128,
    *,
    mean: tuple[float, float, float] | None = None,
    std: tuple[float, float, float] | None = None,
) -> tuple[torch.Tensor, tuple[int, int]]:
    """Load an RGB frame as ``FloatTensor[3,H,W]`` in ``[0, 1]``.

    Returns the tensor and the original image size as ``(height, width)``.
    Target coordinates are not resized by this function; normalized
    coordinates remain valid after image resizing.
    """

    frame_path = Path(path).expanduser()
    if not frame_path.exists():
        raise FileNotFoundError(f"Frame image not found: {frame_path}")

    with Image.open(frame_path) as image:
        image = image.convert("RGB")
        original_size_hw = (int(image.height), int(image.width))
        target_hw = image_size_to_hw(image_size)
        if target_hw is not None:
            target_size_wh = (int(target_hw[1]), int(target_hw[0]))
            if image.size != target_size_wh:
                image = image.resize(target_size_wh, Image.BILINEAR)
        array = np.asarray(image, dtype=np.float32) / 255.0

    tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
    if mean is not None or std is not None:
        if mean is None or std is None:
            raise ValueError("Both mean and std must be provided when normalizing frames.")
        mean_t = torch.tensor(mean, dtype=tensor.dtype).view(3, 1, 1)
        std_t = torch.tensor(std, dtype=tensor.dtype).view(3, 1, 1)
        tensor = (tensor - mean_t) / std_t
    return tensor, original_size_hw
