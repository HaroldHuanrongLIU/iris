from __future__ import annotations

import sys
from pathlib import Path

for parent in Path(__file__).resolve().parents:
    if (parent / "surgwmbench_benchmark").is_dir():
        sys.path.insert(0, str(parent))
        break

import torch
import torch.nn.functional as F
from torch import nn

from iris_surgwmbench.adapter import SurgWMBenchIrisTransformer
from surgwmbench_benchmark.future_model_helpers import normalized_context_time, normalized_future_time
from surgwmbench_benchmark.future_prediction import FutureProtocolConfig, main


class IRISFuturePredictionModel(nn.Module):
    """Future-prediction wrapper around the IRIS tokenizer + Transformer core."""

    def __init__(self, config: FutureProtocolConfig) -> None:
        super().__init__()
        self.token_dim = config.latent_dim
        self.core = SurgWMBenchIrisTransformer(
            token_dim=config.latent_dim,
            num_tokens=128,
            hidden_dim=config.hidden_dim,
            num_layers=2,
            num_heads=4,
        )
        self.hidden_to_token = nn.Linear(config.hidden_dim, config.latent_dim)

    def _decode_tokens(self, tokens: torch.Tensor, size_hw: tuple[int, int]) -> torch.Tensor:
        batch, horizon = tokens.shape[:2]
        flat = tokens.reshape(batch * horizon, self.token_dim)
        frames = self.core.tokenizer.decoder(self.core.tokenizer.decoder_fc(flat).view(flat.shape[0], 64, 4, 4))
        if frames.shape[-2:] != size_hw:
            frames = F.interpolate(frames, size=size_hw, mode="bilinear", align_corners=False)
        return frames.view(batch, horizon, 3, *size_hw)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        frames = batch["context_frames"]
        bsz, context, channels, height, width = frames.shape
        token_out = self.core.tokenizer(frames.reshape(bsz * context, channels, height, width), (height, width))
        context_tokens = token_out["tokens"].view(bsz, context, -1)
        future_tokens = context_tokens[:, -1:].expand(-1, batch["future_frame_indices"].shape[1], -1)
        model_in = torch.cat(
            [
                self.core.input_proj(torch.cat([context_tokens, normalized_context_time(batch)], dim=-1)),
                self.core.input_proj(torch.cat([future_tokens, normalized_future_time(batch)], dim=-1)),
            ],
            dim=1,
        )
        hidden = self.core.transformer(model_in)[:, context:]
        pred_coords = torch.sigmoid(self.core.coord_head(hidden))
        pred_frames = self._decode_tokens(self.hidden_to_token(hidden), (height, width))
        return {"pred_frames": pred_frames, "pred_coords_norm": pred_coords}


def make_model(config: FutureProtocolConfig) -> nn.Module:
    return IRISFuturePredictionModel(config)


if __name__ == "__main__":
    raise SystemExit(main("iris", "IRISFuturePredictionCore", "iris_surgwmbench.data.surgwmbench", make_model))
