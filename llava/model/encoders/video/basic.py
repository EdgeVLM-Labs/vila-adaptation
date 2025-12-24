from functools import partial
from typing import Any, Dict, List, Optional
import warnings

import torch

from llava.model.encoders.base import BaseEncoder

__all__ = ["BasicVideoEncoder"]


class BasicVideoEncoder(BaseEncoder):
    def __init__(
        self,
        parent: torch.nn.Module,
        start_tokens: Optional[str] = None,
        end_tokens: Optional[str] = "\n",
    ) -> None:
        super().__init__(parent)
        self.start_tokens = start_tokens
        self.end_tokens = end_tokens

    def embed_tokens(self, tokens: Optional[str]) -> Optional[torch.Tensor]:
        if tokens is None:
            return None
        token_ids = self.parent.tokenizer(tokens).input_ids
        token_ids = torch.tensor(token_ids, device=self.parent.device)
        return self.parent.llm.model.embed_tokens(token_ids)

    def _process_features(
        self,
        features: torch.Tensor,
        start_token_embeds: Optional[torch.Tensor],
        end_token_embeds: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if start_token_embeds is not None:
            start_embeds = torch.stack([start_token_embeds] * features.shape[0], dim=0)
            features = torch.cat([start_embeds, features], dim=1)
        if end_token_embeds is not None:
            end_embeds = torch.stack([end_token_embeds] * features.shape[0], dim=0)
            features = torch.cat([features, end_embeds], dim=1)
        return features.flatten(0, 1)

    def forward(self, videos: List[torch.Tensor], config: Dict[str, Any]) -> List[torch.Tensor]:
        # Handle empty video list - this can happen with corrupted/missing validation samples
        if not videos or len(videos) == 0:
            import logging
            logging.warning(f"Empty video list encountered in video encoder (received {len(videos) if videos else 0} videos). This may indicate data loading issues.")
            # Return empty list to skip this batch gracefully
            return []

        # Filter out empty tensors and log any that are found
        original_count = len(videos)
        videos = [v for v in videos if v is not None and v.numel() > 0]
        if len(videos) < original_count:
            import logging
            logging.warning(f"Filtered out {original_count - len(videos)} empty video tensors")

        if not videos:
            import logging
            logging.warning("All video tensors were empty after filtering. Skipping batch.")
            return []

        num_frames = [video.shape[0] for video in videos]
        images = torch.cat(videos, dim=0)
        features = self.parent.encode_images(images)
        features = torch.split(features, num_frames)
        process_features = partial(
            self._process_features,
            start_token_embeds=self.embed_tokens(self.start_tokens),
            end_token_embeds=self.embed_tokens(self.end_tokens),
        )
        return [process_features(f) for f in features]
