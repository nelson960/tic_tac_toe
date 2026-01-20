"""
Replay buffers for training.

Stores training samples: (tokens, legal_mask, pi_star, v_star)
"""

import random
from collections import deque
from typing import Tuple

import torch


class PositionReplay:
    """
    Replay buffer for position-level training data.

    Each entry: (tokens[9], legal_mask[9], pi_star[9], v_star)
    """

    def __init__(self, max_items: int = 200_000):
        self.buf = deque(maxlen=max_items)

    def __len__(self) -> int:
        return len(self.buf)

    def add(
        self,
        tokens: torch.Tensor,
        legal_mask: torch.Tensor,
        pi_star: torch.Tensor,
        v_star: torch.Tensor,
    ):
        """Add a training sample."""
        self.buf.append((
            tokens.cpu(),
            legal_mask.cpu(),
            pi_star.cpu(),
            float(v_star.item()) if isinstance(v_star, torch.Tensor) else float(v_star),
        ))

    def sample(
        self, batch_size: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample a batch.

        Returns:
            tokens: [B, 9]
            legal_mask: [B, 9]
            pi_star: [B, 9]
            v_star: [B]
        """
        batch_size = min(batch_size, len(self.buf))
        items = random.sample(self.buf, batch_size)

        tokens = torch.stack([it[0] for it in items], dim=0).to(device)
        masks = torch.stack([it[1] for it in items], dim=0).to(device)
        pi = torch.stack([it[2] for it in items], dim=0).to(device)
        v = torch.tensor([it[3] for it in items], dtype=torch.float32, device=device)

        return tokens, masks, pi, v

    def is_full(self) -> bool:
        """Check if buffer is at capacity."""
        return len(self.buf) == self.buf.maxlen


class HardNegativeReplay(PositionReplay):
    """
    Specialized replay for hard negatives (loss-induced samples).

    Typically populated from positions where model made mistakes.
    """

    def __init__(self, max_items: int = 50_000):
        super().__init__(max_items)
