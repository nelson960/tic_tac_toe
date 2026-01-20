"""
Tiny Transformer model for TicTacToe.

Architecture:
- Token embeddings (3 tokens: empty/self/opp)
- Positional embeddings (9 positions)
- Transformer encoder
- Policy head (9 logits)
- Value head (scalar)
"""

import torch
import torch.nn as nn
from typing import Tuple


class TinyTTTTransformer(nn.Module):
    """
    Tiny transformer for TicTacToe policy-value learning.

    Token encoding (perspective):
    - 0: empty square
    - 1: self piece
    - 2: opponent piece
    """

    def __init__(
        self,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        ff_mult: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers

        # Embeddings
        self.token_emb = nn.Embedding(3, d_model)
        self.pos_emb = nn.Embedding(9, d_model)
        self.dropout = nn.Dropout(dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * ff_mult,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-LN architecture
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Heads
        self.policy_head = nn.Linear(d_model, 9)
        self.value_head = nn.Linear(d_model, 1)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)

    def forward(
        self, tokens: torch.LongTensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            tokens: [B, 9] LongTensor of token IDs (0/1/2)

        Returns:
            logits: [B, 9] policy logits
            value: [B] value predictions (tanh-scaled)
        """
        B = tokens.size(0)
        device = tokens.device

        # Positions
        positions = torch.arange(9, device=device).unsqueeze(0).expand(B, 9)

        # Embeddings
        x = self.token_emb(tokens) + self.pos_emb(positions)
        x = self.dropout(x)

        # Encode
        x = self.encoder(x)

        # Pool (mean over sequence)
        pooled = x.mean(dim=1)

        # Heads
        logits = self.policy_head(pooled)  # [B, 9]
        value = torch.tanh(self.value_head(pooled)).squeeze(-1)  # [B]

        return logits, value

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def board_to_tokens_perspective(board: list, player: int) -> torch.LongTensor:
    """
    Convert board to perspective-relative tokens.

    Args:
        board: [9] list with values in {-1, 0, +1}
        player: Current player (+1 or -1)

    Returns:
        [9] LongTensor with tokens in {0, 1, 2}
    """
    toks = []
    for v in board:
        pv = v * player  # Perspective transform
        if pv == 0:
            toks.append(0)    # empty
        elif pv == 1:
            toks.append(1)    # self
        else:
            toks.append(2)    # opponent
    return torch.tensor(toks, dtype=torch.long)


def legal_move_mask(board: list) -> torch.BoolTensor:
    """Return [9] boolean mask of legal moves."""
    mask = torch.zeros(9, dtype=torch.bool)
    for i, v in enumerate(board):
        if v == 0:
            mask[i] = True
    return mask
