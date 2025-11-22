# ------------------------------------------------------------
# edgar.py
#
# Encoder-Decoder Generative Algorithmic Reasoner
#
# Purpose: Full model implementation based on 2D rope and autoregressive reasoning step
# generation for additions and subtractions
#
# Name: Toy name, feel free to change it to sth fancy
# ------------------------------------------------------------

from typing import final
import torch
import math
from torch import nn, t
import torch.nn.functional as F

from projectlib.transformer.tpe2d_model import TwoDTPERoPEAttention

class FeedForward(nn.Module):
    def __init__(
        self,
        d_model: int,
        hidden_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_model, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


@final  # affects typechecker only
class Encoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_heads: int,
        num_blocks: int,
        dropout: float = 0.1,
    ) -> None:
        super(nn.Module, self).__init__()

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model

        self.transformer_blocks = nn.ModuleList([
            self._EncoderBlock(d_model, num_heads, dropout)
            for _ in range(num_blocks)
        ])

    def forward(
        self,
        input_ids: torch.Tensor,
        pos_row: torch.Tensor,
        pos_col: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        input_ids: [B,L]
        pos_row, pos_col: [B,L]
        key_padding_mask: [B,L] (True where padded)
        targets: [B,L] or None
        """

        x = self.tok_emb(input_ids) * math.sqrt(self.d_model)
        x = self.dropout(x)

        for layer in self.transformer_blocks:
            x = layer(
                x,
                pos_row=pos_row,
                pos_col=pos_col,
                key_padding_mask=key_padding_mask,
            )

        return x

    # private helper block
    class _EncoderBlock(nn.Module):
        def __init__(
            self,
            d_model: int,
            num_heads: int,
            dropout: float = 0.1,
        ) -> None:
            super(nn.Module, self).__init__()

            # building blocks
            self.ln1 = nn.LayerNorm(d_model)
            self.ln2 = nn.LayerNorm(d_model)
            self.dropout = nn.Dropout(dropout)
            self.attn = TwoDTPERoPEAttention(d_model, num_heads, dropout, use_causal_mask=False)
            self.ffn = FeedForward(d_model, 4 * d_model, dropout)

        def forward(
            self,
            x: torch.Tensor,
            pos_row: torch.Tensor,
            pos_col: torch.Tensor,
            key_padding_mask: torch.Tensor | None = None,
        ) -> torch.Tensor:
            # pre-norm + attention
            h = self.ln1(x)
            h = self.attn(h, pos_row, pos_col, key_padding_mask)
            x = x + self.dropout(h)

            # pre-norm + FFN
            h2 = self.ln2(x)
            h2 = self.ffn(h2)
            x = x + self.dropout(h2)
            return x
