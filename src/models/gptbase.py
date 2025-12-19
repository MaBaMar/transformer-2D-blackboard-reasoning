"""
gptbase.py

Implementation of a basic GPT like model for CoT baselines.
"""

import torch
import torch.nn as nn

from projectlib.transformer.tpe2d_model import FeedForward, OutputHead

class CoTBaseline(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_heads: int,
        num_blocks: int,
        pad_id: int,
        dropout: float = 0.1,
        embedding_dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)    # token embedding
        self.pos_emb = nn.Embedding(vocab_size, d_model)    # additive positional embedding

        self.blocks = nn.ModuleList([
            self._TransformerBlock(d_model, num_heads, dropout)
            for _ in range(num_blocks)
        ])

        self.dropout = nn.Dropout(dropout)
        self.embedding_dropout = nn.Dropout(embedding_dropout)

        self.head = OutputHead(vocab_size, d_model, pad_id)

    def forward(self, x: torch.Tensor, y: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        assert x.ndim == 2, "input must be 2D tensor"
        L = x.shape[1]

        # embedding (+ some random embedding dropout)
        x = self.tok_emb(x) + self.pos_emb(torch.arange(L, device=x.device)[None, ...])
        x = self.embedding_dropout(x)

        # pass through transformer blocks
        for block in self.blocks:
            x = block(x)

        logits, loss = self.head(context=x, target=y)
        return logits, loss

    # -----------------------------------------
    # private internal classes/building blocks
    # -----------------------------------------
    class _TransformerBlock(nn.Module):
        def __init__(
            self,
            d_model: int,
            num_heads: int,
            dropout: float,
        ):
            super().__init__()
            assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

            self.ln1 = nn.LayerNorm(d_model)
            self.ln2 = nn.LayerNorm(d_model)
            self.dropout = nn.Dropout(dropout)
            self.attn: nn.MultiheadAttention = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
            self.ffn = FeedForward(d_model, 4 * d_model, dropout)

        def forward(
            self,
            x: torch.Tensor,
        ):
            assert x.ndim == 2, "input must be 2D tensor"
            B, L = x.shape

            h = self.ln1(x)

            # do masked multihead attention:
            causal_mask = torch.triu(
                torch.ones((L, L), device=x.device, dtype=torch.bool),
                diagonal=1
            )
            h, _ = self.attn(h, h, h, attn_mask=causal_mask)
            x = x + h

            h2 = self.ln2(x)
            h2 = self.ffn(h2)
            x = x + self.dropout(h2)

            return x
