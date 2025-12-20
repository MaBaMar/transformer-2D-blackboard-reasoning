"""
gptbase.py

Implementation of a basic GPT like model for CoT baselines.
"""

import torch
import torch.nn as nn

from logging import getLogger

from projectlib.transformer.tpe2d_model import FeedForward, OutputHead
from projectlib.wrappertypes import EndOfComputationException

class CoTBaseline(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        d_model: int,
        num_heads: int,
        num_blocks: int,
        pad_id: int,
        eos_id: int,
        dropout: float = 0.1,
        embedding_dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)    # token embedding
        self.pos_emb = nn.Embedding(max_seq_len, d_model)    # additive positional embedding

        self.blocks = nn.ModuleList([
            self._TransformerBlock(d_model, num_heads, dropout)
            for _ in range(num_blocks)
        ])

        self.dropout = nn.Dropout(dropout)
        self.embedding_dropout = nn.Dropout(embedding_dropout)

        self.head = OutputHead(vocab_size, d_model, pad_id)
        self.max_seq_len = max_seq_len
        self._eos_id = eos_id
        self._pad_id = pad_id
        self.model_logger = getLogger(__name__)

    def forward(self, x: torch.Tensor, y: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        assert x.ndim == 2, f"input must be 2D tensor, got {x.ndim}D"
        L = x.shape[1]

        # embedding (+ some random embedding dropout)
        x = self.tok_emb(x) + self.pos_emb(torch.arange(L, device=x.device).unsqueeze(0))
        x = self.embedding_dropout(x)

        # pass through transformer blocks
        for block in self.blocks:
            x = block(x)

        logits, loss = self.head(context=x, targets=y)
        return logits, loss

    @torch.inference_mode()
    def batch_inference(self, x: torch.Tensor) -> torch.Tensor:

        is_active = torch.ones(x.shape[0], dtype=torch.bool, device=x.device)

        x_new: torch.Tensor = x.clone() # Note: this may be replaced with reference copying, if we do not need to be careful with reference semantics here

        while is_active.any():
            next_tokens = self.forward(x_new[is_active])[0].argmax(dim=1) # greedily select the next token
            is_active[is_active][~self._find_and_append_inplace(x_new[is_active], next_tokens)] = False

        # for debugging, check if all sequences are finished
        if (cnt := (x_new == self._eos_id).any(dim=1).sum()) != x_new.shape[0]:
            self.model_logger.debug(f"Not all computation finished in an EOS token, {cnt} outputs reached maxlen without generating an EOS token")
        else:
            self.model_logger.debug("All computations in batch completed sucessfully")

        return x_new

    def _find_and_append_inplace(self, curr_inputs: torch.Tensor, next_token: torch.Tensor) -> torch.Tensor:
        # find index of first padding token for each sample
        pad_mask = curr_inputs == self._pad_id
        has_eos = (curr_inputs == self._eos_id).any(dim=1)

        can_be_appended = pad_mask.any(dim=1) & ~has_eos

        replace_targets = pad_mask[can_be_appended].to(torch.long).argmax(dim=1, keepdim=True)
        curr_inputs[can_be_appended][replace_targets] = next_token[can_be_appended]

        return can_be_appended

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
            L = x.shape[1]

            h = self.ln1(x)

            # do masked multihead attention:
            causal_mask = torch.triu(
                torch.ones((L, L), device=x.device, dtype=torch.bool),
                diagonal=1
            )
            h, _ = self.attn(h, h, h, attn_mask=causal_mask, need_weights=False)
            x = x + self.dropout(h)

            h2 = self.ln2(x)
            h2 = self.ffn(h2)
            x = x + self.dropout(h2)

            return x
