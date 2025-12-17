# ------------------------------------------------------------
# eogar.py
#
# Encoder-Only Generative Algorithmic Reasoner
#
# Purpose: Full model implementation based on 2D RoPE + routing,
# and (optionally) entropy regularization on the router.
#
# Key design choice:
#   - We DO NOT change the training-loop interface.
#   - EOgar.forward still returns (logits, loss).
#   - If entropy_coef > 0, we add the entropy penalty internally:
#       loss_total = loss_ce + entropy_coef * loss_entropy
#   - (Optional) for logging, we expose:
#       self.last_ent_loss  (None if not computed)
# ------------------------------------------------------------

from __future__ import annotations

from typing import Optional, final, Tuple, List

import torch
from torch import nn
import torch.nn.functional as F

from projectlib.transformer.tpe2d_model import TwoDTPERoPEAttention
from projectlib.wrappertypes import BBChainGenerator


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
        super().__init__()

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.dropout = nn.Dropout(dropout)

        self.transformer_blocks = nn.ModuleList([
            self._EncoderBlock(d_model, num_heads, dropout)
            for _ in range(num_blocks)
        ])

    def forward(
        self,
        input_ids: torch.Tensor,
        pos_row: torch.Tensor,
        pos_col: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        collect_entropy: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            input_ids: [B,L]
            pos_row, pos_col: [B,L]
            key_padding_mask: [B,L] (True where padded)
            collect_entropy: if True, aggregate entropy losses from blocks (training only)

        Returns:
            context: [B,L,d_model]
            ent_loss_total: scalar tensor if collected, else None
        """
        x = self.tok_emb(input_ids)
        x = self.dropout(x)

        ent_losses: List[torch.Tensor] = []

        for block in self.transformer_blocks:
            x, ent_l = block(
                x,
                pos_row=pos_row,
                pos_col=pos_col,
                key_padding_mask=key_padding_mask,
            )
            if collect_entropy and (ent_l is not None):
                ent_losses.append(ent_l)

        ent_loss_total: Optional[torch.Tensor] = None
        if collect_entropy and len(ent_losses) > 0:
            # average across blocks to keep scale stable as you change depth
            ent_loss_total = torch.stack(ent_losses).mean()

        return x, ent_loss_total

    class _EncoderBlock(nn.Module):
        def __init__(
            self,
            d_model: int,
            num_heads: int,
            dropout: float = 0.1,
        ) -> None:
            super().__init__()

            self.ln1 = nn.LayerNorm(d_model)
            self.ln2 = nn.LayerNorm(d_model)
            self.dropout = nn.Dropout(dropout)

            # IMPORTANT: TwoDTPERoPEAttention now returns (out, ent_loss)
            self.attn = TwoDTPERoPEAttention(d_model, num_heads, dropout, use_causal_mask=False)
            self.ffn = FeedForward(d_model, 4 * d_model, dropout)

        def forward(
            self,
            x: torch.Tensor,
            pos_row: torch.Tensor,
            pos_col: torch.Tensor,
            key_padding_mask: torch.Tensor | None = None,
        ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
            # pre-norm + attention
            h = self.ln1(x)
            h_attn, ent_loss = self.attn(
                h,
                pos_row,
                pos_col,
                key_padding_mask=key_padding_mask,
            )
            x = x + self.dropout(h_attn)

            # pre-norm + FFN
            h2 = self.ln2(x)
            h2 = self.ffn(h2)
            x = x + self.dropout(h2)

            return x, ent_loss


class Head(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        pad_id: int,
    ) -> None:
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.pad_id = pad_id

    def forward(
        self,
        context: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        h = self.ln(context)
        logits = self.head(h)

        loss: Optional[torch.Tensor] = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=self.pad_id,
            )

        return logits, loss


class EOgar(BBChainGenerator):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_heads_encoder: int,
        n_encoder_blocks: int,
        pad_id: int,
        rope_mode: str = "2d",
        entropy_coef: float = 0.0,
    ) -> None:
        """
        Args:
            entropy_coef:
                λ in  L_total = L_CE + λ * L_entropy.
                Set to 0.0 to disable entropy regularization.
        """
        super().__init__()

        self.pad_id = pad_id
        self.rope_mode = rope_mode
        self.entropy_coef = float(entropy_coef)

        # for logging without changing the training loop
        self.last_ent_loss: Optional[torch.Tensor] = None

        self.encoder = Encoder(
            vocab_size=vocab_size,
            d_model=d_model,
            num_heads=num_heads_encoder,
            num_blocks=n_encoder_blocks,
        )

        self.head = Head(
            vocab_size=vocab_size,
            d_model=d_model,
            pad_id=pad_id,
        )

    def forward(
        self,
        x: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        y: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Returns:
            logits, loss_total
        """

        # Unpack input
        x_tokens: torch.Tensor = x[0]
        x_pos_row: torch.Tensor = x[1]
        x_pos_col: torch.Tensor = x[2]
        x_key_padding_mask: torch.Tensor = x[3]

        # Unpack target
        y_tokens: torch.Tensor = y[0]
        y_pos_row: torch.Tensor = y[1]
        y_pos_col: torch.Tensor = y[2]
        y_key_padding_mask: torch.Tensor = y[3]  # currently unused; kept for API symmetry

        # Ensure 1D mode positions have correct batch shape
        if self.rope_mode == "1d":
            Bx, Lx = x_tokens.shape
            seq_x = torch.arange(Lx, device=x_tokens.device).unsqueeze(0).expand(Bx, -1)
            x_pos_row = seq_x
            x_pos_col = seq_x

            By, Ly = y_tokens.shape
            seq_y = torch.arange(Ly, device=y_tokens.device).unsqueeze(0).expand(By, -1)
            y_pos_row = seq_y
            y_pos_col = seq_y

        collect_entropy = bool(self.training and (self.entropy_coef > 0.0) and (y_tokens is not None))

        # IMPORTANT: pass key_padding_mask so attention + entropy masking are correct
        context, ent_loss_total = self.encoder(
            input_ids=x_tokens,
            pos_row=x_pos_row,
            pos_col=x_pos_col,
            key_padding_mask=x_key_padding_mask,
            collect_entropy=collect_entropy,
        )

        logits, ce_loss = self.head(
            context=context,
            targets=y_tokens,
        )

        # default: keep old behavior
        loss_total = ce_loss

        # Add entropy regularization internally (NO change to return signature)
        self.last_ent_loss = None
        if ce_loss is not None and (self.entropy_coef > 0.0) and (ent_loss_total is not None):
            loss_total = ce_loss + (self.entropy_coef * ent_loss_total)
            self.last_ent_loss = ent_loss_total.detach()

        return logits, loss_total

    @torch.no_grad()
    def next_state(
        self,
        x: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """
        Greedy next-state prediction from the encoder outputs.
        """

        x_tokens: torch.Tensor = x[0]
        x_pos_row: torch.Tensor = x[1]
        x_pos_col: torch.Tensor = x[2]
        x_key_padding_mask: torch.Tensor = x[3]

        if self.rope_mode == "1d":
            B, L = x_tokens.shape
            seq = torch.arange(L, device=x_tokens.device).unsqueeze(0).expand(B, -1)
            x_pos_row = seq
            x_pos_col = seq

        # No entropy collection during inference
        context, _ = self.encoder(
            input_ids=x_tokens,
            pos_row=x_pos_row,
            pos_col=x_pos_col,
            key_padding_mask=x_key_padding_mask,
            collect_entropy=False,
        )

        logits, _ = self.head(context=context)
        out_tokens = logits.argmax(dim=-1)
        return out_tokens

    @staticmethod
    def load_from_path(model_path: str) -> "EOgar":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        checkpoint = torch.load(model_path, map_location=device)
        config = checkpoint["config"]

        model = EOgar(
            vocab_size=config["vocab_size"],
            d_model=config["d_model"],
            num_heads_encoder=config["num_heads_encoder"],
            n_encoder_blocks=config["n_encoder_blocks"],
            pad_id=config["pad_id"],
            rope_mode=config.get("rope_mode", "2d"),
            entropy_coef=config.get("entropy_coef", 0.0),  # backward compatible
        )

        model.load_state_dict(checkpoint["model_state_dict"])
        return model
