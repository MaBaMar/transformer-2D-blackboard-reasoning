# ------------------------------------------------------------
# 2d_tpe_transformer.py
#
# Minimal GPT-style Transformer with 2D TPE (two 1D RoPE orders
# + per-head router), inspired by the "2D-TPE" mechanism.
#
# Focus:
#   - Implement 2D-TPE logic in PyTorch, without LLaMA/HF deps.
#   - Simple toy dataset so the file runs end-to-end.
#
# Assumptions about input:
#   - We get:
#       input_ids: [B, L]
#       pos_row:   [B, L]  (row-wise / order-1 indices)
#       pos_col:   [B, L]  (col-wise / order-2 indices)
#
#   - pos_row and pos_col can be any integer position indices
#     that encode different meaningful orders over the same
#     tokens (e.g. row-major vs column-major traversal of a table).
#
# ------------------------------------------------------------

# NOTE: Entropy regularization is missing.
# NOTE: Data generation is basic. Paper used Question/Table/Answer formatting and flattens it, which is not implemented here. (but easy to add)


import math
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------------------------------------------------
# RoPE utilities
# -------------------------------------------------------------------

def build_rope_cache(
    head_dim: int,
    max_position: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build cos/sin caches for RoPE:
        cos, sin: [1, 1, max_position, head_dim]
    """
    # half-dim frequencies
    theta = torch.arange(0, head_dim, 2, device=device, dtype=torch.float32)
    theta = 1.0 / (10000 ** (theta / head_dim))  # [head_dim/2]

    seq_pos = torch.arange(max_position, device=device, dtype=torch.float32)  # [max_position]
    freqs = torch.einsum("i,j->ij", seq_pos, theta)  # [max_position, head_dim/2]

    emb = torch.cat([freqs, freqs], dim=-1)  # [max_position, head_dim]
    cos = emb.cos().unsqueeze(0).unsqueeze(0)  # [1,1,max_position,head_dim]
    sin = emb.sin().unsqueeze(0).unsqueeze(0)
    return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    RoPE helper: rotate pairs (x0,x1) -> (-x1,x0)
    """
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    x_rot = torch.stack([-x2, x1], dim=-1)
    return x_rot.flatten(-2)


def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    positions: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary position embedding on q and k given integer positions.

    q, k: [B, H, L, D]
    positions: [B, L] (integer indices in [0, max_position))
    cos, sin: [1,1,max_position,D]
    """
    B, H, L, D = q.shape

    # gather cos/sin per token position
    # positions_expanded: [B, 1, L, 1] -> broadcast to [B, H, L, D]
    pos = positions.unsqueeze(1).unsqueeze(-1).expand(B, H, L, D)
    cos_pos = torch.gather(
        cos.expand(B, H, -1, -1),  # [B,H,max_position,D]
        2,
        pos,
    )
    sin_pos = torch.gather(
        sin.expand(B, H, -1, -1),
        2,
        pos,
    )

    q_out = (q * cos_pos) + (rotate_half(q) * sin_pos)
    k_out = (k * cos_pos) + (rotate_half(k) * sin_pos)
    return q_out, k_out


# -------------------------------------------------------------------
# Core 2D-TPE Attention
# -------------------------------------------------------------------

class TwoDTPERoPEAttention(nn.Module):
    """
    Multi-head causal self-attention with 2D-TPE:

      - We have two 1D position sequences per token:
          pos_row: order-1 (e.g. row-major)
          pos_col: order-2 (e.g. column-major)
      - We apply RoPE twice to Q,K:
          (q_row,k_row) with pos_row
          (q_col,k_col) with pos_col
      - We run attention twice (one per order) with causal masks
        defined over each order (via sorting by positions).
      - We then mix the two outputs with a learned per-head
        router that gives us weights r_row, r_col for each
        (token, head).
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)

        self.attn_dropout = nn.Dropout(dropout)

        # Router MLP: per-head, per-token mixing between row/col.
        # We apply it to head-local queries (before RoPE).
        router_hidden = 4 * self.head_dim
        self.router_up = nn.Linear(self.head_dim, router_hidden)
        self.router_down = nn.Linear(router_hidden, 2)  # 2 orders: row & col

    def _attend_with_order(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        positions: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Run causal self-attention for a single order.

        q,k,v: [B, H, L, D]
        positions: [B, L] integer order indices
        key_padding_mask: [B, L] bool (True where padded)
        """
        B, H, L, D = q.shape
        device = q.device

        # sort tokens according to positions (each batch separately)
        # positions_sorted, sort_idx: [B, L]
        _, sort_idx = torch.sort(positions, dim=-1, stable=True)

        # gather q,k,v in sorted order
        gather_idx = sort_idx.unsqueeze(1).unsqueeze(-1).expand(B, H, L, D)
        q_sorted = torch.gather(q, 2, gather_idx)
        k_sorted = torch.gather(k, 2, gather_idx)
        v_sorted = torch.gather(v, 2, gather_idx)

        # sort padding mask the same way
        if key_padding_mask is not None:
            # [B, L]
            kpm_sorted = torch.gather(key_padding_mask, 1, sort_idx)
        else:
            kpm_sorted = None

        # scaled dot-product attention
        attn_scores = torch.matmul(
            q_sorted, k_sorted.transpose(-2, -1)
        ) / math.sqrt(D)  # [B,H,L,L]

        # causal mask (lower-triangular)
        causal = torch.triu(
            torch.ones((L, L), device=device, dtype=torch.bool),
            diagonal=1,
        )  # [L,L]
        causal = causal.unsqueeze(0).unsqueeze(0)  # [1,1,L,L]

        # key padding mask: mask out padded positions as keys
        if kpm_sorted is not None:
            # [B,1,1,L]
            kpm_mask = kpm_sorted.unsqueeze(1).unsqueeze(2)
            attn_mask = causal | kpm_mask
        else:
            attn_mask = causal

        attn_scores = attn_scores.masked_fill(attn_mask, float("-inf"))
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        out_sorted = torch.matmul(attn_weights, v_sorted)  # [B,H,L,D]

        # invert the sort to map back to original token positions
        inv_idx = torch.argsort(sort_idx, dim=-1)
        inv_gather_idx = inv_idx.unsqueeze(1).unsqueeze(-1).expand(B, H, L, D)
        out = torch.gather(out_sorted, 2, inv_gather_idx)  # [B,H,L,D]

        return out

    def forward(
        self,
        hidden_states: torch.Tensor,
        pos_row: torch.Tensor,
        pos_col: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        hidden_states: [B, L, d_model]
        pos_row, pos_col: [B, L] integer indices
        key_padding_mask: [B, L] (True where padded)
        """
        B, L, _ = hidden_states.shape
        device = hidden_states.device

        # project to q,k,v
        q = self.q_proj(hidden_states)  # [B,L,D]
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # reshape to [B, H, L, D_head]
        q = q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        # build RoPE cache up to max position we see in either order
        max_pos = int(
            max(
                pos_row.max().item(),
                pos_col.max().item(),
            )
        ) + 1
        cos, sin = build_rope_cache(
            head_dim=self.head_dim,
            max_position=max_pos,
            device=device,
        )

        # Apply RoPE for row order
        q_row, k_row = apply_rope(q, k, pos_row, cos, sin)  # [B,H,L,D]
        # Apply RoPE for col order
        q_col, k_col = apply_rope(q, k, pos_col, cos, sin)

        # attention for each order
        out_row = self._attend_with_order(
            q_row, k_row, v, pos_row, key_padding_mask
        )  # [B,H,L,D]
        out_col = self._attend_with_order(
            q_col, k_col, v, pos_col, key_padding_mask
        )  # [B,H,L,D]

        # router: per-head, per-token logits over {row, col}
        # use q (before RoPE) as input
        # q: [B,H,L,D_head]
        router_h = F.silu(self.router_up(q))            # [B,H,L,4*D_head]
        router_logits = self.router_down(router_h)      # [B,H,L,2]
        router_weights = F.softmax(router_logits, dim=-1)

        w_row = router_weights[..., 0].unsqueeze(-1)    # [B,H,L,1]
        w_col = router_weights[..., 1].unsqueeze(-1)

        out_heads = w_row * out_row + w_col * out_col   # [B,H,L,D_head]

        # merge heads
        out = out_heads.transpose(1, 2).contiguous().view(B, L, self.d_model)
        out = self.o_proj(out)  # [B,L,D_model]
        return out



# -------------------------------------------------------------------
# Transformer block
# -------------------------------------------------------------------

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


class TransformerBlock2DTPE(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = TwoDTPERoPEAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(
            d_model=d_model,
            hidden_dim=4 * d_model,
            dropout=dropout,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        pos_row: torch.Tensor,
        pos_col: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # pre-norm + attention
        h = self.ln1(x)
        h = self.attn(h, pos_row=pos_row, pos_col=pos_col,
                      key_padding_mask=key_padding_mask)
        x = x + self.dropout(h)

        # pre-norm + FFN
        h2 = self.ln2(x)
        h2 = self.ffn(h2)
        x = x + self.dropout(h2)
        return x


# -------------------------------------------------------------------
# Causal Transformer with 2D-TPE attention
# -------------------------------------------------------------------

class CausalTransformer2DTPE(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.drop = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            TransformerBlock2DTPE(
                d_model=d_model,
                num_heads=num_heads,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        pos_row: torch.Tensor,
        pos_col: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        input_ids: [B,L]
        pos_row, pos_col: [B,L]
        key_padding_mask: [B,L] (True where padded)
        targets: [B,L] or None
        """

        x = self.tok_emb(input_ids) * math.sqrt(self.d_model)
        x = self.drop(x)

        for layer in self.layers:
            x = layer(
                x,
                pos_row=pos_row,
                pos_col=pos_col,
                key_padding_mask=key_padding_mask,
            )

        x = self.ln_f(x)
        logits = self.head(x)  # [B,L,vocab]

        loss: Optional[torch.Tensor] = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-100,
            )
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        pos_row: torch.Tensor, 
        pos_col: torch.Tensor,
        max_new_tokens: int,
        eos_id: Optional[int] = None,
        pad_id: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Very simple greedy generation just to show end-to-end.
        We assume pos_row/pos_col for new tokens are continued
        monotonically from the last token.
        """
        self.eval()
        device = input_ids.device
        B, L = input_ids.shape
        out_ids = input_ids.clone()
        row = pos_row.clone()
        col = pos_col.clone()

        if pad_id is None:
            pad_id = 0

        for _ in range(max_new_tokens):
            key_padding_mask = (out_ids == pad_id)
            logits, _ = self.forward(
                out_ids,
                pos_row=row,
                pos_col=col,
                key_padding_mask=key_padding_mask,
                targets=None,
            )
            next_token = logits[:, -1].argmax(dim=-1)  # [B]
            out_ids = torch.cat([out_ids, next_token.unsqueeze(-1)], dim=-1)

            # simple continuation of orders: just +1 from last
            last_row = row[:, -1]
            last_col = col[:, -1]
            row = torch.cat([row, (last_row + 1).unsqueeze(-1)], dim=-1)
            col = torch.cat([col, (last_col + 1).unsqueeze(-1)], dim=-1)

            if eos_id is not None and bool((next_token == eos_id).all()):
                break

        return out_ids
