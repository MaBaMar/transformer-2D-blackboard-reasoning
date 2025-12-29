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

# NOTE: Entropy regularization is implemented (enable via entropy_coef).
# NOTE: Data generation is basic. Paper used Question/Table/Answer formatting and flattens it, which is not implemented here. (but easy to add)

import math
from typing import Tuple, Optional, final

import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------------------------------------------------
# RoPE utilities
# -------------------------------------------------------------------
@final  # affects the typechecker only
class RoPECache(nn.Module):
    """
    Cache cos/sin tables for RoPE as buffers, similar in spirit to
    LLaMA's rotary embedding implementation.

    Produces cos, sin of shape [1, 1, max_position, head_dim].
    """

    def __init__(
        self,
        head_dim: int,
        base: float = 10000.0,
        max_position_embeddings: int = 2048,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.base = base
        self.max_seq_len_cached: int = 0

        # inverse frequencies (never change)
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim)
        )
        # non-trainable, moves with the module on .to(...)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # placeholders for caches; remain registered buffers even when reassigned
        self.register_buffer("cos_cached", torch.empty(0), persistent=False)
        self.register_buffer("sin_cached", torch.empty(0), persistent=False)

        # build an initial cache; will be extended on demand
        if max_position_embeddings > 0:
            self._set_cos_sin_cache(
                max_position=max_position_embeddings,
                dtype=torch.get_default_dtype(),
            )

    def _set_cos_sin_cache(
        self,
        max_position: int,
        dtype: torch.dtype,
    ) -> None:
        # always build on the same device as inv_freq
        device = self.inv_freq.device
        self.max_seq_len_cached = max_position

        # positions 0, 1, ..., max_position-1
        t = torch.arange(max_position, device=device, dtype=self.inv_freq.dtype)  # [max_position]
        # couldn't get rid of type problems here
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)  # [max_position, head_dim/2]
        emb = torch.repeat_interleave(freqs, 2, dim=-1)  # [max_position, head_dim]

        cos = emb.cos()[None, None, :, :].to(dtype)        # [1,1,max_position,head_dim]
        sin = emb.sin()[None, None, :, :].to(dtype)

        # update the registered buffers
        self.cos_cached = cos
        self.sin_cached = sin

    def forward(
        self,
        max_position: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Ensure cache covers [0, ..., max_position-1] and return cos/sin slices.
        """
        if max_position > self.max_seq_len_cached:
            self._set_cos_sin_cache(
                max_position=max_position,
                dtype=dtype,
            )

        # slice to requested range and match dtype/device
        cos = self.cos_cached[:, :, :max_position, :].to(device=device, dtype=dtype)
        sin = self.sin_cached[:, :, :max_position, :].to(device=device, dtype=dtype)
        return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    RoPE helper: rotate pairs (x0,x1) -> (-x1,x0)
    There are two equivalent ways to implement rotary position embeddings.

    (1) The paper's formulation views the last dim as interleaved pairs
        (x_0, x_1), (x_2, x_3), … and rotates each pair using
        x_even = x[..., 0::2], x_odd = x[..., 1::2].

    (2) LLaMA / HuggingFace instead store the "real" and "imag" parts in
        two contiguous halves:
            cos/sin cache: emb = cat([freqs, freqs], dim=-1)
        so that each complex pair is (x[..., :d/2], x[..., d/2:]).

    The function below implements (1). To implement (2), we would do:
    d = x.shape[-1]
    x1 = x[..., : d // 2]      # "real" part
    x2 = x[..., d // 2 :]      # "imag" part
    return torch.cat((-x2, x1), dim=-1)

    NOTE: I am not sure yet which is more efficient in practice. -> Something we need to check
    """
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    x_rot = torch.stack([-x2, x1], dim=-1)
    return x_rot.flatten(-2)


def apply_rope(
    qk_tensor: torch.Tensor,
    positions: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """
    Apply rotary position embedding on q and k given integer positions.

    qk_tensor: [B, H, L, D]
    positions: [B, L] (integer indices in [0, max_position))
    cos, sin: [1,1,max_position,D]
    """
    B, H, L, D = qk_tensor.shape

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

    qk_out = (qk_tensor * cos_pos) + (rotate_half(qk_tensor) * sin_pos)
    return qk_out


# -------------------------------------------------------------------
# Core 2D-TPE Attention
# -------------------------------------------------------------------
@final  # affects typecheckers only
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
      - We support causal masks for decoders or no causal masks for encoders.
    """

    # Router MLP: per-head, per-token mixing between row/col.
    # We apply it to a per-head slice of the input hidden state.

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.0,
        use_causal_mask: bool = True,
    ) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.use_causal_mask = use_causal_mask
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

        # RoPE cache shared by all heads in this attention layer
        self.rope_cache = RoPECache(
            head_dim=self.head_dim,
            max_position_embeddings=512,  # can grow on demand
        )

    def _attend_with_order(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        q_positions: torch.Tensor,
        k_positions: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Run causal self-attention for a single order.

        q,k,v: [B, H, L, D]
        q_positions: [B, L] integer order indices
        k_positions: [B, L] integer order indices
        key_padding_mask: [B, L] bool (True where padded)
        """
        B, H, L_q, D = q.shape
        _, _, L_k, _ = k.shape
        device = q.device

        # sort tokens according to positions (each batch separately)
        # q_positions_sorted, sort_idx: [B, L_q]
        _, q_sort_idx = torch.sort(q_positions, dim=-1, stable=True)
        q_gather_idx = q_sort_idx.unsqueeze(1).unsqueeze(-1).expand(B, H, L_q, D)
        q_sorted = torch.gather(q, 2, q_gather_idx)
        # gather q,k,v in sorted order
        _, k_sort_idx = torch.sort(k_positions, dim=-1, stable=True)
        k_gather_idx = k_sort_idx.unsqueeze(1).unsqueeze(-1).expand(B, H, L_k, D)
        k_sorted = torch.gather(k, 2, k_gather_idx)
        v_sorted = torch.gather(v, 2, k_gather_idx)

        # scaled dot-product attention
        attn_scores = torch.matmul(
            q_sorted, k_sorted.transpose(-2, -1)
        ) / math.sqrt(D)  # [B,H,L_q,L_k]

        # generate fill mask for attention scores
        if key_padding_mask is not None:
            # sort padding mask the same way as key and query
            attn_mask = torch.gather(key_padding_mask, 1, k_sort_idx).unsqueeze(1).unsqueeze(2).bool() # [B,1,1,L_k]
        else:
            attn_mask = torch.zeros((B, 1, 1, L_k), device=device, dtype=torch.bool)

        if self.use_causal_mask:
            # time indices (represents the time of token generation) for causal masking in a row-major fashion
            q_time_raw = torch.arange(L_q, device=device).unsqueeze(0).expand(B, -1)
            k_time_raw = torch.arange(L_k, device=device).unsqueeze(0).expand(B, -1)

            q_time_sorted = torch.gather(q_time_raw, 1, q_sort_idx)[:, None, :, None]  # [B,1,L_q,1]
            k_time_sorted = torch.gather(k_time_raw, 1, k_sort_idx)[:, None, None, :]   # [B,1,1,L_k]

            causal_mask = q_time_sorted < k_time_sorted # [B,1,L_q,L_k] (true = mask it)

            attn_mask = attn_mask | causal_mask

        attn_scores = attn_scores.masked_fill(attn_mask, float("-inf"))
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        out_sorted = torch.matmul(attn_weights, v_sorted)  # [B,H,L_q,D]

        # invert the sort to map back to original token positions
        inv_idx = torch.argsort(q_sort_idx, dim=-1)
        inv_gather_idx = inv_idx.unsqueeze(1).unsqueeze(-1).expand(B, H, L_q, D)
        out = torch.gather(out_sorted, 2, inv_gather_idx)  # [B,H,L_q,D]

        return out

    def forward(
        self,
        hidden_states: torch.Tensor,
        pos_row: torch.Tensor,
        pos_col: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        context_pos_row: Optional[torch.Tensor] = None,
        context_pos_col: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass of the Transformer 2D model. CAREFUL: The key_padding_mask is 1 if a token should be ignored and 0 otherwise

        Args:
            hidden_states (torch.Tensor): Input tensor of shape [B, L, d_model].
            pos_row (torch.Tensor): Row position tensor of shape [B, L] with integer indices.
            pos_col (torch.Tensor): Column position tensor of shape [B, L] with integer indices.
            key_padding_mask (Optional[torch.Tensor]): Padding mask tensor of shape [B, L]. A one indicates a padded token.

            context (Optional[torch.Tensor]): Context tensor of shape [B, L_ctx, d_model]. Modification for cross-attention.
            context_pos_row (Optional[torch.Tensor]): Row position tensor for context of shape [B, L_ctx] with integer indices.
            context_pos_col (Optional[torch.Tensor]): Column position tensor for context of shape [B, L_ctx] with integer indices.
        Returns:
            torch.Tensor: Attention result tensor of shape [B, L, d_model].
            Optional[torch.Tensor]: Entropy auxiliary loss for router distribution.
        """

        B, L, _ = hidden_states.shape
        device = hidden_states.device
        if context is None:
            # self-attention
            kv_input = hidden_states
            kv_pos_row = pos_row
            kv_pos_col = pos_col
            kv_B, kv_L, _ = hidden_states.shape
        else:
            # cross-attention modification
            kv_input = context
            kv_pos_row = context_pos_row
            kv_pos_col = context_pos_col
            kv_B, kv_L, _ = context.shape

        q: torch.Tensor = self.q_proj(hidden_states)  # [B,L,D]
        k: torch.Tensor = self.k_proj(kv_input)
        v: torch.Tensor = self.v_proj(kv_input)

        q = q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(kv_B, kv_L, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(kv_B, kv_L, self.num_heads, self.head_dim).transpose(1, 2)

        max_pos = int(
            max(
                kv_pos_row.max().item(),
                kv_pos_col.max().item(),
                pos_row.max().item(),
                pos_col.max().item(),
            )
        ) + 1
        # build / reuse RoPE cache up to max position we see in either order
        cos, sin = self.rope_cache(
            max_position=max_pos,
            device=device,
            dtype=q.dtype,
        )

        # Apply RoPE for all four positions (4 instead of 2 cals because kv_pos_row does not have to be the same as pos_row)
        q_row = apply_rope(q, pos_row, cos, sin)
        q_col = apply_rope(q, pos_col, cos, sin)
        k_row = apply_rope(k, kv_pos_row, cos, sin)
        k_col = apply_rope(k, kv_pos_col, cos, sin)
        # attention for each order
        out_row = self._attend_with_order(
            q_row, k_row, v, q_positions=pos_row, k_positions=kv_pos_row, key_padding_mask=key_padding_mask
        )  # [B,H,L,D]
        out_col = self._attend_with_order(
            q_col, k_col, v, q_positions=pos_col, k_positions=kv_pos_col, key_padding_mask=key_padding_mask
        )  # [B,H,L,D]

        # router: per-head, per-token logits over {row, col}
        # use a per-head view of the *input hidden state* (before Q/K/V projections)
        # q: [B,H,L,D_head]
        router_in = hidden_states.contiguous().view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        router_h = F.silu(self.router_up(router_in))      # [B,H,L,4*D_head]
        router_logits = self.router_down(router_h)        # [B,H,L,2]

        # router_weights is a distribution over the two orders, so we need softmax here
        router_weights = F.softmax(router_logits, dim=-1) # [B,H,L,2]

        # ---- entropy auxiliary loss (paper Eq. 13-15) ----
        # One small numerical improvement (worth doing):
        # entropy should be computed via log_softmax for stability.
        ent_loss: Optional[torch.Tensor] = None
        if self.training:
            logp = F.log_softmax(router_logits, dim=-1)   # [B,H,L,2]
            p = logp.exp()
            entropy = -(p * logp).sum(dim=-1)            # [B,H,L]

            if key_padding_mask is not None and context is None:
                # mask out padded *query* positions (True where padded)
                valid = (~key_padding_mask).unsqueeze(1)  # [B,1,L]
                valid = valid.expand_as(entropy)          # [B,H,L]
                denom = valid.sum()

                if denom.item() == 0:
                    ent_loss = entropy.new_tensor(0.0)
                else:
                    ent_loss = (entropy * valid).sum() / denom
            else:
                ent_loss = entropy.mean()
        # -----------------------------------------------

        w_row = router_weights[..., 0].unsqueeze(-1)      # [B,H,L,1]
        w_col = router_weights[..., 1].unsqueeze(-1)

        out_heads = w_row * out_row + w_col * out_col   # [B,H,L,D_head]

        # merge heads
        out = out_heads.transpose(1, 2).contiguous().view(B, L, self.d_model)
        out = self.o_proj(out)  # [B,L,D_model]
        return out, ent_loss


# -------------------------------------------------------------------
# TODO: Maybe we can remove everything after this line and move it to
# TODO: the actual model implementations as it is no longer universal
# -------------------------------------------------------------------


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
        self.ffn = FeedForward(d_model=d_model, hidden_dim=4 * d_model, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        pos_row: torch.Tensor,
        pos_col: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # pre-norm + attention
        h = self.ln1(x)
        h_attn, ent_loss = self.attn(
            h,
            pos_row=pos_row,
            pos_col=pos_col,
            key_padding_mask=key_padding_mask,
        )
        x = x + self.dropout(h_attn)

        # pre-norm + FFN
        h2 = self.ln2(x)
        h2 = self.ffn(h2)
        x = x + self.dropout(h2)
        return x, ent_loss


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
        entropy_coef: float = 0.0,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.entropy_coef = entropy_coef

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.drop = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            TransformerBlock2DTPE(d_model=d_model, num_heads=num_heads, dropout=dropout)
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
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        input_ids: [B,L]
        pos_row, pos_col: [B,L]
        key_padding_mask: [B,L] (True where padded)
        targets: [B,L] or None
        """

        x = self.tok_emb(input_ids)  # * math.sqrt(self.d_model) not used in RoPE, comes from original Transformer paper
        x = self.drop(x)

        ent_losses = []

        for layer in self.layers:
            x, ent_l = layer(
                x,
                pos_row=pos_row,
                pos_col=pos_col,
                key_padding_mask=key_padding_mask,
            )
            if (targets is not None) and (self.entropy_coef > 0.0) and (ent_l is not None):
                ent_losses.append(ent_l)

        x = self.ln_f(x)
        logits = self.head(x)  # [B,L,vocab]

        loss: Optional[torch.Tensor] = None
        ent_loss_total: Optional[torch.Tensor] = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-100,
            )

            if self.entropy_coef > 0.0 and len(ent_losses) > 0:
                # average across layers to keep scale stable as you change depth
                ent_loss_total = torch.stack(ent_losses).mean()
                loss = loss + self.entropy_coef * ent_loss_total

        return logits, loss, ent_loss_total

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
            logits, _, _ = self.forward(
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
