# ------------------------------------------------------------
# eogar.py
#
# Encoder-Only Generative Algorithmic Reasoner
#
# Purpose: Full model implementation based on 2D rope and autoregressive reasoning step
# generation for additions and subtractions
#
# Name: Toy name, feel free to change it to sth fancy
#
# ADDITION (entropy regularization):
#   - EOgar.forward still returns (logits, loss) so your training loop is unchanged.
#   - If entropy_coef > 0, we add: loss_total = loss_CE + entropy_coef * loss_entropy
#   - loss_entropy is aggregated from ent_loss returned by TwoDTPERoPEAttention in each block.
#   - For logging without changing the training loop, we expose:
#       self.last_ent_loss  (None if not computed)
# ------------------------------------------------------------

from typing import Optional, final, Tuple, List
import torch
from torch import nn

from projectlib.transformer.tpe2d_model import TwoDTPERoPEAttention, FeedForward, OutputHead
from projectlib.wrappertypes import BBChainGenerator

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
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        input_ids: [B,L]
        pos_row, pos_col: [B,L]
        key_padding_mask: [B,L] (True where padded)
        targets: [B,L] or None
        """

        x = self.tok_emb(input_ids) # * math.sqrt(self.d_model) not used in RoPE, comes from original Transformer paper
        x = self.dropout(x)

        ent_losses: List[torch.Tensor] = []

        for layer in self.transformer_blocks:
            x, ent_l = layer(
                x,
                pos_row=pos_row,
                pos_col=pos_col,
                key_padding_mask=key_padding_mask,
            )
            if ent_l is not None:
                ent_losses.append(ent_l.mean())

        ent_loss_total: Optional[torch.Tensor] = None
        if len(ent_losses) > 0:
            ent_loss_total = torch.stack(ent_losses).mean()

        return x, ent_loss_total

    # private helper block
    class _EncoderBlock(nn.Module):
        def __init__(
            self,
            d_model: int,
            num_heads: int,
            dropout: float = 0.1,
        ) -> None:
            super().__init__()

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
        ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
            # pre-norm + attention
            h = self.ln1(x)

            h_attn, ent_loss = self.attn(h, pos_row, pos_col, key_padding_mask=key_padding_mask)
            
            x = x + self.dropout(h_attn)

            # pre-norm + FFN
            h2 = self.ln2(x)
            h2 = self.ffn(h2)
            x = x + self.dropout(h2)

            return x, ent_loss


class EOgar(BBChainGenerator):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_heads_encoder: int,
        n_encoder_blocks: int,
        pad_id: int,
        rope_mode: str = "2d",
        entropy_coef: float = 0.0,   # <-- NEW (default off)
    ) -> None:
        """
        Implementation of our EOgar model.

        Args:
            vocab_size (int): Vocabulary size.
            d_model (int): Model dimension.
            num_heads_encoder (int): Number of attention heads for encoder.
            n_encoder_blocks (int): Number of encoder blocks.
            entropy_coef (float): if >0, add entropy regularization to the loss:
                                  L_total = L_CE + entropy_coef * L_entropy
        """
        super().__init__()  # to torch module

        self.pad_id = pad_id
        self.rope_mode = rope_mode

        # --- NEW: entropy regularization coefficient ---
        self.entropy_coef = float(entropy_coef)

        # --- NEW: expose for logging without changing training loop ---
        self.last_ent_loss: Optional[torch.Tensor] = None

        self.encoder = Encoder(
            vocab_size=vocab_size,
            d_model=d_model,
            num_heads=num_heads_encoder,
            num_blocks=n_encoder_blocks
        )

        self.head = OutputHead(
            vocab_size=vocab_size,
            d_model=d_model,
            pad_id=pad_id,
        )

    def forward(
        self,
        x: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        y: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ):
        """
        Forward pass of the model for training.

        Args:
            x (tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]): Input data (current blackboard state).
            y (tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]): Target data (next blackboard state).

        Note:
            For questions about the input data format, consult projectlib/my_datasets/collators.py.
            Model should be used with dataloaders like:
                from projectlib.my_datasets.collators import collate_blackboards, make_collator_with_args
                collate_fn = make_collator_with_args(collate_blackboards, dataset.pad_id)
                DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        Returns:
            Probably sth suitable for both generation and training
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
        y_key_padding_mask: torch.Tensor = y[3]

        if self.rope_mode == "1d":
            # NOTE: original code created shape [1,L]. That can break for batch_size>1.
            # Keeping behavior but making it safe by expanding to [B,L] (does not change values).
            Bx, Lx = x_tokens.shape
            x_seq_idx = torch.arange(Lx, device=x_tokens.device).unsqueeze(0).expand(Bx, -1)
            x_pos_row = x_seq_idx
            x_pos_col = x_seq_idx

            By, Ly = y_tokens.shape
            y_seq_idx = torch.arange(Ly, device=y_tokens.device).unsqueeze(0).expand(By, -1)
            y_pos_row = y_seq_idx
            y_pos_col = y_seq_idx

        # --- NEW: decide whether to collect entropy ---
        collect_entropy = bool(self.training and (self.entropy_coef > 0.0))

        context, ent_loss = self.encoder(
            input_ids=x_tokens,
            pos_row=x_pos_row,
            pos_col=x_pos_col,
            key_padding_mask=x_key_padding_mask,
        )

        if not collect_entropy:
            ent_loss = None

        logits, loss = self.head(
            context=context,
            targets=y_tokens,
        )

        # --- NEW: add entropy reg internally, but still return only (logits, loss) ---
        self.last_ent_loss = None
        if (loss is not None) and (self.entropy_coef > 0.0) and (ent_loss is not None):
            loss = loss + (self.entropy_coef * ent_loss)
            self.last_ent_loss = ent_loss.detach()

        return logits, loss

    def next_state(
        self,
        x: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ):
        """Generates the next state of the blackboard as follows:
            - feed current state to the decoder
            - feed a blackboard that is empty (except for the BOS token in its first cell) to the decoder and generate the next token
            - update the blackboard state with the generated token
            - pass updated blackboard state to the decoder and generate the next token
            - etc ...
            - stop when we generated L-1 tokens, where L is the length of the input sequence (current blackboard)

        Args:
            x (tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]): Input data (current blackboard state).

        Returns:
            torch.Tensor: Next blackboard state.
        """

        # Unpack input
        x_tokens: torch.Tensor = x[0]
        x_pos_row: torch.Tensor = x[1]
        x_pos_col: torch.Tensor = x[2]
        x_key_padding_mask: torch.Tensor = x[3]

        if self.rope_mode == "1d":
            # same safety fix as in forward()
            B, L = x_tokens.shape
            x_seq_idx = torch.arange(L, device=x_tokens.device).unsqueeze(0).expand(B, -1)
            x_pos_row = x_seq_idx
            x_pos_col = x_seq_idx

        # No entropy collection during inference
        context, _ = self.encoder(
            input_ids=x_tokens,
            pos_row=x_pos_row,
            pos_col=x_pos_col,
            key_padding_mask=x_key_padding_mask,
        )

        logits, _ = self.head(
            context=context,
        )

        out_tokens = logits.argmax(dim=-1)

        return out_tokens

    @staticmethod
    def load_from_path(model_path: str) -> "EOgar":
        """Load a locally stored model at the given path.

        Args:
            model_path (str): Path to the model.

        Returns:
            EOgar: Model initiated with the stored values.
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"

        checkpoint = torch.load(model_path, map_location=device)
        config = checkpoint["config"]

        vocab_size = config["vocab_size"]
        d_model = config["d_model"]
        num_heads_encoder = config["num_heads_encoder"]
        n_encoder_blocks = config["n_encoder_blocks"]
        pad_id = config["pad_id"]
        rope_mode = config["rope_mode"]

        # NEW: backward compatible (old checkpoints won't have this)
        entropy_coef = config.get("entropy_coef", 0.0)

        model = EOgar(
            vocab_size=vocab_size,
            d_model=d_model,
            num_heads_encoder=num_heads_encoder,
            n_encoder_blocks=n_encoder_blocks,
            pad_id=pad_id,
            rope_mode=rope_mode,
            entropy_coef=entropy_coef,
        )

        model.load_state_dict(checkpoint["model_state_dict"])

        return model
