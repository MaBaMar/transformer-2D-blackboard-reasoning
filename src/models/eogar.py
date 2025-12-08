# ------------------------------------------------------------
# eogar.py
#
# Encoder-Only Generative Algorithmic Reasoner
#
# Purpose: Full model implementation based on 2D rope and autoregressive reasoning step
# generation for additions and subtractions
#
# Name: Toy name, feel free to change it to sth fancy
# ------------------------------------------------------------

from typing import Optional, final
import torch
import math
from torch import nn
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



class Head(nn.Module):
        def __init__(
            self,
            vocab_size: int,
            d_model: int,
        ) -> None:
            super().__init__()
            # building blocks
            self.ln = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, vocab_size, bias=False)

        def forward(
            self,
            context: torch.Tensor,
            targets: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            
            h = self.ln(context)
            logits = self.head(h)

            loss: Optional[torch.Tensor] = None
            if targets is not None:
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    targets.view(-1),
                    ignore_index=pad_id
                )

            return logits, loss



class EOgar(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_heads_encoder: int,
        n_encoder_blocks: int,
        pad_id: int,
        rope_mode: str = "2d",
    ) -> None:
        """
        Implementation of our EOgar model.

        Args:
            vocab_size (int): Vocabulary size.
            d_model (int): Model dimension.
            num_heads_encoder (int): Number of attention heads for encoder.
            n_encoder_blocks (int): Number of encoder blocks.
        """
        super().__init__() # to torch module

        self.pad_id = pad_id
        self.rope_mode = rope_mode

        self.encoder = Encoder(
            vocab_size=vocab_size,
            d_model=d_model,
            num_heads=num_heads_encoder,
            num_blocks=n_encoder_blocks
        )   

        self.head = Head(
            vocab_size=vocab_size,
            d_model=d_model,
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
            x_seq_idx = torch.arange(x_tokens.size(1), device=x_tokens.device).unsqueeze(0)
            x_pos_row = x_seq_idx
            x_pos_col = x_seq_idx

            y_seq_idx = torch.arange(y_tokens.size(1), device=y_tokens.device).unsqueeze(0)
            y_pos_row = y_seq_idx
            y_pos_col = y_seq_idx

        context = self.encoder(
            input_ids=x_tokens,
            pos_row=x_pos_row,
            pos_col=x_pos_col,
        )

        logits, loss = self.head(
            context=context,
            targets=y_tokens,
        )

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
            x_seq_idx = torch.arange(x_tokens.size(1), device=x_tokens.device).unsqueeze(0)
            x_pos_row = x_seq_idx
            x_pos_col = x_seq_idx

        context = self.encoder(
            input_ids=x_tokens,
            pos_row=x_pos_row,
            pos_col=x_pos_col,
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

        model = EOgar(
            vocab_size=vocab_size,
            d_model=d_model,
            num_heads_encoder=num_heads_encoder,
            n_encoder_blocks=n_encoder_blocks,
            pad_id=pad_id,
            rope_mode=rope_mode,
        )

        model.load_state_dict(checkpoint["model_state_dict"])

        return model
