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

class Decoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_heads: int,
        num_blocks: int,
        dropout: float = 0.1,
    ) -> None:
        # TODO implement
        super().__init__()
        self.d_model = d_model
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.dropout = nn.Dropout(dropout)

        self.transformer_blocks = nn.ModuleList([
            _DecoderBlock(d_model, num_heads, dropout)
            for _ in range(num_blocks)
        ])
        
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        pos_row: torch.Tensor,
        pos_col: torch.Tensor,
        context: torch.Tensor, 
        context_pos_row: torch.Tensor,
        context_pos_col: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor]= None,       # Mask for input_ids
        context_key_padding_mask: Optional[torch.Tensor] = None, # Mask for context
        targets: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Implementation notes:
        Outputs logits and loss. The target is the same as the input (we use teacher forcing)
        No need to compute "real logits", i.e applying softmax, as we can also argmax over logits without softmax for generation
        """
        # TODO implement
        # dummy placeholders

        x = self.tok_emb(input_ids) * math.sqrt(self.d_model)
        x = self.dropout(x)

        for layer in self.transformer_blocks:
            x = layer(
                x=x, 
                pos_row=pos_row, 
                pos_col=pos_col, 
                key_padding_mask=key_padding_mask,
                context=context, 
                context_pos_row=context_pos_row, 
                context_pos_col=context_pos_col,
                context_key_padding_mask=context_key_padding_mask,
            )

        x = self.ln_f(x)
        logits = self.head(x)

        loss: Optional[torch.Tensor] = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-100 
            )

        return logits, loss

    def generate(
        self,
        input_ids: torch.Tensor,
        pos_row: torch.Tensor,
        pos_col: torch.Tensor, 
        max_new_tokens: int,
        context: torch.Tensor,
        context_pos_row: torch.Tensor,
        context_pos_col: torch.Tensor,
        context_key_padding_mask: Optional[torch.Tensor] = None,
        eos_id: Optional[int] = None,
        pad_id: Optional[int] = None,
    ):
        """
        Outputs the next token for the model
        """
        # TODO: implement (can also move this to somewhere else, especially if we want to use stuff like beam search later on)
        self.eval()

        out_ids = input_ids.clone()
        row = pos_row.clone()
        col = pos_col.clone()

        W = int(context_pos_col.max().item()) + 1

        if pad_id is None:
            pad_id = 0

        for _ in range(max_new_tokens):
            key_padding_mask = (out_ids == pad_id)
            logits, _ = self.forward(
                input_ids=out_ids,
                pos_row=row,
                pos_col=col,
                context=context,
                context_pos_row=context_pos_row,
                context_pos_col=context_pos_col,
                key_padding_mask=key_padding_mask,
                context_key_padding_mask=context_key_padding_mask,
                targets=None,
            )


            next_token = logits[:, -1].argmax(dim=-1)  # greedy
            out_ids = torch.cat([out_ids, next_token.unsqueeze(-1)], dim=-1)

            last_row = row[:, -1]
            last_col = col[:, -1]
            next_col_candidate = last_col + 1
            line_break = (next_col_candidate >= W)
            next_row = torch.where(line_break, last_row + 1, last_row)
            next_col = torch.where(line_break, torch.zeros_like(next_col_candidate), next_col_candidate)
            row = torch.cat([row, (next_row).unsqueeze(-1)], dim=-1) 
            col = torch.cat([col, (next_col).unsqueeze(-1)], dim=-1)

            if eos_id is not None and bool((next_token == eos_id).all()):
                break

        return out_ids

class _DecoderBlock(nn.Module):
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
            self.ln3 = nn.LayerNorm(d_model)
            self.dropout = nn.Dropout(dropout)
            self.masked_attn = TwoDTPERoPEAttention(d_model, num_heads, dropout, use_causal_mask=True)  
            self.cross_attn = TwoDTPERoPEAttention(d_model, num_heads, dropout, use_causal_mask=False) 
            self.ffn = FeedForward(d_model, 4 * d_model, dropout)

        def forward(
            self,
            x: torch.Tensor,
            pos_row: torch.Tensor,
            pos_col: torch.Tensor,
            key_padding_mask: Optional[torch.Tensor] = None,
            context: Optional[torch.Tensor] = None,
            context_pos_row: Optional[torch.Tensor] = None,
            context_pos_col: Optional[torch.Tensor] = None,
            context_key_padding_mask: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            

            # pre-norm + attention
            h = self.ln1(x)
            h = self.masked_attn(hidden_states=h, pos_row=pos_row, 
                                 pos_col=pos_col, key_padding_mask=key_padding_mask)
            x = x + self.dropout(h)

            # pre-norm + cross-attention
            h = self.ln2(x)
            h = self.cross_attn(hidden_states=h, context=context, pos_row=pos_row, pos_col=pos_col,
                                context_pos_row=context_pos_row, context_pos_col=context_pos_col, 
                                 key_padding_mask=context_key_padding_mask)
            
            x = x + self.dropout(h)

            # pre-norm + FFN
            h2 = self.ln3(x)
            h2 = self.ffn(h2)
            x = x + self.dropout(h2)
            return x

class Edgar(BBChainGenerator):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_heads_encoder: int,
        num_heads_decoder: int,
        n_encoder_blocks: int,
        n_decoder_blocks: int,
        pad_id: int,
        eos_id: int,
    ) -> None:
        """
        Implementation of our Edgar model.

        Args:
            vocab_size (int): Vocabulary size.
            d_model (int): Model dimension.
            num_heads_encoder (int): Number of attention heads for encoder.
            num_heads_decoder (int): Number of attention heads for decoder.
            n_encoder_blocks (int): Number of encoder blocks.
            n_decoder_blocks (int): Number of decoder blocks.
        """
        # TODO implement
        super().__init__() # to torch module

        self.pad_id = pad_id
        self.eos_id = eos_id

        self.encoder = Encoder(
            vocab_size=vocab_size,
            d_model=d_model,
            num_heads=num_heads_encoder,
            num_blocks=n_encoder_blocks
        )   
        self.decoder = Decoder(
            vocab_size=vocab_size,
            d_model=d_model,
            num_heads=num_heads_decoder,
            num_blocks=n_decoder_blocks
        )

        pass

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
        # TODO implement
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


        context = self.encoder(
            input_ids=x_tokens,
            pos_row=x_pos_row,
            pos_col=x_pos_col,
        )

        logits, loss = self.decoder(
            input_ids=y_tokens,
            pos_row=y_pos_row,
            pos_col=y_pos_col,
            context=context,
            context_pos_row = x_pos_row,
            context_pos_col = x_pos_col,
            key_padding_mask=y_key_padding_mask,
            context_key_padding_mask = x_key_padding_mask,
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

        x_tokens: torch.Tensor = x[0]
        x_pos_row: torch.Tensor = x[1]
        x_pos_col: torch.Tensor = x[2]
        x_key_padding_mask: torch.Tensor = x[3]

        device = x_tokens.device
        B, L = x_tokens.shape

        context = self.encoder(
            input_ids=x_tokens,
            pos_row=x_pos_row,
            pos_col=x_pos_col,
            )
        
        row = torch.zeros((B, 1), device=device, dtype=torch.long)
        col = torch.zeros((B, 1), device=device, dtype=torch.long)
        out_ids = torch.full((B, 1), fill_value=x_tokens[0, 0].item(), device=device, dtype=torch.long)

        # generate at most one full blackboard
        max_new_tokens = L - 1

        out_tokens = self.decoder.generate(
            input_ids=out_ids,
            pos_row=row,
            pos_col=col,
            max_new_tokens=max_new_tokens,
            context=context,
            context_pos_row=x_pos_row,
            context_pos_col=x_pos_col,
            context_key_padding_mask=x_key_padding_mask,
            pad_id=self.pad_id,
            eos_id=self.eos_id,
        )

        return out_tokens
