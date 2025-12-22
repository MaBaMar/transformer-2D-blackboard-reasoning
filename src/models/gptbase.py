"""
gptbase.py

Implementation of a basic GPT like model for CoT baselines.
"""

# TODO: handle position IDs correctly when doing left-padding!
# FIXME: inference pass
#
import torch
import torch.nn as nn

from transformers import AutoTokenizer, PreTrainedTokenizer

from logging import getLogger

from projectlib.transformer.tpe2d_model import FeedForward, OutputHead

class GPTBaseTokenizer:
    def __init__(self, device: torch.device, tokenizer_name: str = 'gpt2'):
        """
        Initialize the tokenizer with the specified tokenizer name to be used for the GPT baseline. Recommended is 'gpt2'.
        The tokenizer supports padding on a per-batch level, which is useful for batching inputs of different lengths with minimal memory overhead. Moreover,
        the tokenizer differentiates between inference (left padding) and training (right padding) to enable smooth integration with the model.

        Args:
            tokenizer_name (str): The name of the tokenizer to be used. Defaults to 'gpt2'.
        """
        self._tok_internal: PreTrainedTokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self._tok_internal.add_special_tokens({'pad_token': '<pad>', 'eos_token': '<eos>', 'sep_token': '<sep>'}, replace_additional_special_tokens=True)
        self._device = device
        # vocab size of tokenizer fails to consider special tokens, so we need a custom solution
        self.vocab_size: int = len(self._tok_internal)

    def get_token_config(self) -> dict[str, int]:
        pad_id = self._tok_internal.pad_token_id
        eos_id = self._tok_internal.eos_token_id
        sep_id = self._tok_internal.sep_token_id
        # safety check (may simplify debugging)
        assert isinstance(pad_id, int) and isinstance(eos_id, int) and isinstance(sep_id, int), "Unexpected types!, please do not double register special tokens"
        return {'pad': pad_id, 'eos': eos_id, 'sep': sep_id}

    def encode_batch(self, batch: list[str], inference_mode: bool) -> dict[str, torch.Tensor]:
        """Encode a batch of strings into token IDs."""
        self._tok_internal.padding_side = 'left' if inference_mode else 'right'

        tokens = self._tok_internal(
            batch,
            padding=True,   # pads to batch max length
            return_tensors='pt',
        )

        return {'input_ids': tokens.input_ids.to(self._device), 'attention_mask': tokens.attention_mask.to(self._device)}

    def strip_decode(self, tokens: torch.Tensor) -> list[str]:
        """
        Strip special tokens and decode the tokens into a string.

        Args:
            tokens (torch.Tensor): A batch of token IDs to be decoded.

        Returns:
            list[str]: The decoded strings.
        """
        res = []
        for token_ids in tokens:
            decoded = self._tok_internal.decode(token_ids, skip_special_tokens=True)
            res.append(decoded)
        return res

class GPTStyleBaseline(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        d_model: int,
        num_heads: int,
        num_blocks: int,
        token_config: dict,
        dropout: float = 0.1,
        embedding_dropout: float = 0.1,
        max_inference_steps: int = 100
    ) -> None:
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)    # token embedding
        self.pos_emb = nn.Embedding(max_seq_len, d_model)    # additive positional embedding

        self.blocks = nn.ModuleList([
            self._TransformerBlock(d_model, num_heads, dropout, max_seq_len)
            for _ in range(num_blocks)
        ])

        self.dropout = nn.Dropout(dropout)
        self.embedding_dropout = nn.Dropout(embedding_dropout)

        self._eos_id = token_config['eos']
        self._pad_id = token_config['pad']
        self._sep_id = token_config['sep']
        self.head = OutputHead(vocab_size, d_model, self._pad_id)
        self.max_seq_len = max_seq_len
        self._max_inference_steps = max_inference_steps
        self.model_logger = getLogger(__name__)

        # weight linking:
        self.head.head.weight = self.tok_emb.weight

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor, y: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        assert x.ndim == 2, f"input must be 2D tensor, got {x.ndim}D"
        assert attention_mask.shape == x.shape, f"attention mask shape {attention_mask.shape} does not match input shape {x.shape}"

        L = x.shape[1]
        assert L <= self.max_seq_len, f"sequence length {L} exceeds max_seq_len {self.max_seq_len}"

        # patch attention mask in case of left padding
        pos_ids = attention_mask.cumsum(dim=1) - 1
        pos_ids = pos_ids.masked_fill(pos_ids == -1, 0)

        # embedding (+ some random embedding dropout)
        x = self.tok_emb(x) + self.pos_emb(pos_ids)
        x = self.embedding_dropout(x)

        # pass through transformer blocks
        for block in self.blocks:
            x = block(x=x, attention_mask=attention_mask)

        logits, loss = self.head(context=x.contiguous(), targets=y.contiguous() if y is not None else None)
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
            maxlen: int
        ):
            super().__init__()
            assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

            self.ln1 = nn.LayerNorm(d_model)
            self.ln2 = nn.LayerNorm(d_model)
            self.dropout = nn.Dropout(dropout)
            self.attn: nn.MultiheadAttention = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
            self.ffn = FeedForward(d_model, 4 * d_model, dropout)
            self.register_buffer('causal_buffer', torch.triu(
                torch.ones((maxlen, maxlen), dtype=torch.bool),
                diagonal=1
            ))

        def forward(
            self,
            x: torch.Tensor,
            attention_mask: torch.Tensor
        ):
            L = x.shape[1]

            h = self.ln1(x)

            # do masked multihead attention:
            causal_mask = self.causal_buffer[:L, :L]
            h, _ = self.attn(h, h, h, attn_mask=causal_mask, key_padding_mask=(attention_mask==0), need_weights=False)
            x = x + self.dropout(h)

            h2 = self.ln2(x)
            h2 = self.ffn(h2)
            x = x + self.dropout(h2)

            return x

    # -------------------------
    # Model inference mechanism
    # -------------------------
    def set_max_inference_steps(self, steps: int) -> None:
        """
        Set the maximum number of inference steps for the model, i.e. the maximum number of tokens to generate for each sequence.

        Args:
            steps (int): The maximum number of inference steps.
        """
        self._max_inference_steps = steps

    @torch.inference_mode()
    def batch_inference(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Perform batch inference on the model. Supports variable-length sequences.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """

        x_new = torch.ones((x.shape[0], self._max_inference_steps), dtype=torch.long, device=x.device)

        # prepadding

        is_active = torch.ones(x.shape[0], dtype=torch.bool, device=x.device)

        x_new: torch.Tensor = x.clone() # Note: this may be replaced with reference copying, if we do not need to be careful with reference semantics here

        while is_active.any():
            logits, _ = self.forward(x_new[is_active], attention_mask[is_active])
            next_tokens = logits[:, -1, :].argmax(dim=-1) # greedily select the next token
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
