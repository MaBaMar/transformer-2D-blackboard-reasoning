# ------------------------------------------------------------
# gptbase.py
#
# Implementation of a basic GPT like model for CoT baselines. (baseline B)
# ------------------------------------------------------------

import torch
import torch.nn as nn
from typing import TypeAlias, Literal

from transformers import AutoTokenizer, PreTrainedTokenizer
import json

from logging import getLogger

from projectlib.transformer.tpe2d_model import FeedForward, OutputHead, TwoDTPERoPEAttention
from projectlib.my_datasets import ScratchpadDataset, CoTDataset

# -----------------------------------------
# model registry
# -----------------------------------------
# up here for easier scalability if we need to add more datasets
_DATA_T_REGISTRY = {
    "scratchpad": ScratchpadDataset,
    "cot": CoTDataset,
}
dataset_option_t: TypeAlias = Literal["scratchpad", "cot"]


# -----------------------------------------
# custom tokenizer
# -----------------------------------------
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


# -----------------------------------------
# model implementation
# -----------------------------------------
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
        max_inference_steps: int = 100,
        use_weight_linking: bool = False
    ) -> None:
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)    # token embedding

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
        self._d_model = d_model
        self._num_heads = num_heads
        self._num_blocks = num_blocks

        # weight linking:
        if use_weight_linking:
            self.head.head.weight = self.tok_emb.weight

        self._linked_weights = use_weight_linking

    @property
    def device(self):
        return self.tok_emb.weight.device

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor, y: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        assert x.ndim == 2, f"input must be 2D tensor, got {x.ndim}D"
        assert attention_mask.shape == x.shape, f"attention mask shape {attention_mask.shape} does not match input shape {x.shape}"

        L = x.shape[1]
        assert L <= self.max_seq_len, f"sequence length {L} exceeds max_seq_len {self.max_seq_len}"

        # make sure all padding tokens are ignored!
        attention_mask = attention_mask & (x != self._pad_id)

        # embedding (+ some random embedding dropout)
        x = self.tok_emb(x)
        x = self.dropout(x)

        # zero out padding tokens to avoid influencing feed forward, layer norm etc....
        x = x * attention_mask.unsqueeze(-1)
        # x = self.embedding_dropout(x)

        # pass through transformer blocks
        for block in self.blocks:
            x = block(x=x, attention_mask=attention_mask)

        # mask out everything in front of sep token (including sep) for loss computation (= set to pad token)
        if y is not None:
            y_in = y.contiguous()
            # index of sep token
            sep_indices = (y_in == self._sep_id).nonzero(as_tuple=True)[1]

            for i, sep_idx in enumerate(sep_indices):
                y_in[i, :sep_idx + 1] = self._pad_id

            y = y_in

        logits, loss = self.head(context=x.contiguous(), targets=y)
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

            # the 2D rope module also supports 1D rope if data is tailored accordingly. This is slightly wasteful in terms of compute
            # but makes for a guaranteed-fair comparison
            self.attn: TwoDTPERoPEAttention = TwoDTPERoPEAttention(
                d_model=d_model,
                num_heads=num_heads,
                dropout=dropout,
                use_causal_mask=True,
                disable_entropy=True,
            )

            self.ffn = FeedForward(d_model, 4 * d_model, dropout)

        def forward(
            self,
            x: torch.Tensor,
            attention_mask: torch.Tensor
        ):
            B, L = x.shape[:2] # shape is [B,L, model_dimension]

            h = self.ln1(x)

            # do masked multihead attention:
            # causal_mask = self.causal_buffer[:L, :L]
            _1d_rope_indices = torch.arange(0, L, dtype=torch.long, device=x.device).unsqueeze(0).expand(B, -1)
            h, _ = self.attn.forward(
                hidden_states=h,
                pos_row=_1d_rope_indices,
                pos_col=_1d_rope_indices,
                key_padding_mask=(attention_mask==0)
            )

            # currently, pad tokens are completely masked out and cannot even attend to themselves. This produces
            # "floating" values (values with no input, i.e. NaNs)  after attention. The NaN values then propagate and in the final head mix with all
            # other NN values. This is bad and produces only NaN logits. To solve this, I currently just replace the NaNs
            # I'm open for better solutions (like modifying attention masks). However, this fix effectively works and ensures the
            # desired left-padding invariance (notably boosting accuracy).
            h[attention_mask == 0] = 0.0
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
    @torch.no_grad()
    def batch_inference(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Perform batch inference on the model. Supports variable-length sequences, but assumes that the input sequences are padded to the same length (left padding).

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """

        assert x.shape[0] == attention_mask.shape[0], "Input tensor and attention mask must have the same batch size."
        assert x.shape[1] == attention_mask.shape[1], "Input tensor and attention mask must have the same sequence length."

        x_new = torch.full((x.shape[0], self._max_inference_steps), self._pad_id, dtype=torch.long, device=x.device)
        full_mask = torch.full((x.shape[0], self._max_inference_steps), False, dtype=torch.bool, device=x.device)
        x_new[:, :x.shape[1]] = x
        full_mask[:, :x.shape[1]] = attention_mask

        curr_idx = x.shape[1]
        is_active = torch.ones(x.shape[0], dtype=torch.bool, device=x.device)

        while is_active.any() and curr_idx < self._max_inference_steps:
            logits, _ = self.forward(x_new[is_active, :curr_idx], full_mask[is_active, :curr_idx])
            next_tokens = logits[:, -1, :].argmax(dim=-1) # greedily select the next token

            active_indices = torch.where(is_active)[0]
            x_new[active_indices, curr_idx] = next_tokens
            full_mask[active_indices, curr_idx] = True

            # update which batch-samples are still active
            is_active[active_indices[next_tokens == self._eos_id]] = False

            curr_idx += 1

        # for debugging, check if all sequences are finished
        if (cnt := (x_new == self._eos_id).any(dim=1).sum()) != x_new.shape[0]:
            self.model_logger.debug(f"Not all computation finished in an EOS token, {cnt} outputs reached maxlen without generating an EOS token")
        else:
            self.model_logger.debug("All computations in batch completed sucessfully")

        return x_new

    def save_to_path(self, model_path: str, seed: int, **auxiliary_kwargs):
        """Save the model to the given path together with some metadata

        Args:
            model_path (str): Path to save the model.
        """
        torch.save({
            "model_state_dict": self.state_dict(),
            "config": {
                "vocab_size": self.tok_emb.num_embeddings,
                "max_seq_len": self.max_seq_len,
                "d_model": self._d_model,
                "num_heads": self._num_heads,
                "num_blocks": self._num_blocks,
                "token_config": {
                    "eos": self._eos_id,
                    "pad": self._pad_id,
                    "sep": self._sep_id
                },
                "dropout": self.dropout.p,
                "embedding_dropout": self.embedding_dropout.p,
                "max_inference_steps": self._max_inference_steps,
                "use_weight_linking": self._linked_weights,
                "seed": seed,
                **auxiliary_kwargs
            }
        }, model_path)

    @staticmethod
    def load_from_path(model_path: str) -> GPTStyleBaseline:
        """Load a locally stored model at the given path.

        Args:
            model_path (str): Path to the model.

        Returns:
            GPTStyleBaseline: Model initiated with the stored values.
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"

        checkpoint = torch.load(model_path, map_location=device)
        config = checkpoint["config"]

        model = GPTStyleBaseline(
            vocab_size=config["vocab_size"],
            max_seq_len=config["max_seq_len"],
            d_model=config["d_model"],
            num_heads=config["num_heads"],
            num_blocks=config["num_blocks"],
            token_config=config["token_config"],
            dropout=config["dropout"],
            embedding_dropout=config["embedding_dropout"],
            max_inference_steps=config["max_inference_steps"],
            use_weight_linking=config["use_weight_linking"]
        ).to(device)

        model.load_state_dict(checkpoint["model_state_dict"])

        print("Loaded model with configuration:")
        print(json.dumps(config, indent=4))

        return model
