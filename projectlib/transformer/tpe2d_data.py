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
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# -------------------------------------------------------------------
# Simple vocab (demo)
# -------------------------------------------------------------------

class SimpleVocab:
    def __init__(self, specials: Optional[List[str]] = None) -> None:
        specials = specials or ["<pad>", "<bos>", "<eos>"]
        self.stoi: Dict[str, int] = {}
        self.itos: List[str] = []
        for tok in specials:
            self.add(tok)

    def add(self, tok: str) -> int:
        if tok not in self.stoi:
            self.stoi[tok] = len(self.itos)
            self.itos.append(tok)
        return self.stoi[tok]

    def encode(self, tokens: List[str]) -> List[int]:
        return [self.add(t) for t in tokens]

    def decode(self, ids: List[int]) -> List[str]:
        return [self.itos[i] for i in ids]

    @property
    def pad_id(self) -> int:
        return self.stoi["<pad>"]

    @property
    def bos_id(self) -> int:
        return self.stoi["<bos>"]

    @property
    def eos_id(self) -> int:
        return self.stoi["<eos>"]

    @property
    def size(self) -> int:
        return len(self.itos)


# -------------------------------------------------------------------
# Toy 2D grid dataset
#
# We create grids of shape [rows, cols] with dummy token strings.
# We flatten them row-major into a sequence and, for each token,
# we define:
#   - pos_row: row-major order index (0,1,2,...)
#   - pos_col: column-major order index
#
# => These two 1D orders are what 2D-TPE will use as its two
#    RoPE position sequences.
# -------------------------------------------------------------------

class Grid2DDataset(Dataset):
    def __init__(
        self,
        grids: List[List[List[str]]],
        vocab: SimpleVocab,
    ) -> None:
        super().__init__()
        self.vocab = vocab
        self.samples: List[Dict[str, torch.Tensor]] = []

        for grid in grids:
            rows = len(grid)
            cols = max(len(r) for r in grid)

            # make rectangular
            norm = [r + ["<pad_tok>"] * (cols - len(r)) for r in grid]

            for tok in ["<pad_tok>"] + sum(norm, []):
                self.vocab.add(tok)

            flat_tokens: List[str] = []
            pos_row: List[int] = []
            pos_col: List[int] = []

            # row-major and column-major indices
            # NOTE: here we assume one token per cell; if we later
            #       tokenize cells into multiple sub-tokens, we can
            #       just assign the same (row-major, col-major) index
            #       to all sub-tokens in that cell.
            for r in range(rows):
                for c in range(cols):
                    flat_tokens.append(norm[r][c])
                    # row-major index
                    rm_idx = r * cols + c
                    # column-major index
                    cm_idx = c * rows + r
                    pos_row.append(rm_idx)
                    pos_col.append(cm_idx)

            # add EOS at the end
            flat_tokens.append("<eos>")
            ids = vocab.encode(flat_tokens)

            # for EOS, just continue row/col order monotonically
            pos_row.append(pos_row[-1] + 1)
            pos_col.append(pos_col[-1] + 1)

            input_ids = torch.tensor(ids[:-1], dtype=torch.long)
            labels    = torch.tensor(ids[1:],  dtype=torch.long)
            pos_row_t = torch.tensor(pos_row[:-1], dtype=torch.long)
            pos_col_t = torch.tensor(pos_col[:-1], dtype=torch.long)

            self.samples.append({
                "input_ids": input_ids,
                "labels":    labels,
                "pos_row":   pos_row_t,
                "pos_col":   pos_col_t,
            })

        self.max_row_pos = max(s["pos_row"].max().item() for s in self.samples)
        self.max_col_pos = max(s["pos_col"].max().item() for s in self.samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.samples[idx]


# -------------------------------------------------------------------
# Collation / padding helpers
# -------------------------------------------------------------------

def pad_1d(seqs: List[torch.Tensor], pad_value: int) -> torch.Tensor:
    max_len = max(s.size(0) for s in seqs)
    out = torch.full((len(seqs), max_len), pad_value, dtype=seqs[0].dtype)
    for i, s in enumerate(seqs):
        out[i, : s.size(0)] = s
    return out


def collate_grid_batch(
    batch: List[Dict[str, torch.Tensor]],
    pad_id: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    input_ids = pad_1d([b["input_ids"] for b in batch], pad_id)
    labels    = pad_1d([b["labels"]    for b in batch], -100)
    pos_row   = pad_1d([b["pos_row"]   for b in batch], 0)
    pos_col   = pad_1d([b["pos_col"]   for b in batch], 0)

    key_padding_mask = (input_ids == pad_id)
    return input_ids, labels, pos_row, pos_col, key_padding_mask


# Generates a list of synthetic 2D grids with token strings "T{k}_r{r}_c{c}" per cell.
# These grids serve as a simple table-like dataset for testing the 2D-TPE mechanism.
def make_toy_grids(
    n: int = 100,
    rows: int = 3,
    cols: int = 4,
) -> List[List[List[str]]]:
    return [
        [[f"T{k}_r{r}_c{c}" for c in range(cols)] for r in range(rows)]
        for k in range(n)
    ]
