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

# TODO: move somewhere else, should not be in the library
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from tpe2d_data import SimpleVocab, Grid2DDataset, collate_grid_batch, make_toy_grids
from tpe2d_model import CausalTransformer2DTPE


# -------------------------------------------------------------------
# Training demo (toy)
# -------------------------------------------------------------------

@dataclass
class TrainConfig:
    d_model: int = 192
    num_heads: int = 4
    num_layers: int = 3
    batch_size: int = 8
    lr: float = 3e-4
    steps: int = 200
    device: str = "cpu"   # "cuda" / "mps" if available


def main() -> None:
    cfg = TrainConfig()
    device = torch.device(cfg.device)

    vocab = SimpleVocab()
    train_grids = make_toy_grids(n=200, rows=3, cols=4)
    ds = Grid2DDataset(train_grids, vocab)

    collate_fn: Callable[
        [List[Dict[str, torch.Tensor]]],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ] = lambda batch: collate_grid_batch(batch, vocab.pad_id)

    dl = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    model = CausalTransformer2DTPE(
        vocab_size=vocab.size,
        d_model=cfg.d_model,
        num_layers=cfg.num_layers,
        num_heads=cfg.num_heads,
        dropout=0.1,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    model.train()
    step = 0
    for epoch in range(999999):
        for input_ids, labels, pos_row, pos_col, key_padding_mask in dl:
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            pos_row = pos_row.to(device)
            pos_col = pos_col.to(device)
            key_padding_mask = key_padding_mask.to(device)

            _, loss = model(
                input_ids=input_ids,
                pos_row=pos_row,
                pos_col=pos_col,
                key_padding_mask=key_padding_mask,
                targets=labels,
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            step += 1
            if step % 50 == 0:
                print(f"step {step:4d} | loss {loss.item():.4f}")
            if step >= cfg.steps:
                break
        if step >= cfg.steps:
            break

    # generation demo
    model.eval()
    with torch.no_grad():
        sample = ds[0]
        inp_ids = sample["input_ids"].unsqueeze(0).to(device)
        pos_r = sample["pos_row"].unsqueeze(0).to(device)
        pos_c = sample["pos_col"].unsqueeze(0).to(device)

        out = model.generate(
            inp_ids[:, :8],
            pos_r[:, :8],
            pos_c[:, :8],
            max_new_tokens=10,
            eos_id=vocab.eos_id,
            pad_id=vocab.pad_id,
        )
        print("Generated IDs:", out[0].tolist())
        print("Generated tokens:", vocab.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
