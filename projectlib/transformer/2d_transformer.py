# 2d_transformer.py
import math
from dataclasses import dataclass
from typing import List, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ------------------------------
# 1) Toy tokenizer (replace later)
# ------------------------------
class SimpleVocab:
    def __init__(self, specials: List[str] = None):
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


# --------------------------------------------------
# 2) Example dataset: flatten a 2D grid into sequence
#    and produce (px, py) coordinates per token
# --------------------------------------------------
class GridLMExample(Dataset):
    """
    Each sample is a 2D list of tokens, e.g.
    [
      ["A11", "A12", "A13"],
      ["A21", "A22", "A23"],
    ]
    We flatten row-major and append <eos>.
    Coordinates (px, py) are 1-based; 0 reserved for padding.
    """
    def __init__(self, grids: List[List[List[str]]], vocab: SimpleVocab):
        self.vocab = vocab
        self.samples = []
        for grid in grids:
            rows = len(grid)
            cols = max(len(r) for r in grid)
            # normalize to rectangular (pad with a placeholder token if needed)
            norm = [r + ["<pad_tok>"] * (cols - len(r)) for r in grid]
            for tok in ["<pad_tok>"] + sum(norm, []):
                self.vocab.add(tok)  # grow vocab

            # flatten row-major
            flat_tokens = []
            px = []  # col index (1..cols)
            py = []  # row index (1..rows)
            for y, row in enumerate(norm):
                for x, t in enumerate(row):
                    flat_tokens.append(t)
                    px.append(x + 1)  # 1-based
                    py.append(y + 1)  # 1-based
            flat_tokens.append("<eos>")  # next-token target boundary

            ids = self.vocab.encode(flat_tokens)
            # For the <eos>, give a dummy coordinate = last token's coords
            px.append(px[-1] if px else 1)
            py.append(py[-1] if py else 1)

            self.samples.append({
                "input_ids": torch.tensor(ids[:-1], dtype=torch.long),
                "labels":    torch.tensor(ids[1:],  dtype=torch.long),
                "pos_x":     torch.tensor(px[:-1],  dtype=torch.long),
                "pos_y":     torch.tensor(py[:-1],  dtype=torch.long),
            })

        # record max extents for embedding table sizes
        self.max_x = max(s["pos_x"].max().item() for s in self.samples) if self.samples else 1
        self.max_y = max(s["pos_y"].max().item() for s in self.samples) if self.samples else 1

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        return self.samples[idx]


def pad_1d(seqs: List[torch.Tensor], pad: int) -> torch.Tensor:
    L = max(s.size(0) for s in seqs)
    out = torch.full((len(seqs), L), pad, dtype=seqs[0].dtype)
    for i, s in enumerate(seqs):
        out[i, :s.size(0)] = s
    return out


def collate_batch(batch, pad_id: int):
    # pad input_ids, labels, pos_x, pos_y
    input_ids = pad_1d([b["input_ids"] for b in batch], pad_id)
    labels    = pad_1d([b["labels"]    for b in batch], -100)  # ignore index for CE
    pos_x     = pad_1d([b["pos_x"]     for b in batch], 0)     # 0 == "no position" (padding)
    pos_y     = pad_1d([b["pos_y"]     for b in batch], 0)
    # causal mask is created inside the model; here we build an attention key padding mask:
    attn_pad_mask = (input_ids == pad_id)  # [B, L], True where padding
    return input_ids, labels, pos_x, pos_y, attn_pad_mask


# ------------------------------------------
# 3) TwoD positional encoding (learned)
# ------------------------------------------
class TwoDPositionalEncoding(nn.Module):
    """
    pos2d(i) = Ex[ px[i] ] + Ey[ py[i] ]
    with 0 reserved for padding (embedding(0) = 0 vector).
    """
    def __init__(self, d_model: int, max_x: int, max_y: int):
        super().__init__()
        self.ex = nn.Embedding(max_x + 1, d_model)  # index 0 = pad
        self.ey = nn.Embedding(max_y + 1, d_model)
        # initialize padding row to zeros
        with torch.no_grad():
            self.ex.weight[0].zero_()
            self.ey.weight[0].zero_()

    def forward(self, pos_x: torch.Tensor, pos_y: torch.Tensor) -> torch.Tensor:
        return self.ex(pos_x) + self.ey(pos_y)  # [B, L, d]


# ------------------------------------------
# 4) A tiny GPT-style Transformer (encoder-only)
# ------------------------------------------
class CausalTransformer2DPE(nn.Module):
    def __init__(self, vocab_size: int, d_model=256, n_heads=4, n_layers=4,
                 max_x=128, max_y=128, dropout=0.1):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos2d = TwoDPositionalEncoding(d_model, max_x, max_y)

        # store extents for generation-time wrapping
        self.max_x = max_x
        self.max_y = max_y

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,   # [B, L, d]
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.ln = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, vocab_size)
        self.d_model = d_model

    def _causal_mask(self, L: int, device) -> torch.Tensor:
        # [L, L] bool mask: True means "mask out"
        return torch.triu(torch.ones((L, L), device=device, dtype=torch.bool), diagonal=1)

    def forward(self, input_ids, pos_x, pos_y, key_padding_mask=None, targets=None):
        """
        input_ids: [B, L]
        pos_x, pos_y: [B, L] (0 is padding)
        key_padding_mask: [B, L] (True where pad)
        targets: [B, L] or None
        """
        B, L = input_ids.size()
        x = self.tok(input_ids) + self.pos2d(pos_x, pos_y)  # [B, L, d]

        # causal mask for self-attention (Transformer API wants shape [L, L])
        causal = self._causal_mask(L, x.device)  # True above diagonal = masked

        # GPT-style: encoder with causal mask + padding mask
        h = self.encoder(
            x,
            mask=causal,                        # [L, L]
            src_key_padding_mask=key_padding_mask  # [B, L], True where pad
        )  # [B, L, d]

        h = self.ln(h)
        logits = self.out(h)  # [B, L, V]

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-100
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, input_ids, pos_x, pos_y,
                 max_new_tokens=50, eos_id=None, pad_id=None):
        """
        Greedy decoding that *extends* (pos_x, pos_y) row-major, but keeps
        indices within [1..max_x] and [1..max_y] to avoid out-of-range errors.

        Policy:
        - If last_px < max_x: (px+1, same py)
        - If last_px == max_x: (1, min(py+1, max_y))
        """
        self.eval()
        B, L = input_ids.size()
        device = input_ids.device

        out_ids = input_ids.clone()
        px = pos_x.clone()
        py = pos_y.clone()

        if pad_id is None:
            pad_id = 0  # default assumption if not given

        for _ in range(max_new_tokens):
            # Build masks and forward
            pad_mask = (out_ids == pad_id)  # [B, L]
            logits, _ = self.forward(out_ids, px, py,
                                     key_padding_mask=pad_mask,
                                     targets=None)
            next_id = logits[:, -1, :].argmax(dim=-1)  # greedy
            out_ids = torch.cat([out_ids, next_id.unsqueeze(1)], dim=1)

            # Update 2D coords with wrapping
            last_px = px[:, -1]
            last_py = py[:, -1]

            # bool mask: True where we are at end of row
            wrap_row = (last_px == self.max_x)

            # new_px: increment, but wrap to 1 when last_px == max_x
            new_px = last_px + 1
            new_px = torch.where(wrap_row, torch.ones_like(new_px), new_px)

            # new_py: same row, except when wrapping; then increment but clamp at max_y
            new_py = last_py
            inc_py = torch.clamp(last_py + 1, max=self.max_y)
            new_py = torch.where(wrap_row, inc_py, new_py)

            px = torch.cat([px, new_px.unsqueeze(1)], dim=1)
            py = torch.cat([py, new_py.unsqueeze(1)], dim=1)

            if eos_id is not None and bool((next_id == eos_id).all()):
                break

        return out_ids


# ------------------------------------------
# 5) Tiny demo: make data, train, infer
# ------------------------------------------
@dataclass
class TrainConfig:
    d_model: int = 192
    n_heads: int = 4
    n_layers: int = 3
    batch_size: int = 8
    lr: float = 3e-4
    steps: int = 300
    device: str = "cpu"  # "mps" works on Apple Silicon for many ops if you want


def make_toy_grids(n=100, rows=3, cols=4) -> List[List[List[str]]]:
    grids = []
    for k in range(n):
        grid = [[f"T{k}_r{r}_c{c}" for c in range(cols)] for r in range(rows)]
        grids.append(grid)
    return grids


def main():
    # 1) Build toy data
    vocab = SimpleVocab()
    train_grids = make_toy_grids(n=200, rows=3, cols=4)
    ds = GridLMExample(train_grids, vocab)

    # 2) DataLoader
    collate = lambda batch: collate_batch(batch, pad_id=vocab.pad_id)
    dl = DataLoader(ds, batch_size=TrainConfig.batch_size, shuffle=True, collate_fn=collate)

    # 3) Model
    model = CausalTransformer2DPE(
        vocab_size=vocab.size,
        d_model=TrainConfig.d_model,
        n_heads=TrainConfig.n_heads,
        n_layers=TrainConfig.n_layers,
        max_x=ds.max_x,
        max_y=ds.max_y,
    ).to(TrainConfig.device)

    opt = torch.optim.AdamW(model.parameters(), lr=TrainConfig.lr)

    # 4) Train a few steps
    model.train()
    step = 0
    for epoch in range(9999):
        for input_ids, labels, pos_x, pos_y, pad_mask in dl:
            input_ids = input_ids.to(TrainConfig.device)
            labels    = labels.to(TrainConfig.device)
            pos_x     = pos_x.to(TrainConfig.device)
            pos_y     = pos_y.to(TrainConfig.device)
            pad_mask  = pad_mask.to(TrainConfig.device)

            _, loss = model(input_ids, pos_x, pos_y,
                            key_padding_mask=pad_mask,
                            targets=labels)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            step += 1
            if step % 50 == 0:
                print(f"step {step} | loss {loss.item():.4f}")
            if step >= TrainConfig.steps:
                break
        if step >= TrainConfig.steps:
            break

    # 5) Greedy generation demo
    model.eval()
    with torch.no_grad():
        sample = ds[0]
        input_ids = sample["input_ids"].unsqueeze(0).to(TrainConfig.device)
        pos_x     = sample["pos_x"].unsqueeze(0).to(TrainConfig.device)
        pos_y     = sample["pos_y"].unsqueeze(0).to(TrainConfig.device)

        out = model.generate(
            input_ids[:, :8],
            pos_x[:, :8],
            pos_y[:, :8],
            max_new_tokens=10,
            eos_id=vocab.eos_id,
            pad_id=vocab.pad_id,
        )
        print("Generated IDs:", out[0].tolist())
        print("Generated tokens:", vocab.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
