# ------------------------------------------------------------
# 2d_transformer.py: Minimal 2D-aware GPT-style model

# NOTE:
#  - IMPORTANT PART: 2D positional encoding class (TwoDPositionalEncoding) for the positional embeddings and how it is
#    added in the forward() of CausalTransformer2DPE.
#  - REST: Filler for dataset, tokenizer, training loop, generation so that the file can run end-to-end on toy data.

# TODO: Replace dataset, tokenizer, training loop, generation, etc.
#    later with real logic.
# ------------------------------------------------------------

import math
from dataclasses import dataclass
from typing import List, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# Simple tokenizer to get the demo running.
# => Replace this with our real tokenizer later.
class SimpleVocab:
    def __init__(self, specials=None):
        specials = specials or ["<pad>", "<bos>", "<eos>"]

        self.stoi = {}
        self.itos = []

        # add special tokens
        for tok in specials:
            self.add(tok)

    def add(self, tok):
        """grow vocab dynamically"""
        if tok not in self.stoi:
            self.stoi[tok] = len(self.itos)
            self.itos.append(tok)
        return self.stoi[tok]

    # Encoding/decoding (again: demo only)
    def encode(self, tokens): return [self.add(t) for t in tokens]
    def decode(self, ids):     return [self.itos[i] for i in ids]

    @property
    def pad_id(self): return self.stoi["<pad>"]
    @property
    def bos_id(self): return self.stoi["<bos>"]
    @property
    def eos_id(self): return self.stoi["<eos>"]
    @property
    def size(self):   return len(self.itos)


# Tiny dataset: turns a 2D grid of tokens into a sequence.
# => Real project will NOT use this logic.
# => Kept only because we need *something* to train on.
class GridLMExample(Dataset):
    """
    - Input: grid[x][y] tokens
    - Flatten row-major
    - Also produce px, py = 2D coordinates for each token
    - Coordinates start at 1, padding = 0
    """

    def __init__(self, grids, vocab):
        self.vocab = vocab
        self.samples = []

        for grid in grids:
            rows = len(grid)
            cols = max(len(r) for r in grid)

            # make rectangular
            norm = [r + ["<pad_tok>"] * (cols - len(r)) for r in grid]

            # grow vocab
            for tok in ["<pad_tok>"] + sum(norm, []):
                self.vocab.add(tok)

            # flatten
            flat, px, py = [], [], []
            for y, row in enumerate(norm):
                for x, t in enumerate(row):
                    flat.append(t)
                    px.append(x + 1)  # 1-based
                    py.append(y + 1)

            # add eos
            flat.append("<eos>")
            ids = vocab.encode(flat)

            px.append(px[-1])
            py.append(py[-1])

            self.samples.append({
                "input_ids": torch.tensor(ids[:-1]),
                "labels":    torch.tensor(ids[1:]),
                "pos_x":     torch.tensor(px[:-1]),
                "pos_y":     torch.tensor(py[:-1]),
            })

        # extents needed for the embedding tables
        self.max_x = max(s["pos_x"].max().item() for s in self.samples)
        self.max_y = max(s["pos_y"].max().item() for s in self.samples)

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]


# Batch padding helper (demo only)
def pad_1d(seqs, pad):
    L = max(s.size(0) for s in seqs)
    out = torch.full((len(seqs), L), pad, dtype=seqs[0].dtype)
    for i, s in enumerate(seqs): out[i, :s.size(0)] = s
    return out

def collate_batch(batch, pad_id):
    """pads tokens + coords — demo only"""
    input_ids = pad_1d([b["input_ids"] for b in batch], pad_id)
    labels    = pad_1d([b["labels"]    for b in batch], -100)
    pos_x     = pad_1d([b["pos_x"]     for b in batch], 0)
    pos_y     = pad_1d([b["pos_y"]     for b in batch], 0)
    attn_pad_mask = (input_ids == pad_id)
    return input_ids, labels, pos_x, pos_y, attn_pad_mask



# ***** CORE COMPONENT OF THIS FILE *****
#
# TwoDPositionalEncoding = learned embeddings for x and y, added together.
#
#   pe(i) = ex[pos_x(i)] + ey[pos_y(i)]
#
# => this injects 2D structure directly into the sequence.
class TwoDPositionalEncoding(nn.Module):
    """
    2D learned embeddings:
        - ex[k] encodes column-coordinate k
        - ey[k] encodes row-coordinate k
        - pos_x==0 / pos_y==0 => padding => embedding = 0
    """

    def __init__(self, d_model, max_x, max_y):
        super().__init__()

        # allocate one extra for index 0 = padding
        self.ex = nn.Embedding(max_x + 1, d_model)
        self.ey = nn.Embedding(max_y + 1, d_model)

        # enforce padding row = 0
        with torch.no_grad():
            self.ex.weight[0].zero_()
            self.ey.weight[0].zero_()

    def forward(self, pos_x, pos_y):
        # Core: 2D pos embedding happens RIGHT HERE
        return self.ex(pos_x) + self.ey(pos_y)


# GPT-style transformer with 2D positional encoding.
# => This is what we actually keep long term.
class CausalTransformer2DPE(nn.Module):
    def __init__(self, vocab_size, d_model=256, n_heads=4, n_layers=4,
                 max_x=128, max_y=128, dropout=0.1):
        super().__init__()

        # token -> vector embedding
        self.tok = nn.Embedding(vocab_size, d_model)

        # *** Important: add 2D positional encoding layer ****
        self.pos2d = TwoDPositionalEncoding(d_model, max_x, max_y)

        self.max_x = max_x
        self.max_y = max_y

        # Transformer stack
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)

        self.ln = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, vocab_size)

    def _causal_mask(self, L, device):
        return torch.triu(torch.ones((L, L), device=device, dtype=torch.bool), diagonal=1)

    def forward(self, input_ids, pos_x, pos_y, key_padding_mask=None, targets=None):
        """
        forward pass with:
        x = tok_emb + 2D_pos_emb
        """

        B, L = input_ids.size()

        # ----- core injection of 2D structure -----
        x = self.tok(input_ids) + self.pos2d(pos_x, pos_y)

        causal = self._causal_mask(L, x.device)

        h = self.encoder(
            x,
            mask=causal,
            src_key_padding_mask=key_padding_mask,
        )

        h = self.ln(h)
        logits = self.out(h)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-100
            )
        return logits, loss

    # Greedy generation: demo logic only.
    # Replace this with our real generation pipeline later.
    @torch.no_grad()
    def generate(self, input_ids, pos_x, pos_y,
                 max_new_tokens=50, eos_id=None, pad_id=None):

        self.eval()
        B, L = input_ids.size()
        device = input_ids.device

        out_ids = input_ids.clone()
        px = pos_x.clone()
        py = pos_y.clone()

        if pad_id is None: pad_id = 0

        for _ in range(max_new_tokens):
            pad_mask = (out_ids == pad_id)
            logits, _ = self.forward(out_ids, px, py,
                                     key_padding_mask=pad_mask)
            next_id = logits[:, -1].argmax(dim=-1)

            out_ids = torch.cat([out_ids, next_id.unsqueeze(1)], dim=1)

            # update coordinates (simple row-major wrap)
            last_px = px[:, -1]
            last_py = py[:, -1]

            wrap_row = (last_px == self.max_x)

            new_px = torch.where(wrap_row, torch.ones_like(last_px), last_px + 1)
            new_py = torch.where(wrap_row,
                                 torch.clamp(last_py + 1, max=self.max_y),
                                 last_py)

            px = torch.cat([px, new_px.unsqueeze(1)], dim=1)
            py = torch.cat([py, new_py.unsqueeze(1)], dim=1)

            if eos_id is not None and bool((next_id == eos_id).all()):
                break

        return out_ids


# Training demo: again filler logic.
# => Replace with our proper pipeline.
@dataclass
class TrainConfig:
    d_model=192
    n_heads=4
    n_layers=3
    batch_size=8
    lr=3e-4
    steps=300
    device="cpu"   # mps/cuda if available


# Synthetic 2D toy grids (demo only)
def make_toy_grids(n=100, rows=3, cols=4):
    return [[[f"T{k}_r{r}_c{c}" for c in range(cols)] for r in range(rows)]
            for k in range(n)]


def main():
    # -------- BUILD TOY DATA (replace later) --------
    vocab = SimpleVocab()
    train_grids = make_toy_grids(n=200, rows=3, cols=4)
    ds = GridLMExample(train_grids, vocab)

    collate = lambda batch: collate_batch(batch, vocab.pad_id)
    dl = DataLoader(ds, batch_size=TrainConfig.batch_size,
                    shuffle=True, collate_fn=collate)

    # -------- CREATE MODEL (keep this part) --------
    model = CausalTransformer2DPE(
        vocab_size=vocab.size,
        d_model=TrainConfig.d_model,
        n_heads=TrainConfig.n_heads,
        n_layers=TrainConfig.n_layers,
        max_x=ds.max_x,
        max_y=ds.max_y,
    ).to(TrainConfig.device)

    opt = torch.optim.AdamW(model.parameters(), lr=TrainConfig.lr)

    # -------- TRAINING LOOP (demo only) --------
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

    # -------- GENERATION DEMO (also filler) --------
    model.eval()
    with torch.no_grad():
        s = ds[0]
        out = model.generate(
            s["input_ids"].unsqueeze(0)[:, :8].to(TrainConfig.device),
            s["pos_x"].unsqueeze(0)[:, :8].to(TrainConfig.device),
            s["pos_y"].unsqueeze(0)[:, :8].to(TrainConfig.device),
            max_new_tokens=10,
            eos_id=vocab.eos_id,
            pad_id=vocab.pad_id,
        )

        print("Generated IDs:", out[0].tolist())
        print("Generated tokens:", vocab.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
