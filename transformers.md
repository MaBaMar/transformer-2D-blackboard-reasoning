# 1: “From-scratch” Transformer (Python) (Selbst implementieren)

- **PyTorch “Transformer building blocks”**: tutorial to implement transformer
  **URLs:**  
  - Tutorial: https://pytorch.org/tutorials/intermediate/transformer_building_blocks.html  
  - SDPA tutorial: https://pytorch.org/tutorials/intermediate/scaled_dot_product_attention_tutorial.html  
  - FlexAttention blog: https://pytorch.org/blog/flexattention/

- **Other Transformer Implementations (Harvard NLP, Sasha Rush)**: line-by-line PyTorch build (~400 LoC) with math to code mapping; includes article, code, and a runnable notebook.  
  **URLs:**  
  - Article: https://nlp.seas.harvard.edu/2018/04/03/attention.html  
  - GitHub: https://github.com/harvardnlp/annotated-transformer  
  - ACL Anthology (paper): https://aclanthology.org/W18-2509/  
  - Colab notebook: https://colab.research.google.com/github/harvardnlp/annotated-transformer/blob/master/AnnotatedTransformer.ipynb

- **Other Blog Posts**: 
  **URLs:**  
  - Book: https://d2l.ai/  
  - Transformer chapter (PyTorch): https://d2l.ai/chapter_attention-mechanisms-and-transformers/transformer.html  
  - Book paper: https://arxiv.org/abs/2106.11342  
  - Colab (PyTorch): https://colab.research.google.com/github/d2l-ai/d2l-pytorch-colab/blob/master/chapter_attention-mechanisms-and-transformers/transformer.ipynb
  - https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial6/Transformers_and_MHAttention.html



# 2: Custom Positional Encodings with (Mostly) Stock PyTorch

> Goal: add **our own** positional encodings while keeping PyTorch’s `nn.Transformer` / `nn.TransformerEncoder(Layer)` intact as much as possible.

---

## Positional Encoding Variants

1) **Absolute PE (add to token embeddings)**  
Compute a position tensor `P ∈ ℝ^{B×L×d_model}` and add it to token embeddings before the encoder (optionally before *every* layer).  
- Works directly with `nn.Transformer` / `nn.TransformerEncoder(Layer)` (no API changes).
- Typical for 1D sinusoid and ViT-style **2D sin–cos** or learned 2D tables.

**Docs & walkthroughs**
- PyTorch `nn.Transformer` / `TransformerEncoder`:  
  https://docs.pytorch.org/docs/stable/generated/torch.nn.Transformer.html  
  https://docs.pytorch.org/docs/stable/generated/torch.nn.TransformerEncoder.html
- “The Annotated Transformer” (line-by-line, includes PE):  
  https://nlp.seas.harvard.edu/2018/04/03/attention.html
- UvA DL Tutorial 6 (builds Transformer & PE end-to-end):  
  https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial6/Transformers_and_MHAttention.html
- ViT/MAE 2D sin–cos references (ready-made functions):  
  https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py  
  Mirror docs (Lightly) for `get_2d_sincos_pos_embed`: https://docs.lightly.ai/self-supervised-learning/lightly.models.utils.html  
  Simple 2D PE repos: https://github.com/wzlxjtu/PositionalEncoding2D , https://github.com/tatp22/multidim-positional-encoding

---

2) **Relative/bias PE (add bias to attention logits)**  
Provide a **float attention bias** matrix as the mask so PyTorch adds it to the attention scores before softmax. With this we can encode 1D/2D relative distances, ALiBi slopes, Swin-style relative tables, etc., **without changing token vectors**.  
- Use `attn_mask` on `nn.MultiheadAttention`; for encoder-only stacks, pass `src_mask` and it’s forwarded down.
- Shapes: `(L, S)` (broadcasted) **or** `(N·num_heads, L, S)` for **per-head/per-batch** biases.

**Docs & examples**
- PyTorch `nn.MultiheadAttention` mask semantics & shapes:  
  https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html
- “Transformer building blocks” (modern PyTorch 2.x, masks & SDPA paths):  
  https://docs.pytorch.org/tutorials/intermediate/transformer_building_blocks.html
- FlexAttention (write custom score functions—ALiBi/relative—without full `L×L` materialization):  
  https://pytorch.org/blog/flexattention/
- ALiBi (paper + official code snippets):  
  Paper: https://arxiv.org/abs/2108.12409  
  Code:  https://github.com/ofirpress/attention_with_linear_biases
- 2D **relative position bias** (Swin family; small learned table indexed by Δrow,Δcol):  
  Paper: https://arxiv.org/abs/2103.14030  
  TorchVision Swin (shows `relative_position_bias` in code):  
  https://docs.pytorch.org/vision/main/_modules/torchvision/models/swin_transformer.html

> **Minimal-change recipe:** compute `(L, L)` (or `(B·H, L, L)`) bias once per batch and pass it as `src_mask` / `attn_mask`. Keep the rest of the PyTorch blocks unchanged.

---

3) **Rotary PE (RoPE; modify Q/K before dot-product)**  
Apply position-dependent rotations to Q and K (often split channels for x/y on a grid). This **does** require a tiny custom attention layer (or FlexAttention), but the feed-forward/norm/residual part can remain stock.
- Use PyTorch’s fused SDPA in your custom layer for speed: `torch.nn.functional.scaled_dot_product_attention`.

**Docs & references**
- SDPA tutorial (how to wire custom attention that stays fast):  
  https://docs.pytorch.org/tutorials/intermediate/scaled_dot_product_attention_tutorial.html
- RoFormer / RoPE paper: https://arxiv.org/abs/2104.09864
- HF blog “Designing positional encoding” (step-by-step to RoPE):  
  https://huggingface.co/blog/designing-positional-encoding
- LearnOpenCV RoPE explainer (derivation + PyTorch snippet):  
  https://learnopencv.com/rope-position-embeddings/

> **Minimal-change recipe:** subclass just the attention part (QKV + SDPA + output proj) to insert RoPE; keep `TransformerEncoderLayer` structure otherwise identical.

---

## 2D (blackboard) patterns

- **2D Sin–Cos (absolute)**: deterministic, parameter-free, good extrapolation; add once.  
  MAE utility: https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py
- **Learned 2D table (absolute)**: a `(H, W, d)` parameter we add to tokens; interpolate if `H×W` changes.  
  (Common across ViT repos like timm/ViT/MAE.)
- **2D Relative Bias (logits)**: small `(2H−1)×(2W−1)` table indexed by (Δrow,Δcol); pass as `attn_mask`.  
  Swin paper/code: https://arxiv.org/abs/2103.14030 ,  
  TorchVision impl: https://docs.pytorch.org/vision/main/_modules/torchvision/models/swin_transformer.html
- **Axial / 2D RoPE**: rotate half channels by column index and half by row (a standard RoPE-to-2D trick).  
  RoPE paper: https://arxiv.org/abs/2104.09864  
  (Nice overview on 2D/ND RoPE): https://jerryxio.ng/posts/nd-rope/

---

## Which path when we want **minimal** changes?

- **Stay fully stock (`nn.Transformer`)** → **Absolute PE** (add to embeddings) or **Relative/Bias PE** (supply float `src_mask` / `attn_mask`).  
  - `nn.Transformer`: https://docs.pytorch.org/docs/stable/generated/torch.nn.Transformer.html  
  - `MultiheadAttention` mask rules: https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html
- **Need relative geometry without allocating `L×L`** → **RoPE** (small custom attention) or **FlexAttention** (custom score fn).  
  - SDPA tutorial: https://docs.pytorch.org/tutorials/intermediate/scaled_dot_product_attention_tutorial.html  
  - FlexAttention: https://pytorch.org/blog/flexattention/







