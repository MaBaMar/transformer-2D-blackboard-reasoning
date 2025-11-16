The paper doesn't actually build a “2D embedding vector” in the usual sense. Instead they reuse vanilla 1D RoPE several times on different 1D traversals of the table, and then let each attention head route itself to the traversal it finds most useful. 

Step by step implementation written (mostly) by ChatGPT and verified by me.


## 1. Recap: what RoPE is doing (§3.1)

Standard (1D) RoPE works like this: for a sequence of tokens
\(x_1, \dots, x_M\), an attention head \(h\) has queries, keys, values
\(q_m^h, k_n^h, v_n^h \in \mathbb{R}^d\).

Causal self-attention output for head \(h\) and token \(m\) is

\[
o_m^h = \sum_{n \le m} a_{m,n}^h \, v_n^h,
\]

with

\[
a_{m,n}^h =
\frac{\exp\big(f(q_m^h, k_n^h)\big)}
     {\sum_{j \le m} \exp\big(f(q_m^h, k_j^h)\big)}.
\]

RoPE defines

\[
f(q_m^h, k_n^h)
= (q_m^h)^\top R_{b,d}^{\,n-m} k_n^h,
\]

where \(R_{b,d}^m \in \mathbb{R}^{d \times d}\) is a block-diagonal
rotation matrix; each 2-dim subspace is rotated by angle
\(m \, \theta_{b,d,i}\) (with different frequencies per block).

Key property: the attention score depends only on the relative offset
\(n - m\), but you can implement it by rotating queries/keys according to
absolute position indices.

So in plain 1D RoPE you assign a single scalar position index \(m\) per
token, and then use that index to build the rotation.

---

## 2. Their core idea: represent “2D position” as several 1D positions

The paper’s setting: input is

- a question \(Q\),
- a table \(T\),
- a text instruction “Answer:”.

They concatenate those into a single token sequence

\[
X = (x_1, x_2, \dots, x_M).
\]

Then instead of a single position index per token, they assign a vector
of indices

\[
P = (p_1, \dots, p_M), \quad
p_m = (p_{m,1}, \dots, p_{m,J}),
\]

where \(p_{m,j}\) is the position index of token \(x_m\) under
permutation order \(j\).

Each permutation order \(j\) corresponds to a traversal mode over the
table (e.g. row-wise, column-wise, diagonal, Hilbert curve, …).

In this paper they actually use only two: row-wise and column-wise
traversals (\(J = 2\)).

So “2D positional encoding” = give each token multiple 1D RoPE indices,
each reflecting one way of walking through the 2D table.

Important design points from §4.3:

- **Table tokens**: for each table cell, you traverse the table in
  different orders (row-wise, column-wise, etc.) and assign indices
  accordingly.  
  – Row-wise: scan rows left→right, top→bottom.  
  – Column-wise: scan columns top→bottom, left→right.  
  – Tokens inside the same cell keep the same relative order in all
    traversals.

- **Text tokens** (question + “Answer:” + any plain text): they simply
  get the same monotonically increasing index in every permutation
  order; i.e. \(p_{m,1} = p_{m,2} = \dots\). So pure text behaves
  exactly like a normal 1D RoPE LLM.

- **Generated answer tokens**: during generation, they continue
  incrementing the position indices for all permutation orders in
  lockstep, again matching vanilla RoPE for the answer span.

So the whole 2D-ness is in how you assign these index vectors \(p_m\).

---

## 3. 2D-TPE attention: mixture of several RoPE-based attentions

Given:

- token sequence \(X\),
- per-token index vectors
  \(p_m = (p_{m,1}, \dots, p_{m,J})\),

they modify each self-attention layer like this.

### 3.1 Per-head mixture over permutation orders

For a head \(h\) and token \(x_m\), instead of one attention output they
compute one attention output per permutation order, then mix them:

\[
o_m^h = \sum_{j=1}^J r_{m,j}^h \, o_{m,j}^h. \tag{7}
\]

Here:

- \(o_{m,j}^h\) = attention output for head \(h\), token \(m\), using
  order \(j\).
- \(r_{m,j}^h\) = routing weight saying how much this head, for this
  token, trusts permutation \(j\).

The routing weights come from a small MLP router per head:

\[
r_{m,j}^h = \mathrm{Softmax}(\mathrm{MLP}(h_m^h))_j, \tag{8}
\]

where \(h_m\) is the hidden state at that layer and \(h_m^h\) is the
slice for head \(h\).

Router MLP is LLaMA-style gated FFN:

\[
\mathrm{MLP}(h_m^h)
= W_\text{down} \big(
    \mathrm{SiLU}(W_\text{up} \, h_m^h)
    \odot
    (W_\text{gate} \, h_m^h)
  \big),
\]

with

- \(W_\text{up} \in \mathbb{R}^{4d \times d}\),
- \(W_\text{gate} \in \mathbb{R}^{4d \times d}\),
- \(W_\text{down} \in \mathbb{R}^{J \times 4d}\).

So per (head, token) you get a length-\(J\) logit vector, softmax it →
routing distribution.

**Intuition**: each head + token decides “for this query, do I want to
look at the world in row-wise mode, column-wise mode, …?”.

### 3.2 Attention for a fixed permutation order \(j\)

For a given order \(j\), you just do standard causal attention with 1D
RoPE, but using position indices \(p_{m,j}\) instead of plain sequence
indices:

\[
o_{m,j}^h =
\sum_{p_{n,j} \le p_{m,j}} a_{m,n,j}^h \, v_n^h, \tag{10}
\]

with

\[
a_{m,n,j}^h =
\frac{\exp\big((q_m^h)^\top R_{b,d}^{\,p_{n,j} - p_{m,j}} k_n^h\big)}
     {\sum_{p_{i,j} \le p_{m,j}}
        \exp\big((q_m^h)^\top R_{b,d}^{\,p_{i,j} - p_{m,j}} k_i^h\big)}.
\tag{11}
\]

Key points:

- Same projection matrices to get \(q_h, k_h, v_h\): we don’t change
  Q/K/V definitions, only the RoPE and mask.
- **Causal mask in order \(j\)**: token with index \(p_{m,j}\) can
  attend only to tokens with index \(p_{n,j} \le p_{m,j}\). So causality
  is defined along that permutation’s linearization.
- Because the sequence order of \(X\) itself is “inessential” (they say
  this explicitly), they re-rank Q/K/V by increasing \(p_{m,j}\) before
  computing attention, so the causal mask is the usual lower-triangular
  mask in that order. After attention, you map outputs back to the
  original token order.

So for each permutation order:

1. Sort tokens by \(p_{\cdot,j}\) (ascending).
2. Apply standard QKV attention with RoPE on that sorted sequence.
3. Undo the sort to align outputs with original token indices.

Call that \(o_{m,j}^h\).

Then mix across \(j\) using the router weights as in (7).

---

## 4. How the “2D” indices are constructed (candidate permutation orders, §4.3)

They discuss several possibilities: row-wise, column-wise, diagonal,
Hilbert curve, Z-order curve, etc. Any such traversal induces a
permutation of the tokens and hence a scalar index for each token.

In this paper, they fix \(J = 2\):

- Order 1: row-wise traversal from top-left to bottom-right.
- Order 2: column-wise traversal from top-left to bottom-right.

For each traversal \(j\):

- Walk the table, cell by cell.
- Within each cell, traverse WordPiece/BPE tokens in reading order,
  assigning indices \(p_{m,j}\) sequentially.
- Tokens in the same cell stay contiguous and in the same relative order
  for all permutations (only distances between cells change).

For text tokens (question and “Answer:” plus any other plain text around
the table): all permutations share the same incremental index sequence;
so text-only attention is exactly like standard RoPE.

During generation, they “incrementally assign position indices to
generated tokens” in all permutation orders, again mimicking standard 1D
RoPE for the answer span.

So effectively:

- **Table cells** → different \(p_{m,j}\) for different ways of
  linearizing the 2D grid.
- **Question/Answer** → same \(p_{m,j}\) across all \(j\), so 2D-TPE
  degenerates to 1D RoPE there.

---

## 5. Training: making heads specialize to traversal modes (§4.2)

The base loss is standard language modeling loss on the answer:

\[
L_\text{nll} = -\log P(A \mid Q, T). \tag{12}
\]

But if you only use that, router distributions \(r_{m,j}^h\) could stay
diffuse (all permutation orders mixed equally), which makes the model
harder to interpret and possibly inefficient.

So they add an entropy regularization term on the router distributions:

\[
E_m^h = -\sum_{j=1}^J r_{m,j}^h \log r_{m,j}^h,
\]

\[
L_\text{ent}
= \frac{1}{M H} \sum_{m=1}^M \sum_{h=1}^H E_m^h.
\tag{13–14}
\]

Total loss:

\[
L = L_\text{nll} + \lambda \, L_\text{ent}. \tag{15}
\]

They minimize \(L_\text{ent}\), thus pushing \(r_{m,\cdot}^h\) towards
low-entropy distributions — ideally each head+token strongly prefers one
permutation order instead of blending them.

**Intuition**: each head becomes something like “a row head” or “a
column head” (or later “a diagonal head”, etc.), for specific
regions/tokens.
