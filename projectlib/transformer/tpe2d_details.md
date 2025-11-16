The paper doesn't actually build a “2D embedding vector” in the usual sense. Instead they reuse vanilla 1D RoPE several times on different 1D traversals of the table, and then let each attention head route itself to the traversal it finds most useful. 

Step by step implementation written (mostly) by ChatGPT and verified by me.

1. Recap: what RoPE is doing (§3.1)

Standard (1D) RoPE works like this: for a sequence of tokens 
𝑥
1
,
…
,
𝑥
𝑀
x
1
	​

,…,x
M
	​

, an attention head 
ℎ
h has queries, keys, values

𝑞
𝑚
ℎ
,
𝑘
𝑛
ℎ
,
𝑣
𝑛
ℎ
∈
𝑅
𝑑
.
q
m
h
	​

,k
n
h
	​

,v
n
h
	​

∈R
d
.

Causal self-attention output for head 
ℎ
h and token 
𝑚
m is

𝑜
𝑚
ℎ
=
∑
𝑛
≤
𝑚
𝑎
𝑚
,
𝑛
ℎ
 
𝑣
𝑛
ℎ
,
o
m
h
	​

=
n≤m
∑
	​

a
m,n
h
	​

v
n
h
	​

,

with

𝑎
𝑚
,
𝑛
ℎ
=
exp
⁡
(
𝑓
(
𝑞
𝑚
ℎ
,
𝑘
𝑛
ℎ
)
)
∑
𝑗
≤
𝑚
exp
⁡
(
𝑓
(
𝑞
𝑚
ℎ
,
𝑘
𝑗
ℎ
)
)
.
a
m,n
h
	​

=
∑
j≤m
	​

exp(f(q
m
h
	​

,k
j
h
	​

))
exp(f(q
m
h
	​

,k
n
h
	​

))
	​

.

RoPE defines

𝑓
(
𝑞
𝑚
ℎ
,
𝑘
𝑛
ℎ
)
=
(
𝑞
𝑚
ℎ
)
⊤
𝑅
𝑏
,
𝑑
𝑛
−
𝑚
𝑘
𝑛
ℎ
,
f(q
m
h
	​

,k
n
h
	​

)=(q
m
h
	​

)
⊤
R
b,d
n−m
	​

k
n
h
	​

,

where 
𝑅
𝑏
,
𝑑
𝑚
∈
𝑅
𝑑
×
𝑑
R
b,d
m
	​

∈R
d×d
 is a block-diagonal rotation matrix; each 2-dim subspace is rotated by angle 
𝑚
𝜃
𝑏
,
𝑑
,
𝑖
mθ
b,d,i
	​

 (with different frequencies per block).

Key property: the attention score depends only on the relative offset 
𝑛
−
𝑚
n−m, but you can implement it by rotating queries/keys according to absolute position indices.

So in plain 1D RoPE you assign a single scalar position index 
𝑚
m per token, and then use that index to build the rotation.

2. Their core idea: represent “2D position” as several 1D positions

The paper’s setting: input is

a question 
𝑄
Q,

a table 
𝑇
T,

a text instruction “Answer:”.

They concatenate those into a single token sequence

𝑋
=
(
𝑥
1
,
𝑥
2
,
…
,
𝑥
𝑀
)
.
X=(x
1
	​

,x
2
	​

,…,x
M
	​

).

Then instead of a single position index per token, they assign a vector of indices

𝑃
=
(
𝑝
1
,
…
,
𝑝
𝑀
)
,
𝑝
𝑚
=
(
𝑝
𝑚
,
1
,
…
,
𝑝
𝑚
,
𝐽
)
,
P=(p
1
	​

,…,p
M
	​

),p
m
	​

=(p
m,1
	​

,…,p
m,J
	​

),

where

𝑝
𝑚
,
𝑗
 is the position index of token 
𝑥
𝑚
 under permutation order 
𝑗
.
p
m,j
	​

 is the position index of token x
m
	​

 under permutation order j.

Each permutation order 
𝑗
j corresponds to a traversal mode over the table (e.g. row-wise, column-wise, diagonal, Hilbert curve,…).

In this paper they actually use only two: row-wise and column-wise traversals (
𝐽
=
2
J=2).

So “2D positional encoding” = give each token multiple 1D RoPE indices, each reflecting one way of walking through the 2D table.

Important design points from §4.3:

Table tokens: for each table cell, you traverse the table in different orders (row-wise, column-wise, etc.) and assign indices accordingly.
– Row-wise: scan rows left→right, top→bottom.
– Column-wise: scan columns top→bottom, left→right.
– Tokens inside the same cell keep the same relative order in all traversals.

Text tokens (question + “Answer:” + any plain text): they simply get the same monotonically increasing index in every permutation order; i.e. 
𝑝
𝑚
,
1
=
𝑝
𝑚
,
2
=
…
p
m,1
	​

=p
m,2
	​

=…. So pure text behaves exactly like a normal 1D RoPE LLM.

Generated answer tokens: during generation, they continue incrementing the position indices for all permutation orders in lockstep, again matching vanilla RoPE for the answer span.

So the whole 2D-ness is in how you assign these index vectors 
𝑝
𝑚
p
m
	​

.

3. 2D-TPE attention: mixture of several RoPE-based attentions

Given:

token sequence 
𝑋
X,

per-token index vectors 
𝑝
𝑚
=
(
𝑝
𝑚
,
1
,
.
.
.
,
𝑝
𝑚
,
𝐽
)
p
m
	​

=(p
m,1
	​

,...,p
m,J
	​

),

they modify each self-attention layer like this.

3.1 Per-head mixture over permutation orders

For a head 
ℎ
h and token 
𝑥
𝑚
x
m
	​

, instead of one attention output they compute one attention output per permutation order, then mix them:

	
𝑜
𝑚
ℎ
=
∑
𝑗
=
1
𝐽
𝑟
𝑚
,
𝑗
ℎ
 
𝑜
𝑚
,
𝑗
ℎ
.
		
(7)
o
m
h
	​

=
j=1
∑
J
	​

r
m,j
h
	​

o
m,j
h
	​

.
(7)

𝑜
𝑚
,
𝑗
ℎ
o
m,j
h
	​

 = attention output for head 
ℎ
h, token 
𝑚
m, using order 
𝑗
j.

𝑟
𝑚
,
𝑗
ℎ
r
m,j
h
	​

 = routing weight saying how much this head, for this token, trusts permutation 
𝑗
j.

The routing weights come from a small MLP router per head:

	
𝑟
𝑚
,
𝑗
ℎ
=
Softmax
(
MLP
(
ℎ
𝑚
ℎ
)
)
𝑗
,
		
(8)
r
m,j
h
	​

=Softmax(MLP(h
m
h
	​

))
j
	​

,
(8)

where 
ℎ
𝑚
h
m
	​

 is the hidden state at that layer and 
ℎ
𝑚
ℎ
h
m
h
	​

 is the slice for head 
ℎ
h.

Router MLP is LLaMA-style gated FFN:

MLP
(
ℎ
𝑚
ℎ
)
=
𝑊
down
(
SiLU
(
𝑊
up
 
ℎ
𝑚
ℎ
)
⊙
(
𝑊
gate
 
ℎ
𝑚
ℎ
)
)
,
MLP(h
m
h
	​

)=W
down
	​

(SiLU(W
up
	​

h
m
h
	​

)⊙(W
gate
	​

h
m
h
	​

)),

with

𝑊
up
∈
𝑅
4
𝑑
×
𝑑
W
up
	​

∈R
4d×d
,

𝑊
gate
∈
𝑅
4
𝑑
×
𝑑
W
gate
	​

∈R
4d×d
,

𝑊
down
∈
𝑅
𝐽
×
4
𝑑
W
down
	​

∈R
J×4d
.

So per (head, token) you get a length-
𝐽
J logit vector, softmax it → routing distribution.

Intuition: each head + token decides “for this query, do I want to look at the world in row-wise mode, column-wise mode, …?”.

3.2 Attention for a fixed permutation order 
𝑗
j

For a given order 
𝑗
j, you just do standard causal attention with 1D RoPE, but using position indices 
𝑝
𝑚
,
𝑗
p
m,j
	​

 instead of plain sequence indices:

	
𝑜
𝑚
,
𝑗
ℎ
=
∑
𝑝
𝑛
,
𝑗
≤
𝑝
𝑚
,
𝑗
𝑎
𝑚
,
𝑛
,
𝑗
ℎ
 
𝑣
𝑛
ℎ
,
		
(10)
o
m,j
h
	​

=
p
n,j
	​

≤p
m,j
	​

∑
	​

a
m,n,j
h
	​

v
n
h
	​

,
(10)

with

	
𝑎
𝑚
,
𝑛
,
𝑗
ℎ
=
exp
⁡
(
(
𝑞
𝑚
ℎ
)
⊤
𝑅
𝑏
,
𝑑
𝑝
𝑛
,
𝑗
−
𝑝
𝑚
,
𝑗
𝑘
𝑛
ℎ
)
∑
𝑝
𝑖
,
𝑗
≤
𝑝
𝑚
,
𝑗
exp
⁡
(
(
𝑞
𝑚
ℎ
)
⊤
𝑅
𝑏
,
𝑑
𝑝
𝑖
,
𝑗
−
𝑝
𝑚
,
𝑗
𝑘
𝑖
ℎ
)
.
		
(11)
a
m,n,j
h
	​

=
∑
p
i,j
	​

≤p
m,j
	​

	​

exp((q
m
h
	​

)
⊤
R
b,d
p
i,j
	​

−p
m,j
	​

	​

k
i
h
	​

)
exp((q
m
h
	​

)
⊤
R
b,d
p
n,j
	​

−p
m,j
	​

	​

k
n
h
	​

)
	​

.
(11)

Key points:

Same projection matrices to get 
𝑞
ℎ
,
𝑘
ℎ
,
𝑣
ℎ
q
h
,k
h
,v
h
: we don’t change Q/K/V definitions, only the RoPE and mask.

Causal mask in order 
𝑗
j: token with index 
𝑝
𝑚
,
𝑗
p
m,j
	​

 can attend only to tokens with index 
𝑝
𝑛
,
𝑗
≤
𝑝
𝑚
,
𝑗
p
n,j
	​

≤p
m,j
	​

. So causality is defined along that permutation’s linearization.

Because the sequence order of 
𝑋
X itself is “inessential” (they say this explicitly), they re-rank Q/K/V by increasing 
𝑝
𝑚
,
𝑗
p
m,j
	​

 before computing attention, so the causal mask is the usual lower-triangular mask in that order. After attention, you map outputs back to the original token order.

So for each permutation order:

Sort tokens by 
𝑝
⋅
,
𝑗
p
⋅,j
	​

 (ascending).

Apply standard QKV attention with RoPE on that sorted sequence.

Undo the sort to align outputs with original token indices.

Call that 
𝑜
𝑚
,
𝑗
ℎ
o
m,j
h
	​

.

Then mix across 
𝑗
j using the router weights as in (7).

4. How the “2D” indices are constructed (candidate permutation orders, §4.3)

They discuss several possibilities: row-wise, column-wise, diagonal, Hilbert curve, Z-order curve, etc. Any such traversal induces a permutation of the tokens and hence a scalar index for each token.

In this paper, they fix 
𝐽
=
2
J=2:

Order 1: row-wise traversal from top-left to bottom-right.

Order 2: column-wise traversal from top-left to bottom-right.

For each traversal 
𝑗
j:

Walk the table, cell by cell.

Within each cell, traverse WordPiece/BPE tokens in reading order, assigning indices 
𝑝
𝑚
,
𝑗
p
m,j
	​

 sequentially.

Tokens in the same cell stay contiguous and in the same relative order for all permutations (only distances between cells change).

For text tokens (question and “Answer:” plus any other plain text around the table): all permutations share the same incremental index sequence; so text-only attention is exactly like standard RoPE.

During generation, they “incrementally assign position indices to generated tokens” in all permutation orders, again mimicking standard 1D RoPE for the answer span.

So effectively:

Table cells → different 
𝑝
𝑚
,
𝑗
p
m,j
	​

 for different ways of linearizing the 2D grid.

Question/Answer → same 
𝑝
𝑚
,
𝑗
p
m,j
	​

 across all 
𝑗
j, so 2D-TPE degenerates to 1D RoPE there.

5. Training: making heads specialize to traversal modes (§4.2)

The base loss is standard language modeling loss on the answer:

	
𝐿
nll
=
−
log
⁡
𝑃
(
𝐴
∣
𝑄
,
𝑇
)
.
		
(12)
L
nll
	​

=−logP(A∣Q,T).
(12)

But if you only use that, router distributions 
𝑟
𝑚
,
𝑗
ℎ
r
m,j
h
	​

 could stay diffuse (all permutation orders mixed equally), which makes the model harder to interpret and possibly inefficient.

So they add an entropy regularization term on the router distributions:

𝐸
𝑚
ℎ
=
−
∑
𝑗
=
1
𝐽
𝑟
𝑚
,
𝑗
ℎ
log
⁡
𝑟
𝑚
,
𝑗
ℎ
,
E
m
h
	​

=−
j=1
∑
J
	​

r
m,j
h
	​

logr
m,j
h
	​

,
	
𝐿
ent
=
1
𝑀
𝐻
∑
𝑚
=
1
𝑀
∑
ℎ
=
1
𝐻
𝐸
𝑚
ℎ
.
		
(13–14)
L
ent
	​

=
MH
1
	​

m=1
∑
M
	​

h=1
∑
H
	​

E
m
h
	​

.
(13–14)

Total loss:

	
𝐿
=
𝐿
nll
+
𝜆
 
𝐿
ent
.
		
(15)
L=L
nll
	​

+λL
ent
	​

.
(15)

They minimize 
𝐿
ent
L
ent
	​

, thus pushing 
𝑟
𝑚
,
⋅
ℎ
r
m,⋅
h
	​

 towards low-entropy distributions — ideally each head+token strongly prefers one permutation order instead of blending them.

Intuition: each head becomes something like “a row head” or “a column head” (or later “a diagonal head”, etc.), for specific regions/tokens.
