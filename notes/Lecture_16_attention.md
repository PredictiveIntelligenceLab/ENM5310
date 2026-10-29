# Lecture 16 — Attention: From Fixing RNNs to a New Kind of Layer

**ENM 5310 — Data-driven Modeling and Probabilistic Scientific Computing**

**Builds on:** Lecture 9 (GP regression: predictions as weighted sums of training targets), Lecture 12 (MLPs), Lecture 13 (softmax saturation; vanishing gradients through products of Jacobians), Lecture 14 (convolution: fixed, local, shared weights; translation equivariance), Lecture 15 (RNNs, BPTT and vanishing gradients, LSTMs, the autoregressive factorization, teacher forcing).

**Reading:** Murphy I §15.4. Zhang, Lipton, Li & Smola, *Dive into Deep Learning* (d2l.ai), §§11.1–11.6 — the most gentle introduction available, and it also starts from kernel regression. Alammar (2018), *The Illustrated Transformer* (blog), first half. Optional: Bahdanau, Cho & Bengio (2015), §§1–3; Weng (2018), *Attention? Attention!* (Lil'Log).

**The point of this lecture.** Attention looks intimidating because it usually arrives all at once, as one equation with three matrices and a softmax. It is easier to build it in steps, each of which fixes a problem from the previous one:

0. Some tasks map a sequence to a sequence of a different length. The RNN **encoder–decoder** handles them, but squeezes the entire input into one vector.
1. **Fix:** let the decoder look back at every input position, weighted by relevance (Bahdanau attention).
2. "Weighted by relevance" is a **soft dictionary lookup**: match a query against keys, return a weighted average of values.
3. You have seen this before. With a Gaussian weighting it is **kernel regression**, and with normalized inputs it becomes **dot-product attention**.
4. Applied by a sequence to itself, it becomes **self-attention**: a layer in which every element reads from every other, with weights computed from content.
5. Self-attention ignores order, so we add **positional information**, run several attentions in parallel (**multi-head**), and **mask** the future for autoregressive models.

Steps 0–4 are the core of today. Step 5 can spill into Lecture 17, where these pieces are assembled into the transformer.

---

### Notation

Attention is conventionally written with **tokens as rows**, unlike the column-vector convention of Lectures 12–15. We follow the convention so that the formulas match every paper you will read.

| Symbol | Meaning |
|---|---|
| $n$, $d$ | number of tokens (sequence elements), model width |
| $X\in\mathbb{R}^{n\times d}$ | input sequence, one token per row, $x_i^\top$ |
| $q,\ k_j,\ v_j$ | a query, the $j$-th key, the $j$-th value |
| $Q\in\mathbb{R}^{n\times d_k}$, $K\in\mathbb{R}^{m\times d_k}$, $V\in\mathbb{R}^{m\times d_v}$ | queries, keys and values stacked as rows |
| $W_Q,W_K\in\mathbb{R}^{d\times d_k}$, $W_V\in\mathbb{R}^{d\times d_v}$ | learned projections |
| $\alpha_{ij}$ | attention weight of query $i$ on key $j$; $\sum_j\alpha_{ij}=1$ |
| $\operatorname{softmax}(z)_j = e^{z_j}/\sum_{j'}e^{z_{j'}}$ | applied row-wise to matrices |
| $h_j$, $s_t$ | encoder state at input position $j$, decoder state at output step $t$ (§0) |

### Timing budget (90 min)

| # | Segment | Min |
|---|---|---|
| 0 | Sequence-to-sequence: the RNN encoder–decoder and its limitations | 12 |
| 1 | Bahdanau attention: removing the bottleneck | 13 |
| 2 | Attention as a soft dictionary lookup | 12 |
| 3 | You have seen this before: kernel regression | 8 |
| 4 | Scaled dot-product attention | 10 |
| 5 | Self-attention | 15 |
| 6 | Positional information | 8 |
| 7 | Multi-head attention and causal masking | 10 |
| 8 | Summary | 2 |

If time runs short, §§6–7 open Lecture 17.

**Learning objectives.** After this lecture you should be able to:

1. Describe the RNN encoder–decoder for sequence-to-sequence tasks, and state the three limitations that attention addresses.
2. Write Bahdanau attention and interpret its weights as a soft alignment.
3. Describe attention as a soft dictionary lookup with queries, keys and values, and compute it by hand for a small example.
4. Show that Nadaraya–Watson kernel regression is attention with a Gaussian score, and derive dot-product attention from it.
5. Write scaled dot-product attention in matrix form, and explain the $1/\sqrt{d_k}$ factor.
6. Explain self-attention, how it differs from convolution and recurrence, and why it is permutation equivariant.
7. Explain why positional encodings are needed, and what multi-head attention and causal masks do.

---

## 0. Sequence-to-sequence: the RNN encoder–decoder and its limitations (12 min)

### 0.1 The task

Lecture 15's RNNs produced one output per input step (forecasting, tagging) or one output per sequence (classification). Many tasks need something else: an input sequence $x_{1:S}$ mapped to an output sequence $y_{1:T}$ of a **different length, with no step-by-step correspondence**. Translation is the classic case: "I like tea" has three words, "J'aime le thé" has four, and the word order need not match. Scientific versions are common:

- Map a forcing history (a loading path, a rainfall record) to a response sampled on a different time grid.
- Map a short window of observations to a long forecast.
- Map a measured spectrum to the sequence of species or layers that produced it.

An RNN that emits $y_t$ when it reads $x_t$ cannot do this: it has no way to produce four outputs from three inputs, or to produce the first output before reading the last input.

### 0.2 The encoder–decoder

Sutskever, Vinyals and Le (2014) and Cho et al. (2014) split the job between two RNNs.

1. **Encoder.** An RNN reads the whole input, $h_j = f_{\text{enc}}(h_{j-1},x_j)$ for $j=1,\dots,S$. Its final state is the **context vector** $m = h_S$, a fixed-size summary of the entire input.
2. **Decoder.** A second RNN, initialized with $s_0=m$, generates the output autoregressively (Lecture 15 §1). At each step it takes its previous output as input,
$$
s_t = f_{\text{dec}}(s_{t-1},\,y_{t-1}),\qquad p(y_t\mid y_{<t},x_{1:S}) = g(s_t),
$$
and stops when it emits a special end-of-sequence token. The output length is decided by the model, not fixed by the input.
3. **Training** is maximum likelihood of the output sequence given the input, $-\sum_t\log p(y_t\mid y_{<t},x_{1:S})$, with teacher forcing (Lecture 15 §6): the true $y_{t-1}$ is fed to the decoder during training.

Picture it as $x_1\to x_2\to\cdots\to x_S \Rightarrow m \Rightarrow y_1\to y_2\to\cdots\to y_T$: one pipe in, one vector in the middle, one pipe out. With deep LSTMs and large datasets this worked remarkably well for translation, and it is the template for everything in this lecture.

### 0.3 Three limitations

**1. The bottleneck.** Everything the decoder knows about the input must fit into one fixed-size vector $m$, whether the input has 5 tokens or 500. Cho et al. (2014) found that translation quality dropped sharply as sentences got longer. The scientific analogue: summarize a 10,000-step sensor history into one 256-dimensional vector, then reconstruct from it every detail the decoder needs.

**2. Long paths.** Information about $x_1$ reaches the decoder only after passing through $S$ encoder steps, and reaches $y_T$ only after $T$ more. By Lecture 15 §4, the gradient along that path is a product of Jacobians and tends to vanish. Gating helps but does not shorten the path. (Sutskever et al. found, tellingly, that simply *reversing* the input sentence improved results, because it put the first input words close to the first output words.)

**3. Sequential computation.** $h_j$ needs $h_{j-1}$, and $s_t$ needs $s_{t-1}$. A length-$n$ sequence takes $n$ steps that cannot run in parallel, on hardware built for large parallel matrix products.

Attention was invented to fix the first, and turned out to fix all three.

---

## 1. Bahdanau attention: removing the bottleneck (13 min)

### 1.1 The idea

Bahdanau, Cho and Bengio (2015) kept the RNN encoder and decoder but stopped discarding the encoder's intermediate states. The encoder produces one vector $h_j$ per input position. At every output step $t$, the decoder builds its **own** context vector from **all** of them, weighted by how relevant each is right now:

$$
e_{tj} = \operatorname{score}(s_{t-1},\,h_j), \qquad
\alpha_{tj} = \frac{\exp(e_{tj})}{\sum_{j'=1}^{S}\exp(e_{tj'})}, \qquad
c_t = \sum_{j=1}^{S}\alpha_{tj}\,h_j .
$$

The decoder then updates its state using $c_t$ as an additional input, $s_t = f(s_{t-1},\,y_{t-1},\,c_t)$, and predicts $y_t$ from $s_t$ and $c_t$.

In the original paper the score is a small MLP, $\operatorname{score}(s,h) = v^\top\tanh(W s + U h)$ (**additive attention**). Luong et al. (2015) showed a simpler bilinear score, $s^\top W h$ (**multiplicative attention**), works as well and is cheaper. Everything is differentiable and trained end to end by ordinary maximum likelihood.

### 1.2 Reading the weights

For each output step, $(\alpha_{t1},\dots,\alpha_{tS})$ is a probability distribution over input positions: where the decoder is "looking." Plot $\alpha$ as a matrix — outputs on one axis, inputs on the other — and for translation you see a learned **alignment**. It is mostly diagonal, since word order is mostly preserved, with off-diagonal blocks where it is not. In English–French translation, for instance, adjective–noun order flips ("European Economic Area" becomes "zone économique européenne"), and the weights show the decoder looking at "Area" first. Nobody provided alignment labels; they emerge because they reduce the loss.

### 1.3 What it fixed

| Limitation (§0) | With attention |
|---|---|
| Bottleneck | Gone: the context $c_t$ is recomputed at every step from all $S$ encoder states |
| Long paths | Every input position is one weighted sum away from every output step |
| Sequential computation | Still there: the encoder and decoder are still RNNs |

The third row is the opening for Vaswani et al. (2017): if attention already provides every connection that matters, do we need the recurrence at all? Before answering, we strip attention down to what it actually computes.

> **Remember.** Bahdanau attention: at each output step, score every encoder state against the current decoder state, softmax the scores into weights, and take the weighted average as context. The weights are a learned soft alignment.

---

## 2. Attention as a soft dictionary lookup (12 min)

### 2.1 Three roles

Strip away the RNNs, and §1.1 does the following. Something asks a question (the decoder state). A collection of items each advertise what they contain (the encoder states, used in the score). Each item also carries content to hand back (the encoder states again, averaged into $c_t$). Give the three roles names:

- **Query** $q$: what I am looking for.
- **Keys** $k_1,\dots,k_m$: what each item advertises, used for matching.
- **Values** $v_1,\dots,v_m$: what each item returns if selected.

**Attention** is

$$
\operatorname{Attn}(q;\,K,V) = \sum_{j=1}^{m}\alpha_j\,v_j, \qquad \alpha_j = \operatorname{softmax}_j\big(\operatorname{score}(q,k_j)\big).
$$

In Bahdanau's model, the keys and the values are the same vectors $h_j$. Separating them is useful because what you match on need not be what you retrieve. A library catalog is searched by title and subject (keys), but what you take home is the book (values). A network of sensors is searched by location and type (keys), but what you want back is the reading (values).

### 2.2 Hard versus soft lookup

A Python dictionary does a **hard** lookup: find the one key that matches the query exactly and return its value. That is not differentiable — a small change in the query either changes nothing or jumps to a different key — so it cannot be learned by gradient descent.

Attention does a **soft** lookup. Every key matches to some degree, the softmax turns degrees of match into weights that sum to one, and the result is a blend of all the values. As the scores sharpen, the softmax approaches a one-hot vector and the soft lookup approaches the hard one. Because every step is differentiable, the network can learn what to ask for (queries), how to advertise (keys) and what to return (values).

### 2.3 A worked example

Three sensors on a beam advertise their positions as keys and report temperatures as values:

| Sensor | Key $k_j$ (position, m) | Value $v_j$ (temperature, °C) |
|---|---|---|
| 1 | 0.0 | 20 |
| 2 | 1.0 | 30 |
| 3 | 2.0 | 60 |

We want the temperature at $q = 0.9$ m, scoring each sensor by negative squared distance, $\operatorname{score}(q,k_j) = -(q-k_j)^2/(2\ell^2)$, with $\ell=0.5$:

1. Scores: $-(0.9)^2/0.5 = -1.62$, $\ -(0.1)^2/0.5 = -0.02$, $\ -(1.1)^2/0.5 = -2.42$.
2. Exponentials: $0.198,\ 0.980,\ 0.089$; sum $1.267$.
3. Weights: $\alpha = (0.156,\ 0.774,\ 0.070)$.
4. Output: $0.156\cdot20 + 0.774\cdot30 + 0.070\cdot60 \approx 30.5$ °C.

The nearest sensor dominates, but not exclusively. Halve $\ell$ and the weights sharpen toward $(0,1,0)$ — a hard lookup. Double it and they flatten toward a plain average. The width $\ell$ acts as a **temperature**.

> **Remember.** Attention = soft dictionary lookup. Compare the query with every key, softmax the scores into weights, return the weighted average of the values. Keys are for matching; values are for returning.

---

## 3. You have seen this before: kernel regression (8 min)

### 3.1 Nadaraya–Watson

The example in §2.3 is a classical estimator. Given data $(x_j,y_j)$, the **Nadaraya–Watson** kernel regression estimate (Nadaraya, 1964; Watson, 1964) at a point $x^\star$ is

$$
\hat f(x^\star) = \sum_{j=1}^{m}\frac{\kappa(x^\star,x_j)}{\sum_{j'}\kappa(x^\star,x_{j'})}\,y_j .
$$

With a Gaussian kernel $\kappa(x,x') = \exp\big(-\lVert x-x'\rVert^2/2\ell^2\big)$, the normalized weights are exactly a softmax:

$$
\frac{\kappa(x^\star,x_j)}{\sum_{j'}\kappa(x^\star,x_{j'})} = \operatorname{softmax}_j\Big(-\frac{\lVert x^\star-x_j\rVert^2}{2\ell^2}\Big).
$$

**Nadaraya–Watson regression is attention** with query $x^\star$, keys $x_j$, values $y_j$, and a Gaussian score.

Compare the GP predictive mean of Lecture 9, $\mu(x^\star) = k(x^\star,X)\,(K+\sigma^2I)^{-1}y$. It is also a weighted sum of training targets, but its weights account for correlations between training points through the inverse, and they need not be positive or sum to one. Nadaraya–Watson skips the inverse and simply normalizes.

What attention adds is **learning**: the kernel is not fixed in advance but parameterized, and queries and keys are produced by learned maps.

### 3.2 From the Gaussian kernel to the dot product

Expand the Gaussian score:

$$
-\frac{\lVert q-k_j\rVert^2}{2\ell^2} = -\frac{\lVert q\rVert^2}{2\ell^2} + \frac{q^\top k_j}{\ell^2} - \frac{\lVert k_j\rVert^2}{2\ell^2}.
$$

1. The first term is the same for every $j$. A softmax is unchanged by adding a constant to all its inputs, so the term drops out.
2. If all keys have the same norm, the third term is also constant across $j$ and drops out.
3. What remains is

$$
\alpha_j = \operatorname{softmax}_j\Big(\frac{q^\top k_j}{\ell^2}\Big).
$$

**Dot-product attention is Gaussian kernel smoothing on normalized keys, with $\ell^2$ playing the role of a temperature.** The dot product is a similarity: large when $q$ and $k_j$ point the same way. It is also cheap: all query–key pairs together are a single matrix product.

> **Remember.** Attention generalizes kernel regression: a softmax over similarities, applied to values. With a Gaussian kernel it is Nadaraya–Watson; with normalized keys the Gaussian score becomes a dot product.

---

## 4. Scaled dot-product attention (10 min)

### 4.1 Matrix form

Stack $n$ queries as rows of $Q\in\mathbb{R}^{n\times d_k}$, and $m$ keys and values as rows of $K\in\mathbb{R}^{m\times d_k}$ and $V\in\mathbb{R}^{m\times d_v}$. All $n\times m$ scores are the entries of $QK^\top$. Softmax each row, then mix the values:

$$
\boxed{\;\operatorname{Attention}(Q,K,V) = \operatorname{softmax}\!\Big(\frac{QK^\top}{\sqrt{d_k}}\Big)\,V\;}
$$

with shapes $(n\times d_k)(d_k\times m)\to n\times m$ weights, then $(n\times m)(m\times d_v)\to n\times d_v$ outputs. Row $i$ of the output is query $i$'s soft lookup. This is the equation of Vaswani et al. (2017). Every part of it is now familiar except the $\sqrt{d_k}$.

### 4.2 Why divide by $\sqrt{d_k}$

Suppose the entries of $q$ and $k$ are independent with mean 0 and variance 1, as they roughly are at a good initialization (Lecture 13). Then

$$
\operatorname{Var}\big[q^\top k\big] = \sum_{i=1}^{d_k}\operatorname{Var}[q_ik_i] = d_k .
$$

With $d_k=64$, raw scores have standard deviation 8, so typical gaps between scores are of order 10 or more. The softmax of such scores is nearly one-hot, and a saturated softmax has nearly zero gradient — the saturation problem of Lecture 13 in a new place. Dividing by $\sqrt{d_k}$ restores unit variance, so the softmax starts in its responsive regime whatever the width. In the language of §3, it fixes the temperature.

### 4.3 Shapes in one example

A sequence of $n=6$ tokens, model width $d=16$, and $d_k=d_v=8$:

| Quantity | Shape |
|---|---|
| $X$ | $6\times16$ |
| $Q=XW_Q$, $K=XW_K$, $V=XW_V$ | each $6\times8$ |
| $QK^\top/\sqrt8$ | $6\times6$ scores |
| softmax by row | $6\times6$ weights, rows sum to 1 |
| weights $\times V$ | $6\times8$ output |

> **Remember.** $\operatorname{softmax}(QK^\top/\sqrt{d_k})\,V$: all similarities at once, a softmax per row, a weighted average of values. The $\sqrt{d_k}$ keeps the scores at unit variance so the softmax does not saturate.

---

## 5. Self-attention (15 min)

### 5.1 A sequence attending to itself

In Bahdanau attention the queries come from one sequence (the decoder) and the keys and values from another (the encoder): **cross-attention**. In **self-attention**, all three come from the same sequence, through three learned projections:

$$
Q = XW_Q,\qquad K = XW_K,\qquad V = XW_V,\qquad Y = \operatorname{softmax}\!\Big(\frac{QK^\top}{\sqrt{d_k}}\Big)V .
$$

Every token issues a query, every token offers a key and a value, and every token's output $y_i$ is a weighted average of all tokens' values. The output has one row per input token — a new representation of each token, informed by every other.

The three projections let a token ask for something different from what it advertises. In "The animal didn't cross the street because **it** was too tired," a useful representation of "it" should borrow from "animal." Its query can look for "a noun this pronoun could refer to," while "animal" advertises, through its key, "an animate noun." Trained models do learn heads with this behavior.

### 5.2 A hand computation

Three tokens with $d=2$: $x_1=(1,0)$, $x_2=(0,1)$, $x_3=(1,1)$. To keep the arithmetic simple, take $W_Q=W_K=W_V=I$ and skip the $\sqrt{d_k}$ scaling.

1. Scores $XX^\top$:
$$
\begin{pmatrix}1&0&1\\0&1&1\\1&1&2\end{pmatrix}.
$$
2. Softmax of row 1, $(1,0,1)$: $(e,1,e)/(2e+1) \approx (0.42,\ 0.16,\ 0.42)$.
3. Softmax of row 3, $(1,1,2)$: $(e,e,e^2)/(2e+e^2)\approx(0.21,\ 0.21,\ 0.58)$.
4. Outputs: $y_1\approx0.42(1,0)+0.16(0,1)+0.42(1,1) = (0.84,\ 0.58)$, and $y_3\approx0.21(1,0)+0.21(0,1)+0.58(1,1)=(0.79,\ 0.79)$.

Token 1 is pulled toward the tokens that resemble it (itself and token 3) and picks up some of token 3's second component. Each output is a context-dependent blend.

### 5.3 How self-attention differs from convolution and recurrence

Every layer we have seen mixes information across positions. They differ in *how the mixing weights are chosen*:

| Layer | Output at position $i$ | Mixing weights are… |
|---|---|---|
| Dense (MLP) | $\sum_j W_{ij}x_j$ | fixed after training, different for every pair |
| Convolution | $\sum_j w_{j-i}\,x_j$ | fixed, shared, depend only on the offset, local |
| Recurrence | $h_i = f(h_{i-1},x_i)$ | implicit, passed step by step through a state |
| Self-attention | $\sum_j\alpha_{ij}(X)\,v_j$ | **computed from the content**, global |

A convolution applies the same stencil everywhere, whatever the data. Self-attention recomputes its "stencil" for every input: which positions matter, and how much, depends on what is in them. It is global — position 1 can read position 1000 in one step — and since all rows are computed by matrix products, it is fully parallel over the sequence. Of the three limitations in §0, the bottleneck and the long paths were fixed by attention; sequential computation is fixed by dropping recurrence altogether.

**Scientific examples.**
- *Interacting particles.* Treat each atom or particle as a token. Self-attention computes a learned, state-dependent interaction weight between every pair — a learned analogue of a pairwise force law, with no cutoff radius fixed in advance.
- *Sparse sensors to a field.* Use cross-attention with queries at the locations where you want the field and keys and values from the sensors (positions and readings). This is the Nadaraya–Watson interpolation of §3 with a learned kernel, and several neural operators decode fields in exactly this way.
- *Global coupling on a grid.* A CNN needs many layers to connect distant points of a field (Lecture 14 §4). One self-attention layer connects them directly, which matters for problems with long-range coupling such as pressure in incompressible flow.

### 5.4 Self-attention ignores order

Permute the tokens: replace $X$ by $PX$ for a permutation matrix $P$. Then $Q$, $K$, $V$ become $PQ$, $PK$, $PV$, and

$$
\operatorname{softmax}\!\Big(\frac{PQ(PK)^\top}{\sqrt{d_k}}\Big)PV = P\operatorname{softmax}\!\Big(\frac{QK^\top}{\sqrt{d_k}}\Big)P^\top PV = P\,Y,
$$

using $P^\top P = I$ and the fact that permuting the rows and columns of a matrix permutes the rows of its row-wise softmax in the same way. **Self-attention is permutation equivariant**: shuffle the inputs and the outputs shuffle identically, with no other change. It treats its input as a *set*.

That is exactly right for a set of particles, and exactly wrong for a sentence or a time series: "dog bites man" and "man bites dog" contain the same tokens. Compare Lecture 14, where convolution was equivariant to *translations* — a smaller symmetry group, which preserves relative position. Self-attention is equivariant to *all* permutations, which preserves nothing about position. When order matters, it must be supplied (§6).

> **Remember.** Self-attention: $Q$, $K$, $V$ are three projections of the same sequence; every output is a content-weighted average over all positions. Unlike convolution, its weights depend on the data and are global; unlike recurrence, it is parallel. It is permutation equivariant, so it knows nothing about order by itself.

---

## 6. Positional information (8 min)

Since self-attention sees a set, we add position to each token's content before the first layer: $x_i \leftarrow x_i + p_i$, where $p_i\in\mathbb{R}^d$ encodes position $i$.

**Learned positional embeddings** are one trainable vector per position. Simple, but they say nothing about positions never seen in training.

**Sinusoidal encodings** (Vaswani et al., 2017) use sines and cosines at geometrically spaced frequencies:

$$
p_{i,2r} = \sin\big(i\,\omega_r\big),\qquad p_{i,2r+1}=\cos\big(i\,\omega_r\big),\qquad \omega_r = 10000^{-2r/d}.
$$

Low-frequency components distinguish distant positions; high-frequency components distinguish neighbors. For each frequency, the pair $(\sin i\omega,\cos i\omega)$ at position $i+\Delta$ is a fixed rotation of the pair at position $i$, so relative offsets are easy for linear maps to read off.

These are the **Fourier features** that cure spectral bias in coordinate networks (Lecture 14 §0), and the parallel is direct: in scientific problems, the "position" of a token is often a physical coordinate $(x,y,z,t)$, and a Fourier-feature embedding of that coordinate is the natural positional encoding. Many current models instead inject *relative* position directly into the scores (for example, rotary embeddings, which rotate queries and keys by position-dependent angles), so that attention depends on $i-j$ rather than on $i$ and $j$ separately — translation equivariance recovered.

> **Remember.** Self-attention is order-blind, so position is added to the input. Sinusoidal encodings are Fourier features of the position; for physical data, Fourier features of the coordinates play the same role.

---

## 7. Multi-head attention and causal masking (10 min)

### 7.1 Several lookups in parallel

One attention layer gives each token a single weighted average. But a token may need several kinds of information at once: a word needs its subject *and* its neighbors; a particle feels short-range repulsion *and* long-range attraction. **Multi-head attention** runs $h$ attentions in parallel, each with its own projections and a reduced width $d_k = d/h$, concatenates the results, and mixes them with one more matrix:

$$
\text{head}_r = \operatorname{Attention}\big(XW_Q^{(r)},\,XW_K^{(r)},\,XW_V^{(r)}\big),\qquad
\operatorname{MHA}(X) = \big[\text{head}_1,\dots,\text{head}_h\big]\,W_O .
$$

Because each head has width $d/h$, the total cost is about that of one full-width head. Different heads learn different patterns: in language, some track adjacent words, some syntax, some coreference.

### 7.2 Causal masking

For autoregressive generation (Lecture 15 §1), position $i$ must not see positions $j>i$: the model of $p(y_i\mid y_{<i})$ cannot use $y_i$ or anything after it. Enforce this by setting those scores to $-\infty$ before the softmax:

$$
\operatorname{softmax}\!\Big(\frac{QK^\top}{\sqrt{d_k}} + M\Big)V,\qquad M_{ij}=\begin{cases}0 & j\le i\\ -\infty & j>i\end{cases}.
$$

$e^{-\infty}=0$, so future positions get zero weight, and the weight matrix is lower triangular.

This one line resolves the tension of Lecture 15 §6. With teacher forcing, an RNN still had to step through the sequence one position at a time. A causally masked self-attention layer computes **all $n$ next-token predictions in a single parallel pass**, each one seeing exactly its own past — teacher-forced maximum likelihood for the whole sequence at once. This is why transformers train so much faster than RNNs on long sequences.

### 7.3 The price

Every query is compared with every key, so the cost per layer is $O(n^2d)$ time and $O(n^2)$ memory for the weights. For a sentence of a few hundred tokens this is negligible. For a $256\times256$ field treated as one token per grid point, $n = 65{,}536$ and each head of each layer computes $n^2\approx4.3\times10^9$ scores. Grouping grid points into patches, restricting attention to local windows, and approximating the softmax are the main remedies, and they come up again when we reach transformers for images and neural operators.

> **Remember.** Multi-head: several smaller attentions in parallel, concatenated and mixed. Causal mask: $-\infty$ on future scores, which lets an autoregressive model train on all positions in parallel. Cost grows as $n^2$.

---

## 8. Summary (2 min)

Attention began as a fix for the RNN encoder–decoder's bottleneck: score every encoder state against the decoder's current state, softmax, and average. Stripped down, it is a soft dictionary lookup — queries, keys, values — and with a Gaussian score it is the Nadaraya–Watson kernel smoother, which becomes dot-product attention on normalized keys. Scaled by $\sqrt{d_k}$ and written in matrix form, it is one line. Applied by a sequence to itself, it gives a layer whose mixing weights are computed from the data, connect all positions directly, and can be computed in parallel. It is blind to order, so position is added; it is run in several heads at once; and with a causal mask it trains an autoregressive model on all positions at once.

Next lecture assembles these pieces — attention, the MLPs of Lecture 12, and the residual connections and normalization of Lecture 13 — into the transformer block.

---

## Code Checkpoint 16

NumPy only. Each item asks you to report a number.

**A. Attention from scratch.**
1. Implement `attention(Q, K, V, mask=None)` returning the outputs and the weight matrix, using `QK^T/sqrt(d_k)`, an optional additive mask, and a numerically stable softmax (subtract the row maximum before exponentiating).
2. Reproduce the hand computation of §5.2 (without scaling). **Report $y_1$ and $y_3$.**
3. Build self-attention with random $W_Q$, $W_K$, $W_V$ on random $X\in\mathbb{R}^{10\times16}$. Apply a random permutation $P$. **Report $\lVert f(PX)-Pf(X)\rVert$**, then repeat after adding sinusoidal positional encodings to $X$ (and $PX$) and report again. Explain the difference in one sentence.
4. With a causal mask, **verify** that the weight matrix is lower triangular, and that changing $x_{10}$ leaves outputs $y_1,\dots,y_9$ unchanged.

**B. Kernel regression is attention.** Draw 40 noisy samples of $y = \sin(2\pi x)$ on $[0,1]$ (noise standard deviation 0.1).
1. Implement Nadaraya–Watson as a call to your `attention` function: queries are test points, keys are training inputs, values are training targets, scores $-(x^\star-x_j)^2/2\ell^2$ (pass them as a custom score, or note how to obtain them from dot products).
2. For $\ell\in\{0.01,0.03,0.1,0.3\}$, **report the test RMSE**. Explain the trend in terms of temperature (§2.3).
3. Compare with your GP posterior mean from Lecture 9 on the same data, with hyperparameters fitted by marginal likelihood. **Report both RMSEs** and one sentence on what the GP's $(K+\sigma^2I)^{-1}$ buys.

**C. Why $\sqrt{d_k}$.** For $d_k\in\{4,16,64,256\}$, draw 1000 queries and 32 keys with i.i.d. standard normal entries. **Report the mean of the largest attention weight per query**, with and without the $1/\sqrt{d_k}$ scaling. In two sentences, connect the unscaled trend to softmax saturation and vanishing gradients.

---

## Quiz-eligible facts

1. An RNN encoder–decoder maps an input sequence to an output sequence of different length: the encoder compresses the input into a context vector $m=h_S$, and the decoder generates the output autoregressively from it. It has three limitations: a fixed-size context bottleneck, long gradient paths between distant positions, and sequential computation.
2. Bahdanau attention computes, at each output step, $\alpha_{tj}=\operatorname{softmax}_j(\operatorname{score}(s_{t-1},h_j))$ and $c_t=\sum_j\alpha_{tj}h_j$; the weights form a learned soft alignment and sum to one over input positions.
3. Attention is a soft dictionary lookup: queries are matched against keys, and the output is a softmax-weighted average of values.
4. Nadaraya–Watson kernel regression with a Gaussian kernel is attention with query $x^\star$, keys $x_j$ and values $y_j$; with keys of equal norm, the Gaussian score reduces to a scaled dot product.
5. Scaled dot-product attention: $\operatorname{softmax}(QK^\top/\sqrt{d_k})V$; the scaling keeps score variance at 1, because $\operatorname{Var}[q^\top k]=d_k$ for unit-variance entries, and prevents softmax saturation.
6. Self-attention computes $Q$, $K$, $V$ from the same sequence; cross-attention takes queries from one sequence and keys and values from another.
7. Unlike convolution, self-attention's mixing weights depend on content and connect all positions; unlike recurrence, it is computed in parallel.
8. Self-attention is permutation equivariant, $f(PX)=Pf(X)$, so positional information must be added; sinusoidal encodings are Fourier features of position.
9. Multi-head attention runs $h$ attentions of width $d/h$ in parallel and mixes their concatenation with $W_O$.
10. A causal mask sets future scores to $-\infty$, giving a lower-triangular weight matrix and allowing all next-token predictions to be trained in one parallel pass.
11. Self-attention costs $O(n^2d)$ time and $O(n^2)$ memory in the sequence length $n$.

---

## Practice problems

**P1.** In Bahdanau attention, a decoder step produces weights $\alpha_t=(0.05,\ 0.90,\ 0.05)$ over three encoder states. What does this tell you, and what is $c_t$?

*Answer.* The decoder is attending almost entirely to input position 2: it is the input most relevant to producing output $t$ (a near-hard alignment). $c_t = 0.05\,h_1+0.90\,h_2+0.05\,h_3\approx h_2$.

**P2.** Scores for one query are $(2,\ 0,\ 2)$ and values are $v_1=1$, $v_2=4$, $v_3=7$. Compute the attention output. What happens if all scores are multiplied by 10?

*Answer.* Weights $(e^2,1,e^2)/(2e^2+1)\approx(0.47,\ 0.06,\ 0.47)$, output $\approx0.47+0.24+3.29 = 4.0$. Multiplied by 10, the weights become $\approx(0.5,\ 0,\ 0.5)$ and the output $\approx4.0$ — with the middle value now excluded entirely. Larger scores mean lower temperature and a harder lookup.

**P3.** With $d_k=256$ and unit-variance query and key entries, what is the standard deviation of an unscaled score $q^\top k$? Why does this hurt training?

*Answer.* $\sqrt{256}=16$. Score gaps of this size make the softmax nearly one-hot, where its gradient is nearly zero, so the projections that produce queries and keys barely learn. Dividing by $\sqrt{d_k}=16$ restores unit variance.

**P4.** You feed the token sequences "dog bites man" and "man bites dog" to a single self-attention layer with no positional encodings. How are the two outputs related? What does this imply?

*Answer.* The second input is a permutation of the first, so its outputs are the same vectors in the permuted order: "dog" gets the same representation in both sentences. The layer cannot distinguish the meanings, which is why positional information must be added.

**P5.** A $64\times64$ field is fed to self-attention with one token per grid point. How many attention scores does one head compute per layer? What if the field is first split into $8\times8$ patches, one token per patch?

*Answer.* $n=4096$ tokens, so $n^2\approx1.7\times10^7$ scores. With $8\times8$ patches, $n=64$ tokens and $n^2=4096$ scores — about 4000 times fewer, at the cost of coarser tokens.

---

## References

- Alammar, J. (2018). The illustrated transformer. jalammar.github.io.
- Bahdanau, D., Cho, K., & Bengio, Y. (2015). Neural machine translation by jointly learning to align and translate. *ICLR*.
- Cho, K., van Merriënboer, B., Bahdanau, D., & Bengio, Y. (2014). On the properties of neural machine translation: Encoder–decoder approaches. *SSST-8 Workshop*.
- Cho, K., van Merriënboer, B., Gulcehre, C., Bahdanau, D., Bougares, F., Schwenk, H., & Bengio, Y. (2014). Learning phrase representations using RNN encoder–decoder for statistical machine translation. *EMNLP*.
- Luong, M.-T., Pham, H., & Manning, C. D. (2015). Effective approaches to attention-based neural machine translation. *EMNLP*.
- Murphy, K. P. (2022). *Probabilistic Machine Learning: An Introduction*. MIT Press. §15.4.
- Nadaraya, E. A. (1964). On estimating regression. *Theory of Probability and Its Applications*, 9(1), 141–142.
- Sutskever, I., Vinyals, O., & Le, Q. V. (2014). Sequence to sequence learning with neural networks. *NeurIPS*.
- Tsai, Y.-H. H., Bai, S., Yamada, M., Morency, L.-P., & Salakhutdinov, R. (2019). Transformer dissection: A unified understanding of transformer's attention via the lens of kernel. *EMNLP*.
- Vaswani, A., et al. (2017). Attention is all you need. *NeurIPS*.
- Watson, G. S. (1964). Smooth regression analysis. *Sankhyā, Series A*, 26(4), 359–372.
- Weng, L. (2018). Attention? Attention! lilianweng.github.io.
- Zhang, A., Lipton, Z. C., Li, M., & Smola, A. J. (2023). *Dive into Deep Learning*. Cambridge University Press. Ch. 11.
