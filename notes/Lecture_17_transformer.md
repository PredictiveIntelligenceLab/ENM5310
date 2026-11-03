# Lecture 17 — The Transformer

**ENM 5310 — Data-driven Modeling and Probabilistic Scientific Computing**

**Builds on:** Lecture 5 (categorical likelihood and cross-entropy), Lecture 12 (MLPs), Lecture 13 (residual connections, LayerNorm, AdamW, warmup and cosine schedules, gradient clipping), Lecture 14 (patches, pooling, translation equivariance), Lecture 15 (the autoregressive factorization, teacher forcing), Lecture 16 §§0–5 (encoder–decoders, attention as soft lookup, scaled dot-product attention, self-attention and its permutation equivariance).

**Reading:** Vaswani et al. (2017), *Attention Is All You Need*, §3. Alammar (2018), *The Illustrated Transformer* (blog). Phuong & Hutter (2022), *Formal Algorithms for Transformers*, §§1–6 — precise pseudocode for every variant. Murphy I §15.5. Optional: Prince (2023), *Understanding Deep Learning*, Ch. 12; Karpathy (2023), *Let's build GPT: from scratch, in code, spelled out* (video).

**The point of this lecture.** Lecture 16 ended with a layer that lets every token read from every other, with weights computed from content, in parallel. It also ended with a problem: that layer treats its input as a *set*. Today we finish the job. Add **position**, run attention in several **heads**, and wrap it with the three tools from Lecture 13 — a per-token **MLP**, **residual connections**, and **LayerNorm** — and you have the **transformer block**. Stack blocks, choose a **mask**, and you get the three transformer families behind modern language, vision and scientific models. The architecture is less mysterious than its reputation: every piece is something you already know.

---

### Notation

Tokens are rows, as in Lecture 16.

| Symbol | Meaning |
|---|---|
| $n$, $d$, $h$, $L$ | number of tokens, model width, number of heads, number of blocks |
| $d_k = d/h$ | width per head |
| $d_{\text{ff}}$ | hidden width of the per-token MLP, usually $4d$ |
| $X\in\mathbb{R}^{n\times d}$ | sequence of token representations ("residual stream"), rows $x_i^\top$ |
| $P\in\mathbb{R}^{n\times d}$ | positional encodings, rows $p_i^\top$ |
| $W_Q^{(r)},W_K^{(r)},W_V^{(r)}\in\mathbb{R}^{d\times d_k}$, $W_O\in\mathbb{R}^{d\times d}$ | projections of head $r$; output projection |
| $W_1\in\mathbb{R}^{d\times d_{\text{ff}}}$, $W_2\in\mathbb{R}^{d_{\text{ff}}\times d}$ | MLP weights |
| $\operatorname{LN}(\cdot)$ | layer normalization, applied to each row (token) separately |
| $M\in\{0,-\infty\}^{n\times n}$ | attention mask, added to the scores |
| $\mathcal V$, $E\in\mathbb{R}^{|\mathcal V|\times d}$ | vocabulary, token embedding matrix |

Recall from Lecture 16:

$$
\operatorname{Attention}(Q,K,V;M) = \operatorname{softmax}\!\Big(\frac{QK^\top}{\sqrt{d_k}}+M\Big)V,
$$

with the softmax applied row by row.

**Learning objectives.** After this lecture you should be able to:

1. Explain why positional information is needed, write sinusoidal encodings, and show that their dot products depend only on relative position.
2. Write multi-head attention with shapes, and explain why it costs no more than one full-width head.
3. Write a pre-LN transformer block, explain the role of each of its four components, and count its parameters ($\approx12d^2$).
4. Distinguish encoder-only, decoder-only and encoder–decoder transformers by their masks and cross-attention, and match each to a task.
5. Explain how a causal mask turns teacher-forced maximum likelihood into one parallel pass, and why generation is still sequential.
6. Turn text, images and physical fields into tokens, and choose an output head and likelihood.
7. Implement a transformer forward pass from the pseudocode, and state its cost in $n$ and $d$.
8. Compare the inductive biases of MLPs, CNNs, RNNs and transformers.

---

## 0. Where we are, and what is missing (5 min)

Self-attention (Lecture 16 §5) maps a sequence $X$ to

$$
Y = \operatorname{softmax}\!\Big(\frac{(XW_Q)(XW_K)^\top}{\sqrt{d_k}}\Big)XW_V .
$$

Each output row is a content-weighted average of all value rows. It is global and parallel. But a single attention layer, used alone, has four shortcomings, and each section below fixes one.

1. **It ignores order.** It is permutation equivariant, so "dog bites man" and "man bites dog" are the same input. → §1, positional encodings.
2. **It computes one weighted average per token.** A token often needs several different kinds of information at once. → §2, multiple heads.
3. **It only mixes.** Every output is a convex combination of value vectors; nothing transforms each token nonlinearly. Stacks of attention-only layers lose expressive power with depth — their outputs collapse toward rank one (Dong et al., 2021). → §3, a per-token MLP.
4. **It is one layer.** Deep stacks need the signal-scale and gradient-flow tools of Lecture 13. → §3, residual connections and LayerNorm.

---

## 1. Positional information (10 min)

### 1.1 Add position to content

Self-attention sees a set, so we make each token's position part of its content before the first layer:

$$
X^{(0)} = X_{\text{in}} + P, \qquad p_i\in\mathbb{R}^d \text{ encodes position } i .
$$

After this, permuting the tokens changes their representations, and the network can learn to use order. Adding (rather than concatenating) keeps the width at $d$. In high dimensions the content and position vectors are close to orthogonal, so the network can still read each of them separately.

### 1.2 Sinusoidal encodings

Vaswani et al. (2017) use, for $r = 0,\dots,d/2-1$,

$$
p_{i,2r} = \sin(i\,\omega_r), \qquad p_{i,2r+1} = \cos(i\,\omega_r), \qquad \omega_r = 10000^{-2r/d}.
$$

Each pair of coordinates is a clock hand rotating at its own frequency as $i$ increases. The frequencies are geometrically spaced, from one radian per position down to about $10^{-4}$. Fast hands distinguish neighbors; slow hands distinguish positions far apart. Together they work like the digits of a number in base 10 — except that the digits are smooth.

**A small example** with $d=4$, so $\omega_0=1$ and $\omega_1 = 10000^{-1/2}=0.01$:

| Position $i$ | $\sin(i)$ | $\cos(i)$ | $\sin(0.01i)$ | $\cos(0.01i)$ |
|---|---|---|---|---|
| 0 | 0.000 | 1.000 | 0.000 | 1.000 |
| 1 | 0.841 | 0.540 | 0.010 | 1.000 |
| 2 | 0.909 | −0.416 | 0.020 | 1.000 |
| 3 | 0.141 | −0.990 | 0.030 | 1.000 |

The first pair changes quickly; the second barely moves over four positions and would be needed only to tell position 3 from position 300.

### 1.3 Why sinusoids: relative position is built in

Fix one frequency $\omega$ and write $p_i = (\sin i\omega,\ \cos i\omega)$.

1. *Shifting is a rotation.* By the angle-addition formulas,
$$
p_{i+\Delta} = \begin{pmatrix}\cos\Delta\omega & \sin\Delta\omega\\ -\sin\Delta\omega & \cos\Delta\omega\end{pmatrix}p_i ,
$$
a rotation that depends on the offset $\Delta$ but not on $i$. A linear map can therefore express "the token $\Delta$ positions back" in the same way everywhere.
2. *Dot products depend only on the offset.* $p_i\cdot p_j = \sin i\omega\sin j\omega + \cos i\omega\cos j\omega = \cos\big((i-j)\omega\big)$. Summing over frequencies, $p_i^\top p_j$ is a function of $i-j$ alone, large when the positions are close.

So sinusoidal encodings give attention a translation-invariant notion of "nearby" — the property convolutions had by construction (Lecture 14).

### 1.4 Other choices

- **Learned embeddings:** one trainable vector per position. Simple and common, but undefined beyond the longest training sequence.
- **Rotary embeddings (RoPE;** Su et al., 2024**):** instead of adding $p_i$, rotate each query and key pair by angle $i\omega_r$ before the dot product. Then $(R_iq)^\top(R_jk) = q^\top R_{j-i}k$, so attention scores depend on content and on the *relative* offset $j-i$ only. Most current language models use RoPE.
- **Physical coordinates.** For a token that is a grid point, a particle or a sensor, the natural "position" is its coordinate $(x,y,z,t)$. Encoding it with sinusoids of several frequencies is exactly the Fourier-feature embedding that counters spectral bias in coordinate networks. This also lets the model take tokens at arbitrary, irregular locations.

> **Remember.** Add a positional encoding to each token before the first layer. Sinusoidal encodings are multi-frequency clocks: a shift is a rotation, and $p_i^\top p_j$ depends only on $i-j$. For physical data, Fourier features of the coordinates play the same role.

---

## 2. Multi-head attention (10 min)

### 2.1 Several lookups in parallel

One attention layer gives each token a single weighted average. But in "the animal didn't cross the street because it was too tired," the token "it" needs to find its referent, and also its verb, and also its neighbors. A particle in a molecule feels short-range repulsion and long-range electrostatics. One softmax cannot put sharp weight on several unrelated places at once without blurring them together.

**Multi-head attention** runs $h$ attentions in parallel, each with its own projections of reduced width $d_k=d/h$, and mixes their concatenated outputs:

$$
\text{head}_r = \operatorname{Attention}\big(XW_Q^{(r)},\,XW_K^{(r)},\,XW_V^{(r)};\,M\big)\in\mathbb{R}^{n\times d_k},\qquad r=1,\dots,h,
$$

$$
\operatorname{MHA}(X) = \big[\text{head}_1\;\cdots\;\text{head}_h\big]\,W_O\in\mathbb{R}^{n\times d}.
$$

### 2.2 Shapes and cost

Example: $n=10$ tokens, $d=64$, $h=8$, so $d_k=8$.

| Step | Shape |
|---|---|
| $X$ | $10\times64$ |
| $Q^{(r)},K^{(r)},V^{(r)}$ for each head | $10\times8$ |
| scores $Q^{(r)}K^{(r)\top}/\sqrt8$ for each head | $10\times10$ (8 such matrices) |
| head outputs | $10\times8$ each |
| concatenation | $10\times64$ |
| times $W_O$ | $10\times64$ |

All $h$ heads together use $3hd\,d_k = 3d^2$ projection parameters, the same as one head of full width, plus $d^2$ for $W_O$: **$4d^2$ parameters, whatever $h$ is.** Splitting into heads changes what the layer can represent, not what it costs. In practice the heads are computed together by reshaping one $d\times d$ projection into $h$ blocks (§7).

Trained heads specialize. In language models some heads attend to the previous token, some to syntactic dependents, some to earlier occurrences of the current token. You can inspect them by plotting each head's $n\times n$ weight matrix.

> **Remember.** $h$ heads of width $d/h$, run in parallel, concatenated, mixed by $W_O$. Total cost and parameter count ($4d^2$) match a single full-width head; the gain is several independent lookups per token.

---

## 3. The transformer block (20 min)

### 3.1 Communicate, then compute

A transformer block has two sublayers, each wrapped in a residual connection with LayerNorm. In the **pre-LN** form used by essentially all modern transformers:

$$
\begin{aligned}
Z &= X + \operatorname{MHA}\big(\operatorname{LN}(X)\big) &&\text{tokens exchange information}\\
X' &= Z + \operatorname{FFN}\big(\operatorname{LN}(Z)\big) &&\text{each token is processed on its own}
\end{aligned}
$$

where the **feed-forward network** is a two-layer MLP (Lecture 12) applied to every row separately, with shared weights:

$$
\operatorname{FFN}(z) = \sigma\big(zW_1+b_1\big)W_2+b_2,\qquad W_1\in\mathbb{R}^{d\times4d},\ W_2\in\mathbb{R}^{4d\times d},\ \sigma=\text{GELU}.
$$

A useful picture: the rows of $X$ are people at a meeting. In the attention sublayer they talk — each one listens to everyone and takes notes, weighted by relevance. In the FFN sublayer each person goes off and thinks alone about what they heard. A transformer is $L$ rounds of "talk, then think."

### 3.2 What each component is for

| Component | Role | Without it |
|---|---|---|
| Multi-head attention | Mixes information **across tokens**, with content-dependent weights | Tokens never interact: the model is an MLP applied to each token separately |
| FFN | Transforms each token **nonlinearly**; holds about two-thirds of the parameters | Outputs are only weighted averages of values; deep attention-only stacks collapse in rank |
| Residual connections | Each sublayer *adds* a correction to a running representation; the identity path carries gradients through depth (Lecture 13) | Deep stacks become hard to train |
| LayerNorm | Re-standardizes each token before each sublayer, so inputs to attention and FFN stay at unit scale (Lecture 13) | Scales drift with depth; softmax and GELU saturate |

The attention sublayer is the only place where tokens interact. The FFN is the only place with a nonlinearity applied to individual tokens. A CNN alternates the same two operations — a spatial convolution that mixes neighbors and a $1\times1$ convolution plus ReLU that acts on each pixel (Lecture 14 §2.3) — except that attention mixes globally, with weights it computes on the fly.

### 3.3 The residual stream

Unroll the residual connections across all $L$ blocks:

$$
X^{(L)} = X^{(0)} + \sum_{\ell=1}^{L}\Big(\operatorname{MHA}_\ell(\cdots) + \operatorname{FFN}_\ell(\cdots)\Big).
$$

The final representation is the input embedding plus the sum of every sublayer's contribution. Each sublayer *reads* the current stream (through LayerNorm) and *writes* an update back into it. This **residual stream** view (Elhage et al., 2021) explains why transformers train stably at great depth: as in Lecture 13 §3, the backward pass always has a direct identity path from the loss to every layer.

### 3.4 Pre-LN versus post-LN

The original transformer placed LayerNorm *after* the residual addition: $X' = \operatorname{LN}\big(X + \operatorname{Sublayer}(X)\big)$ ("post-LN"). Then the identity path passes through a LayerNorm at every block, gradients near the output are large at initialization, and training is unstable without a careful learning-rate warmup (Xiong et al., 2020). Pre-LN keeps the identity path clean, trains more robustly, and adds one final LayerNorm after the last block. When you read the 2017 paper, keep this difference in mind.

### 3.5 Counting parameters and compute

Ignoring biases and LayerNorm gains, one block has

$$
\underbrace{4d^2}_{\text{attention}} + \underbrace{8d^2}_{\text{FFN}} = 12\,d^2 \text{ parameters.}
$$

$L$ blocks have $12Ld^2$, plus the embedding matrices. As a sanity check, GPT-3 has $L=96$ and $d=12{,}288$: $12\cdot96\cdot12{,}288^2\approx1.74\times10^{11}$, or the familiar "175 billion."

The cost of one block on $n$ tokens is

$$
\underbrace{O(n\,d^2)}_{\text{projections and FFN}} + \underbrace{O(n^2d)}_{\text{scores and weighted sums}} .
$$

For text with $n$ in the hundreds or thousands and $d$ similar, the first term dominates. For scientific fields with $n$ in the tens of thousands or more, the second does — which is why the token count matters so much there (§5).

> **Remember.** Pre-LN block: $Z = X+\operatorname{MHA}(\operatorname{LN}(X))$, then $X' = Z+\operatorname{FFN}(\operatorname{LN}(Z))$. Attention communicates across tokens; the FFN computes within each token; residuals and LayerNorm make depth trainable. About $12d^2$ parameters per block; cost $O(nd^2+n^2d)$.

---

## 4. Masks and the three families (12 min)

The block is the same everywhere. Transformers differ in **who may attend to whom**, set by the mask, and in whether there is a second sequence to read from.

### 4.1 Encoder-only: bidirectional

No mask ($M=0$): every token sees every other, before and after it. The output is a contextual representation of each token, informed by the whole sequence. Use it when the entire input is available and you want to label, classify or regress from it: sentiment, the properties of a molecule, the class of an image, a field reconstructed from a full set of measurements. **BERT** (Devlin et al., 2019) is trained by masking 15% of the input tokens and predicting them from both sides. The **Vision Transformer** (§5) is encoder-only.

### 4.2 Decoder-only: causal

The causal mask of Lecture 16 sets $M_{ij}=-\infty$ for $j>i$: token $i$ sees only itself and its past.

$$
M = \begin{pmatrix}0 & -\infty & -\infty & -\infty\\ 0 & 0 & -\infty & -\infty\\ 0&0&0&-\infty\\0&0&0&0\end{pmatrix}
\quad\Longrightarrow\quad
\alpha = \begin{pmatrix}1&0&0&0\\ \ast&\ast&0&0\\ \ast&\ast&\ast&0\\ \ast&\ast&\ast&\ast\end{pmatrix}.
$$

The output at position $t$ predicts token $t+1$. This is the autoregressive factorization of Lecture 15,

$$
p(y_{1:n}) = \prod_{t=1}^{n}p(y_t\mid y_{<t}),
$$

with every conditional computed by the same network. **GPT** (Radford et al., 2018) and most current large language models are decoder-only.

**Training is parallel.** Feed the whole true sequence; the mask guarantees that position $t$'s prediction uses only $y_{\le t}$. One forward pass yields all $n$ next-token predictions, and the loss is their summed cross-entropy. This is teacher forcing (Lecture 15 §6) for the entire sequence at once — what an RNN needed $n$ sequential steps to do.

**Generation is still sequential.** At test time, sample $y_{n+1}$, append it, and run again. Each new token needs one forward pass. Since the keys and values of earlier tokens never change, they are stored and reused (a "KV cache"), so each step processes only the newest token.

### 4.3 Encoder–decoder: cross-attention

The original transformer (Vaswani et al., 2017) is an encoder–decoder, the architecture of Lecture 16 §0 with every RNN replaced. The encoder is a bidirectional stack that reads the input $x_{1:S}$ and produces $H_{\text{enc}}\in\mathbb{R}^{S\times d}$. Each decoder block has three sublayers instead of two:

$$
\begin{aligned}
Z_1 &= Y + \operatorname{MHA}_{\text{causal}}\big(\operatorname{LN}(Y)\big) &&\text{the output so far attends to itself}\\
Z_2 &= Z_1 + \operatorname{CrossAttn}\big(\underbrace{\operatorname{LN}(Z_1)}_{\text{queries}},\ \underbrace{H_{\text{enc}}}_{\text{keys, values}}\big) &&\text{the output attends to the input}\\
Y' &= Z_2 + \operatorname{FFN}\big(\operatorname{LN}(Z_2)\big)
\end{aligned}
$$

The cross-attention sublayer is Bahdanau attention (Lecture 16 §1) reborn: queries come from the decoder, keys and values from the encoder. Use this family when an input sequence is mapped to an output sequence of different length or type: translation, a forcing history to a response, a set of sensors to a field at query points.

| Family | Mask | Cross-attention | Typical use |
|---|---|---|---|
| Encoder-only | none | no | Understanding a whole input: classification, regression, per-token labels (BERT, ViT) |
| Decoder-only | causal | no | Generating sequences autoregressively (GPT-style language models, autoregressive forecasters) |
| Encoder–decoder | none in encoder; causal in decoder | yes | Mapping one sequence to another (translation, sequence-to-sequence surrogates) |

> **Remember.** Same block, different masks. No mask: encoder (sees everything). Causal mask: decoder (sees the past), trained on all positions in one pass but generating one token at a time. Encoder–decoder: add cross-attention from decoder queries to encoder keys and values.

---

## 5. What goes in and what comes out: tokens and heads (8 min)

A transformer operates on a set of $d$-dimensional vectors. Anything you can turn into such a set, it can process.

### 5.1 Tokenization and embedding

- **Text.** Split into subword units from a fixed vocabulary $\mathcal V$ (tens of thousands of entries), and look up each token's row of a learned embedding matrix $E\in\mathbb{R}^{|\mathcal V|\times d}$.
- **Images: patches.** The Vision Transformer (ViT; Dosovitskiy et al., 2021) cuts an image into $16\times16$ patches, flattens each, and maps it to $d$ dimensions with one linear layer — a convolution with kernel size and stride 16. A $224\times224$ image becomes $14\times14=196$ tokens.
- **Physical fields.** The same patching applies to a simulation snapshot: a $256\times256$ field in $16\times16$ patches gives 256 tokens instead of 65,536. Irregular data — particles, mesh nodes, sensors — can be tokenized point by point, with Fourier features of the coordinates as positional encoding (§1.4).
- **Time series.** Each time step, or each short window of steps, becomes a token.

### 5.2 Output heads and likelihoods

The final representations $X^{(L)}$ are mapped to whatever the likelihood needs, exactly as in Lecture 12 §3.

| Task | Head on each relevant row | Likelihood and loss |
|---|---|---|
| Next-token prediction | Linear map to $|\mathcal V|$ logits | Categorical; cross-entropy |
| Classification of a whole input | Average the rows (or use a dedicated summary token), then a linear map | Categorical; cross-entropy |
| Regression of a field or next state | Linear map to the output dimension per token or per patch | Gaussian; squared error |

Averaging the rows is the global average pooling of Lecture 14 §3.4: it turns permutation-equivariant token features into a permutation-invariant prediction.

---

## 6. Training a transformer (8 min)

Nothing in Lecture 13 is replaced; some of it becomes essential.

- **Objective:** maximum likelihood, as always. For a decoder-only language model, the average cross-entropy of next-token predictions; $\exp(\text{that average})$ is the *perplexity*.
- **Optimizer:** AdamW (Lecture 13 §5). Large language models typically use $\beta=(0.9,\,0.95)$ and weight decay around $0.1$, not applied to biases, LayerNorm gains or embeddings.
- **Schedule:** linear warmup, then cosine decay. Warmup matters more here than for MLPs, especially with post-LN (§3.4).
- **Gradient clipping** at a norm of about 1. Occasional loss spikes are common at scale.
- **Regularization:** dropout on attention weights and sublayer outputs for small datasets; large models trained on huge datasets often use little or none.

**Data hunger and inductive bias.** A CNN is told by its architecture that images are translation equivariant and local. A transformer is told only that its input is a set, plus whatever the positional encodings convey. It must learn locality from data. This is why ViTs trained on ImageNet's 1.3 million images lag behind comparable CNNs unless trained with heavy augmentation, and overtake them when pretrained on hundreds of millions of images (Dosovitskiy et al., 2021). With little data, structure built into the architecture wins. With a lot, a flexible model wins.

**Scaling.** Transformer test loss falls smoothly, as a power law, as model size, data and compute grow together (Kaplan et al., 2020). For a fixed compute budget, the best results come from scaling parameters and training tokens in proportion, roughly 20 tokens per parameter (Hoffmann et al., 2022). This predictability, as much as raw accuracy, is why the field consolidated around this architecture.

---

## 7. The whole thing in pseudocode (8 min)

A complete decoder-only transformer forward pass, in NumPy-style Python. Parameters live in a dictionary-like object `P`; a batch dimension is omitted for clarity.

```python
import numpy as np

def softmax(z):                                  # row-wise, numerically stable
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)

def layer_norm(x, g, b, eps=1e-5):               # normalize each row (token)
    mu, var = x.mean(-1, keepdims=True), x.var(-1, keepdims=True)
    return g * (x - mu) / np.sqrt(var + eps) + b

def gelu(x):
    return 0.5 * x * (1 + np.tanh(0.79788456 * (x + 0.044715 * x**3)))

def positional_encoding(n, d):                   # sinusoidal, Section 1.2
    pos, r = np.arange(n)[:, None], np.arange(d // 2)[None, :]
    w = 10000.0 ** (-2 * r / d)
    pe = np.zeros((n, d))
    pe[:, 0::2], pe[:, 1::2] = np.sin(pos * w), np.cos(pos * w)
    return pe

def mha(x, p, h, mask):                          # x: (n, d) -> (n, d)
    n, d = x.shape; dk = d // h
    split = lambda t: t.reshape(n, h, dk).transpose(1, 0, 2)   # (h, n, dk)
    q, k, v = split(x @ p.Wq), split(x @ p.Wk), split(x @ p.Wv)
    s = q @ k.transpose(0, 2, 1) / np.sqrt(dk) + mask          # (h, n, n)
    y = softmax(s) @ v                                         # (h, n, dk)
    return y.transpose(1, 0, 2).reshape(n, d) @ p.Wo           # concat heads, mix

def ffn(x, p):                                   # per-token MLP, d -> 4d -> d
    return gelu(x @ p.W1 + p.b1) @ p.W2 + p.b2

def block(x, p, h, mask):                        # pre-LN block, Section 3.1
    x = x + mha(layer_norm(x, p.ln1_g, p.ln1_b), p, h, mask)  # communicate
    x = x + ffn(layer_norm(x, p.ln2_g, p.ln2_b), p)           # compute
    return x

def transformer_lm(tokens, P):                   # tokens: (n,) integer ids
    n = len(tokens)
    x = P.E[tokens] + positional_encoding(n, P.d)              # (n, d)
    mask = np.triu(np.full((n, n), -np.inf), k=1)              # causal, Section 4.2
    for p in P.blocks:                                         # L blocks
        x = block(x, p, P.h, mask)
    x = layer_norm(x, P.lnf_g, P.lnf_b)                        # final LN (pre-LN)
    return x @ P.Wu                                            # (n, |V|) logits

def loss(tokens, P):                             # next-token cross-entropy
    logits = transformer_lm(tokens, P)[:-1]                    # rows 0..n-2
    targets = tokens[1:]                                       # predict t+1
    logp = np.log(softmax(logits))                             # log-probabilities
    return -logp[np.arange(len(targets)), targets].mean()
```

Four things to notice:

1. **The whole architecture is about thirty lines.** Most of the complexity in real code is batching, efficiency and distributed training, not the model.
2. **Heads are a reshape.** One $d\times d$ projection is split into $h$ blocks of width $d/h$; no separate weights per head are needed.
3. **The mask is the only difference between an encoder and a decoder.** Replace `mask` by zeros and `transformer_lm` becomes an encoder (with a different head and loss).
4. **The causal loss uses every position at once**: one call computes $n-1$ predictions and their cross-entropies.

For an encoder–decoder, add a cross-attention sublayer to each decoder block, computing queries from the decoder stream and keys and values from the encoder output.

---

## 8. Transformers in science, and their limits (7 min)

### 8.1 Where they are used

- **Protein structure.** AlphaFold 2 (Jumper et al., 2021) is built around attention over residues and pairs of residues.
- **Weather and climate.** Transformer-based models such as Pangu-Weather (Bi et al., 2023) and the Aurora foundation model (Bodnar et al., 2025) tokenize the atmosphere into 3D patches and forecast at accuracy competitive with operational systems, at a small fraction of the computational cost.
- **Learned simulators and neural operators.** Fields, meshes and point clouds are tokenized as in §5, and transformers map them to future states or to solutions of PDEs.

### 8.2 Attention as a learned integral operator

There is a deeper reason transformers suit physical fields. If the tokens are samples $v(x_j)$ of a function at points $x_j$, one attention head computes

$$
y(x_i) = \sum_{j=1}^{n}\alpha(x_i,x_j)\,v(x_j) \;\approx\; \int\kappa(x_i,x')\,v(x')\,dx',
$$

a quadrature approximation of an **integral operator** with a learned, input-dependent kernel $\kappa$. The softmax normalization makes the weights behave like quadrature weights, so refining the grid adds terms to the sum without changing what it approximates. Linear operators of this form — Green's functions, convolutions, projections — are how PDE solutions are written. Attention layers can therefore be seen as discretizations of operators between function spaces (Kovachki et al., 2023), which is the starting point for neural operators later in the course.

### 8.3 Limits

- **Quadratic cost.** $O(n^2)$ in the number of tokens makes fine 3D fields expensive. Remedies include patching, local or windowed attention, and approximations that are linear in $n$.
- **Weak inductive bias.** With little data, CNNs (with locality and translation equivariance built in) or GPs (with an explicit prior) are often better. Small scientific datasets are the norm, not the exception.
- **No physics by default.** Conservation laws, symmetries beyond permutation, and boundary conditions are not built in; they must be added through the data, the loss, or the architecture.

### 8.4 The architectures side by side

| Architecture | Built-in symmetry | How positions interact | Parallel over positions? | Cost per layer |
|---|---|---|---|---|
| MLP | none | fixed dense weights | yes | $O(n^2)$ weights, input size fixed |
| CNN | translation | fixed local stencil | yes | $O(nk)$ per channel pair |
| RNN | time shift (same update each step) | through a recurrent state | no | $O(nd^2)$, $n$ sequential steps |
| Transformer | permutation (order added by encodings) | content-dependent global weights | yes | $O(nd^2+n^2d)$ |

---

## 9. Summary (2 min)

A transformer turns its input into tokens, adds positional information, and passes the sequence through $L$ identical blocks. Each block lets tokens exchange information through multi-head self-attention, then processes each token through an MLP, with residual connections and LayerNorm around both. The mask decides who sees whom: none for encoders, causal for decoders, and an extra cross-attention sublayer for encoder–decoders. A causal decoder computes every conditional of the autoregressive factorization in one parallel pass, which is why it trains so much faster than an RNN. Every ingredient — the attention of Lecture 16, the MLP of Lecture 12, the residuals, LayerNorm and AdamW of Lecture 13, the patches and pooling of Lecture 14, the autoregressive likelihood of Lecture 15 — was already in hand; the transformer is a particular way of putting them together that happens to scale.

---

## Code Checkpoint 17

NumPy for items A and B. Item C may use `jax.grad` for gradients only — implement the model, optimizer and training loop yourself. Each item asks you to report a number.

**A. A transformer forward pass.**
1. Implement the pseudocode of §7 (you may add a batch dimension). Initialize with $d=64$, $h=4$, $L=2$, $|\mathcal V|=50$.
2. **Report the parameter count** and compare it with $12Ld^2$ plus the embedding and unembedding matrices. Account for the difference (biases, LayerNorm).
3. **Causality test.** For a random sequence of 20 tokens, change the last token and **report the maximum change in the logits at positions 1–19.** It should be zero to machine precision. Then remove the mask and report it again.
4. **Order test.** Remove the positional encodings and the mask, permute the input tokens, and **report** $\lVert f(\pi\,\text{tokens}) - \pi f(\text{tokens})\rVert$. Restore the encodings and report it again.

**B. Positional encodings.**
1. For $n=128$ and $d=64$, compute $P$ and the matrix $G=PP^\top$. **Report the maximum deviation of $G$ from a Toeplitz matrix** (entries depending only on $i-j$).
2. Verify the rotation property of §1.3 numerically for $\Delta=5$ and three frequencies. **Report the maximum error.**
3. Plot $G$ and one row of it. In two sentences, describe how far apart two positions must be before their encodings look dissimilar.

**C. A tiny transformer forecaster.** Using your Lorenz-63 data from Lecture 15 (standardized), train a decoder-only transformer ($L=2$, $d=64$, $h=4$) with a linear Gaussian head on windows of 64 consecutive states: the input at step $t$ is the state $x_t$ (mapped to $d$ dimensions by a linear layer), and the target is $x_{t+1}$. Use your Lecture 13 AdamW with warmup and cosine decay.
1. **Report the one-step test error** and the mean **valid prediction time** in Lyapunov times over 20 initial conditions, with mean $\pm$ standard deviation over 3 seeds.
2. Compare both numbers with your echo state network from Lecture 15. In three sentences, say which you would trust for this problem and why, considering data size, training cost and inductive bias.

---

## Quiz-eligible facts

1. Self-attention alone is permutation equivariant; positional encodings $p_i$ added to the input give the model access to order.
2. Sinusoidal encodings use frequencies $\omega_r = 10000^{-2r/d}$; a shift by $\Delta$ is a rotation of each $(\sin,\cos)$ pair, and $p_i^\top p_j$ depends only on $i-j$.
3. Rotary embeddings rotate queries and keys so that scores depend on the relative offset $j-i$.
4. Multi-head attention runs $h$ heads of width $d/h$, concatenates them and applies $W_O$; it has $4d^2$ parameters regardless of $h$.
5. Pre-LN block: $Z = X+\operatorname{MHA}(\operatorname{LN}(X))$, $X' = Z+\operatorname{FFN}(\operatorname{LN}(Z))$, with $\operatorname{FFN}$ a per-token MLP of hidden width $4d$.
6. Attention is the only sublayer in which tokens interact; the FFN acts on each token separately and holds about two-thirds of a block's parameters.
7. A block has about $12d^2$ parameters; its cost is $O(nd^2+n^2d)$.
8. Post-LN (the original placement) puts LayerNorm on the residual path and needs careful warmup; pre-LN keeps a clean identity path.
9. Encoder-only: no mask (BERT, ViT). Decoder-only: causal mask (GPT). Encoder–decoder: causal decoder plus cross-attention with queries from the decoder and keys and values from the encoder.
10. A causal decoder computes all $n$ next-token predictions in one parallel pass during training; generation is sequential, one token per forward pass, with cached keys and values.
11. ViT tokenizes an image into $16\times16$ patches with a linear map; a $224\times224$ image gives 196 tokens.
12. Transformers have weak inductive bias: with limited data, CNNs or GPs often do better; with large data, transformers overtake them.
13. On sampled functions, an attention head is a quadrature of an integral operator with a learned, input-dependent kernel.

---

## Practice problems

**P1.** A transformer encoder has $L=6$ blocks of width $d=512$. Ignoring biases, LayerNorm and embeddings, how many parameters do the blocks have? What fraction is in the FFNs?

*Answer.* $12\cdot6\cdot512^2 = 18{,}874{,}368\approx1.9\times10^7$. The FFNs hold $8d^2/12d^2 = 2/3$.

**P2.** With $n=10$ tokens, $d=64$ and $h=8$, give the shape of each head's query matrix, of the score tensor for all heads, and of the multi-head output.

*Answer.* $Q^{(r)}\in\mathbb{R}^{10\times8}$; scores $8\times10\times10$; output $10\times64$.

**P3.** For a single frequency $\omega$, compute $p_i^\top p_j$ for the encoding $p_i = (\sin i\omega,\cos i\omega)$. What does this mean for attention?

*Answer.* $\sin i\omega\sin j\omega+\cos i\omega\cos j\omega = \cos((i-j)\omega)$. The similarity between two positional encodings depends only on how far apart the positions are, not on where they are, giving attention a translation-invariant notion of nearness.

**P4.** A $256\times256$ temperature field is fed to a ViT-style encoder with $16\times16$ patches. How many tokens are there, and how many attention scores per head per layer? Compare with one token per grid point.

*Answer.* $16\times16=256$ tokens, $256^2 = 65{,}536$ scores. One token per grid point would give $65{,}536$ tokens and $\approx4.3\times10^9$ scores, $65{,}536$ times more.

**P5.** You train a decoder-only language model and forget to apply the causal mask. Training loss drops to nearly zero within a few hundred steps, but generated text is gibberish. Explain.

*Answer.* Without the mask, position $t$ can attend to position $t+1$, whose token is exactly the target. The model learns to copy the answer from the future rather than predict it. At generation time the future is not available, so the learned shortcut fails.

---

## References

- Alammar, J. (2018). The illustrated transformer. jalammar.github.io.
- Bi, K., Xie, L., Zhang, H., Chen, X., Gu, X., & Tian, Q. (2023). Accurate medium-range global weather forecasting with 3D neural networks. *Nature*, 619, 533–538.
- Bodnar, C., et al. (2025). A foundation model for the Earth system. *Nature*.
- Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. *NAACL*.
- Dong, Y., Cordonnier, J.-B., & Loukas, A. (2021). Attention is not all you need: Pure attention loses rank doubly exponentially with depth. *ICML*.
- Dosovitskiy, A., et al. (2021). An image is worth 16×16 words: Transformers for image recognition at scale. *ICLR*.
- Elhage, N., et al. (2021). A mathematical framework for transformer circuits. transformer-circuits.pub.
- Hoffmann, J., et al. (2022). Training compute-optimal large language models. *NeurIPS*.
- Jumper, J., et al. (2021). Highly accurate protein structure prediction with AlphaFold. *Nature*, 596, 583–589.
- Kaplan, J., et al. (2020). Scaling laws for neural language models. arXiv:2001.08361.
- Karpathy, A. (2023). Let's build GPT: from scratch, in code, spelled out. YouTube.
- Kovachki, N., Li, Z., Liu, B., Azizzadenesheli, K., Bhattacharya, K., Stuart, A., & Anandkumar, A. (2023). Neural operator: Learning maps between function spaces. *JMLR*, 24(89), 1–97.
- Murphy, K. P. (2022). *Probabilistic Machine Learning: An Introduction*. MIT Press. §15.5.
- Phuong, M., & Hutter, M. (2022). Formal algorithms for transformers. arXiv:2207.09238.
- Prince, S. J. D. (2023). *Understanding Deep Learning*. MIT Press. Ch. 12.
- Radford, A., Narasimhan, K., Salimans, T., & Sutskever, I. (2018). Improving language understanding by generative pre-training. OpenAI technical report.
- Su, J., Ahmed, M., Lu, Y., Pan, S., Bo, W., & Liu, Y. (2024). RoFormer: Enhanced transformer with rotary position embedding. *Neurocomputing*, 568, 127063.
- Vaswani, A., et al. (2017). Attention is all you need. *NeurIPS*.
- Xiong, R., et al. (2020). On layer normalization in the transformer architecture. *ICML*.
