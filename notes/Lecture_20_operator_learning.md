# Lecture 20 — Operator Learning Architectures: DeepONet, Graph and Fourier Neural Operators, and Attention

**ENM 5310 — Data-driven Modeling and Probabilistic Scientific Computing**

**Builds on:** Lecture 4 (PCA and the SVD), Lecture 14 (the convolution theorem; resolution dependence of CNNs), Lectures 16–17 (attention, cross-attention, the encoder–decoder bottleneck), Lecture 19 §§0–4 (the operator-learning problem and its integral loss; neural fields; spectral bias and the NTK; conditional neural fields and conditioning mechanisms).

**Reading:** Kovachki et al. (2023), *Neural Operator: Learning Maps Between Function Spaces*, JMLR, §§3–5. Lu et al. (2021), *DeepONet*, Nature Machine Intelligence. Li et al. (2021), *Fourier Neural Operator for Parametric PDEs*, ICLR. Seidman et al. (2022), *NOMAD*, NeurIPS, §§1–3. Optional: Lanthaler, Mishra & Karniadakis (2022), §3; Cao (2021), *Choose a Transformer: Fourier or Galerkin*.

**The point of this lecture.** Lecture 19 ended with one equation: an operator model is an encoder followed by a conditional neural field,

$$
\mathcal G_\theta(a)(y) = f_\theta\big(y;\,E_\phi(a)\big).
$$

Today we use it to organize the main architectures. The organizing question is **what the decoder at a query point $y$ gets to see**. If it sees one global code, we get DeepONet and its relatives, and a provable limitation. If it sees a query-dependent summary of the input — a kernel integral — we get graph neural operators, Fourier neural operators, and attention-based neural operators, which differ only in how they parameterize the kernel.

---

### Notation

As in Lecture 19.

| Symbol | Meaning |
|---|---|
| $D\subset\mathbb{R}^d$ | domain |
| $a:D\to\mathbb{R}^{d_a}$, $u:D\to\mathbb{R}^{d_u}$ | input and output functions; $u=\mathcal G(a)$ |
| $\mu$ | distribution of input functions; $\mathcal G_\#\mu$ the induced distribution of outputs |
| $\{x_j\}_{j=1}^n$, $\{y_k\}_{k=1}^m$ | input sample locations, output query locations |
| $z\in\mathbb{R}^p$ | global latent code; $Z\in\mathbb{R}^{M\times d}$ a set of latent tokens |
| $v_\ell:D\to\mathbb{R}^{d_v}$ | latent field after layer $\ell$ |
| $\kappa(x,x')\in\mathbb{R}^{d_v\times d_v}$ | integral kernel |
| $\gamma(x)$ | Fourier-feature embedding of a coordinate |
| $\mathcal F$, $\hat g(k)$ | Fourier transform, Fourier coefficient at wavenumber $k$ |


**Learning objectives.** After this lecture you should be able to:

1. Define global and local conditioning, and classify an operator architecture by which one it uses.
2. Write DeepONet, state its universal approximation guarantee, and derive the PCA/$n$-width lower bound on any linear decoder.
3. Compute that bound for a translated profile, and explain why transport problems defeat linear decoders while nonlinear decoders do not suffer this limit.
4. Derive the neural operator layer from the Green's-function solution of a linear PDE.
5. Construct graph and Fourier neural operators as choices of kernel, and state their costs, strengths and failure modes.
6. Explain attention as an input-dependent kernel integral, write a transformer neural operator with a cross-attention decoder, and name the main ways to reduce its cost.
7. Choose an architecture for a given data format and problem, and evaluate it honestly.

---

## 0. The lens: operators as conditional neural fields (12 min)

### 0.1 The decomposition

Every architecture in this lecture fits

$$
\mathcal G_\theta(a)(y) = f_\theta\big(y;\ \text{conditioning}(a,y)\big),
$$

an **encoder** that processes the input function, observed at $\{(x_j,a(x_j))\}_{j=1}^n$, and a **decoder** that is a conditional neural field in the query coordinate $y$ (Lecture 19 §4). The decisive design choice is whether the conditioning depends on $y$.

### 0.2 Global conditioning

The encoder produces one vector, and every query sees the same one:

$$
\mathcal G_\theta(a)(y) = f_\theta(y;z),\qquad z=E_\phi(a)\in\mathbb{R}^p .
$$

All information about $a$ that the output can use passes through $p$ numbers. This is the RNN encoder–decoder of Lecture 16 §0, with its single context vector: a **bottleneck**.

### 0.3 Local conditioning

Each query sees its own summary of the input, built by weighting input features by their relevance to $y$:

$$
\mathcal G_\theta(a)(y) = f_\theta\big(y;\,c(y)\big),\qquad c(y)=\sum_{j=1}^{n}w_j\,\kappa_\theta(y,x_j)\,h_j\;\approx\;\int_D\kappa_\theta(y,x)\,h(x)\,dx ,
$$

where $h_j$ are encoded features of the input near $x_j$ and $w_j$ are quadrature weights. This is attention (Lecture 16 §1): a fresh context for every output location, so there is no fixed bottleneck.

| | Global conditioning | Local conditioning |
|---|---|---|
| Decoder at $y$ sees | the same $z\in\mathbb{R}^p$ | a query-dependent integral $c(y)$ over the input |
| Information limit | $p$ numbers | grows with the number of input samples |
| Cost per query | $O(p)$ after encoding | $O(n)$ or $O(M)$ per query, depending on the kernel |
| Typical failure | fine, localized or moving features | cost, and a kernel that is hard to parameterize |
| Examples | PCA-Net, DeepONet, NOMAD | graph NOs, FNO, transformer NOs |

> **Remember.** Ask of any operator model: does the decoder at $y$ see one global code, or a $y$-dependent integral of the input? The first gives a bottleneck; the second gives kernel-integral architectures, distinguished by how they parameterize the kernel.

---

## 1. DeepONet, its linear decoder, and the $n$-width barrier (18 min)

### 1.1 The architecture

DeepONet (Lu et al., 2021) has two networks. The **branch** network takes the input function at $m$ fixed sensor locations and returns $p$ coefficients; the **trunk** network takes a query location and returns $p$ basis functions:

$$
\mathcal G_\theta(a)(y) = \sum_{k=1}^{p}\underbrace{B_k\big(a(x_1),\dots,a(x_m)\big)}_{\text{branch}}\;\underbrace{T_k(y)}_{\text{trunk}}\;+\;b_0 .
$$

In the language of §0: the branch is the encoder, $z=B(a)\in\mathbb{R}^p$; the trunk is a neural field; and the decoder is **linear in $z$** — conditioning enters only at the last layer, as coefficients of a learned basis. Global conditioning, linear decoder.

It is backed by a **universal approximation theorem for operators** (Chen & Chen, 1995): for any continuous operator $\mathcal G$ on a compact set of input functions and any $\varepsilon>0$, there are $p$, $m$, sensor locations and networks $B,T$ such that the error above is below $\varepsilon$ uniformly. As in Lecture 12, this says nothing about how large $p$ must be. The rest of this section answers that.

Three practical notes.

- **The trunk is a neural field** and inherits spectral bias (Lecture 19 §3); Fourier features on $y$ are standard.
- **The output side is discretization-invariant** — query anywhere — but the **input side is not**: the branch requires the input at the same $m$ sensors for every example. Data on varying meshes must be interpolated to the sensors first.
- **Variants** fix the basis instead of learning it: POD-DeepONet uses the leading principal components of the training outputs as the trunk (Lu et al., 2022), which is PCA-Net (Bhattacharya et al., 2021) with a neural encoder.

### 1.2 The barrier: any linear decoder is limited by PCA

Whatever the branch and trunk, every DeepONet output lies in the fixed affine subspace $b_0+\operatorname{span}\{T_1,\dots,T_p\}$. So its error can be no smaller than the error of the best possible $p$-dimensional subspace. On average over the data distribution, that best subspace is found by PCA (Lecture 4):

1. Let $\lambda_1\ge\lambda_2\ge\cdots$ be the eigenvalues of the covariance operator of the output distribution $\mathcal G_\#\mu$, with eigenfunctions $\varphi_k$ (the principal components of the output functions).
2. Among all $p$-dimensional subspaces $V_p$, the expected squared distance $\mathbb E_u\big[\min_{v\in V_p}\lVert u-v\rVert^2\big]$ is minimized by $V_p=\operatorname{span}\{\varphi_1,\dots,\varphi_p\}$, with value $\sum_{k>p}\lambda_k$ — the Eckart–Young result in function space.
3. A linear decoder outputs a point of a fixed $p$-dimensional subspace (after subtracting the mean), so it cannot beat the best one:

$$
\mathbb E_{a\sim\mu}\big\lVert\mathcal G(a)-\mathcal G_\theta(a)\big\rVert_{L^2}^2\;\ge\;\sum_{k>p}\lambda_k .
$$

This is the lower bound of Lanthaler, Mishra & Karniadakis (2022). Its worst-case version is the **Kolmogorov $p$-width** of the output set $\mathcal K=\mathcal G(\mathcal A)$,

$$
d_p(\mathcal K) = \inf_{\dim V_p=p}\ \sup_{u\in\mathcal K}\ \min_{v\in V_p}\lVert u-v\rVert ,
$$

the best uniform error achievable by any $p$-dimensional linear space. No training, data or branch network can push a linear decoder below these numbers. **The eigenvalue decay of the output distribution sets the price of linear decoding.**

### 1.3 When the barrier bites: transport

Diffusion smooths, so its solutions have rapidly decaying spectra and few principal components suffice. Transport does the opposite: it moves features without smoothing them. Take the simplest case, a fixed profile $\varphi$ translated by a random shift on the periodic domain $[0,1)$:

$$
u_s(x) = \varphi(x-s),\qquad s\sim\mathcal U[0,1).
$$

This family is one-dimensional — a single number $s$ identifies each function. But:

1. The covariance $C(x,x')=\mathbb E_s[u_s(x)u_s(x')] = \int_0^1\varphi(x-s)\varphi(x'-s)\,ds$ depends only on $x-x'$. It is stationary.
2. A stationary covariance on a periodic domain is diagonalized by Fourier modes, with eigenvalue $|\hat\varphi(k)|^2$ for each of $\cos 2\pi kx$ and $\sin 2\pi kx$ (Lecture 14 §1.2, again).
3. So the PCA tail is the tail of $\varphi$'s power spectrum: keeping all frequencies up to $K$ uses $p\approx2K$ basis functions and leaves error $\sum_{|k|>K}|\hat\varphi(k)|^2$.

For a **smooth bump** the spectrum decays exponentially, and a few dozen modes suffice. For a **square pulse** of width $\tfrac12$, $|\hat\varphi(k)|^2 = \sin^2(\pi k/2)/(\pi k)^2$, which decays only like $k^{-2}$. Summing the tail and dividing by $\lVert\varphi\rVert^2=\tfrac12$ gives a relative error of about

$$
\frac{\lVert u-\mathcal G_\theta(a)\rVert}{\lVert u\rVert}\;\gtrsim\;\frac{2}{\pi\sqrt p}.
$$

Reaching 1% requires $p\approx4{,}000$ basis functions — to represent a family that has one degree of freedom. Shocks, fronts, interfaces and traveling waves all behave this way.

### 1.4 The fix: a nonlinear decoder

A nonlinear decoder can represent the same family with $p=1$: $f_\theta(y;z)=\varphi(y-z)$ with $z=s$. The family is a **one-dimensional nonlinear manifold** embedded in function space; it just isn't close to any low-dimensional *linear* subspace. **NOMAD** (Seidman et al., 2022) keeps DeepONet's global code and replaces the linear decoder by a conditional neural field — by concatenation $f_\theta([\gamma(y);z])$ or by modulation (Lecture 19 §4.3). The $n$-width bound no longer applies, and on transport-dominated problems the same code size $p$ gives much smaller errors.

The other way around the barrier is to drop the single global code altogether: local conditioning, the subject of the rest of the lecture.

> **Remember.** DeepONet: branch → $p$ coefficients, trunk → $p$ basis functions, output = their dot product. Global conditioning with a linear decoder. Any linear decoder's mean-squared error is at least the PCA tail $\sum_{k>p}\lambda_k$ of the output distribution; for translated sharp features this decays like $p^{-1}$, so $p$ must be huge. Nonlinear decoders (NOMAD) or local conditioning avoid the barrier.

---

## 2. From Green's functions to the neural operator layer (12 min)

### 2.1 Linear PDEs are integral operators

Consider a linear boundary-value problem, $\mathcal Lu=f$ on $D$ with $u=0$ on $\partial D$. Its solution operator $f\mapsto u$ is an integral operator with the **Green's function** $G$ as kernel:

$$
u(x) = \int_D G(x,x')\,f(x')\,dx',\qquad \mathcal L_xG(x,x')=\delta(x-x').
$$

Example: $-u''=f$ on $(0,1)$ with $u(0)=u(1)=0$ has $G(x,x') = \min(x,x')\big(1-\max(x,x')\big)$. For $f\equiv1$, integrating gives $u(x)=x(1-x)/2$, as you can check directly.

Three features of Green's functions guide the architectures that follow:

1. **They are global.** $G(x,x')$ is generally nonzero for all pairs, so the value at $x$ depends on the forcing everywhere. (For the free-space Laplacian in 3D, $G = 1/(4\pi\lVert x-x'\rVert)$ decays slowly.)
2. **For translation-invariant operators they are stationary:** constant-coefficient problems on unbounded or periodic domains have $G(x,x')=G(x-x')$, so the solution is a convolution.
3. **For variable coefficients they depend on the coefficients.** For $-(k(x)u')'=f$, the Green's function depends on $k$; the map $k\mapsto u$ is therefore *nonlinear* in $k$, even though $f\mapsto u$ is linear.

### 2.2 The neural operator layer

To learn nonlinear operators, Li et al. (2020) and Kovachki et al. (2023) combine learned integral operators with pointwise nonlinearities, by analogy with an MLP layer $\sigma(Wh+b)$:

$$
\boxed{\;v_{\ell+1}(x) = \sigma\Big(W_\ell\,v_\ell(x) + \int_D\kappa_\ell(x,x')\,v_\ell(x')\,dx' + b_\ell(x)\Big)\;}
$$

The full model is

$$
v_0(x) = P\big(a(x),\gamma(x)\big)\ \ (\text{lift}),\qquad v_1,\dots,v_L\ \ (\text{kernel layers}),\qquad u(y)=Q\big(v_L(y)\big)\ \ (\text{project}),
$$

with $P$ and $Q$ applied pointwise. The pointwise term $W_\ell v_\ell(x)$ is a local, skip-like path; the integral is a nonlocal, Green's-function-like interaction; the nonlinearity and depth let the effective kernel depend on the input, as in point 3. A stack of such layers can approximate any continuous operator (Kovachki et al., 2023).

### 2.3 Discretization and the CNF view

On data, the integral is replaced by a quadrature over the available samples,

$$
\int_D\kappa(x,x')\,v(x')\,dx'\;\approx\;\sum_{j=1}^{n}w_j\,\kappa(x,x_j)\,v(x_j),
$$

and the same parameters apply to any discretization: the layer is defined on functions, and the sample points only determine the quadrature. The output $u(y)=Q(v_L(y))$ depends on the whole input through $L$ integrals, with weights determined by $y$. This is **local conditioning**: the decoder at $y$ sees the latent field at $y$, a query-dependent summary of $a$.

What remains is the hard part: how to parameterize $\kappa$, and how to compute the integral for all $x$ without $O(n^2)$ cost. The next three sections give three answers.

> **Remember.** The solution of a linear PDE is $\int G(x,x')f(x')dx'$: global, stationary for constant coefficients, input-dependent for variable ones. A neural operator layer is $v\mapsto\sigma(Wv+\int\kappa v+b)$, an MLP layer on functions, discretized by quadrature. Each architecture is a choice of kernel and quadrature.

---

## 3. Graph neural operators (10 min)

**Kernel choice:** a small MLP of the two locations (and optionally the input values), restricted to a neighborhood.

$$
v_{\ell+1}(x_i) = \sigma\Big(W v_\ell(x_i)+\frac{1}{|\mathcal N(i)|}\sum_{x_j\in\mathcal N(i)}\kappa_\phi\big(x_i,x_j,a(x_i),a(x_j)\big)\,v_\ell(x_j)\Big),\qquad \mathcal N(i)=\{x_j:\lVert x_j-x_i\rVert\le r\}.
$$

The average is a Monte Carlo estimate of the integral over the ball $B_r(x_i)$, and the computation is message passing on the graph that connects points within distance $r$ (Li et al., 2020). The kernel network $\kappa_\phi$ outputs a $d_v\times d_v$ matrix for each pair.

**Strengths.** It works on any mesh or point cloud, including irregular geometries and adaptive refinement, and the kernel can depend on the input coefficients, like a variable-coefficient Green's function.

**Weaknesses.** Cost is $O(n\bar k\,d_v^2)$ for $\bar k$ neighbors per point, and it grows quickly with $r$. A truncated kernel misses the long-range part of the Green's function (§2.1, point 1): reaching distance $R$ takes about $R/r$ layers, the same limitation as CNNs. Multipole graph neural operators address this with a hierarchy of graphs at different resolutions, in the spirit of the fast multipole method (Li et al., 2020b).

**Its modern role** is often as an adapter: map irregular input points to a regular latent grid with a GNO layer, process on the grid with an FNO, and map back to arbitrary output points with another GNO layer (GINO; Li et al., 2023). The final GNO layer is a kernel integral from grid nodes to the query $y$ — local conditioning, made explicit.

---

## 4. Fourier neural operators (13 min)

### 4.1 Construction

**Kernel choice:** stationary, $\kappa(x,x')=\kappa(x-x')$, motivated by §2.1 point 2. The integral becomes a convolution, which the convolution theorem turns into a product in Fourier space:

$$
\int_D\kappa(x-x')\,v(x')\,dx' = \mathcal F^{-1}\big(\hat\kappa(k)\,\hat v(k)\big)(x).
$$

The FNO (Li et al., 2021) parameterizes $\hat\kappa$ directly: a learned complex matrix $R_\ell(k)\in\mathbb{C}^{d_v\times d_v}$ for each wavenumber up to a cutoff $k_{\max}$ per dimension, and zero beyond:

$$
v_{\ell+1}(x) = \sigma\Big(W_\ell v_\ell(x) + \mathcal F^{-1}\big[\,R_\ell(k)\,\hat v_\ell(k)\,\mathbf 1_{|k|\le k_{\max}}\big](x)\Big).
$$

On a regular grid of $n$ points: FFT each channel, multiply the retained modes by $R_\ell(k)$ (mixing channels), zero the rest, inverse FFT, add the pointwise path $W_\ell v_\ell$, apply $\sigma$. A typical model uses lifting, four Fourier layers and projection.

### 4.2 Properties

- **Global in one layer.** Every output point depends on every input point, like a Green's function. Compare GNOs and CNNs, which need depth to reach far.
- **Cost** $O(n\log n)$ per channel for the FFTs, plus $O(n\,d_v^2)$ for the pointwise path.
- **Parameters** $O(k_{\max}^{d}\,d_v^2)$ per layer, independent of the grid.
- **Resolution transfer.** A CNN stencil is tied to grid spacing (Lecture 14 §7). FNO weights are tied to physical wavenumbers, so the same weights define the same operator on any grid fine enough to resolve $k_{\max}$.
- **Spectral regularization.** Truncating at $k_{\max}$ removes high-frequency content from the kernel. The nonlinearity and the pointwise path reintroduce high frequencies in the output, so the model is not band-limited overall — but the global interactions are.

### 4.3 Limitations, and the usual fixes

| Limitation | Why | Common fix |
|---|---|---|
| Requires regular grids | the FFT does | deform the domain to a grid (Geo-FNO; Li et al., 2023b), or wrap with GNO layers (GINO) |
| Assumes periodicity | the DFT does | pad the domain with zeros before the FFT and crop after |
| Stationary kernel per layer | by construction | depth and nonlinearity make the overall operator input-dependent, but strongly heterogeneous problems can still be hard |
| Aliasing across resolutions | pointwise nonlinearities create frequencies a coarse grid cannot represent | test at the target resolution; alias-free designs (Bartolucci et al., 2023) |

### 4.4 The CNF view

The FNO outputs a latent field on the grid, and the prediction at $y$ is $Q(v_L(y))$: local conditioning. At grid points this is direct. At arbitrary $y$, the last latent field can be evaluated by its truncated Fourier series (spectral interpolation) or by a cross-attention or GNO decoder from the grid, which makes the conditional-neural-field decoder explicit.

> **Remember.** FNO: stationary kernel, applied by FFT, with learned complex weights on the lowest $k_{\max}$ modes. Global in one layer, $O(n\log n)$, parameters independent of resolution, weights tied to physical frequencies. Needs regular grids and padding for non-periodic domains; test resolution transfer rather than assume it.

---

## 5. Attention-based neural operators (18 min)

### 5.1 Attention is an input-dependent kernel integral

One attention head applied to a sampled function computes

$$
\mathrm{Attn}(v)(x_i) = \sum_{j=1}^{n}\frac{e^{s(x_i,x_j)}}{\sum_{j'}e^{s(x_i,x_{j'})}}\,W_Vv(x_j),\qquad s(x,x')=\frac{\big(W_Qv(x)\big)^\top W_Kv(x')}{\sqrt{d_k}} .
$$

That is the neural operator layer of §2.2 with the kernel

$$
\kappa(x,x') = \frac{e^{s(x,x')}}{\int_D e^{s(x,x'')}\,dx''}\,W_V ,
$$

which is **computed from the current latent field**. Compare the three kernels:

| | Kernel depends on | Support | Grid needed? |
|---|---|---|---|
| GNO | locations (and input values) through an MLP | local, radius $r$ | no |
| FNO | relative position only | global | regular |
| Attention | the content of the latent field at both points | global | no |

Attention is the only one whose kernel changes with the input at every layer — the closest analogue of a variable-coefficient Green's function (§2.1, point 3).

**Discretization.** The softmax is a ratio of two quadratures. With samples drawn from a density $\rho$, it converges as $n\to\infty$ to $\int e^{s}W_Vv\,\rho\,dx'/\int e^{s}\rho\,dx'$, a well-defined operator on functions. If the sampling is non-uniform (a refined mesh), adding $\log w_j$ to the scores, with $w_j$ the quadrature weights, makes it integrate against the intended measure: $\operatorname{softmax}_j(s_{ij}+\log w_j) = w_je^{s_{ij}}/\sum_{j'}w_{j'}e^{s_{ij'}}$.

### 5.2 The template: encoder, latent tokens, cross-attention decoder

Most attention-based neural operators share one structure, which is a conditional neural field with local conditioning.

**Encoder.** Turn the input samples into tokens, $h_j = P\,a(x_j)+\gamma(x_j)W_{\text{pos}}$, with Fourier features of the physical coordinates as positional encoding (Lecture 17 §1.4). Then apply transformer blocks: self-attention, which is the input-dependent integral of §5.1, plus per-token MLPs, residuals and LayerNorm.

**Decoder.** For any query $y$, embed it, $e(y)=\gamma(y)W_e$, and cross-attend to the encoder's tokens $Z=\{z_m\}$:

$$
c(y) = \sum_{m}\operatorname{softmax}_m\Big(\frac{(e(y)W_Q)^\top z_mW_K}{\sqrt{d_k}}\Big)\,z_mW_V,\qquad \mathcal G_\theta(a)(y) = g_\theta\big(e(y)+c(y)\big).
$$

This is the local-conditioning formula of §0.3, with attention as the kernel: each query computes its own weights over the encoded input. Queries do not attend to one another, so the output can be evaluated at any points, any number of them, in parallel — including a random subset at each training step (Lecture 19 §0.3). It is Bahdanau attention (Lecture 16 §1) with spatial coordinates as the queries.

```python
def operator(xs, a_vals, ys, P):                   # input samples -> any query points
    h = a_vals @ P.W_in + fourier_features(xs, P.B) @ P.W_pos     # (n, d) tokens
    z = encoder(h, P.enc)                                          # (M, d) tokens
    e = fourier_features(ys, P.Bq) @ P.W_q                         # (m, d) queries
    for p in P.dec_blocks:                                         # queries are independent
        e = e + cross_attention(q=layer_norm(e), kv=z, p=p)       # local conditioning
        e = e + mlp(layer_norm(e), p)
    return e @ P.W_out                                             # (m, d_u) = G(a)(ys)
```

### 5.3 Making it affordable

Self-attention over $n$ input points costs $O(n^2d)$: about $2.7\times10^{11}$ operations per layer for a $256\times256$ field with $d=64$. Four ideas are widely used, often combined.

| Idea | How | Cost | Example |
|---|---|---|---|
| **Patches** | On regular grids, group $P\times P$ points into one token (as in ViT) | $O\big((n/P^2)^2d\big)$ | CViT (Wang et al., 2024) |
| **Latent tokens** | $M\ll n$ learned queries cross-attend to the input, then self-attend among themselves | $O(nMd+M^2d)$ | Perceiver IO (Jaegle et al., 2022) |
| **Physics-aware slices** | Softly assign the $n$ points to $M$ learned "slices", attend among slice tokens, then broadcast back | $O(nMd+M^2d)$ | Transolver (Wu et al., 2024) |
| **Linear (Galerkin-type) attention** | Drop the softmax and reassociate: $Q(K^\top V)/n$ instead of $(QK^\top)V$ | $O(nd^2)$ | Galerkin transformer (Cao, 2021) |

The last row has a clean interpretation. Without the softmax, $(K^\top V)/n\approx\int k(x')^\top v(x')\,dx'$ is a $d\times d$ matrix of inner products between learned functions, and multiplying by $Q$ projects onto a learned basis — a Petrov–Galerkin projection. It is a rank-$d$ kernel, $\kappa(x,x')=q(x)^\top k(x')$: linear-cost, but low-rank, much like DeepONet's kernel (Lecture 19 §6.4). Speed is traded for the expressiveness of a full kernel.

### 5.4 Two examples of the template

- **Grids in, any points out.** A vision-transformer encoder on patches of a gridded input, followed by a cross-attention decoder at continuous query coordinates (CViT; Wang et al., 2024). The patch tokens are the conditioning set $Z$.
- **Irregular samples in, any points out.** Tokens per input point or per learned latent, with a cross-attention decoder (OFormer, Li et al., 2023c; GNOT, Hao et al., 2023, which also handles multiple input functions). Attention-based decoders over output queries also appear in LOCA (Kissas et al., 2022), which couples the attention weights of nearby query points through a kernel.

Whichever variant: if the decoder averaged all tokens into one vector before reading them, it would become global conditioning, and the bottleneck of §1 would return. What makes these models expressive is that each query **selects** what it reads.

> **Remember.** Attention is a kernel integral whose kernel is computed from the input at every layer: global, mesh-free, and input-dependent. Transformer NOs encode input samples into tokens and decode with cross-attention from Fourier-embedded query coordinates — a conditional neural field with local conditioning. Patches, latent tokens, slices and linear attention reduce the $O(n^2)$ cost; linear attention does so by making the kernel low-rank.

---

## 6. Synthesis: one table, and how to choose (7 min)

### 6.1 The family

| Model | Conditioning | Kernel / decoder | Input sampling | Output sampling | Main limitation |
|---|---|---|---|---|---|
| PCA-Net, POD-DeepONet | global | linear, fixed basis | fixed | anywhere the basis is defined | $n$-width barrier |
| DeepONet | global | linear, learned basis (rank $p$) | fixed sensors | anywhere | $n$-width barrier; fixed sensors |
| NOMAD | global | nonlinear conditional field | fixed sensors | anywhere | fixed-size code |
| Graph NO | local | learned, compact support | any mesh | mesh nodes, or a final kernel layer to any $y$ | local reach; cost grows with $r$ |
| FNO | local | stationary, global, low-pass | regular grid | grid (or spectral interpolation) | grids, periodicity, aliasing |
| Transformer NO | local | input-dependent (attention) | any points | anywhere (cross-attention) | $O(n^2)$ unless reduced; data hunger |

### 6.2 How to choose

| If your data look like… | Start with | Because |
|---|---|---|
| Regular grid, roughly periodic or smooth boundaries, moderate resolution | **FNO** | global, cheap, a strong baseline |
| Smooth outputs with fast-decaying PCA spectrum, fixed sensors | **DeepONet / POD-DeepONet** | simplest and cheapest; the $n$-width barrier will not bite |
| Moving fronts, shocks, transport-dominated outputs | **nonlinear decoder** (NOMAD) or **local conditioning** | the PCA spectrum decays slowly (§1.3) |
| Irregular meshes or complex geometry | **GNO/GINO**, or a **transformer NO** with slices or latents | mesh-free kernels |
| Scattered sensors in, arbitrary query points out | **transformer NO** with a cross-attention decoder | handles both sides natively |
| Small datasets | the **simplest model that fits**, plus strong baselines | inductive bias beats flexibility with little data (Lecture 17 §6) |

**Check the spectrum first.** Before training anything, compute the PCA spectrum of your training outputs (Lecture 4). If $\sum_{k>p}\lambda_k$ is already small for modest $p$, a linear decoder suffices and anything fancier must justify itself. If it decays slowly, no linear decoder will do well, whatever its encoder.

**Evaluate like an experimentalist** (Lecture 13 §8, Lecture 19 §8): report relative $L^2$ error with mean $\pm$ std over seeds; test at resolutions and on input distributions you did not train on; and compare not only against other networks but against the cost of running the solver, *including* the cost of generating the training data. A neural operator pays off only after it has been queried many times.

---

## Code Checkpoint 20

JAX permitted for gradients. Each item asks you to report a number.

**A. The $n$-width barrier.**
1. Generate 2000 shifted profiles $u_s(x)=\varphi(x-s)$ on 256 periodic grid points for two profiles: a Gaussian bump of width $0.05$ and a square pulse of width $\tfrac12$. Compute the PCA spectrum of each. **Report the number of components needed for 1% relative error**, and compare the pulse with the estimate $2/(\pi\sqrt p)$ of §1.3.
2. Train a DeepONet ($p=32$) and a NOMAD-style decoder with concatenation ($p=32$) to map $a = u_0$ to $u = u_0(\cdot-cT)$ for advection with random shifts, on both profiles. **Report test relative $L^2$ error** (mean $\pm$ std over 3 seeds), and compare the DeepONet error with the PCA bound for $p=32$.

**B. FNO resolution transfer.** Implement a 1D FNO (four layers, $k_{\max}=16$, $d_v=64$) for viscous Burgers' equation, $u_0\mapsto u(\cdot,1)$, trained at 256 points. **Report test error** at 128, 256 and 1024 points. Repeat with $k_{\max}=64$ and explain any change in transfer with §4.3.

---

## Quiz-eligible facts

1. Global conditioning passes one code $z$ to every query; local conditioning gives each query its own kernel-weighted integral of the input.
2. DeepONet: $\mathcal G_\theta(a)(y)=\sum_{k=1}^pB_k(a(x_1),\dots,a(x_m))T_k(y)+b_0$ — global conditioning, linear decoder; universal (Chen & Chen, 1995), but fixed input sensors.
3. Any linear decoder has mean-squared error at least $\sum_{k>p}\lambda_k$, the tail of the output distribution's PCA spectrum; the worst-case analogue is the Kolmogorov $p$-width.
4. Translated profiles have a stationary covariance, so their PCA basis is Fourier modes with eigenvalues $|\hat\varphi(k)|^2$; for a discontinuous profile the relative error decays only like $p^{-1/2}$.
5. A nonlinear decoder (NOMAD) can represent a one-parameter family of translations with $p=1$.
6. Linear PDE solutions are $u(x)=\int G(x,x')f(x')dx'$; Green's functions are global, stationary for constant coefficients, input-dependent for variable coefficients.
7. Neural operator layer: $v_{\ell+1}(x)=\sigma(W_\ell v_\ell(x)+\int\kappa_\ell(x,x')v_\ell(x')dx'+b_\ell)$, discretized by quadrature.
8. GNO: learned kernel on a radius-$r$ neighborhood, message passing on any mesh; reaching distance $R$ takes about $R/r$ layers.
9. FNO: stationary kernel, learned complex weights on the lowest $k_{\max}$ Fourier modes; global in one layer, $O(n\log n)$, resolution-independent parameters; needs regular grids and padding for non-periodic problems.
10. Attention is a kernel integral whose kernel depends on the latent field; adding $\log w_j$ to the scores applies quadrature weights for non-uniform samples.
11. Transformer NOs decode by cross-attention from Fourier-embedded query coordinates: local conditioning, output at any points.
12. Linear (Galerkin-type) attention computes $Q(K^\top V)/n$ in $O(nd^2)$; its kernel $q(x)^\top k(x')$ has rank at most $d$.

---

## Practice problems

**P1.** For translated square pulses (§1.3), how many basis functions does a linear decoder need for 5% relative error? For a 1-parameter family, what is the minimum code size for a nonlinear decoder?

*Answer.* $2/(\pi\sqrt p)=0.05$ gives $\sqrt p\approx12.7$, so $p\approx160$. A nonlinear decoder needs $p=1$: $f(y;z)=\varphi(y-z)$.

**P2.** Using $G(x,x')=\min(x,x')(1-\max(x,x'))$, compute the solution of $-u''=1$, $u(0)=u(1)=0$.

*Answer.* $u(x)=\int_0^x x'(1-x)\,dx'+\int_x^1x(1-x')\,dx' = (1-x)\tfrac{x^2}{2}+x\tfrac{(1-x)^2}{2} = \tfrac{x(1-x)}{2}$.

**P3.** Your training inputs come from simulations on different unstructured meshes. Which of DeepONet, FNO, GNO and a transformer NO can take them without preprocessing, and what would the others need?

*Answer.* GNO and a transformer NO accept arbitrary point sets directly. DeepONet needs the inputs interpolated to fixed sensor locations; FNO needs them interpolated to a regular grid (or a GNO/Geo-FNO front end).

**P4.** An FNO trained on $64\times64$ grids with $k_{\max}=12$ is applied to $256\times256$ grids. Which parameters change, and why might the error still increase?

*Answer.* None: the spectral weights are tied to wavenumbers, not grid points. The error can still change because pointwise nonlinearities generate frequencies above what the training grid could represent; at training resolution these aliased onto lower modes, and the learned weights compensated for that aliasing, which no longer happens at the finer resolution.

**P5.** For $n=65{,}536$ input points and width $d=64$, compare the cost of softmax self-attention with Galerkin-type linear attention. What is given up?

*Answer.* $n^2d\approx2.7\times10^{11}$ versus $nd^2\approx2.7\times10^{8}$ — about $n/d=1024$ times cheaper. The kernel becomes $q(x)^\top k(x')$, of rank at most $d$, so it loses the full-rank, sharply input-dependent kernel of softmax attention.

---

## References

- Bartolucci, F., de Bézenac, E., Raonić, B., Molinaro, R., Mishra, S., & Alaifari, R. (2023). Representation equivalent neural operators: A framework for alias-free operator learning. *NeurIPS*.
- Bhattacharya, K., Hosseini, B., Kovachki, N. B., & Stuart, A. M. (2021). Model reduction and neural networks for parametric PDEs. *SMAI J. Comput. Math.*, 7, 121–157.
- Cao, S. (2021). Choose a transformer: Fourier or Galerkin. *NeurIPS*.
- Chen, T., & Chen, H. (1995). Universal approximation to nonlinear operators by neural networks with arbitrary activation functions and its application to dynamical systems. *IEEE Trans. Neural Networks*, 6(4), 911–917.
- Hao, Z., et al. (2023). GNOT: A general neural operator transformer for operator learning. *ICML*.
- Jaegle, A., et al. (2022). Perceiver IO: A general architecture for structured inputs & outputs. *ICLR*.
- Kissas, G., Seidman, J. H., Guilhoto, L. F., Preciado, V. M., Pappas, G. J., & Perdikaris, P. (2022). Learning operators with coupled attention. *JMLR*, 23(215), 1–63.
- Kovachki, N., Li, Z., Liu, B., Azizzadenesheli, K., Bhattacharya, K., Stuart, A., & Anandkumar, A. (2023). Neural operator: Learning maps between function spaces. *JMLR*, 24(89), 1–97.
- Lanthaler, S., Mishra, S., & Karniadakis, G. E. (2022). Error estimates for DeepONets: A deep learning framework in infinite dimensions. *Trans. Math. Appl.*, 6(1).
- Li, Z., Kovachki, N., Azizzadenesheli, K., Liu, B., Bhattacharya, K., Stuart, A., & Anandkumar, A. (2020). Neural operator: Graph kernel network for partial differential equations. arXiv:2003.03485.
- Li, Z., et al. (2020b). Multipole graph neural operator for parametric partial differential equations. *NeurIPS*.
- Li, Z., Kovachki, N., Azizzadenesheli, K., Liu, B., Bhattacharya, K., Stuart, A., & Anandkumar, A. (2021). Fourier neural operator for parametric partial differential equations. *ICLR*.
- Li, Z., et al. (2023). Geometry-informed neural operator for large-scale 3D PDEs. *NeurIPS*.
- Li, Z., Huang, D. Z., Liu, B., & Anandkumar, A. (2023b). Fourier neural operator with learned deformations for PDEs on general geometries. *JMLR*, 24(388), 1–26.
- Li, Z., Meidani, K., & Barati Farimani, A. (2023c). Transformer for partial differential equations' operator learning. *TMLR*.
- Lu, L., Jin, P., Pang, G., Zhang, Z., & Karniadakis, G. E. (2021). Learning nonlinear operators via DeepONet based on the universal approximation theorem of operators. *Nature Machine Intelligence*, 3, 218–229.
- Lu, L., Meng, X., Cai, S., Mao, Z., Goswami, S., Zhang, Z., & Karniadakis, G. E. (2022). A comprehensive and fair comparison of two neural operators (with practical extensions) based on FAIR data. *Comput. Methods Appl. Mech. Eng.*, 393, 114778.
- Seidman, J. H., Kissas, G., Perdikaris, P., & Pappas, G. J. (2022). NOMAD: Nonlinear manifold decoders for operator learning. *NeurIPS*.
- Wang, S., Seidman, J. H., Sankaran, S., Wang, H., Pappas, G. J., & Perdikaris, P. (2024). CViT: Continuous vision transformer for operator learning. arXiv:2405.13998.
- Wu, H., Luo, H., Wang, H., Wang, J., & Long, M. (2024). Transolver: A fast transformer solver for PDEs on general geometries. *ICML*.
