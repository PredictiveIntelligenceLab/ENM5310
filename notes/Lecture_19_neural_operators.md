# Lecture 19 — Neural Operators: Learning Maps Between Functions

**ENM 5310 — Data-driven Modeling and Probabilistic Scientific Computing**

**Builds on:** Lecture 6 (gradient descent on a quadratic in the Hessian eigenbasis; the stability threshold), Lectures 8–9 (kernels, GP regression), Lecture 12 (MLPs; infinitely wide networks as GPs; the NTK remark in §6), Lecture 13 (automatic differentiation; early stopping as a spectral filter), Lecture 14 (convolution, the convolution theorem, resolution dependence of CNNs), Lectures 16–17 (attention as kernel smoothing, cross-attention, encoder–decoders and the context bottleneck, Fourier-feature positional encodings), Lecture 18 (PINNs; the operator-learning picture).

**Reading:** Kovachki et al. (2023), *Neural Operator: Learning Maps Between Function Spaces*, JMLR, §§1–5. Tancik et al. (2020), *Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains*, §§1–4. Lu et al. (2021), *DeepONet*, Nature Machine Intelligence. Li et al. (2021), *Fourier Neural Operator*, ICLR. Optional: Xie et al. (2022), *Neural Fields in Visual Computing and Beyond*, §§1–3; Wang et al. (2024), *CViT* (arXiv:2405.13998), §§2–3.

**The point of this lecture.** Many scientific datasets are not vectors but **functions**: fields sampled on meshes, sensors at scattered locations, solutions at whatever resolution the solver used. This lecture builds models that take functions in and give functions out, in five steps:

1. A function can be represented by a network $x\mapsto f_\theta(x)$: a **neural field**.
2. Neural fields learn smooth components before fine ones — **spectral bias** — and the **neural tangent kernel** explains why and how to fix it.
3. One network can represent *many* functions if it is told which one: a **conditional neural field** $f_\theta(y;z)$.
4. An operator $\mathcal G: a\mapsto u$ is then an encoder $a\mapsto z$ plus a conditional neural field $y\mapsto f_\theta(y;z)$. How the decoder is conditioned — **globally** or **locally** — separates DeepONet from graph, Fourier and transformer neural operators.
5. **Transformer neural operators** use continuous self-attention to encode and cross-attention from continuous queries to decode.

---

### Notation

| Symbol | Meaning |
|---|---|
| $D\subset\mathbb{R}^{d}$ | spatial (or space–time) domain |
| $a: D\to\mathbb{R}^{d_a}$, $u: D\to\mathbb{R}^{d_u}$ | input function, output function |
| $\mathcal A$, $\mathcal U$ | spaces of input and output functions (e.g. subsets of $L^2(D)$) |
| $\mathcal G:\mathcal A\to\mathcal U$ | the operator to be learned; $\mathcal G_\theta$ its approximation |
| $\{x_j\}_{j=1}^{n}$ | locations where the input is observed ("sensors") |
| $\{y_k\}_{k=1}^{m}$ | locations where the output is queried |
| $f_\theta: D\to\mathbb{R}^{d_u}$ | a neural field (coordinate network) |
| $f_\theta(y;z)$ | a conditional neural field with conditioning $z$ |
| $z\in\mathbb{R}^p$ | a global latent code; $Z\in\mathbb{R}^{M\times d}$ a set of $M$ latent tokens |
| $\gamma(x)$ | Fourier-feature embedding of a coordinate |
| $\Theta(x,x')$ | neural tangent kernel |
| $\kappa(x,x')$ | kernel of an integral operator |
| $\mathcal F$ | Fourier transform |

**Learning objectives.** After this lecture you should be able to:

1. State the operator-learning problem, its loss as an integral over the output domain, and what "discretization-invariant" means.
2. Represent a function with a neural field and fit it by regression on samples.
3. Derive the training dynamics of a neural field under gradient flow in terms of the NTK, and explain spectral bias from its eigendecomposition.
4. Explain how Fourier features change the NTK, and why their bandwidth matters.
5. Write a conditional neural field with concatenation, FiLM, hypernetwork, linear and cross-attention conditioning.
6. Define global and local conditioning, and place DeepONet, PCA-Net, NOMAD, graph neural operators, FNO and transformer neural operators in this picture.
7. Write the kernel-integral neural operator layer, and derive the graph and Fourier neural operators as choices of kernel.
8. Describe a transformer neural operator: continuous self-attention as quadrature in the encoder, and cross-attention from continuous queries in the decoder.

---

## 0. Why functions, not vectors (8 min)

### 0.1 The data

A PDE solver returns $u$ on its mesh; change the mesh and the vector changes length. A weather station network observes the atmosphere at a few thousand irregular points; the satellite record observes it on a different grid. An experiment measures a velocity field with PIV at one resolution and pressure with taps at a dozen points. In all of these, the **object** is a function $u: D\to\mathbb{R}^{d_u}$; the vector we store is an accident of how it was sampled.

Every model in this course so far took a fixed-length vector in. A CNN (Lecture 14) is better — it shares weights across positions — but its $3\times3$ stencil encodes the grid spacing, so a model trained on a $64\times64$ grid computes a different physical operator on a $128\times128$ grid (Lecture 14 §7). We want models that are defined on functions and merely *evaluated* on whatever samples we have.

### 0.2 The problem

Many scientific tasks have the form "given one function, predict another":

| Input function $a$ | Output function $u=\mathcal G(a)$ |
|---|---|
| permeability $k(x)$ of a porous medium | pressure $p(x)$ solving $-\nabla\cdot(k\nabla p)=f$ (Darcy flow) |
| initial condition $u_0(x)$ | solution $u(x,T)$ at a later time |
| boundary shape or forcing $f(x,t)$ | stress, temperature or velocity field |
| atmospheric state today | atmospheric state in six hours |

The map $\mathcal G:\mathcal A\to\mathcal U$ is an **operator**: a map between infinite-dimensional function spaces. A classical solver evaluates $\mathcal G(a)$ for one $a$ at a time, often at great cost. **Operator learning** fits an approximation $\mathcal G_\theta$ from examples $\{(a_i,u_i)\}_{i=1}^N$, usually produced by such a solver, so that new inputs can be mapped in milliseconds.

### 0.3 The loss is an integral

The natural risk compares functions, not vectors:

$$
\mathcal L(\theta) = \frac1N\sum_{i=1}^{N}\big\lVert u_i - \mathcal G_\theta(a_i)\big\rVert_{L^2(D)}^2 = \frac1N\sum_{i=1}^{N}\int_D\big\lvert u_i(y)-\mathcal G_\theta(a_i)(y)\big\rvert^2\,dy .
$$

We never have the integrand everywhere, only at query points $y_k$, so in practice we use a quadrature or Monte Carlo estimate: $\int_D g(y)\,dy\approx\frac{|D|}{m}\sum_{k=1}^m g(y_k)$ with $y_k$ uniform. This is the Gaussian log-likelihood of Lecture 5, summed over points of a function. Two consequences:

- The query points can be different for every example and every minibatch — and **sampling a random subset of points is an unbiased estimate of the loss**, which makes training on large fields cheap (§8).
- A model that can be *evaluated at any $y$* fits this loss naturally. That is the motivation for neural fields.

**Discretization invariance.** Informally, a neural operator should (i) accept its input at any set of sample points, (ii) produce its output at any query points, and (iii) converge to a single operator on functions as the samples become dense (Kovachki et al., 2023). The architectures below achieve these to different degrees, and §6 says which.

> **Remember.** Operator learning approximates $\mathcal G: a\mapsto u$ between function spaces from solved examples. The loss is an integral over the output domain, estimated by sampling query points; the model should not care how its input and output were sampled.

---

## 1. Neural fields (8 min)

### 1.1 A function as a network

A **neural field** (or coordinate network, or implicit neural representation) represents a single function by a network that takes a coordinate in and returns the function's value:

$$
f_\theta: D\to\mathbb{R}^{d_u},\qquad x\mapsto f_\theta(x)=\text{MLP}_\theta(x).
$$

Given samples $\{(x_j,u(x_j))\}_{j=1}^n$, fit $\theta$ by least squares, $\min_\theta\sum_j\lvert f_\theta(x_j)-u(x_j)\rvert^2$ — Lecture 12's regression with the coordinate as the input. The function is now stored in $\theta$, not on a grid.

**What this buys.**

- **Any resolution.** Evaluate $f_\theta$ at any point, on any grid, as many times as you like.
- **Any sampling.** Fit from scattered, irregular or partial observations.
- **Derivatives.** $\nabla_x f_\theta$, $\Delta_x f_\theta$ are exact by automatic differentiation (Lecture 13 §1), so a neural field can be constrained by a PDE. A PINN (Lecture 18) is a neural field fitted to a PDE residual instead of, or in addition to, data.
- **Compactness.** For smooth signals the weights can be much smaller than a dense grid.

Neural fields are now standard for representing images, 3D shapes (signed distance functions), and radiance fields for view synthesis (NeRF; Mildenhall et al., 2020).

### 1.2 The catch

Fit $u(x) = \sin(2\pi x)+\tfrac12\sin(2\pi\cdot10x)$ on $[0,1]$ with a ReLU or `tanh` MLP on raw $x$. Within a few hundred steps it captures the slow sinusoid. After many thousands it is still missing most of the fast one. Trained on a photograph, the same network produces a blurry version of it. This is not a capacity problem: by universal approximation (Lecture 12 §5) the network can represent the target. It is a property of how gradient descent fits it.

---

## 2. Spectral bias (6 min)

**Spectral bias** (or frequency bias) is the empirical observation that neural networks trained by gradient descent fit the low-frequency components of a target first and high-frequency components much later (Rahaman et al., 2019). Decompose the training residual $r = f_\theta-u$ into Fourier modes and track each over training: the low-$k$ coefficients drop quickly, the high-$k$ coefficients barely move.

It has two faces.

- **A good inductive bias** for generalization in ordinary regression: among the many functions that fit noisy data, the network prefers smooth ones (compare early stopping as regularization, Lecture 13 §6).
- **A serious obstacle** in scientific computing, where the high-frequency content *is* the physics: boundary layers, shocks, turbulence, fine microstructure. A PINN with a raw-coordinate MLP will often converge to a smooth wrong solution of a multiscale problem.

Two practical fixes are common: feed the network **Fourier features** of the coordinates, $\gamma(x)=[\cos(2\pi Bx),\sin(2\pi Bx)]$ (Tancik et al., 2020), or use **sinusoidal activations** (SIREN; Sitzmann et al., 2020). To see why these work — and how to choose $B$ — we need a model of the training dynamics.

---

## 3. The neural tangent kernel explanation (15 min)

### 3.1 Training dynamics in function space

Take a scalar neural field $f(x;\theta)$, training data $\{(x_j,y_j)\}_{j=1}^n$, and the loss $L(\theta)=\frac12\sum_{j}\big(f(x_j;\theta)-y_j\big)^2$. Analyze **gradient flow**, $\dot\theta = -\nabla_\theta L$, the small-step limit of gradient descent.

1. *Parameter dynamics.* With residuals $r_j = f(x_j;\theta)-y_j$,
$$
\dot\theta = -\sum_{j=1}^{n}\nabla_\theta f(x_j;\theta)\,r_j .
$$
2. *Function dynamics.* By the chain rule, for any input $x$,
$$
\frac{d}{dt}f(x;\theta_t) = \nabla_\theta f(x;\theta_t)^\top\dot\theta = -\sum_{j=1}^{n}\Theta_t(x,x_j)\,r_j,\qquad
\Theta_t(x,x') = \nabla_\theta f(x;\theta_t)^\top\nabla_\theta f(x';\theta_t).
$$
$\Theta_t$ is the **neural tangent kernel** (NTK; Jacot et al., 2018). It is a kernel in the sense of Lecture 8 — an inner product of feature vectors, here the parameter gradients — so the matrix $\Theta_t(X,X)$ on the training inputs is positive semi-definite.
3. *Residual dynamics.* Evaluating step 2 at the training points gives, exactly and for any width,
$$
\dot r = -\Theta_t\,r, \qquad \Theta_t = \Theta_t(X,X)\in\mathbb{R}^{n\times n}.
$$

So far nothing is approximated. The difficulty is that $\Theta_t$ changes as $\theta$ moves.

### 3.2 The infinite-width limit

Jacot et al. (2018) showed that for MLPs in the "NTK parameterization," as all widths go to infinity, $\Theta_t$ converges to a deterministic kernel $\Theta$ that **stays constant during training**: each weight moves by a vanishing amount, but there are so many that the function still changes by $O(1)$. In this regime the network trains like its linearization around initialization (Lee et al., 2019). This is the same limit as Lecture 12 §6, now for training rather than for the prior.

With $\Theta$ constant, step 3 is a linear ODE:

$$
r(t) = e^{-\Theta t}\,r(0).
$$

Diagonalize $\Theta = Q\Lambda Q^\top$, with eigenvalues $\lambda_1\ge\lambda_2\ge\dots\ge0$ and orthonormal eigenvectors $q_i$:

$$
q_i^\top r(t) = e^{-\lambda_i t}\,q_i^\top r(0).
$$

**The component of the residual along eigenvector $q_i$ decays at rate $\lambda_i$.** Directions with large eigenvalues are fitted fast; directions with small eigenvalues are fitted slowly or, in finite training time, not at all. With gradient descent at step size $\eta$, stability requires $\eta<2/\lambda_1$ (Lecture 6), so mode $i$ needs on the order of $\lambda_1/\lambda_i$ steps — a condition number per mode.

This is Lecture 6's analysis of gradient descent on a quadratic, now in function space, and Lecture 13's observation that early stopping acts as a spectral filter: at time $t$, modes with $\lambda_i\gg1/t$ are fitted and modes with $\lambda_i\ll1/t$ are not.

The prediction at a new point also has a closed form (Lee et al., 2019):

$$
f_t(x) = f_0(x) - \Theta(x,X)\,\Theta^{-1}\big(I-e^{-\Theta t}\big)\,r(0),
$$

which as $t\to\infty$ becomes kernel regression with the NTK — the noise-free GP mean of Lecture 9, with $\Theta$ as the kernel.

### 3.3 Where the frequencies come in

The remaining question is what the eigenvectors of $\Theta$ look like. For inputs spread uniformly on a circle or sphere, the NTK of a standard MLP depends only on the angle between inputs, so it is rotation invariant. Rotation-invariant kernels are diagonalized by Fourier modes on the circle (spherical harmonics on the sphere) — the same reason convolutions are diagonalized by the DFT (Lecture 14 §1.2). And the eigenvalues **decay with frequency**: for a two-layer ReLU network on the circle, $\lambda_k\propto k^{-2}$ (Basri et al., 2019); on the sphere $S^{d-1}$ the decay is polynomial, of order $k^{-d}$ (Bietti & Mairal, 2019).

Putting it together, the frequency-$k$ component of the residual behaves like

$$
\hat r_k(t)\approx e^{-c\,t/k^{2}}\,\hat r_k(0)\qquad(\text{two-layer ReLU, 1D}),
$$

so learning frequency 10 takes about 100 times longer than frequency 1. That is spectral bias, derived.

### 3.4 Why Fourier features fix it

Feed the network $\gamma(x) = \big[\cos(2\pi b_r^\top x),\ \sin(2\pi b_r^\top x)\big]_{r=1}^{R}$, with $b_r\sim\mathcal N(0,\sigma^2I)$. Two facts:

1. $\gamma(x)^\top\gamma(x') = \sum_r\cos\big(2\pi b_r^\top(x-x')\big)$ depends only on $x-x'$. This is the identity behind sinusoidal positional encodings (Lecture 17 §1.3), and it makes the NTK of the composed network approximately **stationary**, $\Theta(x,x')\approx h(x-x')$.
2. A stationary kernel is diagonalized by Fourier modes, with eigenvalues given by its Fourier transform $\hat h(\omega)$ (Bochner's theorem, Lecture 8). The frequencies $b_r$ set the width of $\hat h$: with larger $\sigma$, the spectrum extends to higher frequencies, so high-frequency residuals are no longer starved.

So **$\sigma$ is a bandwidth, i.e. a prior on how much fine structure the target has.** Too small: spectral bias persists, and the fit is blurry. Too large: the kernel is so narrow that the network interpolates the training points with high-frequency noise and generalizes badly between them (Tancik et al., 2020). For multiscale targets, mixing several bandwidths helps (Wang, Wang & Perdikaris, 2021). The same NTK analysis explains why the different loss terms of a PINN — residual, boundary, initial condition — can train at very different rates (Wang, Yu & Perdikaris, 2022).

### 3.5 What the NTK does and does not say

The NTK regime is "lazy": features hardly move, and the network is a fixed kernel machine (Lecture 12 §6). Real, finite networks do learn features, and their NTK changes during training. Quantitative NTK predictions for practical networks should therefore be treated with care. The qualitative picture — residuals decay mode by mode at rates set by a kernel's spectrum, and input embeddings reshape that spectrum — holds up well in practice, and it is what you need to diagnose a neural field or a PINN that will not fit fine scales.

> **Remember.** Under gradient flow, $\dot r = -\Theta_t r$ exactly. In the infinite-width limit $\Theta$ is constant, so each eigencomponent of the residual decays like $e^{-\lambda_i t}$. For standard MLPs the eigenfunctions are Fourier modes with eigenvalues decaying in frequency: spectral bias. Fourier features make the NTK stationary with a tunable bandwidth $\sigma$.

---

## 4. Conditional neural fields (10 min)

### 4.1 Many functions, one network

A neural field represents one function. A dataset contains thousands: $N$ solutions $u_1,\dots,u_N$, one per input. Fitting $N$ separate networks shares nothing between them. Instead, give each function a **latent code** $z_i\in\mathbb{R}^p$ and train one network on all of them:

$$
u_i(y)\approx f_\theta(y;z_i),\qquad i=1,\dots,N .
$$

The weights $\theta$ capture what all functions have in common; the code $z_i$ says which one we want. This is a **conditional neural field** (Xie et al., 2022).

### 4.2 Where the code comes from

- **Auto-decoding** (Park et al., 2019): treat each $z_i$ as a free parameter and optimize it jointly with $\theta$, $\min_{\theta,\{z_i\}}\sum_i\sum_k|f_\theta(y_k;z_i)-u_i(y_k)|^2+\lambda\lVert z_i\rVert^2$. The penalty is a Gaussian prior on codes — MAP, as in Lecture 5. A new function requires a new optimization over $z$.
- **Amortized encoding**: learn an encoder $z = E_\phi(\text{data})$ that maps observations directly to a code. A new function requires one forward pass. **Operator learning is this case, with the input function $a$ as the data:** $z=E_\phi(a)$.

### 4.3 How the code enters the network

The choice of conditioning mechanism matters as much as the choice of network.

| Mechanism | Form | Comment |
|---|---|---|
| Concatenation | $f_\theta([\gamma(y);z])$ | First layer computes $W_y\gamma(y)+W_zz$: the code only shifts the first layer's bias |
| FiLM / modulation (Perez et al., 2018) | $h_{\ell+1} = \sigma\big(s_\ell(z)\odot(W_\ell h_\ell+b_\ell)+t_\ell(z)\big)$ | The code scales and shifts every layer; shift-only modulation is used in "functa" (Dupont et al., 2022) |
| Hypernetwork (Ha et al., 2017) | $f_{\theta(z)}(y)$, $\theta(z)=H_\psi(z)$ | The code generates the weights; most expressive, most expensive |
| Linear, last layer | $f_\theta(y;z) = \sum_{k=1}^{p}z_k\,t_k(y)$ | The output lies in the span of $p$ learned basis functions $t_k$; the code only chooses coefficients |
| Cross-attention | $f_\theta(y;Z) = g\Big(\sum_{m}\alpha_m(y)\,v_m\Big)$, $\alpha_m(y)=\operatorname{softmax}_m\frac{q(y)^\top k_m}{\sqrt d}$ | The code is a *set* of tokens $Z=\{z_m\}$, and each query point $y$ reads its own weighted combination |

Linear conditioning is the most restrictive. Every output is a linear combination of the same $p$ functions, so the best achievable error over a set of outputs is bounded below by its **Kolmogorov $p$-width** — how well the set can be approximated by *any* $p$-dimensional linear subspace. For transport-dominated problems (a moving front, a traveling wave), the $p$-width decays slowly and many basis functions are needed, even though each solution is a simple shifted profile. Nonlinear decoders $f_\theta(y;z)$ can represent such a low-dimensional *nonlinear* family with small $p$ (Seidman et al., 2022).

> **Remember.** A conditional neural field $f_\theta(y;z)$ represents a family of functions with one network; the code comes from per-sample optimization (auto-decoding) or from an encoder. Conditioning ranges from linear (a span of $p$ basis functions, limited by the $p$-width) through concatenation, FiLM and hypernetworks, to cross-attention over a set of tokens.

---

## 5. Operator learning as conditional neural fields (10 min)

### 5.1 The decomposition

Any operator model that can be queried at arbitrary output points can be written as

$$
\boxed{\;\mathcal G_\theta(a)(y) = f_\theta\big(y;\;E_\phi(a)\big)\;}
$$

an **encoder** $E_\phi$ that turns the input function, given at sensors $\{(x_j,a(x_j))\}_{j=1}^n$, into conditioning information, followed by a **conditional neural field** decoder evaluated at the query $y$. Neural operators differ in how they encode, and — more fundamentally — in what the decoder at $y$ receives.

### 5.2 Global versus local conditioning

**Global conditioning.** The encoder produces one vector $z=E_\phi(a)\in\mathbb{R}^p$, and the decoder sees the *same* $z$ at every query point:

$$
\mathcal G_\theta(a)(y) = f_\theta(y;z),\qquad z = E_\phi(a).
$$

All information about $a$ that the output will ever use passes through $p$ numbers.

**Local conditioning.** The decoder at $y$ sees a **query-dependent** summary of the input,

$$
\mathcal G_\theta(a)(y) = f_\theta\big(y;\,c(y)\big),\qquad c(y) = \sum_{j=1}^{n}\kappa_\theta(y,x_j)\,h_j\;\approx\;\int_D\kappa_\theta(y,x)\,h(x)\,dx,
$$

where $h_j$ are encoded features of the input near $x_j$ and $\kappa_\theta$ decides which input locations matter for output location $y$.

This is exactly the distinction of Lecture 16 §§0–1. Global conditioning is the RNN encoder–decoder: one context vector $m$ for the whole output, a **bottleneck**. Local conditioning is attention: a fresh context $c_t$ for every output step, built from the input positions relevant to it.

| | Global conditioning | Local conditioning |
|---|---|---|
| Decoder at $y$ sees | the same $z\in\mathbb{R}^p$ | $c(y)$, an aggregate of input features weighted by relevance to $y$ |
| Information limit | fixed $p$, a bottleneck | grows with the number of input samples |
| Strengths | cheap per query; compact; easy to sample anywhere | resolves local features, fronts, multiscale structure |
| Weaknesses | struggles when outputs need fine, localized detail (§4.3) | costlier; needs a kernel or attention over inputs |
| Examples | PCA-Net, DeepONet, NOMAD | graph NOs, FNO, transformer NOs |

### 5.3 The global family

**PCA-Net** (Bhattacharya et al., 2021). Compute principal components $\{\varphi_k\}$ of the training outputs and $\{\psi_k\}$ of the training inputs. Encode $a$ by its first PCA coefficients, map them through an MLP to output coefficients, and decode linearly: $\mathcal G_\theta(a)(y)=\sum_{k=1}^{p}z_k(a)\,\varphi_k(y)$. Global and linear, with a fixed basis — model reduction with a learned map in between.

**DeepONet** (Lu et al., 2021). Learn the basis too. A *branch* network maps the input sampled at $m$ fixed sensors to coefficients, and a *trunk* network maps the query point to basis functions:

$$
\mathcal G_\theta(a)(y) = \sum_{k=1}^{p}\underbrace{B_k\big(a(x_1),\dots,a(x_m)\big)}_{\text{branch: }z=E(a)}\;\underbrace{T_k(y)}_{\text{trunk: basis}} + b_0 .
$$

In our language: global conditioning, linear last-layer conditioning, with a neural-field trunk as the learned basis. It rests on a universal approximation theorem for operators (Chen & Chen, 1995), and its error is bounded below by the $p$-width of the output set (Lanthaler et al., 2022).

**NOMAD** (Seidman et al., 2022). Keep the global code but replace the linear decoder with a nonlinear conditional neural field, $\mathcal G_\theta(a)(y) = f_\theta(y;E(a))$, e.g. by concatenation or modulation. Same bottleneck size $p$, much smaller errors on transport-dominated problems.

The trunk of a DeepONet is a neural field and suffers from spectral bias (§3): Fourier features on $y$ are standard practice.

> **Remember.** $\mathcal G_\theta(a)(y)=f_\theta(y;E_\phi(a))$. Global conditioning sends one code $z$ to every query — PCA-Net, DeepONet (linear in a learned basis) and NOMAD (nonlinear) — and inherits an encoder–decoder bottleneck. Local conditioning gives every query its own aggregate of input features, like attention.

---

## 6. Integral-operator layers: graph and Fourier neural operators (15 min)

### 6.1 The kernel integral layer

The local family is built from one layer type (Li et al., 2020; Kovachki et al., 2023). Lift the input pointwise to a $d_v$-dimensional latent field, $v_0(x) = P\big(a(x),\gamma(x)\big)$. Then apply $L$ layers of the form

$$
v_{\ell+1}(x) = \sigma\Big(W_\ell\,v_\ell(x) + \int_D\kappa_\ell(x,x')\,v_\ell(x')\,dx' + b_\ell\Big),\qquad \kappa_\ell(x,x')\in\mathbb{R}^{d_v\times d_v},
$$

and project pointwise, $u(y) = Q(v_L(y))$. Compare an MLP layer $\sigma(Wh+b)$: the matrix $W$ acting on a vector becomes a pointwise linear map **plus an integral operator** acting on a function. Linear integral operators are how PDE solutions are written (Green's functions), so this is a natural building block, and a stack of such layers with nonlinearities is universal for continuous operators (Kovachki et al., 2023).

The layer is defined on functions. On data, the integral becomes a quadrature over whatever samples are available, $\int_D\kappa(x,x')v(x')\,dx'\approx\sum_j w_j\,\kappa(x,x_j)\,v(x_j)$, and the same parameters apply to any discretization. Each choice of kernel parameterization and quadrature gives a different neural operator.

Every layer produces a latent *field* $v_\ell(x)$, and the output at $y$ is read from $v_L(y)$ — which aggregated information from the whole input through $L$ integral layers. That is **local conditioning**: the decoder at $y$ sees $c(y) = v_L(y)$, a query-dependent summary of $a$.

### 6.2 Graph neural operators: learn the kernel, restrict its support

Parameterize $\kappa$ by a small MLP of the two points and the input values, and restrict it to a ball of radius $r$:

$$
v_{\ell+1}(x_i) = \sigma\Big(W v_\ell(x_i) + \frac{1}{|\mathcal N(i)|}\sum_{x_j\in\mathcal N(i)}\kappa_\phi\big(x_i,x_j,a(x_i),a(x_j)\big)\,v_\ell(x_j)\Big),\qquad \mathcal N(i)=\{x_j:\lVert x_j-x_i\rVert\le r\}.
$$

The sum is a Monte Carlo estimate of $\frac{1}{|B_r|}\int_{B_r(x_i)}\kappa\,v\,dx'$, and it is message passing on the graph connecting points within distance $r$ (Li et al., 2020). It works on any mesh or point cloud, with cost $O(n\bar k\,d_v^2)$ for $\bar k$ neighbors per point. Its weakness is the one of CNNs: interactions beyond radius $r$ require many layers.

### 6.3 Fourier neural operators: assume stationarity, compute in Fourier space

Assume the kernel is stationary, $\kappa(x,x')=\kappa(x-x')$. Then the integral is a convolution, and by the convolution theorem (Lecture 14 §1.2):

$$
\int_D\kappa(x-x')\,v(x')\,dx' = \mathcal F^{-1}\big(\hat\kappa\cdot\mathcal Fv\big)(x).
$$

The FNO (Li et al., 2021) parameterizes $\hat\kappa$ directly: one complex $d_v\times d_v$ matrix $R_\ell(k)$ for each of the lowest $k_{\max}$ Fourier modes per dimension, and zero above:

$$
v_{\ell+1}(x) = \sigma\Big(W_\ell v_\ell(x) + \mathcal F^{-1}\big[R_\ell(k)\,(\mathcal Fv_\ell)(k)\big]_{|k|\le k_{\max}}(x)\Big).
$$

- **Global receptive field in one layer:** the kernel spans the whole domain.
- **Cost:** $O(n\log n)$ with the FFT on a uniform grid.
- **Parameters:** about $k_{\max}^{D}d_v^2$ per layer in $D$ dimensions, independent of the grid.
- **Resolution transfer:** the weights are attached to physical frequencies, not to grid offsets. A CNN's stencil changes meaning when the grid is refined (Lecture 14 §7); the FNO's $R_\ell(k)$ does not, so the same weights apply on any grid with enough points to represent the $k_{\max}$ modes.

Two caveats. The FFT requires a regular grid (irregular geometries need a mapping to one). And pointwise nonlinearities create frequencies above $k_{\max}$ that a coarse grid cannot represent, so outputs can alias; "zero-shot super-resolution" claims should be tested, not assumed (Bartolucci et al., 2023).

### 6.4 Where DeepONet fits

A single kernel layer with a **low-rank** kernel, $\kappa(y,x) = \sum_{k=1}^{p}t_k(y)\,\beta_k(x)$, gives

$$
\int_D\kappa(y,x)\,a(x)\,dx = \sum_{k=1}^{p}t_k(y)\underbrace{\int_D\beta_k(x)\,a(x)\,dx}_{z_k(a)} ,
$$

which is a linear DeepONet: the integral against $\beta_k$ produces global coefficients, and $t_k(y)$ is the trunk. Low rank means the query sees the input only through $p$ projections — global conditioning, viewed as an integral operator.

| Model | Conditioning | Kernel | Input sampling | Cost per layer |
|---|---|---|---|---|
| PCA-Net | global, linear, fixed basis | — | fixed (PCA on a grid) | $O(np)$ |
| DeepONet | global, linear, learned basis | rank $p$ | fixed sensors | $O(mp)$ encode; $O(p)$ per query |
| NOMAD | global, nonlinear decoder | — | fixed sensors | as DeepONet |
| Graph NO | local | learned, compact support | any mesh | $O(n\bar k\,d_v^2)$ |
| FNO | local | stationary, global, low-pass | regular grid | $O(n\log n\,d_v+n\,d_v^2)$ |
| Transformer NO | local | data-dependent (attention) | any points | $O(n^2d)$, or $O(nMd)$ with $M$ latents |

> **Remember.** The kernel integral layer $v\mapsto\sigma(Wv+\int\kappa\,v+b)$ is an MLP layer on functions. Graph NOs learn $\kappa$ on a neighborhood (any mesh, local reach); FNOs assume stationarity and learn $\hat\kappa$ on low Fourier modes (global reach, regular grids, $O(n\log n)$); DeepONet is the rank-$p$, global special case.

---

## 7. Transformer neural operators (13 min)

### 7.1 Attention is a kernel integral layer

Lecture 17 §8.2 noted that one attention head on sampled functions computes

$$
\mathrm{Attn}(v)(x_i) = \sum_{j=1}^{n}\frac{e^{s(x_i,x_j)}}{\sum_{j'}e^{s(x_i,x_{j'})}}\,V v(x_j),\qquad s(x,x') = \frac{(W_Qv(x))^\top(W_Kv(x'))}{\sqrt{d_k}} .
$$

This is the kernel integral layer of §6.1 with a **data-dependent, normalized** kernel. Two observations make it precise.

1. *It is a ratio of quadratures.* With sample points drawn from a density $\rho$ on $D$,
$$
\mathrm{Attn}(v)(x)\;\xrightarrow{\,n\to\infty\,}\;\frac{\int_D e^{s(x,x')}\,Vv(x')\,\rho(x')\,dx'}{\int_D e^{s(x,x')}\,\rho(x')\,dx'} ,
$$
a well-defined operator on functions — so self-attention is discretization-invariant in the sense of §0.3, as long as the sampling density is fixed.
2. *Non-uniform sampling biases it, and the fix is one line.* If the samples cluster (a refined mesh near a boundary layer), the limit above weights the dense region more. To target the uniform measure instead, use quadrature weights $w_j$ (e.g. cell volumes): $\sum_j w_je^{s_{ij}}Vv_j/\sum_jw_je^{s_{ij}}$. Since $w_je^{s_{ij}} = e^{s_{ij}+\log w_j}$, this is ordinary softmax attention with $\log w_j$ added to the scores.

Compared with graph and Fourier kernels, attention's kernel is global like the FNO's, works on any point set like the GNO's, and is recomputed from the content of the input, so it can concentrate where the solution has structure. Its price is the $O(n^2)$ cost of Lecture 17.

### 7.2 The encoder: continuous self-attention

The input arrives as a set of samples $\{(x_j,a(x_j))\}_{j=1}^n$ — any number, any locations.

1. **Tokenize.** $h_j = P\,a(x_j) + \gamma(x_j)\,W_{\text{pos}}$. The positional encoding is a Fourier-feature embedding of the physical coordinate, so §3's spectral-bias analysis applies here too. On a regular grid, tokens can instead be patches (as in ViT), which reduces $n$.
2. **Mix.** Apply transformer blocks (Lecture 17 §3): continuous self-attention, as in §7.1, then per-token MLPs, with residuals and LayerNorm.
3. **Compress (optional).** To avoid $O(n^2)$, introduce $M\ll n$ learned latent tokens $Z_0\in\mathbb{R}^{M\times d}$ and **cross-attend** from them to the input tokens, $Z = Z_0+\mathrm{CrossAttn}(Z_0;\,H)$, then apply self-attention among the latents only. Cost $O(nMd+M^2d)$. This is the Perceiver idea (Jaegle et al., 2022), and it decouples model cost from input resolution.

The encoder's output is a set of tokens $Z=\{z_m\}_{m=1}^{M}$: either one per input sample, one per patch, or $M$ latents.

### 7.3 The decoder: cross-attention from continuous queries

For any query point $y\in D$:

$$
\begin{aligned}
e(y) &= \gamma(y)\,W_e && \text{embed the query coordinate}\\
c(y) &= \sum_{m=1}^{M}\operatorname{softmax}_m\!\Big(\frac{(e(y)W_Q)^\top(z_mW_K)}{\sqrt{d_k}}\Big)\,z_mW_V && \text{cross-attend to the encoded tokens}\\
\mathcal G_\theta(a)(y) &= g_\theta\big(e(y)+c(y)\big) && \text{small MLP head}
\end{aligned}
$$

(in practice multi-head, with one or two such layers). Read in the language of §§4–5:

- It is a **conditional neural field**: a function of $y$, evaluable anywhere, conditioned on the token set $Z$.
- The conditioning is **local**: each query computes its own weights over the tokens, so a point in a boundary layer can attend to the tokens describing that boundary layer. It is Bahdanau attention (Lecture 16 §1) with spatial queries instead of decoder states.
- **Queries do not attend to each other.** Each $y$ is decoded independently, so the output can be produced on any grid, at any resolution, at scattered points, or on a random subset of points during training, at cost $O(m\,M\,d)$ for $m$ queries.

If all $M$ tokens were averaged into one vector before decoding, this would collapse to global conditioning — the bottleneck returns. The decoder's ability to *select* among tokens is what removes it.

This encoder–decoder template covers several current models: attention-based encoders with cross-attention decoders (OFormer; Li et al., 2023), latent-token encoders with query decoders (Perceiver IO; Jaegle et al., 2022), models with heterogeneous and multiple inputs (GNOT; Hao et al., 2023), and vision-transformer encoders on gridded inputs combined with continuous cross-attention decoders (CViT; Wang et al., 2024). Linear-cost variants of the attention itself (Cao, 2021) and attention over learned "physical states" (Transolver; Wu et al., 2024) address the $n^2$ cost.

### 7.4 In pseudocode

```python
def encode(xs, a_vals, P):                     # xs: (n, d) locations, a_vals: (n, d_a)
    h = a_vals @ P.W_in + fourier_features(xs, P.B) @ P.W_pos     # (n, d) tokens
    z = P.latents                                                  # (M, d) learned
    z = z + cross_attention(q=z, kv=h, p=P.enc_x)                  # read the input: O(nM)
    for p in P.enc_blocks:                                         # mix latents: O(M^2)
        z = transformer_block(z, p)
    return z                                                       # (M, d) token set

def decode(ys, z, P):                          # ys: (m, d) ANY query points
    e = fourier_features(ys, P.B_q) @ P.W_q                        # (m, d)
    for p in P.dec_blocks:                                         # queries never see each other
        e = e + cross_attention(q=layer_norm(e), kv=z, p=p)       # local conditioning
        e = e + mlp(layer_norm(e), p)
    return e @ P.W_out                                             # (m, d_u) = G(a)(ys)

def loss(batch, P):                            # random query subsets each step
    z = encode(batch.xs, batch.a, P)
    pred = decode(batch.ys, z, P)
    return relative_l2(pred, batch.u_at_ys)
```

`cross_attention` and `transformer_block` are the functions of Lecture 17 §7, with queries and keys taken from different sets.

> **Remember.** Self-attention on sampled functions is a normalized kernel integral, discretization-invariant for a fixed sampling density; add $\log w_j$ to the scores for non-uniform samples. A transformer NO encodes the input samples (optionally into $M$ latent tokens) and decodes with cross-attention from Fourier-embedded query coordinates: a conditional neural field with local conditioning, evaluable anywhere.

---

## 8. Training, evaluation, summary (5 min)

**Training.** The loss of §0.3 is usually the relative $L^2$ error, $\frac1N\sum_i\lVert u_i-\mathcal G_\theta(a_i)\rVert/\lVert u_i\rVert$, so that every example counts equally regardless of its amplitude. For models with neural-field decoders, sample a random subset of query points for each example at each step: the loss stays unbiased and memory drops from the full field to the subset. Normalize inputs and outputs (Lecture 13 §3); for multiscale outputs, tune the Fourier-feature bandwidth on validation data (§3.4).

**Evaluation.** Apply Lecture 13's checklist, with three additions specific to operators.

1. **Baselines.** Compare against a simple model trained on the same data — a linear or PCA-based map, or a DeepONet — and against the cost of simply running the solver.
2. **Resolution.** If you claim discretization invariance, test at resolutions you did not train on, and report how the error changes, not just one number.
3. **Distribution shift.** Report error on inputs from the training distribution and on inputs outside it (different coefficients, geometries, longer horizons). Neural operators interpolate well within their training family; outside it, they offer no guarantee.

**Summary.** A neural field represents a function by a network over coordinates. Under gradient flow its residual decays mode by mode at rates set by the neural tangent kernel, which for standard MLPs favors low frequencies; Fourier features make the kernel stationary with a tunable bandwidth. A conditional neural field $f_\theta(y;z)$ represents a family of functions, and an operator is an encoder $a\mapsto z$ followed by such a field. With global conditioning — one code for every query — we recover PCA-Net, DeepONet and NOMAD, and their bottleneck. With local conditioning — a query-dependent aggregate of the input — we get the kernel integral layer: graph NOs (learned kernel on neighborhoods), FNOs (stationary kernel in Fourier space) and transformer NOs (data-dependent kernel by attention), the last decoding by cross-attention from continuous queries.

---

## Code Checkpoint 19

JAX permitted for gradients and Jacobians; models, training loop and optimizer are yours. Each item asks you to report a number.

**A. Spectral bias and the NTK.**
1. Fit $u(x)=\sin(2\pi x)+\tfrac12\sin(20\pi x)$ on 256 points in $[0,1]$ with a 3-layer `tanh` MLP of width 128, (i) on raw $x$ and (ii) on Fourier features with $\sigma\in\{1,10,100\}$. Every 100 steps, compute the residual's Fourier coefficients at frequencies 1 and 10. **Report the number of steps** for each to fall below $10^{-2}$ of its initial value, or "not reached" after $10^4$ steps.
2. At initialization, compute the empirical NTK $\Theta = JJ^\top$ on the 256 points with `jax.jacrev`, for cases (i) and (ii, $\sigma=10$). **Report the five largest eigenvalues**, and plot the leading eigenvectors. In two sentences, relate what you see to §3.3–3.4.

**B. Global versus local conditioning.** Generate data for 1D linear advection $u_t+cu_x=0$ on a periodic domain: inputs are random smooth or step-like initial conditions $a=u(\cdot,0)$, outputs $u(\cdot,T)$ (exact by shifting). Train (i) a DeepONet with $p\in\{16,64\}$ and Fourier-feature trunk, and (ii) a cross-attention decoder (§7.3) on input tokens, with similar parameter counts. **Report test relative $L^2$ error** (mean $\pm$ std over 3 seeds) for smooth and for step-like inputs, and explain the gap with §4.3.

**C. Resolution.** Train model B(ii) on inputs sampled at 128 points and queries at random subsets of 128 points. **Report test error** with inputs at 64, 128 and 512 points and outputs at 1024 points. Repeat with inputs sampled non-uniformly (density $\propto1+0.9\sin2\pi x$), with and without the $\log w_j$ correction of §7.1, and **report both errors**.

---

## Quiz-eligible facts

1. Operator learning approximates $\mathcal G:a\mapsto u$ between function spaces; the loss is an integral over the output domain, estimated without bias by sampling query points.
2. A neural field represents a function as $x\mapsto\mathrm{MLP}_\theta(x)$: evaluable at any point, fitted from any samples, differentiable in $x$ by AD.
3. Spectral bias: networks trained by gradient descent fit low frequencies before high ones.
4. Under gradient flow, the training residual obeys $\dot r = -\Theta_t r$ with $\Theta_t(x,x')=\nabla_\theta f(x)^\top\nabla_\theta f(x')$; in the infinite-width limit $\Theta$ is constant and each eigencomponent decays like $e^{-\lambda_it}$.
5. For standard MLPs on uniformly spread inputs, NTK eigenfunctions are Fourier modes (spherical harmonics) with eigenvalues decaying in frequency, e.g. $\lambda_k\propto k^{-2}$ for two-layer ReLU nets on the circle.
6. Fourier features $\gamma(x)$ satisfy $\gamma(x)^\top\gamma(x') = \sum_r\cos(2\pi b_r^\top(x-x'))$, making the NTK approximately stationary; their scale $\sigma$ sets its bandwidth — too small blurs, too large overfits.
7. A conditional neural field $f_\theta(y;z)$ represents many functions with one network; codes come from auto-decoding (per-sample optimization) or an encoder.
8. Linear conditioning $\sum_kz_kt_k(y)$ is limited by the Kolmogorov $p$-width of the output set; nonlinear decoders are not.
9. Global conditioning passes one code $z$ to every query (PCA-Net, DeepONet, NOMAD): a bottleneck. Local conditioning gives each query its own aggregate $c(y)=\sum_j\kappa(y,x_j)h_j$ (graph NOs, FNO, transformer NOs).
10. The neural operator layer is $v_{\ell+1}(x)=\sigma(W_\ell v_\ell(x)+\int\kappa_\ell(x,x')v_\ell(x')dx'+b_\ell)$, discretized by quadrature.
11. FNO: stationary kernel, parameterized by complex matrices on the lowest $k_{\max}$ Fourier modes; global, $O(n\log n)$, regular grids, parameters independent of resolution.
12. Graph NO: learned kernel restricted to a radius-$r$ neighborhood; any mesh; local reach per layer.
13. DeepONet is a rank-$p$ kernel integral: $\sum_k t_k(y)\int\beta_k(x)a(x)dx$.
14. Softmax attention over samples converges to a ratio of integrals under the sampling density; adding $\log w_j$ to the scores applies quadrature weights.
15. A transformer NO decoder cross-attends from Fourier-embedded query coordinates to encoded tokens; queries are independent, so outputs can be evaluated anywhere.

---

## Practice problems

**P1.** In the NTK regime, a two-layer ReLU network on the circle has $\lambda_k\propto k^{-2}$. If the frequency-1 component of the residual falls by a factor $e$ in 100 steps, roughly how many steps does the frequency-8 component need?

*Answer.* The time constant scales as $1/\lambda_k\propto k^2$, so $64\times100 = 6400$ steps.

**P2.** Show that $\gamma(x)^\top\gamma(x')$ depends only on $x-x'$ for $\gamma(x) = [\cos(2\pi b^\top x),\ \sin(2\pi b^\top x)]$. Why does this matter for spectral bias?

*Answer.* $\cos\alpha\cos\beta+\sin\alpha\sin\beta = \cos(\alpha-\beta) = \cos(2\pi b^\top(x-x'))$. The induced kernel is stationary, so it is diagonalized by Fourier modes with eigenvalues given by its spectrum, which the frequencies $b$ control. Choosing $b$ with large enough scale gives high-frequency modes large eigenvalues, so gradient descent no longer starves them.

**P3.** A DeepONet with $p=32$ fits diffusion problems well but performs poorly on advection of a sharp front with a random initial position. Explain, and propose two fixes.

*Answer.* The outputs are shifted copies of a sharp profile, a set with slowly decaying Kolmogorov $p$-width: no 32-dimensional linear space approximates all shifts well, so any linear decoder — DeepONet included — has a large error floor. Fixes: a nonlinear decoder (NOMAD-style conditioning of the field on $z$), or local conditioning (a cross-attention decoder, FNO, or graph NO), which lets each query read the relevant part of the input.

**P4.** A 2D FNO layer keeps $12\times12$ Fourier modes with $d_v=32$ channels. How many real parameters does its spectral weight have, and how does this change if the grid goes from $64^2$ to $256^2$?

*Answer.* $144\times32\times32 = 147{,}456$ complex entries, $294{,}912$ real parameters (ignoring the conjugate-symmetric half retained in practice). It does not change with the grid: the weights are attached to frequencies, not grid points.

**P5.** Input samples are twice as dense on the left half of the domain as on the right. Show that adding $\log w_j$ to the attention scores, with $w_j$ proportional to the local cell size, turns softmax attention into a weighted quadrature, and say what goes wrong without it.

*Answer.* $\operatorname{softmax}_j(s_{ij}+\log w_j) = w_je^{s_{ij}}/\sum_{j'}w_{j'}e^{s_{ij'}}$, which is $\int e^{s}Vv\,dx'/\int e^{s}dx'$ discretized with weights $w_j$. Without it, the limit integrates against the sampling density, so the left half counts twice as much as the right: the operator learned depends on the mesh.

---

## References

- Bartolucci, F., de Bézenac, E., Raonić, B., Molinaro, R., Mishra, S., & Alaifari, R. (2023). Representation equivalent neural operators: A framework for alias-free operator learning. *NeurIPS*.
- Basri, R., Jacobs, D., Kasten, Y., & Kritchman, S. (2019). The convergence rate of neural networks for learned functions of different frequencies. *NeurIPS*.
- Bhattacharya, K., Hosseini, B., Kovachki, N. B., & Stuart, A. M. (2021). Model reduction and neural networks for parametric PDEs. *SMAI J. Comput. Math.*, 7, 121–157.
- Bietti, A., & Mairal, J. (2019). On the inductive bias of neural tangent kernels. *NeurIPS*.
- Cao, S. (2021). Choose a transformer: Fourier or Galerkin. *NeurIPS*.
- Chen, T., & Chen, H. (1995). Universal approximation to nonlinear operators by neural networks with arbitrary activation functions and its application to dynamical systems. *IEEE Trans. Neural Networks*, 6(4), 911–917.
- Dupont, E., Kim, H., Eslami, S. M. A., Rezende, D., & Rosenbaum, D. (2022). From data to functa: Your data point is a function and you can treat it like one. *ICML*.
- Ha, D., Dai, A., & Le, Q. V. (2017). HyperNetworks. *ICLR*.
- Hao, Z., et al. (2023). GNOT: A general neural operator transformer for operator learning. *ICML*.
- Jacot, A., Gabriel, F., & Hongler, C. (2018). Neural tangent kernel: Convergence and generalization in neural networks. *NeurIPS*.
- Jaegle, A., et al. (2022). Perceiver IO: A general architecture for structured inputs & outputs. *ICLR*.
- Kissas, G., Seidman, J. H., Guilhoto, L. F., Preciado, V. M., Pappas, G. J., & Perdikaris, P. (2022). Learning operators with coupled attention. *JMLR*, 23(215), 1–63.
- Kovachki, N., Li, Z., Liu, B., Azizzadenesheli, K., Bhattacharya, K., Stuart, A., & Anandkumar, A. (2023). Neural operator: Learning maps between function spaces. *JMLR*, 24(89), 1–97.
- Lanthaler, S., Mishra, S., & Karniadakis, G. E. (2022). Error estimates for DeepONets: A deep learning framework in infinite dimensions. *Trans. Math. Appl.*, 6(1).
- Lee, J., Xiao, L., Schoenholz, S. S., Bahri, Y., Novak, R., Sohl-Dickstein, J., & Pennington, J. (2019). Wide neural networks of any depth evolve as linear models under gradient descent. *NeurIPS*.
- Li, Z., Kovachki, N., Azizzadenesheli, K., Liu, B., Bhattacharya, K., Stuart, A., & Anandkumar, A. (2020). Neural operator: Graph kernel network for partial differential equations. arXiv:2003.03485.
- Li, Z., Kovachki, N., Azizzadenesheli, K., Liu, B., Bhattacharya, K., Stuart, A., & Anandkumar, A. (2021). Fourier neural operator for parametric partial differential equations. *ICLR*.
- Li, Z., Meidani, K., & Barati Farimani, A. (2023). Transformer for partial differential equations' operator learning. *TMLR*.
- Lu, L., Jin, P., Pang, G., Zhang, Z., & Karniadakis, G. E. (2021). Learning nonlinear operators via DeepONet based on the universal approximation theorem of operators. *Nature Machine Intelligence*, 3, 218–229.
- Mildenhall, B., Srinivasan, P. P., Tancik, M., Barron, J. T., Ramamoorthi, R., & Ng, R. (2020). NeRF: Representing scenes as neural radiance fields for view synthesis. *ECCV*.
- Park, J. J., Florence, P., Straub, J., Newcombe, R., & Lovegrove, S. (2019). DeepSDF: Learning continuous signed distance functions for shape representation. *CVPR*.
- Perez, E., Strub, F., de Vries, H., Dumoulin, V., & Courville, A. (2018). FiLM: Visual reasoning with a general conditioning layer. *AAAI*.
- Rahaman, N., et al. (2019). On the spectral bias of neural networks. *ICML*.
- Seidman, J. H., Kissas, G., Perdikaris, P., & Pappas, G. J. (2022). NOMAD: Nonlinear manifold decoders for operator learning. *NeurIPS*.
- Sitzmann, V., Martel, J., Bergman, A., Lindell, D., & Wetzstein, G. (2020). Implicit neural representations with periodic activation functions. *NeurIPS*.
- Tancik, M., et al. (2020). Fourier features let networks learn high frequency functions in low dimensional domains. *NeurIPS*.
- Wang, S., Seidman, J. H., Sankaran, S., Wang, H., Pappas, G. J., & Perdikaris, P. (2024). CViT: Continuous vision transformer for operator learning. arXiv:2405.13998.
- Wang, S., Wang, H., & Perdikaris, P. (2021). On the eigenvector bias of Fourier feature networks: From regression to solving multi-scale PDEs with physics-informed neural networks. *Comput. Methods Appl. Mech. Eng.*, 384, 113938.
- Wang, S., Yu, X., & Perdikaris, P. (2022). When and why PINNs fail to train: A neural tangent kernel perspective. *J. Comput. Phys.*, 449, 110768.
- Wu, H., Luo, H., Wang, H., Wang, J., & Long, M. (2024). Transolver: A fast transformer solver for PDEs on general geometries. *ICML*.
- Xie, Y., et al. (2022). Neural fields in visual computing and beyond. *Computer Graphics Forum (Eurographics STAR)*, 41(2), 641–676.
