# Lecture 13 — Training Deep Networks: Automatic Differentiation, Scale, Optimizers, Regularization

**ENM 5310 — Data-driven Modeling and Probabilistic Scientific Computing**

**Builds on:** Lectures 4–5 (MLE, MAP, ridge as a Gaussian prior, the SVD), Lecture 6 (GD on a quadratic in the Hessian eigenbasis, stability threshold, standardization and conditioning, SGD noise floor, momentum, Adam as a diagonal preconditioner), Lecture 12 (the MLP and its likelihoods).

**Reading:** Murphy I §13.3–13.5. Glorot & Bengio (2010) — all eight pages. Baydin et al. (2018), *Automatic differentiation in machine learning: a survey*, §§1–3. Loshchilov & Hutter (2019), §§1–2. Karpathy (2019), *A Recipe for Training Neural Networks*.

**The point of this lecture.** Lecture 12 showed that an MLP *can* represent what we need. Whether it *will* depends on four things: computing gradients (§§1–2), keeping signals at the right scale (§§3–4), configuring the optimizer (§5), and controlling overfitting (§§6–7). Each section ends with what you should remember.

---

### Notation

| Symbol | Meaning |
|---|---|
| $z^{(\ell)}=W^{(\ell)}h^{(\ell-1)}+b^{(\ell)}$, $h^{(\ell)}=\sigma(z^{(\ell)})$ | pre-activation, activation; $h^{(0)}=x$, $f=z^{(L)}$ |
| $W^{(\ell)}\in\mathbb{R}^{H_\ell\times H_{\ell-1}}$; $n_{\text{in}}=H_{\ell-1}$, $n_{\text{out}}=H_\ell$ | weights of layer $\ell$; fan-in, fan-out |
| $\ell(\theta)$, $\mathcal{L}(\theta)=\frac1N\sum_n\ell_n$ | per-example loss $-\log p(y\mid x,\theta)$; training loss |
| $J_f(x)\in\mathbb{R}^{m\times n}$ | Jacobian of $f:\mathbb{R}^n\to\mathbb{R}^m$ |
| $\dot x$ (tangent), $\bar x$ (cotangent) | forward-mode and reverse-mode derivative carriers |
| $\delta^{(\ell)}=\partial\ell/\partial z^{(\ell)}$ | backpropagated error at layer $\ell$ |
| $\eta$, $\lambda$ | learning rate, weight decay |

**Learning objectives.** After this lecture you should be able to:

1. Define JVP, VJP and grad; state the cost of each and of a full Jacobian; explain why training uses reverse mode.
2. Derive backpropagation for an MLP as a chain of VJPs, and explain why gradients vanish or explode with depth.
3. Derive the Glorot and He initialization variances, and the shrinkage factor of the old "standard" initialization.
4. Explain why L2 regularization and weight decay differ under Adam, and write the warmup-plus-cosine schedule.
5. Identify weight decay as MAP, early stopping as approximate ridge with $\alpha\approx1/(\eta k)$, dropout as noise injection (ridge for linear models), and augmentation as an invariance assumption.
6. Train a network as an experiment: baseline, sanity checks, multiple seeds, reading the training curve.

---


## 2. Backpropagation through an MLP (8 min)

Backpropagation is reverse-mode AD applied to the MLP, using the two VJP rules above. Let $\delta^{(\ell)}=\partial\ell/\partial z^{(\ell)}$.

1. **Output.** $\delta^{(L)}=\partial\ell/\partial f$ depends only on the likelihood: $\operatorname{softmax}(f)-e_y$ for softmax cross-entropy, $(f-y)/\sigma^2$ for a Gaussian. These are the "$\mu-y$" residuals of Lecture 5.
2. **Recursion.** VJP through $W^{(\ell+1)}$, then through $\sigma$:
$$
\delta^{(\ell)} = \sigma'(z^{(\ell)})\odot\big(W^{(\ell+1)\top}\delta^{(\ell+1)}\big).
$$
3. **Parameter gradients.** $\dfrac{\partial\ell}{\partial W^{(\ell)}}=\delta^{(\ell)}h^{(\ell-1)\top}$, $\ \dfrac{\partial\ell}{\partial b^{(\ell)}}=\delta^{(\ell)}$: error arriving at the output times activation arriving at the input.

**Why depth is dangerous.** Unrolling step 2 with $D^{(\ell)}=\operatorname{diag}(\sigma'(z^{(\ell)}))$,

$$
\delta^{(\ell)} = D^{(\ell)}W^{(\ell+1)\top}D^{(\ell+1)}W^{(\ell+2)\top}\cdots D^{(L-1)}W^{(L)\top}\,\delta^{(L)} .
$$

A product of $L-\ell$ matrices behaves like a product of scalars: unless the typical factor is near one, the gradient shrinks or grows **exponentially in depth**. The logistic sigmoid has $\sigma'\le1/4$, so across 10 layers the first-layer gradient can be smaller by $\sim4^{-9}\approx4\times10^{-6}$ — the vanishing-gradient problem that stalled the second wave. The forward pass has the same structure, so activations can vanish or explode too. §§3–4 choose scales that keep both products near one.

---

## 2. Automatic differentiation: JVP, VJP, grad (20 min)

**Three ways to get a derivative.** *Symbolic* differentiation produces exact expressions that grow uncontrollably for long programs. *Finite differences* need one extra evaluation per input — $10^6$ evaluations for $10^6$ parameters — and trade truncation error against roundoff. *Automatic differentiation* (AD) gives derivatives exact to machine precision at a cost that is a small multiple of evaluating the function. It does this by applying the chain rule to the program, one primitive operation at a time.

**A program is a composition.** Write $f = f_K\circ\cdots\circ f_1$ with $x_0=x$ and $x_k = f_k(x_{k-1})$. By the chain rule,

$$
J_f(x) = J_K\,J_{K-1}\cdots J_1, \qquad J_k = \frac{\partial f_k}{\partial x_{k-1}}.
$$

AD never forms $J_f$. It only multiplies by it, from one side or the other.

**Forward mode: the Jacobian–vector product (JVP).** Given a tangent direction $v\in\mathbb{R}^n$, compute $J_f v = J_K(\cdots(J_2(J_1v)))$, right to left. Carry a tangent $\dot x_k$ alongside each value:

$$
x_k = f_k(x_{k-1}), \qquad \dot x_k = J_k\,\dot x_{k-1}, \qquad \dot x_0 = v.
$$

One forward pass gives $f(x)$ and the directional derivative $J_fv$, at a cost of a small constant times evaluating $f$, with no extra memory. A full Jacobian takes $n$ passes, one per input direction.

**Reverse mode: the vector–Jacobian product (VJP).** Given a cotangent $u\in\mathbb{R}^m$, compute $u^\top J_f = ((u^\top J_K)J_{K-1})\cdots J_1$, left to right. Run the program forward, **storing every $x_k$**, then sweep backward:

$$
\bar x_K = u, \qquad \bar x_{k-1} = J_k^\top\,\bar x_k .
$$

One forward-plus-backward pass gives $u^\top J_f$, again at a small constant times the cost of $f$, but with memory proportional to the number of stored intermediates. A full Jacobian takes $m$ passes, one per output.

**grad is one VJP.** A training loss is a scalar, $\mathcal L:\mathbb{R}^n\to\mathbb{R}$, so $m=1$ and

$$
\nabla\mathcal L(\theta)^\top = 1\cdot J_{\mathcal L}(\theta).
$$

**One reverse pass gives the entire gradient, whatever the number of parameters.** Forward mode would need $n$ passes. This asymmetry is why every deep learning framework is built on reverse mode, and why training is limited by activation memory rather than arithmetic.

| | Computes | Passes for full Jacobian | Memory | Best when |
|---|---|---|---|---|
| Forward (JVP) | $J_fv$ | $n$ (inputs) | constant | few inputs, many outputs |
| Reverse (VJP) | $u^\top J_f$ | $m$ (outputs) | stores intermediates | many inputs, few outputs — e.g. a loss |

**Worked example.** $y = x_1x_2+\sin x_1$ at $(x_1,x_2)=(2,3)$, with intermediates $v_1=x_1x_2$, $v_2=\sin x_1$, $y=v_1+v_2$.

- *Forward*, $\dot x=(1,0)$: $\dot v_1=\dot x_1x_2+x_1\dot x_2=3$, $\dot v_2=\cos(x_1)\dot x_1=\cos2$, $\dot y=3+\cos2$. That is $\partial y/\partial x_1$ only; $\partial y/\partial x_2$ needs a second pass.
- *Reverse*, $\bar y=1$: $\bar v_1=\bar v_2=1$; then $\bar x_1=\bar v_1x_2+\bar v_2\cos x_1=3+\cos2$ and $\bar x_2=\bar v_1x_1=2$. Both partials in one pass. Note that $\bar x_1$ *accumulates* contributions from every path through which $x_1$ influences $y$.

**Each primitive needs only its local rule.** A framework knows the JVP and VJP of each primitive and composes them. The two that make up an MLP:

| Primitive | JVP | VJP |
|---|---|---|
| $y=Wx$ | $\dot y=W\dot x+\dot W x$ | $\bar x=W^\top\bar y$, $\ \bar W=\bar y\,x^\top$ |
| $y=\sigma(x)$, elementwise | $\dot y=\sigma'(x)\odot\dot x$ | $\bar x=\sigma'(x)\odot\bar y$ |

**In code** (JAX; PyTorch's `torch.func` has the same three functions):

```python
y, Jv        = jax.jvp(f, (x,), (v,))     # forward mode
y, f_vjp     = jax.vjp(f, x); (uJ,) = f_vjp(u)   # reverse mode
g            = jax.grad(loss)(theta)       # = VJP with u = 1
Hv = jax.jvp(jax.grad(loss), (theta,), (v,))[1]  # Hessian-vector product
```

The last line composes the two modes — forward over reverse — to get $\nabla^2\mathcal L\,v$ for a few times the cost of a gradient, without forming the Hessian. That is what makes Newton-CG (Lecture 6) feasible for large models. Forward mode also matters in scientific ML whenever you differentiate a network with respect to its few *input* coordinates, as in a PDE residual.

> **Remember.** JVP pushes a tangent forward; VJP pulls a cotangent back. A gradient is one VJP, so it costs a small multiple of one function evaluation regardless of the number of parameters — paid for in memory.

---


## 3. Data normalization (5 min)

Standardize every input feature with its **training-set** mean and standard deviation, and apply the same transformation to validation and test data. For regression, standardize the targets too and invert at prediction time.

- **Conditioning.** The first-layer gradient is $\delta^{(1)}x^\top$, so the first layer sees its inputs exactly as linear regression does: mismatched scales mean a large condition number (Lecture 6).
- **Centering.** If every $x_j>0$, all components of a unit's weight gradient $\delta_ix_j$ share the sign of $\delta_i$, so the weights can only move together, and the optimizer zig-zags. The same argument condemns always-positive hidden activations such as the logistic sigmoid.
- **Signal scale.** The initialization analysis of §4 assumes unit-variance inputs. A feature with standard deviation 1000 saturates `tanh` units at once, and $\tanh'(10)\approx8\times10^{-9}$.

For physical data, nondimensionalize first, then standardize; log-transform strongly skewed positive quantities.

**Normalizing inside the network.** Input normalization and initialization fix the scale at step 0 only. As the weights train, hidden activations drift, and the analysis of §4 no longer holds. Two architectural ingredients keep deep networks trainable throughout: normalization layers fix the *forward* scale at every step, and residual connections fix the *backward* product of §2.

**Layer normalization** (Ba et al., 2016) re-standardizes each example's hidden vector $h\in\mathbb{R}^H$:

$$
\operatorname{LN}(h)=\gamma\odot\frac{h-\mu}{\sqrt{s^2+\epsilon}}+\beta,\qquad
\mu=\frac1H\sum_{j=1}^H h_j,\quad s^2=\frac1H\sum_{j=1}^H(h_j-\mu)^2 .
$$

The statistics are taken across the units of one example, so LN behaves identically at training and test time and for any batch size. The learned gain $\gamma$ and shift $\beta$ let the network undo the normalization where that helps. 

**Batch normalization** (Ioffe & Szegedy, 2015) takes the same statistics across the examples of a minibatch instead. It works well for convolutional networks with large batches, but it couples the examples in a batch, degrades at small batch sizes, and must switch to stored running averages at test time — a common source of train/test discrepancies. LayerNorm is the standard choice in transformers and most scientific ML models.

**Residual connections** (He et al., 2016) attack the vanishing gradient directly. Instead of $h^{(\ell)}=F_\ell(h^{(\ell-1)})$, a residual block computes

$$
h^{(\ell)} = h^{(\ell-1)} + F_\ell\big(h^{(\ell-1)}\big),
$$

so each layer learns a *correction* to its input rather than a whole new representation. Now look at backpropagation. In §2 the error reaching layer $\ell$ was a product of Jacobians $D^{(k)}W^{(k+1)\top}$, and any product of many factors shrinks or grows exponentially. A residual block has Jacobian $I+J_{F_\ell}$, so its VJP is

$$
\bar h^{(\ell-1)} = \bar h^{(\ell)} + J_{F_\ell}^\top\,\bar h^{(\ell)},
$$

and unrolling over the remaining layers gives

$$
\bar h^{(\ell)} = \prod_{k=\ell+1}^{L}\big(I+J_{F_k}^\top\big)\,\bar h^{(L)}
= \bar h^{(L)} + \sum_{k}J_{F_k}^\top\bar h^{(L)} + \cdots .
$$

The first term is the output error delivered to layer $\ell$ *unchanged*, through the chain of identity paths. However deep the network and however small the $J_{F_k}$, that term cannot vanish multiplicatively: the gradient has a highway around every block. If the blocks start small — near-zero $F_k$ at initialization — every factor is close to $I$ and the product neither vanishes nor explodes. This is why residual networks train at depths of hundreds of layers where plain MLPs of the kind in §2 fail at a few dozen. The one constraint is that $F_\ell$ must preserve the dimension, so the identity can be added.

**The two combined.** Modern deep architectures, transformers included, stack blocks of the form

$$
h^{(\ell)} = h^{(\ell-1)} + F_\ell\big(\operatorname{LN}(h^{(\ell-1)})\big),
$$

with the normalization inside the branch and the identity path left untouched. LN keeps each block's input at unit scale; the residual keeps the gradient path open. For the shallow MLPs of this lecture both are optional. For anything deep, this block is the default.

> **Remember.** Initialization fixes the scale at step 0. LayerNorm restores unit scale in the forward pass at every step. Residual connections replace the backprop product $\prod D\,W^\top$ by $\prod(I+J_F^\top)$, which always contains an identity path, so gradients reach the early layers intact.

---

## 4. Initialization (17 min)

**Not zero.** Units that start identical receive identical gradients and stay identical forever. Weights must be random; biases can start at zero.

**With what variance?** In a width-$n$ linear network with weight variance $v$, each layer multiplies the signal variance by $nv$, so $L$ layers multiply it by $(nv)^L$. Only $nv\approx1$ is stable. Make this precise.

*Assumptions at initialization:* zero biases; i.i.d. zero-mean weights with variance $v_\ell$, independent of the layer input.

**Forward.**

1. $z^{(\ell)}_i=\sum_{j=1}^{n_{\text{in}}}W_{ij}h_j$ is a sum of independent zero-mean terms, each with variance $\mathbb{E}[W_{ij}^2]\mathbb{E}[h_j^2]=v_\ell\,\mathbb{E}[h_j^2]$.
2. So $\operatorname{Var}[z^{(\ell)}]=n_{\text{in}}\,v_\ell\,\mathbb{E}[(h^{(\ell-1)})^2]$.
3. For `tanh` near zero, $h\approx z$, so $\operatorname{Var}[z^{(\ell)}]\approx n_{\text{in}}v_\ell\operatorname{Var}[z^{(\ell-1)}]$.
4. Preserving variance requires $v_\ell=1/n_{\text{in}}$ (LeCun).

**Backward.** By step 2 of §2 with $\sigma'\approx1$, $\operatorname{Var}[\delta^{(\ell-1)}]\approx n_{\text{out}}v_\ell\operatorname{Var}[\delta^{(\ell)}]$, which requires $v_\ell=1/n_{\text{out}}$.

**Glorot (Xavier).** The two conditions conflict unless $n_{\text{in}}=n_{\text{out}}$. Glorot and Bengio (2010) compromise:

$$
v_\ell=\frac{2}{n_{\text{in}}+n_{\text{out}}},\qquad
W_{ij}\sim\mathcal U\Big[-\sqrt{\tfrac{6}{n_{\text{in}}+n_{\text{out}}}},\ \sqrt{\tfrac{6}{n_{\text{in}}+n_{\text{out}}}}\Big]
$$

(a uniform $\mathcal U[-a,a]$ has variance $a^2/3$).

**What they observed.** The earlier "standard" initialization $\mathcal U[-1/\sqrt{n_{\text{in}}},1/\sqrt{n_{\text{in}}}]$ has variance $1/(3n_{\text{in}})$, so $nv=1/3$: the signal variance falls by a factor of 3 per layer, and in their `tanh` networks the layers saturated one after another. With the logistic sigmoid, which is not zero-centered and has $\sigma'(0)=1/4$, the top hidden layer saturated at once and stayed stuck. With normalized initialization, activation and gradient statistics stayed similar across layers and training was faster. They also found cross-entropy far better behaved than squared error for classification — the Lecture 5 lesson again.

**He (Kaiming), for ReLU.** ReLU zeroes half its input, so for symmetric $z$, $\mathbb{E}[\operatorname{ReLU}(z)^2]=\tfrac12\operatorname{Var}[z]$. Step 3 picks up a factor $\tfrac12$, and preservation requires

$$
v_\ell=\frac{2}{n_{\text{in}}}.
$$

He et al. (2015) showed a 30-layer ReLU network failed to train with Glorot's variance and trained with this one.

**Check your framework's default.** Keras uses Glorot, Flax uses LeCun ($1/n_{\text{in}}$), and PyTorch's `nn.Linear` uses $\mathcal U[\pm1/\sqrt{n_{\text{in}}}]$ — the "standard" initialization above. In architectures with residual connections and normalization layers this rarely matters; in a plain deep MLP it can.

> **Remember.** Variance $1/n_{\text{in}}$ preserves the forward signal for `tanh`, $2/n_{\text{in}}$ for ReLU; Glorot averages the forward and backward conditions. Initialization fixes the scale only at step 0 — residual connections and normalization layers, coming later, keep it fixed during training.

---

## 5. Optimizers (12 min)

Everything in Lecture 6 still applies: the stability threshold explains divergence, Adam's diagonal preconditioner handles the very different gradient scales of different layers, and the SGD noise floor is why schedules end with a decay. What is lost is convexity. In practice, large networks usually reach low training loss; the obstacles are plateaus and bad scaling more often than bad local minima. That is an empirical observation, not a theorem.

**Tune the learning rate first.** Sweep it on a log grid ($10^{-4}$ to $10^{-2}$) before touching anything else. The best value is usually just below where training goes unstable.

**AdamW, not Adam with L2.**

1. Adding $\frac\lambda2\lVert\theta\rVert^2$ to the loss adds $\lambda\theta$ to the gradient.
2. Under SGD, $\theta\leftarrow(1-\eta\lambda)\theta-\eta g$: every weight shrinks by the same factor. L2 regularization *is* weight decay.
3. Under Adam, $\lambda\theta$ is divided by $\sqrt{\hat v}$, so weights with large gradient history are decayed less and others more: no longer uniform, no longer a fixed Gaussian prior.
4. AdamW applies the decay outside the preconditioner:
$$
\theta\leftarrow\theta-\eta_t\Big(\frac{\hat m}{\sqrt{\hat v}+\epsilon}+\lambda\theta\Big).
$$

**Schedule.** Linear warmup over $T_w$ steps (a few percent of $T$), then cosine decay:

$$
\eta_t=\eta_{\max}\frac{t}{T_w}\ \ (t\le T_w),\qquad
\eta_t=\tfrac12\eta_{\max}\Big(1+\cos\frac{\pi(t-T_w)}{T-T_w}\Big)\ \ (t>T_w).
$$

The decay has a reason (Lecture 6's noise floor). Warmup is a well-tested heuristic: early updates rely on Adam statistics averaged over very few steps, and can push the network into badly scaled regions.

**Gradient clipping.** $g\leftarrow g\min(1,c/\lVert g\rVert)$ keeps one bad batch from undoing many steps.

**Starting point** for small MLPs — to be tuned, not trusted:

| Choice | Default |
|---|---|
| Optimizer | AdamW, $\beta=(0.9,0.999)$ |
| Peak learning rate | sweep $\{10^{-4},3\times10^{-4},10^{-3},3\times10^{-3}\}$ |
| Schedule | ~5% warmup, cosine to zero |
| Weight decay | sweep $\{0,10^{-4},10^{-3},10^{-2}\}$ |
| Batch size | 32–256 |
| Initialization | He (ReLU), Glorot (`tanh`), zero biases |

---

## 6. Regularization as priors (18 min)

**Seeing overfitting** requires three splits. *Training* data compute gradients; *validation* data make every decision (learning rate, decay, architecture, when to stop); *test* data are used once, to report. A test set that influenced any decision is a second validation set, and its number is optimistic. The symptom is a widening gap between training and validation loss.

Each standard regularizer has a probabilistic reading.

**Weight decay is MAP.** With prior $\theta\sim\mathcal N(0,\tau^2I)$, the negative log posterior divided by $N$ is $\mathcal L(\theta)+\frac\lambda2\lVert\theta\rVert^2$ with $\lambda=1/(N\tau^2)$ — Lecture 5's ridge derivation, which never used linearity. Exact under SGD; only approximate under AdamW (§5). Biases are usually not decayed.

**Early stopping is approximately ridge.** Least squares with Hessian $A=Q\Lambda Q^\top$ and minimizer $\theta^\star$; gradient descent from $\theta_0=0$.

1. $\theta_k-\theta^\star=(I-\eta A)^k(0-\theta^\star)$, so in the eigenbasis $\tilde\theta_{k,i}=\big[1-(1-\eta\lambda_i)^k\big]\tilde\theta^\star_i\approx\big(1-e^{-k\eta\lambda_i}\big)\tilde\theta^\star_i$.
2. Ridge with penalty $\frac\alpha2\lVert\theta\rVert^2$ gives $\tilde\theta_{\alpha,i}=\dfrac{\lambda_i}{\lambda_i+\alpha}\tilde\theta^\star_i$.
3. Both filters are $\approx1$ for large $\lambda_i$. For small $\lambda_i$ they are $k\eta\lambda_i$ and $\lambda_i/\alpha$, which match when
$$
\alpha\approx\frac{1}{\eta k}.
$$

Training time is an inverse regularization strength. Low-curvature directions — those the data determine worst (Lecture 4) — are fitted last, and choosing the stopping step on validation data is choosing $\alpha$.

**Dropout is noise injection.** During training, zero each hidden unit with probability $p$ and rescale the survivors: $\tilde h_j=m_jh_j/(1-p)$, $m_j\sim\operatorname{Ber}(1-p)$. At test time, switch it off. For linear regression with dropout on the inputs, $\operatorname{Var}[\tilde x_{nj}]=\frac{p}{1-p}x_{nj}^2$, and averaging the squared error over masks gives

$$
\mathbb{E}_m\lVert y-\tilde Xw\rVert^2=\lVert y-Xw\rVert^2+\frac{p}{1-p}\sum_jw_j^2\lVert x_{:,j}\rVert^2,
$$

a ridge penalty weighted by feature energy. In deep networks it penalizes reliance on any single unit. Typical $p\in[0.1,0.5]$; most useful when data are scarce.

**Data augmentation encodes an invariance.** Adding transformed copies of the training data tells the model the label is unchanged by the transformation. The invariance must actually hold: a rotated "6" is a "9," and rotating a velocity field requires rotating its vectors too. An augmentation that breaks the true symmetry is label noise, not regularization.

> **Remember.** Weight decay = Gaussian prior; early stopping ≈ ridge with $\alpha\approx1/(\eta k)$; dropout = multiplicative noise (ridge for linear models); augmentation = an invariance assumption. Decide everything on validation data; touch the test set once.

---

## 7. Overparameterization and implicit bias (4 min)

Networks often have more parameters than data and still generalize. Part of the reason is that **the optimizer chooses among the many solutions that fit the data**. The simplest case: least squares with $\Phi\in\mathbb{R}^{N\times p}$, $p>N$, run from $\theta_0=0$.

1. The gradient $\Phi^\top(\Phi\theta-y)$ lies in the row space of $\Phi$, so every iterate does too.
2. Every interpolant is $\theta_{\text{mn}}+n$ with $n$ in the null space, orthogonal to the row space, so the only interpolant in the row space is the minimum-norm one, $\Phi^+y$.

Gradient descent returns the minimum-norm interpolant without being asked to; the optimizer and the initialization are part of the model. This also explains **double descent**: test error peaks near $p\approx N$, where the smallest singular value of $\Phi$ approaches zero and noise is amplified by $1/s_{\min}$ (Lecture 4). Past that point it falls again, and a small ridge penalty removes the peak.

In practice: if a large network overfits, try more data, augmentation, decay and early stopping before shrinking it. With only a few hundred samples, compare against a GP — you may be in the classical regime.

---

## 8. Training checklist (6 min)

**Before training.**

1. Fit a baseline on the same inputs — logistic regression for classification (Coding Assignment 1), a GP for small regression problems.
2. Check the loss at step 0: about $\log C$ for $C$-class cross-entropy ($2.30$ for $C=10$), order one for standardized regression targets. Anything else is a scaling or label bug.
3. Overfit one batch of 32 examples to near-zero loss. If you can't, there is a bug.

**During and after training.**

4. Plot training and validation loss together; change one thing at a time.
5. Run at least 5 seeds and report mean $\pm$ std. Differences smaller than the seed spread are not results.

| Symptom | First suspect |
|---|---|
| NaN or inf early | learning rate too high; $\log0$ (compute cross-entropy from logits) |
| Initial loss far from $\log C$ | initialization or input scale |
| Flat from the start | learning rate too low; saturated or dead units; a gradient bug |
| Training ↓, validation ↑ | overfitting: stop earlier, decay, augment |
| Both high and close | underfitting: bigger model, longer, retune the learning rate |

---

## Code Checkpoint 13

Implement in NumPy from the equations above. JAX may be used **only** to check item A.

**A. Autodiff by hand.**
1. Write JVP and VJP functions for the two primitives of §1 (linear map, elementwise `tanh`), and compose them into `mlp_jvp(params, x, v)` and `mlp_vjp(params, x, u)`.
2. Check against central finite differences and `jax.jvp`/`jax.vjp` in float64. **Report the maximum relative error.**
3. Verify the identity $u^\top(J v) = (u^\top J)v$ for random $u, v$.
4. For a network $\mathbb{R}^{100}\to\mathbb{R}$, time forming the full gradient with 100 JVPs versus one VJP. **Report the ratio.**

**B. Signal propagation.** In a 20-layer, width-256 MLP with standard-normal inputs, plot the standard deviation of $z^{(\ell)}$ and $\delta^{(\ell)}$ against $\ell$ for {standard, Glorot, He} × {`tanh`, ReLU}. **Report the fitted per-layer factor** where §4 predicts geometric change, and compare with $\sqrt{nv}$ (`tanh`) or $\sqrt{nv/2}$ (ReLU).

**C. MLP against its baseline.** On the Coding Assignment 1 dataset, train with your own AdamW and warmup-plus-cosine schedule, keeping the best-validation checkpoint. Compare 0, 1 and 2 hidden layers of width 256, each with a learning-rate sweep. **Report test accuracy as mean $\pm$ std over 5 seeds**, and run one ablation — input standardization off, or standard instead of He initialization.

---

## Quiz-eligible facts

1. A JVP costs one forward pass and gives $J_fv$; a VJP costs one forward-plus-backward pass and gives $u^\top J_f$. A full Jacobian needs $n$ JVPs or $m$ VJPs.
2. The gradient of a scalar loss is one VJP with $u=1$, so it costs a small multiple of one function evaluation regardless of the number of parameters; reverse mode pays for this in memory.
3. Forward-over-reverse gives Hessian–vector products without forming the Hessian.
4. Backprop: $\delta^{(\ell)}=\sigma'(z^{(\ell)})\odot W^{(\ell+1)\top}\delta^{(\ell+1)}$ and $\partial\ell/\partial W^{(\ell)}=\delta^{(\ell)}h^{(\ell-1)\top}$; gradient size scales exponentially with depth unless each factor is near one.
5. Inputs are standardized with training statistics only; all-positive inputs force a unit's weight gradients to share a sign.
6. Initialization variances: $1/n_{\text{in}}$ (forward, `tanh`), $2/(n_{\text{in}}+n_{\text{out}})$ (Glorot), $2/n_{\text{in}}$ (He, ReLU). The standard $\mathcal U[\pm1/\sqrt{n_{\text{in}}}]$ has variance $1/(3n_{\text{in}})$ and shrinks the signal variance about 3× per layer.
7. Under SGD, L2 regularization equals weight decay and is MAP with $\tau^2=1/(N\lambda)$; under Adam it does not, and AdamW decouples the decay.
8. Early stopping on a quadratic is approximately ridge with $\alpha\approx1/(\eta k)$.
9. Input dropout in linear regression is, in expectation, ridge with penalty $\frac{p}{1-p}\sum_jw_j^2\lVert x_{:,j}\rVert^2$; dropout is off at test time.
10. Gradient descent from zero on underdetermined least squares converges to the minimum-norm interpolant.
11. The initial cross-entropy loss for $C$ classes should be close to $\log C$.

---

## Practice problems

**P1.** $f:\mathbb{R}^{10^6}\to\mathbb{R}$ is a training loss and $g:\mathbb{R}\to\mathbb{R}^{10^6}$ maps a time to a discretized field. Which AD mode gives each full Jacobian in one pass?

*Answer.* Reverse mode for $f$ (one output: one VJP gives the gradient). Forward mode for $g$ (one input: one JVP gives the whole Jacobian column).

**P2.** Reverse mode on $y=x_1x_2+\sin x_1$ at $(1,0)$: give $\bar x_1$ and $\bar x_2$.

*Answer.* $\bar x_1=x_2+\cos x_1=\cos1\approx0.54$; $\bar x_2=x_1=1$.

**P3.** A 10-class classifier starts at loss 24.7. What should it be, and what do you check?

*Answer.* About $\log10\approx2.30$. The logits are far too large: check the input normalization and the initialization scale.

**P4.** A plain 10-layer `tanh` MLP uses $\mathcal U[\pm1/\sqrt{n_{\text{in}}}]$. In the linear regime, by what factor does the output variance shrink relative to the input?

*Answer.* $nv=1/3$ per layer, so $3^{-10}\approx1.7\times10^{-5}$. Use variance $1/n_{\text{in}}$ or Glorot.

**P5.** Gradient descent with $\eta=0.01$ is stopped after $k=1000$ steps. What ridge penalty does this approximate? Is a Hessian direction with $\lambda_i=10^{-3}$ fitted?

*Answer.* $\alpha\approx1/(\eta k)=0.1$. Since $\lambda_i\ll\alpha$, it is barely fitted: the filter is about $k\eta\lambda_i=0.01$.

---

## References

- Baydin, A. G., Pearlmutter, B. A., Radul, A. A., & Siskind, J. M. (2018). Automatic differentiation in machine learning: a survey. *JMLR*, 18(153), 1–43.
- Belkin, M., Hsu, D., Ma, S., & Mandal, S. (2019). Reconciling modern machine-learning practice and the classical bias–variance trade-off. *PNAS*, 116(32), 15849–15854.
- Glorot, X., & Bengio, Y. (2010). Understanding the difficulty of training deep feedforward neural networks. *AISTATS*.
- He, K., Zhang, X., Ren, S., & Sun, J. (2015). Delving deep into rectifiers. *ICCV*.
- Karpathy, A. (2019). A recipe for training neural networks. karpathy.github.io.
- LeCun, Y., Bottou, L., Orr, G. B., & Müller, K.-R. (1998). Efficient backprop. In *Neural Networks: Tricks of the Trade*. Springer.
- Loshchilov, I., & Hutter, F. (2019). Decoupled weight decay regularization. *ICLR*.
- Murphy, K. P. (2022). *Probabilistic Machine Learning: An Introduction*. MIT Press. Ch. 13.
- Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R. (2014). Dropout. *JMLR*, 15, 1929–1958.
- Wager, S., Wang, S., & Liang, P. (2013). Dropout training as adaptive regularization. *NeurIPS*.
