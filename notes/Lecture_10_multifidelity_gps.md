# Lecture 10 — Multi-output and Multi-fidelity Gaussian Processes

**ENM 5310 — Data-driven Modeling and Probabilistic Scientific Computing**

**Builds on:** Lecture 8 (kernels, the GP model), Lecture 9 (training by marginal likelihood, inference by conditioning, kernel composition).

**Reading:** Álvarez, Rosasco & Lawrence (2012), *Kernels for vector-valued functions: a review*, §§1–4 — multi-output GPs and coregionalization. Kennedy & O'Hagan (2000) — the autoregressive multi-fidelity model. Perdikaris et al. (2017) — the nonlinear generalization. Background: Murphy I §17.2, Rasmussen & Williams Ch. 2.

**The idea in one sentence.** Everything in Lectures 8 and 9 goes through unchanged if we enlarge the input space to include a label saying *which* function we are talking about; multi-fidelity modeling is then the special case where the labels are ordered by cost and accuracy, and the cross-covariance between them has a particular structure.

**Learning objectives.** After this lecture you should be able to:

1. Write a multi-output GP as an ordinary GP on the augmented input space $\mathcal{X}\times\{1,\dots,T\}$, and explain why every formula from Lecture 9 then applies unchanged.
2. Write down the intrinsic coregionalization model, state what the coregionalization matrix means, and say what it cannot express.
3. State the autoregressive multi-fidelity model and derive its three covariance functions.
4. Exhibit that model as a linear model of coregionalization with two latent processes.
5. Assemble the block covariance matrix for data observed at two fidelities, and write the predictive distribution at the high fidelity.
6. Explain the recursive training scheme, what it costs, and what a nested design buys.
7. Diagnose from fitted hyperparameters whether the low-fidelity data is helping, and recognize when a nonlinear model is needed.

### Notation

$T$ fidelity levels, indexed $t = 1,\dots,T$ from cheapest to most accurate; for two levels write $L$ (low) and $H$ (high). Training inputs $X_t$ with $N_t$ points, targets $y_t$. Kernels $k_L$, $k_\delta$; scaling factor $\rho$; noise variances $\sigma_t^2$. The full data vector is the stack $y = (y_1^\top,\dots,y_T^\top)^\top$ of length $N = \sum_t N_t$.

---

## 1. Why multi-fidelity: the budget problem

The situation is completely standard in engineering. You have two ways to evaluate the same quantity:

| | Cost per run | Accuracy | Runs you can afford |
|---|---|---|---|
| Low fidelity | minutes | biased, but captures the trends | thousands |
| High fidelity | hours or days | what you actually want | tens |

A coarse mesh versus a fine one; RANS versus LES; a two-dimensional idealization versus the full three-dimensional geometry; a surrogate physics model versus an experiment. The pattern is always the same: **$N_L \gg N_H$, and it is $f_H$ you care about.**

Two obvious strategies, both bad. Fit a GP to the high-fidelity data alone and you have a handful of points in a possibly multi-dimensional space — Lecture 9 §3.3 tells you exactly what happens, namely a posterior that reverts to the prior almost everywhere. Fit a GP to the low-fidelity data alone and you get a beautiful, confident surrogate for the wrong function.

The multi-fidelity idea is to model both functions **jointly**, so that the cheap data constrains the expensive function through whatever correlation exists between them. The engineering intuition that makes it work:

> You do not have to learn the expensive function from scratch. Learn its shape cheaply, then use the few expensive runs to learn only the **correction**.

If the correction is simpler than the function — smoother, smaller, slowly varying — then a handful of high-fidelity points is enough to pin it down. That "if" is a modeling assumption, it is testable, and §8 shows how to test it.

---

## 2. Multi-output GPs: a GP on an augmented input space

Before specializing to fidelities, do the general case: $T$ related functions $f_1,\dots,f_T$ on the same input space, modeled jointly.

**The move that makes this easy.** Instead of thinking of a vector-valued function $\mathbf{f}(x)\in\mathbb{R}^T$, think of a **scalar** function on a bigger input space:

$$\tilde f:\ \mathcal{X}\times\{1,\dots,T\}\ \to\ \mathbb{R}, \qquad \tilde f(x,t) \triangleq f_t(x).$$

An input is now a pair: *where*, and *which function*. A GP on this augmented space needs a kernel taking two such pairs,

$$\boxed{\ k\big((x,t),(x',t')\big) = \operatorname{Cov}\big[f_t(x),\,f_{t'}(x')\big],\ }$$

which must be positive semi-definite in the usual sense — every Gram matrix built from any finite set of (input, label) pairs must be a valid covariance matrix. Nothing else changes.

**Why this is worth stating explicitly.** Every formula from Lecture 9 now applies verbatim, with no new derivations:

- **Training** is still $\log p(y\mid\theta) = -\frac12y^\top C^{-1}y-\frac12\log|C|-\frac N2\log2\pi$, where $y$ stacks all observations from all outputs and $C$ is built from the augmented kernel plus per-output noise.
- **Inference** is still conditioning: to predict $f_t(x^\star)$, form $k^\star$ with entries $k((x_n,t_n),(x^\star,t))$ and apply $\mu^\star = k^{\star\top}C^{-1}y$, $v^\star = k^{\star\star}-k^{\star\top}C^{-1}k^\star$.
- **Computation** is still one Cholesky factorization, now of an $N\times N$ matrix with $N = \sum_t N_t$.

The only new modeling question is what the cross-covariances $k((x,t),(x',t'))$ for $t\ne t'$ should be. That is the whole subject.

**Block structure.** Order the stacked data by output. Then $C$ has $T\times T$ blocks, the $(t,t')$ block being the $N_t\times N_{t'}$ matrix of cross-covariances between output $t$ at its inputs and output $t'$ at its inputs:

$$C \;=\; \begin{pmatrix} K_{11}(X_1,X_1)+\sigma_1^2 I & \cdots & K_{1T}(X_1,X_T)\\ \vdots & \ddots & \vdots \\ K_{T1}(X_T,X_1) & \cdots & K_{TT}(X_T,X_T)+\sigma_T^2 I\end{pmatrix}.$$

The diagonal blocks are ordinary single-output GPs. **The off-diagonal blocks are the entire point**: they are what lets an observation of output $t'$ change the posterior for output $t$. Set them to zero and the whole thing decouples into $T$ independent GPs, which is exactly the "fit them separately" strategy §1 rejected.

Note the inputs need not be shared: output $t$ is observed at $X_t$, and the sets may be different sizes and disjoint. This matters immediately, because low- and high-fidelity runs are rarely at the same design points.

---

## 3. Separable covariance, and its limitation

The simplest valid construction takes one ordinary kernel $k$ on $\mathcal{X}$ and one $T\times T$ matrix $B$ describing how the outputs relate, and multiplies them:

> **Intrinsic coregionalization model (ICM).**
> $$k\big((x,t),(x',t')\big) = B_{tt'}\;k(x,x'), \qquad B\succeq0 .$$
> $B$ is the **coregionalization matrix**: $B_{tt}$ is the variance of output $t$, and $B_{tt'}$ its covariance with output $t'$.

Validity is easy to see. Write $B = WW^\top$ (always possible for $B\succeq0$), with $w_t$ the $t$-th row of $W$. Then $B_{tt'} = w_t^\top w_{t'}$, and the model is exactly

$$f_t(x) = \sum_{q=1}^{Q} W_{tq}\,u_q(x), \qquad u_q\sim\mathcal{GP}(0,k)\ \text{ independent}.$$

**Every output is a different linear combination of the same $Q$ shared latent functions.** That is where the correlation comes from, and it is why $B$ must be positive semi-definite: it is a Gram matrix of the mixing weights.

Allowing each latent process its own kernel gives the general form:

> **Linear model of coregionalization (LMC).**
> $$k\big((x,t),(x',t')\big) = \sum_{q=1}^{Q} B^{(q)}_{tt'}\;k_q(x,x'), \qquad B^{(q)} = w^{(q)}\big(w^{(q)}\big)^\top .$$

**What ICM cannot do, and why it matters here.** With a single shared $k$, all outputs have the same length scale and the same smoothness, and the correlation between outputs is the same constant everywhere in input space. For multi-fidelity that is too rigid: a high-fidelity simulation typically has *more* fine structure than its coarse counterpart, and the two often agree well in some regions and poorly in others. The next section's model is an LMC with $Q = 2$ and a specific choice of weights, designed precisely to relax this.

---

## 4. Step 1 — Model: the autoregressive multi-fidelity GP

### 4.1 The model

Kennedy and O'Hagan's proposal — also called **co-kriging** — is one line:

$$\boxed{\ f_H(x) \;=\; \rho\,f_L(x)\;+\;\delta(x), \qquad f_L\sim\mathcal{GP}(0,k_L),\quad \delta\sim\mathcal{GP}(0,k_\delta),\quad f_L\perp\delta .\ }$$

In words: **the expensive function is a rescaled copy of the cheap one, plus an independent correction.** The scalar $\rho$ absorbs a systematic difference in magnitude; the **discrepancy** $\delta$ absorbs everything else, and carries its own kernel with its own amplitude and length scale.

Observations at the two levels are noisy:

$$y_L = f_L(X_L)+\epsilon_L,\quad \epsilon_L\sim\mathcal{N}(0,\sigma_L^2 I), \qquad y_H = f_H(X_H)+\epsilon_H,\quad \epsilon_H\sim\mathcal{N}(0,\sigma_H^2 I).$$

For a deterministic simulator, take the corresponding $\sigma_t^2\to0$ (Lecture 9 §4.4) and expect to need jitter.

### 4.2 The three covariance functions

Everything follows from bilinearity of covariance and the independence of $f_L$ and $\delta$:

$$\operatorname{Cov}\big[f_L(x),f_L(x')\big] = k_L(x,x'),$$
$$\operatorname{Cov}\big[f_H(x),f_L(x')\big] = \operatorname{Cov}\big[\rho f_L(x)+\delta(x),\,f_L(x')\big] = \rho\,k_L(x,x'),$$
$$\operatorname{Cov}\big[f_H(x),f_H(x')\big] = \rho^2 k_L(x,x') + k_\delta(x,x').$$

Three readings worth pausing on. The cross-covariance is $\rho k_L$, so **$\rho$ is the entire channel through which the cheap data speaks about the expensive function** — at $\rho = 0$ the off-diagonal blocks vanish and the two models decouple. The high-fidelity kernel $\rho^2k_L+k_\delta$ is a *sum*, so by Lecture 9 §4.1 it inherits structure from both: the coarse shape from $k_L$ and any additional fine structure from $k_\delta$. And the ICM restriction of §3 is gone, since the two terms may have different length scales.

### 4.3 It is an LMC with two latent processes

Collect the covariances above. With the label set $\{L,H\}$,

$$k\big((x,t),(x',t')\big) = a_t\,a_{t'}\,k_L(x,x') \;+\; b_t\,b_{t'}\,k_\delta(x,x'), \qquad a = \begin{pmatrix}1\\\rho\end{pmatrix},\quad b = \begin{pmatrix}0\\1\end{pmatrix},$$

which is exactly the LMC of §3 with $Q = 2$, latent processes $u_1 = f_L$ and $u_2 = \delta$, and coregionalization matrices

$$B^{(1)} = aa^\top = \begin{pmatrix}1&\rho\\ \rho&\rho^2\end{pmatrix}, \qquad B^{(2)} = bb^\top = \begin{pmatrix}0&0\\0&1\end{pmatrix}.$$

So the multi-fidelity model is **a multi-output GP whose coregionalization structure has been specialized in two ways**: the first latent process is shared with weights $(1,\rho)$, and the second is exclusive to the high-fidelity output. The rank-one $B^{(1)}$ is the assumption that one shared latent function explains all the commonality; the triangular pattern of $B^{(2)}$ is the assumption that fidelities are *ordered* — the correction belongs to the expensive level only. Neither is forced on you by the mathematics; both encode real beliefs about the physics.

### 4.4 More than two levels

For $T$ ordered fidelities, apply the same relation recursively:

$$f_t(x) = \rho_{t-1}f_{t-1}(x) + \delta_t(x), \qquad t = 2,\dots,T,$$

with all $\delta_t$ independent of each other and of $f_1$. Each level has its own scaling and its own discrepancy kernel. The **Markov property** implicit here is worth naming: given $f_{t-1}(x)$, no other value $f_{t-1}(x')$ tells you anything more about $f_t(x)$. All the information passes through the same input location. Section 5 turns that into a training algorithm.

---

## 5. Step 2 — Training

### 5.1 The joint approach

The hyperparameters are $\theta = (\theta_L,\ \theta_\delta,\ \rho,\ \sigma_L^2,\ \sigma_H^2)$. Stack the data and build the block covariance from §4.2:

$$y = \begin{pmatrix}y_L\\y_H\end{pmatrix}, \qquad C_\theta = \begin{pmatrix} k_L(X_L,X_L)+\sigma_L^2I & \rho\,k_L(X_L,X_H)\\[2pt] \rho\,k_L(X_H,X_L) & \rho^2k_L(X_H,X_H)+k_\delta(X_H,X_H)+\sigma_H^2I \end{pmatrix}.$$

Then train exactly as in Lecture 9 §2 — nothing is new:

$$\hat\theta = \arg\max_\theta\ \left[-\tfrac12y^\top C_\theta^{-1}y-\tfrac12\log|C_\theta|-\tfrac N2\log2\pi\right],$$

by L-BFGS on the log-hyperparameters, with $\rho$ left unconstrained since it may be negative. Store the Cholesky factor $L$ of $C_{\hat\theta}$ and $\alpha = C_{\hat\theta}^{-1}y$.

**The cost is the problem.** $C$ is $(N_L+N_H)\times(N_L+N_H)$, so the factorization is $O\big((N_L+N_H)^3\big)$ — and $N_L$ is large by construction, which is the whole reason the low-fidelity model was attractive. The joint fit also couples all the hyperparameters into one non-convex optimization, making the local optima of Lecture 9 §2.7 considerably worse.

### 5.2 The recursive approach

The Markov structure of §4.4 suggests fitting one level at a time.

> ### Algorithm: recursive multi-fidelity training (two levels)
>
> 1. **Fit the low fidelity alone.** Train an ordinary GP on $(X_L,y_L)$ for $(\theta_L,\sigma_L^2)$, exactly as in Lecture 9. This gives a posterior $f_L\mid y_L\sim\mathcal{GP}(\mu_L,v_L)$.
> 2. **Fit the discrepancy.** Train an ordinary GP on the residuals
> $$\big(X_H,\ \ y_H-\rho\,\mu_L(X_H)\big)$$
> for $(\theta_\delta,\rho,\sigma_H^2)$, treating $\rho$ as one more hyperparameter of that second fit.

Two fits of ordinary GPs, at a cost of $O(N_L^3)+O(N_H^3)$ instead of $O\big((N_L+N_H)^3\big)$, with two small optimizations instead of one large one. Le Gratiet and Garnier showed that when the designs are **nested**, $X_H\subseteq X_L$, the recursive scheme reproduces the joint model exactly rather than approximating it.

This is a strong argument for choosing nested designs when you control the experiment: evaluate the expensive model at a subset of the points where you evaluated the cheap one. The result is cheaper to fit, better conditioned as an optimization problem, and identical in what it predicts.

*One caution.* Step 2 uses the posterior *mean* $\mu_L(X_H)$ as if it were the truth. Doing this properly requires propagating $v_L$ into the second stage; with a nested design and dense low-fidelity data the difference is small, which is why the simple version is used in practice.

---

## 6. Step 3 — Inference, and what the cheap data bought you

To predict the high-fidelity function at $x^\star$, condition as always. The cross-covariance vector between $f_H(x^\star)$ and the stacked data follows from §4.2:

$$k^\star = \begin{pmatrix}\rho\,k_L(X_L,x^\star)\\[2pt] \rho^2k_L(X_H,x^\star)+k_\delta(X_H,x^\star)\end{pmatrix}, \qquad k^{\star\star} = \rho^2k_L(x^\star,x^\star)+k_\delta(x^\star,x^\star),$$

and then, with $L$ and $\alpha$ from training,

$$\boxed{\ \mu_H^\star = k^{\star\top}\alpha, \qquad v_H^\star = k^{\star\star}-\big\|L^{-1}k^\star\big\|^2 .\ }$$

Identical in form to Lecture 9 §3. Only the kernel changed.

**Where the gain comes from.** Look at the top block of $k^\star$: it is $\rho\,k_L(X_L,x^\star)$, nonzero wherever low-fidelity data sits near $x^\star$. Every cheap run therefore reduces $v_H^\star$, in proportion to $\rho$. The high-fidelity posterior gets sharper in regions you only explored cheaply.

But the reduction is not unlimited, and the formula says why. Even with infinitely dense low-fidelity data, $f_L$ becomes known exactly and the remaining uncertainty about $f_H$ is the uncertainty about $\delta$:

$$v_H^\star \ \longrightarrow\ \operatorname{Var}\big[\delta(x^\star)\mid y_H\big] \qquad\text{as } N_L\to\infty .$$

**That is the entire economics of multi-fidelity modeling.** Cheap data buys you $f_L$; it cannot buy you $\delta$. Expensive data is what learns the discrepancy, and the number of expensive runs you need is set by how complicated $\delta$ is — its amplitude and its length scale — not by how complicated $f_H$ is. When the coarse model captures the physics and errs smoothly, $\delta$ is simple and a few high-fidelity runs suffice. When the coarse model is wrong in an intricate way, $\delta$ is as hard as $f_H$ and you have gained nothing.

---

## 7. When the linear relation fails

The model assumes the two fidelities are related by one scalar $\rho$, uniformly across the input space. Real pairs of models often are not: a coarse mesh may track the fine one well at low Reynolds number and poorly at high, or the relation between them may simply be curved.

The generalization replaces the linear map by a learned nonlinear one:

$$f_t(x) = g_t\big(x,\ f_{t-1}(x)\big), \qquad g_t\sim\mathcal{GP}(0,k_t),$$

where $g_t$ takes both the input and the lower-fidelity value as arguments. The standard kernel for $g_t$ is a composition built with the rules of Lecture 9 §4.1,

$$k_t\big((x,f),(x',f')\big) = k_{t,x}(x,x')\cdot k_{t,f}(f,f') \;+\; k_{t,\delta}(x,x'),$$

a product term expressing "close in input **and** in low-fidelity value" plus an additive discrepancy. Setting $k_{t,f}$ to a linear kernel recovers the autoregressive model, so this is a strict generalization.

**The price.** Composing a GP with a GP is no longer a GP: the low-fidelity input to $g_t$ is itself uncertain, so the predictive distribution at a test point is not Gaussian and must be obtained by sampling — draw $f_{t-1}(x^\star)$ from its posterior, push each draw through $g_t$, and accumulate. Exact conditioning is lost; everything else about the workflow survives.

---

## 8. Design, diagnostics, and where this goes

**Check the assumption before trusting the model.** If you have evaluated both fidelities at some common inputs, plot $y_H$ against $y_L$ at those points. A roughly linear scatter supports the autoregressive model; a visibly curved or fanning scatter says the relation is nonlinear and §7 is needed. This takes one line of code and settles a modeling question that would otherwise be argued about.

**Read the fitted hyperparameters.** They are interpretable, as always:

- $\hat\rho\approx0$ — the fidelities are uncorrelated and the cheap data is buying nothing. Check whether the models really are related, and whether the relation might be nonlinear.
- $\hat\rho<0$ — legitimate, meaning the coarse model errs in the opposite direction. Do not constrain $\rho$ to be positive.
- **Discrepancy amplitude comparable to the amplitude of $f_H$** — the correction is as large as the function, so the low-fidelity model is contributing little. The multi-fidelity apparatus is not helping.
- **Discrepancy length scale much shorter than $\ell_L$** — the correction has fine structure, so §6's argument fails and you will need many high-fidelity points.

**Design.** Prefer nested designs, $X_H\subseteq X_L$, both for §5.2's exactness and because it gives you the shared points the scatter plot needs. Allocating a budget between levels is a real optimization: with per-run costs $c_L\ll c_H$, you want to keep adding cheap runs while they still reduce $v_H^\star$ per unit cost more than an expensive run would. That criterion — variance reduction per unit cost — is the bridge to the next lecture.

**Where this goes.** Bayesian optimization and active learning choose the next evaluation using $\mu^\star$ and $v^\star$. In the multi-fidelity setting the choice is richer: not only *where* to evaluate but *at which fidelity*, trading a small reduction in uncertainty now for a cheap price against a large reduction for an expensive one.

| | Multi-fidelity GP regression |
|---|---|
| **Step 1 — Model** | $f_H = \rho f_L+\delta$, with $f_L\sim\mathcal{GP}(0,k_L)$ and $\delta\sim\mathcal{GP}(0,k_\delta)$ independent; an LMC with two latent processes |
| **Step 2 — Training** | Maximize the marginal likelihood of the stacked data under the block covariance — jointly, or recursively one level at a time |
| **Step 3 — Inference** | Condition as in Lecture 9, with the block $k^\star$; cheap data reduces $v_H^\star$ through $\rho$, down to the uncertainty in $\delta$ |

---

## Checkpoint F (take-home)

`numpy` only, plus `scipy.optimize`. Use the one-dimensional benchmark pair
$$f_H(x) = (6x-2)^2\sin(12x-4), \qquad f_L(x) = 0.5\,f_H(x)+10(x-0.5)-5, \qquad x\in[0,1],$$
which is linearly related by construction — a fair test of the autoregressive model.

1. **Baselines.** Fit an ordinary GP to $N_H = 5$ high-fidelity points alone, and another to $N_L = 50$ low-fidelity points alone. Report the RMSE of each against $f_H$ on a dense grid, and plot both with $\pm2\sigma$ bands. Explain in one sentence what each gets wrong.
2. **The joint model.** Implement the block covariance of §5.1 with nested designs ($X_H\subseteq X_L$), train by maximizing the marginal likelihood, and report $\hat\rho$, both length scales, and the discrepancy amplitude. Compare $\hat\rho$ against the true value of $0.5$ implied by the construction.
3. **The payoff.** Report the RMSE of the multi-fidelity posterior mean against $f_H$, and plot it beside the two baselines. Then sweep $N_H\in\{3,5,10,20\}$ with $N_L = 50$ fixed, and plot RMSE against $N_H$ for the high-fidelity-only GP and for the multi-fidelity GP. How many high-fidelity points does the single-fidelity model need to match the multi-fidelity model at $N_H = 5$?
4. **Where the uncertainty goes.** For fixed $N_H = 5$, sweep $N_L\in\{10,25,50,200\}$ and report the average $v_H^\star$ over the grid. Does it approach zero? Explain what it approaches, using §6.
5. **Breaking it.** Replace the low-fidelity function by a nonlinear transformation of the high-fidelity one, for instance $f_L(x) = \tanh\!\big(f_H(x)/10\big)$ rescaled. Plot $y_H$ against $y_L$ at the shared points, refit the autoregressive model, and report $\hat\rho$ and the discrepancy amplitude. State in two sentences what the diagnostics of §8 are telling you.

---

## Quiz-eligible facts

1. A multi-output GP is an ordinary GP on the augmented input space $\mathcal{X}\times\{1,\dots,T\}$, with kernel $k((x,t),(x',t')) = \operatorname{Cov}[f_t(x),f_{t'}(x')]$; training, inference, and computation are unchanged from the single-output case.
2. The stacked covariance has $T\times T$ blocks; the off-diagonal blocks are what let data on one output inform another. Zero off-diagonals means $T$ independent GPs.
3. Outputs need not be observed at the same inputs; the blocks are simply rectangular.
4. ICM: $k((x,t),(x',t')) = B_{tt'}k(x,x')$ with $B\succeq0$. Writing $B = WW^\top$ shows every output is a linear combination of shared latent GPs, which is why $B$ must be positive semi-definite.
5. ICM forces all outputs to share a length scale and smoothness, with correlation constant across input space. The LMC relaxes this by summing several separable terms with different kernels.
6. The autoregressive multi-fidelity model is $f_H = \rho f_L+\delta$ with $f_L\perp\delta$, giving $\operatorname{Cov}[f_L,f_L'] = k_L$, $\operatorname{Cov}[f_H,f_L'] = \rho k_L$, and $\operatorname{Cov}[f_H,f_H'] = \rho^2k_L+k_\delta$.
7. It is an LMC with two latent processes, weights $a = (1,\rho)$ and $b = (0,1)$, so $B^{(1)} = aa^\top$ is rank one and $B^{(2)} = bb^\top$ assigns the discrepancy to the high fidelity alone.
8. $\rho$ is the only channel through which low-fidelity data informs the high-fidelity posterior; at $\rho = 0$ the model decouples. Negative $\rho$ is legitimate.
9. Joint training costs $O((N_L+N_H)^3)$. The recursive scheme — fit $f_L$, then fit a GP to the residuals $y_H-\rho\mu_L(X_H)$ — costs $O(N_L^3)+O(N_H^3)$ and is exact for nested designs $X_H\subseteq X_L$.
10. As $N_L\to\infty$, the high-fidelity predictive variance tends to the uncertainty in $\delta$, not to zero: cheap data buys $f_L$, expensive data buys $\delta$.
11. The number of high-fidelity runs needed is set by the complexity of the discrepancy, not of $f_H$.
12. Diagnostics: $\hat\rho\approx0$, a discrepancy amplitude comparable to that of $f_H$, or a discrepancy length scale much shorter than $\ell_L$ all mean the low-fidelity data is not helping. A curved scatter of $y_H$ against $y_L$ at shared inputs means the linear model is wrong.
13. The nonlinear generalization takes $f_t(x) = g_t(x,f_{t-1}(x))$ with $g_t$ a GP on the augmented input; the composition is no longer a GP, so predictions require sampling.

---

## Practice problems

*Five questions in the style of the quizzes: each should take two or three minutes, with no computer. The answer follows each one — cover it and try first. Derivations and implementation are in Checkpoint F.*

**P1.** In what sense is the multi-fidelity model a multi-output GP? Give the two coregionalization matrices and say what each one assumes.

*Answer.* Its kernel is $a_ta_{t'}k_L+b_tb_{t'}k_\delta$ with $a = (1,\rho)$, $b = (0,1)$ — an LMC with two latent processes. $B^{(1)} = aa^\top = \binom{1\ \ \rho}{\rho\ \ \rho^2}$ is rank one, assuming a single shared latent function explains everything the two levels have in common; $B^{(2)} = bb^\top$ has a single nonzero entry, assuming the discrepancy belongs to the high-fidelity level only. (§4.3)

**P2.** You fit the model and obtain $\hat\rho = 0.02$, with a discrepancy amplitude about equal to the standard deviation of $y_H$. What has the model concluded, and what should you check next?

*Answer.* That the two fidelities are essentially unrelated — with $\rho\approx0$ the off-diagonal blocks vanish and the fit has decoupled into two independent GPs, so the cheap data is contributing nothing. Before abandoning the cheap model, plot $y_H$ against $y_L$ at shared inputs: a curved relation would mean the fidelities *are* informative but not linearly so, calling for the nonlinear model rather than for more high-fidelity runs. (§§7–8)

**P3.** With $N_H$ fixed, you increase $N_L$ from 50 to 50,000. Does the high-fidelity predictive variance go to zero? What does it approach?

*Answer.* No. In the limit $f_L$ becomes known exactly, so the remaining uncertainty is uncertainty about the discrepancy: $v_H^\star\to\operatorname{Var}[\delta(x^\star)\mid y_H]$. Cheap data buys $f_L$ only; $\delta$ can be learned only from high-fidelity data. (§6)

**P4.** Two coarse models are available for the same expensive simulation. Model A is biased but its error is smooth and slowly varying; model B is unbiased on average but its error oscillates rapidly. With a fixed budget of five high-fidelity runs, which makes the better low-fidelity level?

*Answer.* Model A. What the high-fidelity data must learn is $\delta$, so the relevant question is how complicated the *error* is, not how accurate the model is. A smooth, slowly varying discrepancy is pinned down by a few points; a rapidly oscillating one needs about as many points as learning $f_H$ from scratch, however small it is on average. (§6)

**P5.** Why does a nested design $X_H\subseteq X_L$ make recursive training exact, and what else does it buy you?

*Answer.* The recursive scheme fits the discrepancy to $y_H-\rho\mu_L(X_H)$, which requires the low-fidelity model's prediction at every high-fidelity input; when those inputs are a subset of $X_L$ the low-fidelity posterior there is pinned down by actual data rather than interpolated, and the two-stage fit reproduces the joint model. It also gives you the shared evaluations needed for the $y_H$-versus-$y_L$ diagnostic of §8. (§§5.2, 8)
