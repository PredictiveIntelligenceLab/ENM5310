# Lecture 6 — Optimization: A Practical Primer

**ENM 5310 — Data-driven Modeling and Probabilistic Scientific Computing**

**Builds on:** Lecture 4 (normal equations, the SVD and conditioning, MAP, ridge and lasso), Lecture 5 (the logistic model, its gradient $\Phi^\top(\mu-y)$ and Hessian $\Phi^\top S\Phi$, convexity, separability).

**Reading:** Murphy I §8.1–8.4 and §8.6; §10.2.3, §10.2.6, §10.2.8.

**Purpose.** Lecture 5 ended where maximum likelihood stopped being solvable in closed form. This lecture is the toolkit for everything that follows: it should leave you able to *choose* a method for a given problem, *tune* it, and *diagnose* it when it fails. It follows Murphy Chapter 8 closely, and the correspondence is given below so you can read the two together.

**Learning objectives.** After this lecture you should be able to:

1. Classify an optimization problem along Murphy's four axes and say which methods that classification admits.
2. Define a descent direction, derive steepest descent with respect to a norm, and implement backtracking line search.
3. Derive the convergence rate of gradient descent on a quadratic, give the exact stability threshold for the step size, and express iteration count in terms of the condition number.
4. Explain how standardization and preconditioning change $\kappa$, and why Newton's method is the limiting case.
5. Derive the Newton step, prove affine invariance, and explain damping; state what L-BFGS buys and when it fails.
6. Derive and implement IRLS for logistic regression.
7. Explain why a fixed step size cannot converge under stochastic gradients, derive the noise floor, and state the Robbins–Monro conditions.
8. Write down the Adam update and identify it as a diagonal preconditioner.
9. Solve an $\ell_1$-penalized problem by proximal gradient, and derive soft thresholding.
10. Diagnose a training curve: step size above threshold, poor conditioning, or a stochastic noise floor.

### Notation

$f:\mathbb{R}^D\to\mathbb{R}$ is the objective, almost always $\mathrm{NLL}(\theta)$ plus a regularizer. Write $g(\theta) = \nabla f(\theta)$, $\mathbf{H}(\theta) = \nabla^2f(\theta)$, with $g_k,\mathbf{H}_k$ their values at $\theta_k$. For logistic regression, parameters $w$, with $\Phi$, $\mu = \sigma(\Phi w)$, $S = \operatorname{diag}(\mu_n(1-\mu_n))$.

---

## 1. What kind of problem is this?

Before choosing a method, classify the problem. Murphy §8.1 uses four axes, and each one eliminates or admits whole families of algorithms.

**Constrained or unconstrained?** Everything in this course is unconstrained in form: constraints enter as penalties, and priors are the statistical name for those penalties. Genuine constraints ($\theta\ge0$, $\|\theta\|\le c$) are handled by projection, which §10 shows is the same operation as the proximal step.

**Convex or nonconvex?** $f$ is convex iff $\mathbf{H}(\theta)\succeq0$ everywhere, strictly convex iff $\mathbf{H}\succ0$. Convexity buys two things: **every stationary point is a global minimum**, so $g(\theta) = 0$ is a complete certificate of optimality; and the minimizer set is convex, a single point under strict convexity. Neither survives into the deep learning module.

| Objective | $\mathbf{H}$ | Status |
|---|---|---|
| Linear regression NLL | $\sigma^{-2}\Phi^\top\Phi$ | Convex; strictly if $\Phi$ full column rank |
| Logistic NLL | $\Phi^\top S\Phi$ | Convex; strictly if $\Phi$ full column rank |
| Either, with Gaussian prior | $\;\cdot\;+\tau^{-2}I$ | Strictly convex, always |
| Lasso | — | Convex, **nonsmooth** |
| Neural network | — | Nonconvex |

**Smooth or nonsmooth?** All methods in §§2–9 assume $f$ is differentiable. The lasso objective is not — $\|w\|_1$ has a kink at every axis — and needs §10. This is not a technicality: applying a gradient method to a nonsmooth objective produces coefficients that hover near zero without ever reaching it, which destroys the sparsity that was the point.

**Local or global?** For convex problems the distinction is empty. Otherwise, every method here finds a stationary point and nothing more.

**Two facts to carry.** First, *stationarity*: if $\theta^\star$ is a local minimum then $g(\theta^\star) = 0$, because otherwise $f(\theta^\star-\eta g) = f(\theta^\star)-\eta\|g\|^2+O(\eta^2) < f(\theta^\star)$ for small $\eta$. A nonzero gradient is a constructive certificate of improvement. Stationary points are minima, maxima, or **saddles** (where $\mathbf{H}$ has mixed signs).

Second, *existence*: a continuous **coercive** $f$ — one with $f(\theta)\to\infty$ as $\|\theta\|\to\infty$ — attains its minimum. This is exactly what separability destroys (Lecture 5 §8): the logistic NLL decreases along the ray $\alpha w_0$, so it is not coercive and the infimum is never attained. Adding $\frac{1}{2\tau^2}\|w\|^2$ restores coercivity. **In optimization language, the prior makes the objective coercive and strictly convex**, so the minimizer exists and is unique; §4 adds that it also improves conditioning.

---

## 2. Descent directions and step sizes

Every method in §§2–9 has the form $\theta_{k+1} = \theta_k + \eta_k d_k$. Two choices: the direction $d_k$, and the step size $\eta_k$.

### 2.1 Descent directions

> **Descent direction.** $d$ is a descent direction at $\theta$ if $d^\top g(\theta) < 0$.

The name is earned by a one-line Taylor expansion: $f(\theta+\eta d) = f(\theta) + \eta\,d^\top g + O(\eta^2)$, so a descent direction guarantees the objective *can* be decreased by a small enough step.

Where do descent directions come from? Take the first-order model $f(\theta+\delta)\approx f(\theta)+g^\top\delta$, which is unbounded below and so must be minimized over a trust region. With $\delta^\top A\delta\le r^2$ for $A\succ0$, the Lagrangian $g^\top\delta+\lambda(\delta^\top A\delta-r^2)$ gives $g+2\lambda A\delta = 0$, so

$$\boxed{\ \delta^\star \;\propto\; -\,A^{-1}g .\ } \tag{2.1}$$

**Every method in this lecture is a choice of $A$.** Gradient descent takes $A = I$, asserting all coordinates of $\theta$ are in comparable units — usually false (§4). Newton takes $A = \mathbf{H}$. L-BFGS approximates $\mathbf{H}$. Adam takes $A$ diagonal, estimated online. And *any* $A\succ0$ gives a descent direction, since $(-A^{-1}g)^\top g = -g^\top A^{-1}g < 0$ — so preconditioning is always safe, and the only question is whether it helps.

### 2.2 Step sizes

Three strategies, in increasing order of robustness and cost.

**Constant $\eta$.** Simplest, and what §3 analyzes. Requires knowing the stability threshold, which §3.2 supplies.

**Decaying $\eta_k$.** Necessary once gradients are stochastic (§8.2). Unnecessary overhead when they are not.

**Line search.** Choose $\eta_k$ per step by looking at the function. Exact line search, $\eta_k = \arg\min_\eta f(\theta_k+\eta d_k)$, is rarely worth its cost. What is worth it is **backtracking with the Armijo condition**: accept $\eta$ only if it delivers a fraction $c$ of the decrease the linear model promised,

$$f(\theta_k+\eta d_k) \;\le\; f(\theta_k) + c\,\eta\,d_k^\top g_k, \qquad c\approx10^{-4}.$$

> **Algorithm: backtracking line search.** Given $\theta_k$, descent direction $d_k$, $\eta\leftarrow\eta_{\text{init}}$ (use $1$ for Newton-type directions), $c = 10^{-4}$, $\beta = 0.5$:
> repeat $\eta\leftarrow\beta\eta$ until the Armijo condition holds; return $\eta$.

This costs one or two extra function evaluations per step and removes step-size tuning entirely for full-batch problems. Use it with Newton (§5) and L-BFGS (§6). It is *not* usable with stochastic gradients, since $f$ itself is then only estimated — which is why the deep learning half of the course is full of hand-tuned learning rates while the classical half is not.

---

## 3. Convergence rates and the condition number

This is the analytical core; everything later refers to it. Analyze the exact quadratic

$$f(\theta) = f^\star + \tfrac12(\theta-\theta^\star)^\top\mathbf{H}(\theta-\theta^\star), \qquad \mathbf{H}\succ0, \qquad g(\theta) = \mathbf{H}(\theta-\theta^\star).$$

This is not a toy: it is what any smooth objective looks like near a minimum, and for logistic regression it is exactly the model with $\mathbf{H} = \Phi^\top S\Phi$.

### 3.1 The error recursion decouples

With $e_k = \theta_k-\theta^\star$ and gradient descent, $e_{k+1} = (I-\eta\mathbf{H})e_k$. Diagonalize $\mathbf{H} = Q\Lambda Q^\top$ with eigenvalues $\lambda_1\ge\cdots\ge\lambda_D>0$, and write $\tilde e_k = Q^\top e_k$. Since $Q^\top(I-\eta\mathbf{H})Q = I-\eta\Lambda$ is diagonal, the recursion separates into $D$ independent scalar recursions:

$$\boxed{\ \tilde e_{k,j} = (1-\eta\lambda_j)^k\,\tilde e_{0,j}.\ } \tag{3.1}$$

Each eigendirection contracts at its own rate $|1-\eta\lambda_j|$, and the directions do not interact. **One scalar $\eta$ must serve all $D$ of them at once** — that tension is the entire content of what follows.

### 3.2 Stability: the step-size threshold

Convergence requires $|1-\eta\lambda_j|<1$ for every $j$, hence

$$\boxed{\ 0<\eta<\frac{2}{\lambda_{\max}}\ }$$

Three regimes, each recognizable from a loss curve:

- $0<\eta\lambda_1<1$: the stiffest component decays monotonically; the curve is smooth and decreasing.
- $1<\eta\lambda_1<2$: that component *alternates in sign* while shrinking; the curve falls but visibly ripples.
- $\eta\lambda_1>2$: it alternates and *grows*; the curve oscillates with increasing amplitude and overflows within a few dozen iterations.

**A diverging, oscillating loss is not evidence that the model is wrong or the data bad.** It means $\eta$ violated a bound set by $\lambda_{\max}$ — computable in advance. For logistic regression, $\lambda_{\max}(\mathbf{H})\le\frac14 d_1^2$ with $d_1$ the largest singular value of $\Phi$ (Lecture 5 §7), so $\eta<8/d_1^2$ suffices.

### 3.3 Rate

The slowest component governs: $\rho(\eta) = \max_j|1-\eta\lambda_j| = \max(|1-\eta\lambda_{\min}|, |1-\eta\lambda_{\max}|)$, since $|1-\eta\lambda|$ is V-shaped in $\lambda$ and maximized at an endpoint. Balancing the two endpoints,

$$\eta^\star = \frac{2}{\lambda_{\min}+\lambda_{\max}}, \qquad \rho^\star = \frac{\lambda_{\max}-\lambda_{\min}}{\lambda_{\max}+\lambda_{\min}} = \boxed{\ \frac{\kappa-1}{\kappa+1}\ }, \qquad \kappa\triangleq\frac{\lambda_{\max}}{\lambda_{\min}} .$$

Since $\log\frac{\kappa+1}{\kappa-1}\approx2/\kappa$ for large $\kappa$, reducing the error by $\varepsilon$ takes

$$k \approx \frac\kappa2\log\frac1\varepsilon\ : \quad \textbf{iteration count grows linearly in }\kappa .$$

For $\varepsilon = 10^{-6}$: $\kappa = 10$ needs about $70$ iterations; $\kappa = 10^2$, about $690$; $\kappa = 10^4$, about $69{,}000$.

**The mechanism, in one sentence:** $\lambda_{\max}$ sets a ceiling on the step size, $\lambda_{\min}$ determines how much progress that step makes, and $\kappa$ is the ratio — so a single stiff direction throttles every other direction. Geometrically, the level sets are ellipsoids of aspect ratio $\sqrt\kappa$; in an elongated valley the gradient points mostly *across* rather than along, so the iterates bounce between walls and creep along the floor. Equation (3.1) is that picture algebraically.

No tuning of $\eta$ escapes this — $\eta^\star$ was already optimal. Escaping requires changing the method: §4, §5, §9.

---

## 4. Standardization and preconditioning

Section 3 says iteration count is proportional to $\kappa$. Where does $\kappa$ come from? Uncomfortably: **from how you chose to write down your features.**

**Feature scaling.** For both our models $\mathbf{H}$ is built from $\Phi$. Rescale the $j$-th column by $c$ — measuring a length in nanometres instead of metres — so $\Phi\to\Phi C$ with $C = \operatorname{diag}(1,\dots,c,\dots,1)$ and

$$\mathbf{H}\ \longrightarrow\ C\,\mathbf{H}\,C .$$

The eigenvalues move and $\kappa$ can change by orders of magnitude through a change of units alone. Take one dataset, one solver, one initialization, two scalings: the same problem statistically — identical models, minimizers related by $w\to C^{-1}w$, identical predictions — yet one run converges in tens of iterations and the other in tens of thousands. A practitioner who does not know this diagnoses "the model won't train," changes the architecture, and never finds out.

**So standardize the columns of $\Phi$ to zero mean and unit variance before fitting anything** (Murphy §10.2.8). This is not cosmetic tidying; it is a direct intervention on $\kappa$ and the cheapest optimization improvement available. It is also incomplete — it equalizes the *diagonal* of $\Phi^\top\Phi$, while correlated features leave the off-diagonal structure and hence the conditioning bad. The same singular values $d_j$ that governed ridge shrinkage in Lecture 4 §9 govern $\kappa$ here.

**Preconditioning.** Rather than rescaling the data, rescale the steps. From (2.1) with $A = P\succ0$, the recursion becomes $e_{k+1} = (I-\eta P^{-1}\mathbf{H})e_k$, so the rate is governed by $\kappa(P^{-1}\mathbf{H})$. Choose $P$ so that $P^{-1}\mathbf{H}$ is well conditioned *and* solving $Pz = g$ is cheap — requirements that pull against each other:

| $P$ | $\kappa(P^{-1}\mathbf{H})$ | Cost per step | Method |
|---|---|---|---|
| $I$ | $\kappa(\mathbf{H})$ | $O(D)$ | gradient descent |
| diagonal, estimated online | improved per coordinate | $O(D)$ | Adam (§9) |
| low-rank secant approximation | approaches $1$ | $O(mD)$ | L-BFGS (§6) |
| $\mathbf{H}$ | exactly $1$ | $O(D^3)$ | Newton (§5) |

That table is the map for §§5–9. Seeing these as one idea at different price points is more useful than learning them as separate algorithms.

---

## 5. Newton's method

Section 2 built a *linear* model, which has no minimum, so a trust radius had to be imposed from outside and survived into the algorithm as $\eta$. The repair is a model that does have a minimum. Expand to second order, discard the $O(\|\delta\|^3)$ remainder, and minimize what remains exactly: $\nabla_\delta\big[g_k^\top\delta+\frac12\delta^\top\mathbf{H}_k\delta\big] = g_k+\mathbf{H}_k\delta = 0$, giving the **Newton step**

$$\boxed{\ \mathbf{H}_k\,\delta_k = -g_k, \qquad \theta_{k+1} = \theta_k + \eta_k\delta_k\ \ (\eta_k = 1 \text{ by default}).\ }$$

**No step size was invented.** A positive-definite quadratic model is bounded below, so the step *length* is an output of the curvature rather than an input. In the Hessian eigenbasis, $-\mathbf{H}^{-1}g$ multiplies the $j$-th gradient component by $1/\lambda_j$ — long steps along flat directions, short along steep. Compare §3, where the whole difficulty was one scalar $\eta$ serving both $\lambda_{\max}$ and $\lambda_{\min}$.

**Solve the system; do not invert.** Implement as a Cholesky factorization and two triangular solves — the same reflex as the normal equations in Lecture 4 §6.

**What exactness buys.** On an exact quadratic the remainder vanishes and one step lands on $\theta^\star$ from any start; against §3, where the same problem cost $\Theta(\kappa)$ iterations. The condition number has not disappeared — it now prices the linear solve, a cost paid once per step rather than an iteration count paid repeatedly. Near a minimum with $\mathbf{H}\succ0$, convergence is **quadratic**, $\|e_{k+1}\|\le C\|e_k\|^2$: correct digits roughly double per iteration, $10^{-2}\to10^{-4}\to10^{-8}$, and the run terminates.

**Damping, and it is not optional.** Far from the optimum the quadratic model may describe $f$ poorly and the raw step can overshoot or increase the objective. Use backtracking (§2.2) starting from $\eta = 1$; near the optimum $\eta_k = 1$ is always accepted, so the quadratic rate survives where it matters. If $\mathbf{H}_k$ is singular or indefinite, the model is flat or unbounded below and $\delta_k$ is undefined or ascending; solve $(\mathbf{H}_k+\gamma I)\delta_k = -g_k$ instead, which interpolates Newton ($\gamma\to0$) and gradient descent with step $1/\gamma$ ($\gamma\to\infty$). **A Gaussian prior contributes exactly this term with $\gamma = \tau^{-2}$: the prior damps Newton for free.**

**Affine invariance.** Reparameterize $\theta = A\tilde\theta$. Then $\tilde g = A^\top g$, $\tilde{\mathbf{H}} = A^\top\mathbf{H}A$, so

$$\tilde\delta = -\big(A^\top\mathbf{H}A\big)^{-1}A^\top g = -A^{-1}\mathbf{H}^{-1}g = A^{-1}\delta,$$

exactly the original step in the new coordinates. **Newton takes the same iterates however features are scaled or linearly recombined** — the pathology of §4 vanishes identically. In the language of (2.1), Newton is $A = \mathbf{H}$: it measures smallness by the curvature of the function itself, the only choice importing no convention about the units of $\theta$.

**Why it is abandoned at scale.** Forming $\mathbf{H}$ costs $O(ND^2)$, storing it $O(D^2)$, factoring it $O(D^3)$ per iteration. Free at $D = 10$, painful at $D = 10^4$, impossible at $D = 10^9$. Hence the trade that recurs all semester: **gradient descent is cheap per iteration and needs $O(\kappa)$ iterations; Newton is expensive per iteration and needs a handful.** Which wins is a question about $D$ and $\kappa$.

---

## 6. Quasi-Newton: BFGS and L-BFGS

The obvious middle ground: approximate $\mathbf{H}^{-1}$ without ever forming $\mathbf{H}$.

The information is already available. Consecutive iterates give $s_k = \theta_{k+1}-\theta_k$ and $y_k = g_{k+1}-g_k$, and by the mean value theorem $y_k\approx\mathbf{H}s_k$. So any approximation $B_{k+1}\approx\mathbf{H}$ should satisfy the **secant condition** $B_{k+1}s_k = y_k$ — a set of $D$ linear constraints on $\mathbf{H}$, obtained free of charge from gradients you computed anyway.

**BFGS** picks the update satisfying the secant condition that is closest to the previous $B_k$ and preserves positive definiteness, which turns out to be a rank-two correction. Maintaining the inverse directly makes each step $O(D^2)$ with no linear solve. **L-BFGS** ("limited memory") stores only the last $m$ pairs $(s_k,y_k)$, typically $m = 10$–$20$, and reconstructs the action of $B_k^{-1}$ on a vector by a two-loop recursion costing $O(mD)$ — no $D\times D$ matrix is ever formed.

**Practical guidance.** For smooth, deterministic, full-batch problems of moderate size, **L-BFGS with backtracking is the default**, and it requires no learning rate. It is what `scipy.optimize.minimize(method='L-BFGS-B')` runs, and it will beat hand-tuned gradient descent on essentially every problem in the first half of this course.

**When it fails.** Quasi-Newton methods depend on $y_k$ being an accurate gradient difference. Under minibatch sampling, $y_k$ is dominated by sampling noise rather than curvature, and the approximation degrades badly. That is the central reason the deep learning module uses the SGD family instead — not that second-order information is unhelpful, but that it cannot be estimated cheaply from noisy gradients.

---

## 7. IRLS for logistic regression

For logistic regression the Newton step has a closed form requiring no new machinery: each step is a weighted linear regression of the kind solved in Lecture 4. Substitute $g = \Phi^\top(\mu-y)$ and $\mathbf{H} = \Phi^\top S\Phi$, then factor $\Phi^\top S$ out of the bracket:

$$
\begin{aligned}
w^{+} &= w-\big(\Phi^\top S\Phi\big)^{-1}\Phi^\top(\mu-y)
= \big(\Phi^\top S\Phi\big)^{-1}\Big[\big(\Phi^\top S\Phi\big)w-\Phi^\top(\mu-y)\Big] \\[2pt]
&= \big(\Phi^\top S\Phi\big)^{-1}\Phi^\top S\Big[\Phi w+S^{-1}(y-\mu)\Big]
= \big(\Phi^\top S\Phi\big)^{-1}\Phi^\top S\,z, \qquad \boxed{\ z\triangleq\Phi w+S^{-1}(y-\mu).\ }
\end{aligned}
$$

Compare the weighted least squares estimator $\hat w = (\Phi^\top W\Phi)^{-1}\Phi^\top Wy$: identical, with $W = S$ and targets $z$. Hence **iteratively reweighted least squares**.

Neither $z$ nor $S$ is arbitrary. The **working response** $z_n = a_n+(y_n-\mu_n)/s_n$, with $s_n = \mu_n(1-\mu_n) = \sigma'(a_n)$, is the observation linearized onto the logit scale: we see $y_n$ on the probability scale but the model is linear in the logit, so we ask what logit would have produced $y_n$, to first order. The **weights** are that response's precisions — holding $a_n,\mu_n,s_n$ fixed, $\operatorname{Var}[z_n] = \operatorname{Var}[y_n]/s_n^2 = 1/s_n$. Points predicted with near-certainty get tiny weight, because linearizing the link there amplifies noise enormously; points with $\mu_n\approx\frac12$ get the maximum $\frac14$. IRLS uses exactly the weights the Bernoulli noise model dictates — as it must, having been derived from it.

---

> ### Algorithm: IRLS for logistic regression (with optional Gaussian prior)
>
> **Input:** $\Phi\in\mathbb{R}^{N\times D}$, $y\in\{0,1\}^N$, tolerance $\texttt{tol}$, prior variance $\tau^2$ ($\infty$ for none), weight floor $\varepsilon\approx10^{-10}$, cap $K$. **Initialize** $w_0 = 0$.
>
> For $k = 0,1,\dots,K-1$:
> 1. **Logits.** $a\leftarrow\Phi w_k$.
> 2. **Probabilities.** $\mu\leftarrow\sigma(a)$, in the stable form of Lecture 5 §5.
> 3. **Weights.** $s_n\leftarrow\max\big(\mu_n(1-\mu_n),\,\varepsilon\big)$; $S = \operatorname{diag}(s)$.
> 4. **Working response.** $z\leftarrow a+(y-\mu)/s$ (elementwise).
> 5. **Weighted least squares.** Set $w_{k+1}$ to the minimizer of $\sum_n s_n(z_n-\phi(x_n)^\top w)^2+\tau^{-2}\|w\|_2^2$, i.e. solve $(\Phi^\top S\Phi+\tau^{-2}I)w_{k+1} = \Phi^\top Sz$.
> 6. **Stop** if $\|\Phi^\top(\mu-y)+\tau^{-2}w_{k+1}\|_\infty<\texttt{tol}$ or $\|w_{k+1}-w_k\|_2\le\texttt{tol}(1+\|w_k\|_2)$.
>
> **Return** $w_{k+1}$.

**Three implementation notes.** *Step 5 must not form the normal equations*: row-scale instead, $\tilde\Phi = S^{1/2}\Phi$, $\tilde z = S^{1/2}z$, and solve $\min_w\|\tilde z-\tilde\Phi w\|_2^2$ by QR, since forming $\Phi^\top S\Phi$ squares the condition number precisely in the ill-conditioned problems that motivated Newton. With a prior, append $\tau^{-1}I_D$ as extra rows of $\tilde\Phi$ and $D$ zeros to $\tilde z$. *The weight floor is not cosmetic*: under separability $s_n\to0$, the division in step 4 blows up, and $\Phi^\top S\Phi$ collapses toward singularity — without a prior, IRLS on separable data should fail loudly rather than silently return a large arbitrary $w$. *Step 6 tests the gradient, not the objective*: near the optimum the objective changes by amounts that say nothing about proximity to the solution.

---

## 8. Stochastic gradient descent

Every objective so far is a **finite sum**, $f(\theta) = \frac1N\sum_n f_n(\theta)$, so one gradient costs $O(N)$. When $N$ is $10^6$, a single gradient descent step is unaffordable and §3 says thousands are needed.

**Minibatch gradients.** Sample $\mathcal{B}$ of size $B$ uniformly and use $g_{\mathcal{B}} = \frac1B\sum_{n\in\mathcal{B}}\nabla f_n(\theta)$. It is **unbiased**, $\mathbb{E}[g_{\mathcal{B}}] = \nabla f$, with covariance $\Sigma/B$ where $\Sigma$ is the covariance of a single $\nabla f_n$ across $n$. That is the Monte Carlo rate of Lecture 1 again — the gradient is itself now a Monte Carlo estimate with error decaying as $B^{-1/2}$. The consequence is structural: **the search direction is a random variable**, and §3 no longer applies as written.

### 8.1 The noise floor

Redo §3 with noise. With $g_k = \mathbf{H}e_k+\xi_k$, $\xi_k$ zero-mean with covariance $\Sigma$,

$$e_{k+1} = (I-\eta\mathbf{H})e_k-\eta\,\xi_k .$$

In the Hessian eigenbasis, writing $\sigma_j^2$ for the noise variance along $q_j$, and using independence of $\xi_k$ and $e_k$ so the cross term vanishes,

$$\mathbb{E}\big[\tilde e_{k+1,j}^2\big] = (1-\eta\lambda_j)^2\,\mathbb{E}\big[\tilde e_{k,j}^2\big]+\eta^2\sigma_j^2 .$$

This affine recursion has contraction factor $(1-\eta\lambda_j)^2<1$, so it converges to a fixed point rather than zero. Setting successive terms equal and solving,

$$v_j = \frac{\eta^2\sigma_j^2}{1-(1-\eta\lambda_j)^2} = \frac{\eta^2\sigma_j^2}{\eta\lambda_j(2-\eta\lambda_j)} \;\approx\; \boxed{\ \frac{\eta\,\sigma_j^2}{2\lambda_j}\ }\quad\text{for small }\eta .$$

**This explains most of what training curves do.** The iterates do not converge to $\theta^\star$; they reach a **noise floor** whose squared radius is proportional to $\eta$. Halving $\eta$ halves it — a flattened curve that drops sharply the moment the learning rate is reduced is not something being "unstuck," the floor shrank. The floor is largest along *low-curvature* directions, since $v_j\propto\sigma_j^2/\lambda_j$, so flat directions are both slowest to converge and noisiest at equilibrium. And since $\sigma_j^2\propto1/B$, the controlling quantity is $\eta/B$ — the origin of the rule that changing batch size requires changing the learning rate proportionally.

### 8.2 Step-size schedules and iterate averaging

To converge, $\eta$ must decay: fast enough that the floor vanishes, slowly enough that the iterates can still travel the initial distance. The **Robbins–Monro conditions** are

$$\sum_k\eta_k = \infty \quad\text{(steps can cover any distance)}, \qquad \sum_k\eta_k^2<\infty \quad\text{(accumulated noise is finite)},$$

satisfied by $\eta_k = \eta_0(1+k)^{-a}$ precisely for $a\in(\frac12,1]$. In practice one uses a constant rate followed by step or cosine decay — a coarse approximation chosen because the constant phase makes rapid progress while the floor is still smaller than the distance remaining.

**Iterate averaging** is the cheap alternative. Rather than decaying $\eta$, run at a constant rate and return $\bar\theta_K = \frac{1}{K-K_0}\sum_{k>K_0}\theta_k$, discarding a burn-in. Since the iterates are wandering in a ball around $\theta^\star$ with roughly stationary statistics, averaging cancels much of the noise — the same variance reduction Monte Carlo gives in Lecture 1. It costs one extra vector and is worth trying before elaborate schedules.

**The batch-size trade.** Variance falls as $1/B$ but cost grows as $B$, so per unit computation small batches often make more progress — until hardware parallelism means a larger batch costs no more wall-clock time. The right batch size is a statement about your hardware as much as your problem.

---

## 9. Momentum and preconditioned SGD

Both families answer §3 and §4, and framing them that way is more durable than memorizing update rules.

**Momentum** (Murphy §8.2.4). $\ v_{k+1} = \beta v_k+g_k$, $\ \theta_{k+1} = \theta_k-\eta v_{k+1}$, with $\beta\in[0,1)$, typically $0.9$. The effect is visible in the zig-zag picture of §3.3: along high-curvature directions the gradient reverses sign every step, so contributions cancel in the running sum; along low-curvature directions it points consistently one way, so contributions accumulate with geometric weights summing to $1/(1-\beta)$. On a quadratic with tuned $\beta$, the rate improves from $\frac{\kappa-1}{\kappa+1}$ to roughly $\frac{\sqrt\kappa-1}{\sqrt\kappa+1}$, dropping iteration count from $O(\kappa)$ to $O(\sqrt\kappa)$ — a change in exponent. At $\kappa = 10^4$, the difference between $\sim\!10^5$ and $\sim\!10^3$ iterations.

**Preconditioned SGD** (Murphy §8.4.6). AdaGrad, RMSProp, and Adam maintain a running estimate of each coordinate's squared gradient and scale the step by its reciprocal square root. **Adam** combines this with momentum:

$$
\begin{aligned}
m_k &= \beta_1 m_{k-1}+(1-\beta_1)g_k, &\qquad \hat m_k &= m_k/(1-\beta_1^k), \\
v_k &= \beta_2 v_{k-1}+(1-\beta_2)g_k^{\,2}, &\qquad \hat v_k &= v_k/(1-\beta_2^k), \\
\theta_{k+1} &= \theta_k-\eta\,\hat m_k/\big(\sqrt{\hat v_k}+\epsilon\big) & &\text{(elementwise)},
\end{aligned}
$$

with defaults $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$. The bias corrections are there because $m_0 = v_0 = 0$ makes both estimates too small in early iterations; dividing by $1-\beta^k$ removes exactly that bias and matters most in the first few hundred steps.

In the language of (2.1) and §4, this is preconditioned gradient descent with $P = \operatorname{diag}(\sqrt{\hat v}+\epsilon)$ — **a diagonal preconditioner whose curvature estimate comes from gradient statistics rather than the Hessian.** It costs $O(D)$ per step instead of $O(D^3)$ and recovers invariance to per-coordinate rescaling, precisely the failure mode of §4, though not to general linear recombination.

---

## 10. Nonsmooth objectives: proximal gradient

Lecture 5 §2.2 derived lasso as MAP under a Laplace prior and deferred its optimization to here. The objective

$$f(w) = \underbrace{L(w)}_{\text{smooth}}+\underbrace{\lambda\|w\|_1}_{\text{nonsmooth}}$$

has a kink wherever a coordinate is zero, which is exactly where the solution wants to be. Gradient descent cannot be applied — and the naive fix, **subgradient descent** using any element of $\partial|w| = \operatorname{sign}(w)$ for $w\ne0$, $[-1,1]$ at $0$, converges at a dismal $O(1/\sqrt k)$ and **never produces exact zeros**, so the sparsity that motivated lasso is lost to floating-point noise.

The right tool splits the objective. Define the **proximal operator** of a convex function $R$:

$$\operatorname{prox}_{\eta R}(v) \;\triangleq\; \arg\min_u\Big\{\tfrac12\|u-v\|_2^2+\eta R(u)\Big\},$$

which asks for a point near $v$ that also makes $R$ small. The **proximal gradient method** then takes a gradient step on the smooth part and a prox step on the rest:

$$\boxed{\ w_{k+1} = \operatorname{prox}_{\eta\lambda\|\cdot\|_1}\big(w_k-\eta\nabla L(w_k)\big).\ }$$

**For the $\ell_1$ norm the prox is available in closed form.** The objective separates across coordinates, so solve the scalar problem $\min_u\frac12(u-v)^2+\eta\lambda|u|$. Its subgradient condition is $u-v+\eta\lambda\,\partial|u|\ni0$. If $u>0$ then $u = v-\eta\lambda$, consistent when $v>\eta\lambda$; if $u<0$ then $u = v+\eta\lambda$, consistent when $v<-\eta\lambda$; otherwise $0\in\partial$ at $u = 0$. Hence **soft thresholding**:

$$\operatorname{prox}_{\eta\lambda\|\cdot\|_1}(v)_d = \operatorname{sign}(v_d)\,\max\big(|v_d|-\eta\lambda,\ 0\big).$$

Each iteration is therefore: one gradient step, then shrink every coordinate toward zero by $\eta\lambda$ and clip at zero. This is **ISTA**, and adding momentum in the manner of §9 gives **FISTA**, which improves the rate to $O(1/k^2)$. Coordinates land on *exactly* zero and stay there, which is the whole point.

**The same formula handles constraints.** Take $R$ to be the indicator function of a constraint set $\mathcal{C}$ — zero inside, $+\infty$ outside. Then $\operatorname{prox}_{\eta R}$ is Euclidean projection onto $\mathcal{C}$, and the method becomes projected gradient descent. Constrained and nonsmooth problems are one framework.

---

## 11. A practical checklist

1. **Standardize the columns of $\Phi$.** Always, before anything else (§4).
2. **Check your gradient** against central finite differences before every experiment. This catches more bugs than any other single habit.
3. **Choose a method by $D$, $N$, and smoothness:**
   - Smooth, $D$ small, full batch $\to$ Newton or IRLS with backtracking (§5, §7). No learning rate to tune.
   - Smooth, $D$ moderate, full batch $\to$ L-BFGS with backtracking (§6). Still no learning rate.
   - Smooth, $N$ large $\to$ SGD family; Adam as a default (§9).
   - Nonsmooth penalty $\to$ proximal gradient / ISTA (§10).
   - Hard constraints $\to$ projected gradient (§10).
4. **Set the learning rate from the threshold, not by guessing.** Find the largest $\eta$ that does not diverge, then use a fraction of it (§3.2).
5. **Terminate on $\|g\|_\infty$**, not on the change in objective (§7).
6. **Diagnose failures in this order:**
   - Loss oscillates and grows $\to$ $\eta$ above $2/\lambda_{\max}$. Reduce it (§3.2).
   - Loss decreases monotonically but far too slowly $\to$ conditioning. Standardize, precondition, or switch to a second-order method (§4).
   - Loss plateaus while $\|g\|$ remains large and noisy $\to$ stochastic noise floor. Decay $\eta$, raise $B$, or average iterates (§8).
   - Loss plateaus with $\|g\|\to0$ but a poor objective value $\to$ a saddle or a genuine stationary point; restart from a different initialization.
7. **Never blame the model before ruling out 1, 4, and 6.**

**A closing note on nonconvexity.** Everything above was derived for convex problems, and nothing in the second half of this course is convex. At a critical point of a generic high-dimensional function the Hessian's eigenvalues have effectively arbitrary signs, and a minimum requires *all* $D$ to be positive, so most critical points are saddles rather than local minima. What is lost is global optimality, uniqueness, and any certificate. What survives — and this is why the lecture was worth doing — is every *local* statement: the threshold $\eta<2/\lambda_{\max}$ still governs divergence, conditioning still governs progress along a valley, and the floor $v_j\approx\eta\sigma_j^2/2\lambda_j$ still describes where a stochastic run plateaus. When a network refuses to train, the diagnosis is almost always one of these three.

---

## Coding Checkpoint

`numpy` only; no library optimizers except where item 4 says otherwise. Report fitted numbers, not pictures that look about right. Use the logistic model of Lecture 5.

1. **Implement and verify.** Code $f$, $g$, $\mathbf{H}$ for the logistic NLL with optional Gaussian prior. Verify $g$ against central finite differences and $\mathbf{H}$ against finite differences of $g$; report the maximum relative error over 20 random $\theta$.
2. **The stability threshold.** On a well-conditioned, non-separable dataset ($N = 500$, $D = 5$), run gradient descent from $\theta = 0$ across several orders of magnitude of $\eta$. Report iterations to $\|g\|_\infty<10^{-6}$ against $\eta$, identify the largest convergent $\eta$, and compare against $2/\lambda_{\max}(\mathbf{H}^\star)$ and the a-priori bound $8/d_1^2$. Plot one loss curve from each of the three regimes of §3.2 and label them.
3. **Conditioning and standardization.** Rescale one feature column by $10^3$. Report $\kappa(\mathbf{H}^\star)$ and the iteration count for both scalings, and compare their ratio to the $\frac\kappa2\log\frac1\varepsilon$ prediction. Standardize all columns and report both a third time. State in one sentence why all three runs are the same statistical problem.
4. **Newton, IRLS, and L-BFGS.** Implement §7 with backtracking. Report iteration counts for *both* scalings of item 3, confirm they agree to floating point, and name the equation in §5 that predicts this. Then run `scipy.optimize.minimize(method='L-BFGS-B')` on the same problem supplying your analytic gradient, and compare iteration counts and wall-clock time against your IRLS at $D = 5$ and at $D = 500$.
5. **The noise floor.** Implement minibatch SGD. With fixed $\eta$ and $B$, run well past the point where the loss flattens, then estimate $\mathbb{E}\|\theta_k-\theta^\star\|^2$ by averaging over the last 2000 iterates ($\theta^\star$ from IRLS at high precision). Repeat with $\eta$ and $B$ each varied over a factor of 16; verify the predicted proportionalities to $\eta$ and $1/B$ and report fitted exponents. Show that a single step decay of $\eta$ produces the characteristic drop, then show that iterate averaging at constant $\eta$ achieves a comparable reduction, and report both.

---

## Quiz-eligible facts

1. Convexity means $\mathbf{H}\succeq0$ everywhere; it buys that every stationary point is a global minimum and that $g(\theta) = 0$ is a complete certificate. Lasso is convex but nonsmooth; neural networks are nonconvex.
2. A continuous coercive function attains its minimum; separability makes the logistic NLL non-coercive, and a Gaussian prior restores coercivity and strict convexity.
3. $d$ is a descent direction iff $d^\top g<0$; $-A^{-1}g$ is one for any $A\succ0$, so preconditioning is always safe.
4. Steepest descent in the norm $\|\delta\|_A$ is $-A^{-1}g$; every method here is a choice of $A$ — $I$, diagonal, secant approximation, or $\mathbf{H}$.
5. Backtracking with the Armijo condition $f(\theta+\eta d)\le f(\theta)+c\,\eta\,d^\top g$, $c\approx10^{-4}$, removes step-size tuning for full-batch problems but cannot be used with stochastic gradients.
6. On a quadratic the error decouples in the Hessian eigenbasis: $\tilde e_{k,j} = (1-\eta\lambda_j)^k\tilde e_{0,j}$.
7. Gradient descent converges iff $0<\eta<2/\lambda_{\max}$; for $1<\eta\lambda_{\max}<2$ the stiffest component alternates while decaying; above $2/\lambda_{\max}$ it alternates and grows.
8. Optimal fixed step $2/(\lambda_{\min}+\lambda_{\max})$, rate $(\kappa-1)/(\kappa+1)$, iteration count $\approx\frac\kappa2\log\frac1\varepsilon$ — linear in $\kappa$. $\lambda_{\max}$ sets the step ceiling, $\lambda_{\min}$ the progress; $\sqrt\kappa$ is the level-set aspect ratio.
9. Rescaling feature columns maps $\mathbf{H}\to C\mathbf{H}C$ and changes $\kappa$ without changing the model or its predictions. Standardize before fitting; it equalizes the diagonal only.
10. Preconditioned gradient descent has rate governed by $\kappa(P^{-1}\mathbf{H})$; $P = \mathbf{H}$ is Newton, $P$ diagonal is Adam, $P$ from secant pairs is L-BFGS.
11. The Newton step is the exact minimizer of the second-order Taylor model, from solving $\mathbf{H}\delta = -g$; no step size is introduced because a positive-definite quadratic model is bounded below. It is exact in one step on a quadratic and converges quadratically near a minimum.
12. Newton is affine invariant ($\tilde\delta = A^{-1}\delta$), so feature scaling does not change its iterates. Damping via $(\mathbf{H}+\gamma I)$ interpolates Newton and gradient descent; a Gaussian prior supplies it with $\gamma = \tau^{-2}$. Cost is $O(D^3)$ per iteration.
13. Quasi-Newton methods impose the secant condition $B_{k+1}s_k = y_k$ with $s_k = \theta_{k+1}-\theta_k$, $y_k = g_{k+1}-g_k$; L-BFGS stores $m\approx10$–$20$ pairs at $O(mD)$ cost and is the default for smooth full-batch problems. It degrades under minibatch noise.
14. The IRLS step is weighted least squares with weights $S$ and working response $z = \Phi w+S^{-1}(y-\mu)$; $z$ is the observation linearized onto the logit scale and $s_n = 1/\operatorname{Var}[z_n]$. Solve by QR on $S^{1/2}\Phi$ against $S^{1/2}z$; terminate on the gradient norm.
15. Minibatch gradients are unbiased with covariance $\Sigma/B$. At fixed $\eta$, SGD reaches a noise floor $\approx\eta\sigma_j^2/(2\lambda_j)$ — proportional to $\eta$ and $1/B$, largest along low-curvature directions — which is why decaying the learning rate produces a sudden drop.
16. Robbins–Monro requires $\sum\eta_k = \infty$ and $\sum\eta_k^2<\infty$, satisfied by $\eta_0(1+k)^{-a}$ for $a\in(\frac12,1]$. Iterate averaging at constant $\eta$ is a cheap alternative.
17. Momentum improves iteration count from $O(\kappa)$ to $O(\sqrt\kappa)$ by cancelling oscillation along stiff directions and accumulating along flat ones. Adam is momentum plus a diagonal preconditioner, with bias corrections $1-\beta_1^k$ and $1-\beta_2^k$ because $m_0 = v_0 = 0$.
18. Proximal gradient: $w_{k+1} = \operatorname{prox}_{\eta\lambda R}(w_k-\eta\nabla L(w_k))$. For $R = \|\cdot\|_1$ the prox is soft thresholding $\operatorname{sign}(v)\max(|v|-\eta\lambda,0)$, which yields exact zeros; subgradient descent does not. With $R$ an indicator function the prox is projection.

---

## Practice problems

**P1.** Prove (2.1): minimizing $g^\top\delta$ subject to $\delta^\top A\delta\le r^2$ with $A\succ0$ gives $\delta^\star = -r\,A^{-1}g/\sqrt{g^\top A^{-1}g}$. Recover the Euclidean case as $A = I$, and verify $-A^{-1}g$ is a descent direction for any $A\succ0$.

**P2.** For the quadratic of §3 with $D = 2$, $\lambda_1 = 100$, $\lambda_2 = 1$, $\eta = \eta^\star$: compute the iterations to reduce $\|e\|$ by $10^{-6}$, plot the trajectory over the level sets, and explain the zig-zag using (3.1). Then show gradient descent with $\eta = 2/\lambda_{\max}$ exactly neither converges nor diverges, describing the behavior along $q_1$ and $q_2$.

**P3.** Prove that preconditioned gradient descent converges at a rate governed by $\kappa(P^{-1}\mathbf{H})$. What $P$ makes this $1$, and what does the method reduce to?

**P4 (damping).** Implement damped Newton with backtracking. Construct a logistic problem and initialization at which the undamped step *increases* the objective; report the values before and after. Confirm $\eta_k = 1$ is accepted once near the optimum, and say why that must be so.

**P5 (working response).** Derive $z_n = a_n+(y_n-\mu_n)/\sigma'(a_n)$ by linearizing $a\mapsto\sigma(a)$ about the current $a_n$. Verify $\operatorname{Var}[z_n] = 1/s_n$, and say what happens to $z_n$ and $s_n$ for a point predicted with near-certainty.

**P6 (noise floor).** Derive the fixed point $v_j$ of §8.1 without the small-$\eta$ approximation, and show it diverges as $\eta\to2/\lambda_j$. Interpret that divergence in terms of §3.2.