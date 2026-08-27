# ENM 5310 — Lecture 3
## Distributions as Modeling Choices, Transformations of Random Variables, and Information

**Reading:** Murphy, *PML: An Introduction*, §2.4–2.8; §6.1–6.3 for the information-theoretic material.
**Builds on:** Lectures 1–2 (laws, densities, expectation, conditioning, Bayes).

**Learning objectives.** You should be able to (i) construct a likelihood for a supervised problem by choosing an output distribution and a link, and write its negative log in numerically stable form; (ii) apply the change-of-variables formula in $\mathbb{R}^d$ and state why the log-determinant appears; (iii) define entropy, cross-entropy, and KL divergence, prove $\mathrm{KL}\ge0$, and show that maximum likelihood is minimization of $\mathrm{KL}(\hat p_N \,\|\, p_\theta)$; (iv) explain what mutual information measures and use it to separate epistemic from aleatoric uncertainty.

---

## 1. The recipe

Lecture 2 established that supervised learning is the study of $p(y\mid x)$. Here is how essentially every such model in this course is built:

$$\boxed{\ p(y \mid x, \theta) = \mathcal{D}\big(y \;\big|\; f(x;\theta)\big)\ }$$

Pick a distribution family $\mathcal{D}$ appropriate to the **type** of $y$ (binary, categorical, real, positive, count, on a simplex, on a manifold). Let a function $f$ — linear map, MLP, transformer, neural operator — predict its **parameters** as a function of $x$. Train by maximizing $\sum_n \log p(y_n|x_n,\theta)$.

Two design decisions, and only two:

1. **What is the output distribution?** This is a modeling claim about the noise and the type of $y$. It determines the loss; you do not choose a loss separately.
2. **How do we map an unconstrained network output to the constrained parameters?** Networks emit arbitrary reals; probabilities must lie in $[0,1]$ and sum to one, variances must be positive. The fix is a **link function**: $\sigma$, softmax, softplus, $\exp$.

Everything else — architecture, optimizer, schedule — is machinery for optimizing the objective this choice defines. Get in the habit of reading any paper's model section by asking: *what is $\mathcal{D}$, what is $f$, and what is the link?*

---

## 2. Bernoulli → sigmoid → logistic regression

**Distribution.** For $y\in\{0,1\}$ with success probability $\theta$:

$$\mathrm{Ber}(y\mid\theta) = \theta^{y}(1-\theta)^{1-y}, \qquad \mathbb{E}[y]=\theta,\ \ \mathbb{V}[y]=\theta(1-\theta).$$

Summing $N$ iid Bernoulli trials gives the **binomial**, $\mathrm{Bin}(s\mid N,\theta) = \binom{N}{s}\theta^s(1-\theta)^{N-s}$.

**Link.** We need $f(x;\theta) \in [0,1]$. Let the model emit an unconstrained **logit** $a = f(x;\theta)\in\mathbb{R}$ and squash it:

$$\sigma(a) \triangleq \frac{1}{1+e^{-a}} = \frac{e^a}{1+e^a}, \qquad p(y\mid x,\theta) = \mathrm{Ber}\big(y \mid \sigma(a)\big).$$

Properties to know without looking up (derive $\sigma'$ on the board — it is one line and it is the reason the whole thing is convenient):

$$\sigma'(a) = \sigma(a)\big(1-\sigma(a)\big), \qquad 1-\sigma(a) = \sigma(-a), \qquad \sigma^{-1}(p) = \log\frac{p}{1-p} \triangleq \mathrm{logit}(p),$$
$$\sigma_+(a) \triangleq \log(1+e^a) = \mathrm{softplus}(a), \qquad \sigma_+'(a) = \sigma(a).$$

**Interpretation.** The logit *is* the log-odds:
$$\log\frac{p}{1-p} = \log\left(\frac{e^a}{1+e^a}\cdot\frac{1+e^a}{1}\right) = a.$$
So a linear model $a = w^\top x + b$ asserts that the log-odds are linear in the features; $w_j$ is the change in log-odds per unit $x_j$. That is **logistic regression**, and $\sigma$ is a smoothed Heaviside — the decision boundary is $\{x : a(x)=0\}$, i.e. $p=1/2$.

Why not fit $y$ with linear regression instead? Because a linear function leaves $[0,1]$ as soon as you move far enough in either direction, and you would be asserting probabilities above 1 and below 0 with confidence proportional to distance from the data.

**The loss, written so it does not overflow.** The negative log-likelihood for one example:

$$\ell(a,y) = -\big[y\log\sigma(a) + (1-y)\log\big(1-\sigma(a)\big)\big].$$

This is the **binary cross-entropy** (§7 explains the name). Implemented naively — compute $\sigma(a)$, then take its log — it produces `-inf` for $|a|\gtrsim 40$ in float32. Instead, note $-\log\sigma(a) = \log(1+e^{-a}) = \mathrm{softplus}(-a)$ and $-\log(1-\sigma(a)) = \mathrm{softplus}(a)$, so

$$\boxed{\ \ell(a,y) = \mathrm{softplus}(a) - y\,a\ } \qquad\text{and stably}\qquad \mathrm{softplus}(a) = \max(a,0) + \log\big(1+e^{-|a|}\big).$$

This is exactly what `binary_cross_entropy_with_logits` does, and it is why you should pass logits, never probabilities, to your loss function. **You will implement this from these equations, not import it.** Deriving stable forms of standard losses is a recurring exercise in this course, because the difference between a model that trains and a model that produces NaNs at epoch 3 is usually two lines of this kind.

---

## 3. Categorical → softmax → multiclass classification

**Distribution.** For $y \in \{1,\dots,C\}$, with one-hot encoding $y_c$:

$$\mathrm{Cat}(y\mid\theta) = \prod_{c=1}^{C}\theta_c^{\,\mathbb{I}(y=c)} = \prod_{c=1}^C \theta_c^{y_c}, \qquad \theta_c\ge0,\ \textstyle\sum_c\theta_c=1,$$

so there are $C-1$ free parameters. Counting outcomes over $N$ trials gives the **multinomial**.

**Link.** Given logits $a = f(x;\theta)\in\mathbb{R}^C$,

$$\mathrm{softmax}(a)_c = \frac{e^{a_c}}{\sum_{c'} e^{a_{c'}}}.$$

Three properties worth stating explicitly:

- **Shift invariance:** $\mathrm{softmax}(a + \kappa\mathbf{1}) = \mathrm{softmax}(a)$. The model is over-parameterized by one degree of freedom per example. Harmless for prediction; it means logits are not identifiable, so do not interpret their absolute magnitudes.
- **Temperature:** $\mathrm{softmax}(a/T)$ interpolates between uniform ($T\to\infty$) and a one-hot argmax ($T\to0$, "winner takes all"). This single knob reappears as sampling temperature in generative models, as the temperature in knowledge distillation, and as a post-hoc calibration parameter.
- **Reduction to binary:** with $C=2$, $\mathrm{softmax}(a)_0 = \sigma(a_0-a_1)$. Only the difference matters — consistent with shift invariance.

**Log-sum-exp.** Define $\mathrm{lse}(a) \triangleq \log\sum_c e^{a_c}$, so that $\log p(y=c\mid x) = a_c - \mathrm{lse}(a)$. Computed naively, $a = (1000,1001,1000)$ overflows to `inf` and $a=(-1000,-999,-1000)$ underflows to 0. The fix, for any $m$:

$$\mathrm{lse}(a) = m + \log\sum_c e^{a_c - m}, \qquad m = \max_c a_c,$$

which guarantees the largest exponent is $e^0=1$: no overflow, and any underflow is harmless. **The identity holds for every $m$; choosing the max is what makes it numerically safe.** Same principle appears in the forward–backward algorithm, in importance weights, and in every ELBO you will compute.

A fact that pays dividends later: $\nabla_a \,\mathrm{lse}(a) = \mathrm{softmax}(a)$. The gradient of a log-partition function is the expected sufficient statistic — the organizing identity of exponential families, which we return to in the estimation lecture. It also gives the clean gradient of the categorical NLL: $\nabla_a \ell = \mathrm{softmax}(a) - y_{\text{onehot}}$, i.e. **predicted minus observed**. Derive this once; you will recognize it in backprop for the rest of your life.

---

## 4. Gaussian → regression

**Distribution.**

$$\mathcal{N}(y\mid\mu,\sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}}\exp\left(-\frac{(y-\mu)^2}{2\sigma^2}\right),$$

with cdf $\Phi$ (no closed form; implemented via `erf`), mean $=$ mode $=\mu$, variance $\sigma^2$, precision $\lambda = 1/\sigma^2$.

**Why the Gaussian is everywhere.** Four independent reasons, and it is worth separating them because only some apply in any given situation: (i) two interpretable parameters that are exactly the first two moments; (ii) the central limit theorem (§6) makes it the right model for aggregated small errors; (iii) it is the **maximum-entropy** distribution with a given mean and variance (§8) — the least-committal choice given only those constraints; (iv) it is analytically convenient beyond any competitor. Reason (iv) is not a scientific justification, and you should be able to tell when you are invoking it.

**Conditional model.** $p(y\mid x,\theta) = \mathcal{N}\big(y \mid f_\mu(x;\theta), f_\sigma(x;\theta)^2\big)$.

*Homoscedastic* ($\sigma$ constant, $\mu$ linear) is ordinary **linear regression**, $p(y|x,\theta)=\mathcal{N}(y|w^\top x + b,\sigma^2)$. Its NLL over a dataset:

$$-\log p(\mathcal{D}\mid\theta) = \frac{1}{2\sigma^2}\sum_n \big(y_n - w^\top x_n - b\big)^2 + \frac{N}{2}\log(2\pi\sigma^2).$$

For fixed $\sigma$ the second term is constant and minimizing NLL **is** least squares. State the converse plainly: *whenever you minimize MSE you have assumed additive, homoscedastic, Gaussian noise*, whether or not you meant to. If your residuals are skewed, heavy-tailed, or input-dependent, that assumption is doing damage, and residual diagnostics are how you find out.

*Heteroscedastic* lets the noise depend on the input, $p(y|x,\theta) = \mathcal{N}\big(y \mid \mu(x), \sigma_+(s(x))^2\big)$ with softplus (or $\exp$) ensuring positivity. Per-example NLL:

$$\ell(x,y) = \frac{(y-\mu(x))^2}{2\sigma^2(x)} + \tfrac12\log\sigma^2(x) + \text{const}.$$

Read the two terms as a negotiation: the first is a *precision-weighted* squared error, the second penalizes claiming large uncertainty. This is genuinely more expressive — it is how you get input-dependent aleatoric error bars — and it has a well-known failure mode you should expect on your first attempt: early in training the model reduces the loss fastest by inflating $\sigma(x)$ on hard examples rather than fitting $\mu$ there, effectively deleting them from the gradient signal, and it can collapse in the other direction ($\sigma\to0$ on a memorized point, loss $\to-\infty$) if the mean network is flexible enough. Standard mitigations: warm up with fixed $\sigma$, lower-bound $\sigma$, or train $\mu$ and $\sigma$ on separate splits. **When we do the empirical-methods module, this is one of the failure modes you will be asked to diagnose from a training curve.**

**Robust likelihoods ★.** The Gaussian's exponentially decaying tails make it fragile to outliers: a single point at $10\sigma$ contributes $50\times$ the loss of a point at $\sqrt2\,\sigma$, so the fit moves to accommodate it.

| Family | Density kernel | NLL in $r = y-\mu$ | Behavior |
|---|---|---|---|
| Gaussian | $e^{-r^2/2\sigma^2}$ | $r^2$ | fits the mean; outlier-sensitive |
| Student-$t$, $\nu$ dof | $\big(1+\tfrac{1}{\nu}(r/\sigma)^2\big)^{-(\nu+1)/2}$ | $\tfrac{\nu+1}{2}\log(1+r^2/\nu\sigma^2)$ | polynomial tails; robust |
| Laplace | $e^{-\|r\|/b}$ | $\|r\|/b$ | fits the median; $L_1$ loss |

The Student-$t$ has mean $\mu$ (for $\nu>1$) and variance $\nu\sigma^2/(\nu-2)$ (for $\nu>2$); $\nu=1$ is the **Cauchy**, which has *no mean at all* — 95% of standard normal mass lies in $(-1.96,1.96)$ but for a standard Cauchy the corresponding interval is roughly $(-12.7,12.7)$. By $\nu\gtrsim5$ the Student is effectively Gaussian and its robustness is gone; $\nu=4$ is a common default. Because the log-density grows logarithmically rather than quadratically, a far-out point has bounded influence on the gradient. That is the whole mechanism, and it is worth seeing: robustness is a statement about the *tail of your log-likelihood*, not a preprocessing step.

**Other families you will meet ★.** Gamma $\mathrm{Ga}(x|a,b)\propto x^{a-1}e^{-bx}$ for positive quantities (mean $a/b$, variance $a/b^2$), with the exponential ($a=1$) and $\chi^2_\nu$ ($a=\nu/2, b=1/2$) as special cases; inverse-Gamma as the conjugate prior for a Gaussian variance; Beta on $[0,1]$ (Lecture 2); half-Cauchy as a heavy-tailed prior on scale parameters. The point is not to memorize the table but to recognize that **choosing a distribution is choosing a support and a tail**, and those two choices carry more modeling content than anything else in the model.

---

## 5. Transformations of random variables

If $x\sim p_x$ and $y = f(x)$, what is $p_y$?

**Discrete:** just accumulate mass, $p_y(y) = \sum_{x: f(x)=y} p_x(x)$.

**Continuous:** you cannot sum densities. Work with the cdf. For monotone (hence invertible) $f$ with $g = f^{-1}$:

$$P_y(y) = \Pr(f(X)\le y) = \Pr(X\le g(y)) = P_x(g(y)) \ \Longrightarrow\ p_y(y) = p_x(g(y))\left|\frac{dg}{dy}\right|.$$

The absolute value handles decreasing $f$. Derive it once by the conservation-of-mass argument, which is the version you should remember: probability in corresponding infinitesimal intervals must match, $p_x(x)\,|dx| = p_y(y)\,|dy|$, hence $p_y = p_x\,|dx/dy|$. A density is not a scalar you can transport; it is mass per unit volume, so it must pick up the volume distortion factor.

**Multivariate.** For invertible $f:\mathbb{R}^n\to\mathbb{R}^n$ with inverse $g$ and Jacobian $J_g = \partial g/\partial y^\top$:

$$\boxed{\ p_y(y) = p_x\big(g(y)\big)\,\big|\det J_g(y)\big|\ }$$

$|\det J|$ is the local volume-change factor: for an affine map $f(x)=Ax+b$ the unit square is mapped to a parallelogram of area $|\det A|$. Worked example to do on the board — Cartesian to polar, $g(r,\phi) = (r\cos\phi, r\sin\phi)$:

$$J_g = \begin{pmatrix}\cos\phi & -r\sin\phi\\ \sin\phi & r\cos\phi\end{pmatrix}, \quad |\det J_g| = r \quad\Longrightarrow\quad p_{r,\phi}(r,\phi) = p_{x_1,x_2}(r\cos\phi, r\sin\phi)\,r,$$

which is the familiar $r\,dr\,d\phi$ area element. Geometry and probability agree, as they must.

**Three places this becomes essential later:**

1. **Normalizing flows.** Build a complicated density by pushing a simple one ($\mathcal{N}(0,I)$) through an invertible network: $\log p_y(y) = \log p_x(g(y)) + \log|\det J_g(y)|$. Exact likelihoods, at the cost of architectures whose Jacobian determinant is cheap (triangular, low-rank, or coupling structures). The entire design space of flows is "make $\log|\det J|$ tractable."
2. **The reparameterization trick.** $y = \mu + \sigma\varepsilon$ with $\varepsilon\sim\mathcal{N}(0,1)$ gives $y\sim\mathcal{N}(\mu,\sigma^2)$ while keeping $\mu,\sigma$ differentiable — this is what makes VAE gradients low-variance and it is a change of variables in disguise.
3. **Inverse-cdf sampling.** If $U\sim\mathrm{Unif}(0,1)$ then $X = P^{-1}(U)$ has cdf $P$. One line of the formula above; it is the base case of every sampler you will write.

**Moments of a linear map ★.** For $y = Ax+b$ with $\mathbb{E}[x]=\mu$, $\mathrm{Cov}[x]=\Sigma$:

$$\mathbb{E}[y] = A\mu + b, \qquad \mathrm{Cov}[y] = A\Sigma A^\top, \qquad \mathbb{V}[a^\top x + b] = a^\top\Sigma a.$$

Taking $a = (1,1)^\top$ recovers $\mathbb{V}[x_1+x_2] = \mathbb{V}[x_1]+\mathbb{V}[x_2]+2\mathrm{Cov}[x_1,x_2]$. Note these give you moments only — for a general distribution the *full* law of $y$ still requires the Jacobian formula. The Gaussian is special: it is closed under affine maps, so for Gaussians the moments are the whole answer. That closure property is why linear-Gaussian models (Kalman filters, GP regression) admit closed-form inference.

---

## 6. Sums, convolution, and the CLT

For independent $x_1, x_2$ and $y = x_1 + x_2$, the density of the sum is a **convolution**:

$$p(y) = \int p_1(x_1)\,p_2(y-x_1)\,dx_1 \quad \equiv \quad p = p_1 \circledast p_2,$$

discretely, $p(y=j) = \sum_k p_1(k)p_2(j-k)$ — the familiar "flip and drag." Two dice give the triangular distribution on $\{2,\dots,12\}$; already at $N=2$ it is starting to look bell-shaped. For Gaussians the convolution stays in the family:

$$\mathcal{N}(\mu_1,\sigma_1^2)\circledast\mathcal{N}(\mu_2,\sigma_2^2) = \mathcal{N}(\mu_1+\mu_2,\ \sigma_1^2+\sigma_2^2).$$

**Central limit theorem.** For iid $X_n$ with finite mean $\mu$ and variance $\sigma^2$, and $S_N = \sum_{n=1}^N X_n$,

$$Z_N = \frac{S_N - N\mu}{\sigma\sqrt N} = \frac{\bar X - \mu}{\sigma/\sqrt N} \ \xrightarrow{\ d\ }\ \mathcal{N}(0,1).$$

Three uses, and one caution:

- It justifies Gaussian noise models whenever the error is an aggregate of many small independent contributions.
- It gives the sampling distribution of the sample mean, hence the error bars on every Monte Carlo estimate from Lecture 1 — $\bar X \pm 1.96\,\hat\sigma/\sqrt N$. **When you report an experimental result in this course, this is the interval I expect to see.**
- Combined with Lecture 1's $\sigma_f/\sqrt N$, it explains why the MC error is not just $O(N^{-1/2})$ in magnitude but approximately *normally distributed* around the truth.
- **Caution:** finite variance is required (Cauchy sums do not concentrate — they stay Cauchy), independence is required, and convergence in the *tails* is much slower than in the bulk. Rare-event probabilities are exactly where the CLT approximation fails, and rare events are often what safety-critical engineering cares about.

---

## 7. Information: entropy, cross-entropy, and KL

We now have the vocabulary to say what a learning objective *is*. The organizing idea: information is measured in units of surprise.

**Surprisal.** Observing an outcome of probability $p$ carries information $-\log p$. The requirements that surprise be decreasing in $p$, that certainty ($p=1$) carry zero surprise, and that independent events have *additive* surprise force the logarithm; the base fixes the units (base 2: bits; base $e$: nats — we use nats).

> **Entropy.** $\displaystyle \mathbb{H}[p] \triangleq -\sum_{x} p(x)\log p(x) = \mathbb{E}_p[-\log p(X)]$: the expected surprisal, i.e. the average uncertainty in $p$.

Facts to know: $\mathbb{H}\ge0$ for discrete $p$, maximized by the uniform distribution ($\log K$ for $K$ states), zero for a point mass. For a Bernoulli, $\mathbb{H}(\theta) = -\theta\log\theta - (1-\theta)\log(1-\theta)$, maximal at $\theta=1/2$. Operationally (Shannon's source coding theorem) entropy is the minimum average code length needed to transmit draws from $p$ — which is why "bits per dimension" is the standard metric for generative image models.

**Differential entropy ★ and its trap.** For continuous $p$, $h[p] = -\int p(x)\log p(x)\,dx$. This is *not* a limit of the discrete entropy, it can be **negative** ($\mathcal{N}(0,\sigma^2)$ has $h = \tfrac12\log(2\pi e\sigma^2)$, negative for small $\sigma$), and it is **not invariant under reparameterization** — rescale $x$ and $h$ shifts by $\log|\det A|$, by §5. Consequence: "the entropy of my latent variable" is a units-dependent quantity, and any claim resting on the sign or magnitude of a differential entropy deserves scrutiny. KL and mutual information, being ratios of densities, do not suffer this defect. Prefer them.

> **Cross-entropy.** $\displaystyle \mathbb{H}(p,q) \triangleq -\mathbb{E}_p[\log q(X)]$: the average surprise when the world is $p$ but your model is $q$ — the expected code length using a code optimized for $q$.

> **Kullback–Leibler divergence.**
> $$\mathrm{KL}(p\,\|\,q) \triangleq \mathbb{E}_p\!\left[\log\frac{p(X)}{q(X)}\right] = \mathbb{H}(p,q) - \mathbb{H}[p] \;\ge\; 0,$$
> with equality iff $p=q$ (a.e.). The excess code length from using the wrong model.

*Proof of non-negativity (Gibbs' inequality), worth the board space:* $-\log$ is convex, so by Jensen,
$$\mathrm{KL}(p\|q) = \mathbb{E}_p\!\left[-\log\frac{q(X)}{p(X)}\right] \ \ge\ -\log \mathbb{E}_p\!\left[\frac{q(X)}{p(X)}\right] = -\log\int p(x)\frac{q(x)}{p(x)}dx = -\log 1 = 0. \qquad\blacksquare$$

KL is **not a metric**: it is asymmetric and violates the triangle inequality. Call it a divergence and mean it.

### 7.1 Maximum likelihood is distribution matching

Here is the identity that unifies the whole course's supervised material. Recall the empirical distribution $\hat p_N$ from Lecture 1. Then

$$\mathrm{KL}(\hat p_N \,\|\, p_\theta) = \underbrace{\mathbb{E}_{\hat p_N}[\log \hat p_N(X)]}_{\text{independent of }\theta} - \mathbb{E}_{\hat p_N}[\log p_\theta(X)] = \text{const} - \frac{1}{N}\sum_{n=1}^N \log p_\theta(x_n),$$

so

$$\boxed{\ \arg\min_\theta\ \mathrm{KL}(\hat p_N\|p_\theta) \;=\; \arg\min_\theta\ \mathbb{H}(\hat p_N, p_\theta) \;=\; \arg\max_\theta\ \sum_n \log p_\theta(x_n) \;=\; \hat\theta_{\mathrm{MLE}}\ }$$

**Maximum likelihood is minimization of the KL divergence from the empirical distribution to the model, and equivalently minimization of a cross-entropy.** Three consequences:

1. It explains the name: the "cross-entropy loss" in classification is literally $\mathbb{H}(\hat p, p_\theta)$, and the binary version is §2's $\mathrm{softplus}(a) - ya$.
2. It explains why MSE is a special case: under a homoscedastic Gaussian likelihood, cross-entropy $=$ scaled MSE $+$ constant (§4). **Squared error is a cross-entropy in disguise.** Almost every loss you will write is a negative log-likelihood, and if it is not, you should be able to say what it is instead.
3. It says exactly what MLE targets: the model closest in KL to the *empirical* distribution, not to $p^\star$. The gap between $\hat p_N$ and $p^\star$ is where overfitting lives — the same gap Lecture 1 named as the plug-in error, now with a divergence attached to it.

---

## 8. Mutual information, maximum entropy, and the two KLs

> **Mutual information.**
> $$\mathbb{I}(X;Y) \triangleq \mathrm{KL}\big(p(x,y)\,\|\,p(x)p(y)\big) = \mathbb{H}[X] - \mathbb{H}[X\mid Y] = \mathbb{H}[Y] - \mathbb{H}[Y\mid X] \ \ge 0,$$
> where $\mathbb{H}[X|Y] = \mathbb{E}_{p(y)}\big[\mathbb{H}[p(x|y)]\big]$ is the conditional entropy.

Read it as: how far the joint is from the independent factorization; equivalently, the expected reduction in uncertainty about $X$ from learning $Y$. $\mathbb{I}(X;Y)=0$ iff $X\perp Y$. Unlike correlation, it detects *any* dependence, not just monotone or linear ones — an Anscombe-style example where $\rho=0$ but $\mathbb{I}>0$ is easy to construct, and worth doing.

**Why we care in this course.** Take $X = \theta$ (parameters) and $Y = y_\star$ (a prediction at a candidate input $x_\star$). Then

$$\underbrace{\mathbb{I}(\theta; y_\star \mid x_\star, \mathcal{D})}_{\text{epistemic}} = \underbrace{\mathbb{H}\big[p(y_\star\mid x_\star,\mathcal{D})\big]}_{\text{total predictive}} - \underbrace{\mathbb{E}_{p(\theta|\mathcal{D})}\big[\mathbb{H}[p(y_\star\mid x_\star,\theta)]\big]}_{\text{expected aleatoric}}.$$

This is the clean formalization of the split we introduced in Lecture 1: total uncertainty minus the part that survives even when $\theta$ is known. It is the acquisition function behind Bayesian active learning (BALD) and it is what optimal experimental design maximizes — *choose the experiment whose outcome tells you the most about the parameters.* For an audience that runs physical experiments, that is the single most useful thing information theory gives you, and we will build on it in the Bayesian optimization module.

**Caution ★.** Estimating $\mathbb{I}$ from samples in high dimensions is hard — variational bounds like InfoNCE saturate at $\log(\text{batch size})$, and reported MI values in the representation-learning literature are often uninformative about the quantity they claim to measure. Treat sample-based MI estimates as lower bounds with unknown slack.

**Maximum entropy.** Among all distributions satisfying a set of moment constraints, the maximum-entropy one is the least committal — it assumes nothing beyond the constraints. Results worth knowing: with support on a finite set and no constraints, uniform; with fixed mean on $[0,\infty)$, exponential; **with fixed mean and variance on $\mathbb{R}$, Gaussian.** This is reason (iii) from §4, now with content: choosing a Gaussian noise model when you know only a scale is not laziness, it is the formally minimal assumption. Choosing one when you *do* know more (bounded support, positivity, skew) is throwing information away.

**Forward vs. reverse KL ★ — a preview of variational inference.** The two orderings behave completely differently when $q$ cannot represent $p$:

- $\mathrm{KL}(p\|q)$, **forward**, "mode-covering": the expectation is under $p$, so wherever $p$ has mass and $q$ does not, the integrand blows up. $q$ is forced to spread out and cover everything. This is what MLE minimizes.
- $\mathrm{KL}(q\|p)$, **reverse**, "mode-seeking": the expectation is under $q$, so $q$ is only penalized where it puts mass; it can safely ignore modes of $p$ and lock onto one. This is what variational inference minimizes, and it is why mean-field VI famously *underestimates* posterior variance.

Neither is right; they answer different questions. When we get to VI you will see the identity

$$\log p(x) = \underbrace{\mathbb{E}_{q(z)}\!\left[\log\frac{p(x,z)}{q(z)}\right]}_{\text{ELBO}} + \mathrm{KL}\big(q(z)\,\|\,p(z\mid x)\big),$$

which is now readable with the tools from this lecture: since KL $\ge0$, the ELBO lower-bounds the log evidence, and maximizing it is minimizing the reverse KL to the posterior. Everything in that identity is a definition from today.

---

## 9. Code checkpoint

1. **Stable BCE.** Implement `bce_with_logits(a, y)` from $\mathrm{softplus}(a) - ya$ using the $\max(a,0)+\log(1+e^{-|a|})$ form. Compare against the naive `-y*log(sigmoid(a)) - ...` at $a = \pm10, \pm50, \pm500$ in float32. Report where the naive version fails.
2. **Stable LSE and softmax gradient.** Implement `lse` with the max shift. Verify $\nabla_a\mathrm{lse}(a) = \mathrm{softmax}(a)$ by finite differences, and verify $\nabla_a\ell = \mathrm{softmax}(a) - y_{\text{onehot}}$ for the categorical NLL.
3. **Heteroscedastic failure mode.** Generate $y=\sin(2\pi x) + \varepsilon(x)$ with $\sigma(x) = 0.05 + 0.3x$. Fit (a) a homoscedastic and (b) a heteroscedastic MLP. Plot the learned $\sigma(x)$ against the truth. Then train (b) from scratch with a poor initialization and *find the variance-inflation pathology* — plot the loss and $\sigma(x)$ that produced it.
4. **Robustness.** Fit Gaussian, Student-$t$ ($\nu=4$), and Laplace likelihoods to the same 1-D data, then add a single point at $10\sigma$ and refit. Report the shift in $\hat\mu$ for each. One number each; that table is the argument.
5. **Change of variables.** Sample $x\sim\mathcal{N}(0,1)$, set $y = \tanh(x)$. Derive $p_y$ analytically and overlay it on a histogram of transformed samples. Confirm your Jacobian factor is right by checking that $\int p_y = 1$ numerically.
6. **MLE $=$ min KL.** For a 1-D Gaussian model, compute $\mathrm{KL}(\hat p_N\|p_\theta)$ (via the cross-entropy term only) on a grid of $(\mu,\sigma)$ and confirm the minimizer coincides with the closed-form MLE.
7. **MI without correlation.** Construct $(X,Y)$ with $\rho \approx 0$ but $\mathbb{I}(X;Y)$ clearly positive (e.g. $Y = X^2 + $ noise, $X$ symmetric). Estimate both by binning. Comment on the sensitivity to bin count — this is the practical version of the caution in §8.

---

## 10. What you must know cold (quiz-eligible)

- The recipe $p(y|x,\theta) = \mathcal{D}(y|f(x;\theta))$ and the role of a link function.
- $\sigma$, its derivative, $1-\sigma(a)=\sigma(-a)$, logit as log-odds; softplus and $\sigma_+' = \sigma$.
- Binary cross-entropy in stable form $\mathrm{softplus}(a) - ya$, and why the naive form fails.
- Softmax: shift invariance, temperature, reduction to $\sigma$ at $C=2$; the log-sum-exp identity and why $m=\max_c a_c$.
- Gaussian NLL; MSE $\Leftrightarrow$ homoscedastic Gaussian likelihood; the heteroscedastic NLL and its two competing terms.
- Change of variables in 1-D and $\mathbb{R}^d$, including why $|\det J|$ appears.
- CLT statement with its hypotheses; $\bar X \pm 1.96\hat\sigma/\sqrt N$.
- Definitions of entropy, cross-entropy, KL, mutual information; $\mathrm{KL}\ge0$ with the Jensen proof; KL is not symmetric.
- MLE $=\arg\min \mathrm{KL}(\hat p_N\|p_\theta)$ — the derivation, not just the statement.
- Gaussian as maximum-entropy given mean and variance.

---

## 11. Practice Set 0.3 (ungraded; quiz-eligible)

1. Show that the convolution of two Gaussians is Gaussian with means and variances adding. *Murphy Ex. 2.4.*
2. Derive the normalization constant of the 1-D Gaussian by squaring the integral and changing to polar coordinates. Identify exactly where the Jacobian factor $r$ enters. *Murphy Ex. 2.12.*
3. Let $X\sim\mathrm{Ga}(a,b)$ and $Y = 1/X$. Derive the density of $Y$ using §5. *Murphy Ex. 2.7.*
4. Derive $\nabla_w$ of the logistic-regression NLL with $a = w^\top x$ and show it equals $\sum_n (\sigma(a_n) - y_n)x_n$. Compare its form to the gradient of the linear-regression NLL and explain why they look the same.
5. Prove $\mathbb{I}(X;Y) = \mathbb{H}[X] - \mathbb{H}[X|Y]$ directly from the definition of KL.
6. Show that among densities on $\mathbb{R}$ with fixed mean $\mu$ and variance $\sigma^2$, the Gaussian maximizes differential entropy. (Lagrange multipliers on $-\int p\log p$ with three constraints; or, more slickly, show $\mathrm{KL}(p\|\mathcal{N})\ge0$ and expand.)
7. A model reports 92% accuracy and a mean predictive standard deviation of 0.03. State two distinct things that could be badly wrong with it that neither number would reveal, and name the diagnostic you would run for each.
8. Take a 2-component Gaussian mixture as the target $p$ and a single Gaussian as $q$. Numerically minimize $\mathrm{KL}(p\|q)$ and $\mathrm{KL}(q\|p)$ over $(\mu,\sigma)$. Plot both solutions against $p$ and explain the difference in one sentence each.

---
