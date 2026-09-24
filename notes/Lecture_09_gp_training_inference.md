# Lecture 9 — Gaussian Processes: Training, Inference, and Practice

**ENM 5310 — Data-driven Modeling and Probabilistic Scientific Computing**

**Builds on:** Lecture 2 (Gaussian conditioning), Lecture 4 (maximum likelihood), Lecture 5 (the ways maximum likelihood fails), Lecture 6 (L-BFGS and gradient-based optimization), Lecture 7 (Bayesian linear regression), Lecture 8 §§1–5 (the failure of localized bases, weight space to function space, kernels, Mercer, and the GP model).

**Reading:** Rasmussen & Williams §§2.2–2.3, §5.1, §5.4 (freely available at gaussianprocess.org/gpml). Companion: Murphy I §17.2.

**Where we are.** Lecture 8 reached the end of Step 1 — the model. This lecture completes Steps 2 and 3, then covers what you need to actually use a GP on a real problem. Sections 1 through 3 finish the Lecture 8 notes; sections 4 and 5 are new.

**Learning objectives.** After this lecture you should be able to:

1. Explain why maximizing the likelihood over the function fails, why integrating it out fixes this, and why the result is an ordinary likelihood for the hyperparameters.
2. Write down the GP marginal likelihood, identify its data-fit and complexity terms, and explain why maximizing it prefers the simplest model that fits.
3. Fit hyperparameters by empirical Bayes, and say why you optimize logarithms and restart from several initializations.
4. Derive the GP predictive mean and variance by Gaussian conditioning, and explain why the variance reverts to the prior away from data and does not depend on $y$.
5. Implement GP training and inference with a Cholesky factorization.
6. Build a kernel for a structured problem by adding and multiplying simpler kernels, and use ARD to handle several input dimensions.
7. Diagnose a GP fit from its fitted hyperparameters.

### Notation

Training inputs $x_1,\dots,x_N$, targets $y\in\mathbb{R}^N$. Kernel $k_\theta(x,x')$ with hyperparameters $\theta$; Gram matrix $K$ with $K_{nm} = k(x_n,x_m)$; at a test input $x^\star$, the vector $k^\star$ with entries $k(x_n,x^\star)$ and the scalar $k^{\star\star} = k(x^\star,x^\star)$. Signal amplitude $\sigma_f^2$, length scale $\ell$, noise variance $\sigma^2$. Write $C_\theta = K_\theta+\sigma^2I$.

---

## 1. Recap: the model, and the three steps

Every method in this course has three steps: **model**, **training**, **inference**. Lecture 8 completed the first.

> **Step 1 — Model.** $\ f\sim\mathcal{GP}(0,k_\theta)$, and $\ y_n = f(x_n)+\epsilon_n$ with $\epsilon_n\sim\mathcal{N}(0,\sigma^2)$.

At the training inputs this is two lines. Writing $f = (f(x_1),\dots,f(x_N))^\top$,

$$\text{prior:}\quad f\sim\mathcal{N}(0,K_\theta), \qquad\qquad \text{likelihood:}\quad y\mid f\sim\mathcal{N}(f,\sigma^2 I).$$

There are no weights. The unknowns are the **function** $f$ and the **hyperparameters** $\theta = (\sigma_f^2,\ell,\sigma^2)$.

Three facts from Lecture 8 that the rest of this lecture uses. A Gaussian prior on weights is a Gaussian prior on function values, with covariance $k(x,x') = \tau^2\phi(x)^\top\phi(x')$ — so the kernel *is* the prior. A kernel is valid when every Gram matrix it produces is positive semi-definite, and Mercer's theorem says every valid kernel is an inner product of features. And a **stationary** kernel, one depending only on $x-x'$, has constant prior variance $k(x,x) = \sigma_f^2$, which is what prevents the confident-and-wrong extrapolation that sank the fixed RBF basis.

---

## 2. Step 2 — Training: the marginal likelihood

### 2.1 What is there to learn, and how?

Two kinds of unknown came out of Step 1, and they get different treatment.

- **The function $f$.** Infinitely many values, of which the data touches $N$. Following Lecture 7, we keep a *full distribution* over it and never commit to one function. That happens in Step 3.
- **The hyperparameters $\theta$.** Three numbers describing the *kind* of function — its amplitude, its smoothness, and how noisy the observations are. These are few, and the data typically pins them down, so we choose single best values in the spirit of Lecture 4.

So Step 2 reduces to: **find good values of $\theta$.** The natural first attempt is the principle from Lecture 4 — maximize the likelihood.

### 2.2 Why not maximize the likelihood over everything?

Try maximizing jointly over the function and the noise:

$$p(y\mid f,\sigma^2) = \prod_{n=1}^{N}\mathcal{N}\big(y_n\,\big|\,f(x_n),\,\sigma^2\big).$$

The best function is immediate: set $f(x_n) = y_n$ at every training input, making every residual zero. The likelihood is then

$$p(y\mid f,\sigma^2) = (2\pi\sigma^2)^{-N/2} \;\longrightarrow\; \infty \qquad\text{as } \sigma^2\to0 .$$

The maximum does not exist. This is exactly the collapsing-variance degeneracy of Lecture 5: the model "explains" the data by memorizing it and then declaring there was no noise at all. Any hyperparameters fitted this way are meaningless.

The lesson is not that the objective was wrong but that **the function is far too flexible an object to choose by maximization.** A GP has, in effect, infinitely many parameters. Optimizing them all will always produce interpolation and zero noise.

### 2.3 The fix: average over the function instead of maximizing over it

Replace the question

> *How well does the **best** function explain the data?*

with

> *How well does a **typical** function drawn from my prior explain the data, on average?*

That is, average the likelihood over the prior rather than maximizing it:

$$\boxed{\ p(y\mid\theta) \;=\; \int p(y\mid f,\sigma^2)\;p(f\mid\theta)\;df .\ }$$

This is the **marginal likelihood**, or the **evidence**. "Marginal" because $f$ has been marginalized — integrated out — rather than optimized.

Two properties make it the right training objective.

**It cannot memorize.** A single perfectly-fitting function no longer wins the day, because it is one function among all those the prior allows, weighted by how plausible the prior considers it. No amount of flexibility lets a model cheat an average.

**It is an ordinary likelihood.** It is the probability of the observed data as a function of the remaining unknowns $\theta$ — precisely the kind of object Lecture 4 maximized. So maximizing it is simply maximum likelihood, applied one level up. The name for this is **type-II maximum likelihood**, or **empirical Bayes**.

In one line: **Bayes for the function, maximum likelihood for the hyperparameters.**

### 2.4 Computing it

The integral looks forbidding and is not. From Step 1, at the training inputs,

$$y = f+\epsilon, \qquad f\sim\mathcal{N}(0,K_\theta), \qquad \epsilon\sim\mathcal{N}(0,\sigma^2 I), \qquad f\perp\epsilon .$$

A sum of independent Gaussians is Gaussian, with covariances adding. So without doing any integral at all,

$$\boxed{\ p(y\mid\theta) = \mathcal{N}\big(y\,\big|\,0,\ C_\theta\big), \qquad C_\theta = K_\theta+\sigma^2 I .\ }$$

Taking logarithms,

$$\log p(y\mid\theta) \;=\; \underbrace{-\tfrac12\,y^\top C_\theta^{-1}y}_{\text{data fit}}\;\underbrace{-\;\tfrac12\log\big|C_\theta\big|}_{\text{complexity penalty}}\;-\;\tfrac N2\log2\pi .$$

### 2.5 Why this picks sensible hyperparameters

Go back to the question in §2.3 — how well does a *typical* prior function explain the data? — and run it for three length scales.

- **Length scale too short.** Prior draws wiggle violently. Almost every particular draw wiggles the wrong way at most of the data points. Averaged over draws: a poor fit.
- **Length scale too long.** Prior draws are nearly flat. If the data vary at all, almost every draw misses them. Averaged: also a poor fit.
- **Length scale about right.** A good fraction of prior draws follow the general shape of the data. Averaged: a good fit.

The marginal likelihood is largest in the third case. It selects **the simplest model that still explains the data** — Occam's razor — and nobody added a complexity penalty by hand. The penalty is a consequence of averaging.

The same idea explains the two terms in the formula. $p(y\mid\theta)$ is a probability distribution over *all possible datasets* $y\in\mathbb{R}^N$, so it integrates to one. A flexible model can produce a huge variety of datasets, spreads its probability thinly over them, and therefore assigns only a little to the one actually observed: that shows up as a large $\log|C_\theta|$, penalized. A rigid model concentrates probability on a narrow set of smooth datasets; if the observed data is not among them, $y^\top C_\theta^{-1}y$ is large, penalized. The maximum balances the two.

### 2.6 The same recipe closes a gap from Lecture 7

Lecture 7 left $\sigma^2$ and $\tau^2$ unexplained: the posterior formulas assumed them known. Bayesian linear regression is a GP with the linear kernel $k = \tau^2\phi^\top\phi'$, whose Gram matrix is $K = \tau^2\Phi\Phi^\top$, so its marginal likelihood is

$$p(y\mid\sigma^2,\tau^2) = \mathcal{N}\big(y\,\big|\,0,\ \tau^2\Phi\Phi^\top+\sigma^2 I\big).$$

Maximizing over $(\sigma^2,\tau^2)$ chooses them both. Since the ridge penalty is $\lambda = \sigma^2/\tau^2$, this also chooses the ridge penalty — with no cross-validation and no data held out.

### 2.7 Doing the optimization

Training is now an optimization problem, and Lecture 6 supplies the tools.

**Optimize the logarithms.** All three hyperparameters are positive, and they often differ by orders of magnitude. Optimizing $(\log\sigma_f^2,\log\ell,\log\sigma^2)$ makes the problem unconstrained and much better behaved.

**Use L-BFGS or gradient descent with the exact gradient.** With $\alpha = C_\theta^{-1}y$,

$$\frac{\partial \log p(y\mid\theta)}{\partial\theta_j} \;=\; \frac12\operatorname{tr}\!\Big(\big(\alpha\alpha^\top - C_\theta^{-1}\big)\frac{\partial C_\theta}{\partial\theta_j}\Big),$$

which needs only $\partial C_\theta/\partial\theta_j$, an elementwise derivative of the kernel. Check it against finite differences before trusting it.

**Restart from several initializations.** The marginal likelihood is not concave, and a specific pair of competing explanations shows up repeatedly: a **short length scale with small noise**, in which the function wiggles through every observation, and a **long length scale with large noise**, in which a smooth trend is seen through heavy scatter. Both can be local maxima of the same objective on the same data. Compare their marginal likelihoods, and look at both fits — occasionally the data genuinely cannot decide, and that is worth knowing rather than hiding.

**Two cautions.** This is maximum likelihood, so with many hyperparameters it can overfit. And the evidence only compares the kernels you offered it: it cannot tell you that all of them are wrong. It does not replace a held-out test set.

### 2.8 What training leaves behind

Never compute $C_\theta^{-1}$ explicitly. Factor once, and keep the pieces for Step 3.

> ### Algorithm: GP training
>
> **Input:** inputs $x_{1:N}$, targets $y$, kernel family, initial $\theta$.
>
> 1. Maximize $\log p(y\mid\theta)$ over $\log\theta$, evaluating it as in step 4 below.
> 2. At the optimum $\hat\theta$: form $C = K_{\hat\theta}+\hat\sigma^2 I$ and compute its **Cholesky factor** $L$, lower triangular with $C = LL^\top$.
> 3. Solve $\alpha = L^{-\top}\big(L^{-1}y\big)$ by two triangular solves.
> 4. The log marginal likelihood is $\ -\tfrac12 y^\top\alpha-\sum_n\log L_{nn}-\tfrac N2\log2\pi$, using $\log|C| = 2\sum_n\log L_{nn}$.
> 5. **Store $\hat\theta$, $L$, and $\alpha$.**

**Cost.** The Cholesky factorization is $O(N^3)$ and dominates everything. This is what limits exact GPs to roughly tens of thousands of data points.

**Jitter.** When $\sigma^2$ is very small, $K$ is often numerically singular — nearby inputs give nearly identical rows — and the factorization fails. Add a small constant, around $10^{-6}\sigma_f^2$, to the diagonal. This is a numerical necessity, not a modeling choice, and should be reported.

---

## 3. Step 3 — Inference: predictions at test points

### 3.1 The joint, and the conditional

We want the distribution of $f^\star = f(x^\star)$ given the data. Under the model, the training targets and the test function value are jointly Gaussian; their covariances come straight from the kernel, with noise on the training targets only:

$$\begin{pmatrix}y\\f^\star\end{pmatrix}\sim\mathcal{N}\left(0,\ \begin{pmatrix}K+\sigma^2 I & k^\star\\ k^{\star\top} & k^{\star\star}\end{pmatrix}\right).$$

Apply the Gaussian conditioning formula from Lecture 2 — for zero-mean jointly Gaussian $(z_1,z_2)$,
$z_1\mid z_2\sim\mathcal{N}\big(\Sigma_{12}\Sigma_{22}^{-1}z_2,\ \Sigma_{11}-\Sigma_{12}\Sigma_{22}^{-1}\Sigma_{21}\big)$ — with $z_1 = f^\star$ and $z_2 = y$:

$$\boxed{\ f^\star\mid\mathcal{D}\sim\mathcal{N}(\mu^\star,v^\star), \quad \mu^\star = k^{\star\top}\big(K+\sigma^2I\big)^{-1}y, \quad v^\star = k^{\star\star}-k^{\star\top}\big(K+\sigma^2I\big)^{-1}k^\star .\ }$$

For a new **noisy observation**, add the noise back: $y^\star\mid\mathcal{D}\sim\mathcal{N}\big(\mu^\star,\ v^\star+\sigma^2\big)$.

That is the whole of GP prediction. No weights, no optimization, no step size — just the conditional of a joint Gaussian.

For several test inputs at once, replace $k^\star$ by the $N\times M$ matrix $K_\star$ and $k^{\star\star}$ by the $M\times M$ matrix $K_{\star\star}$; the mean becomes a vector and the variance a full covariance $K_{\star\star}-K_\star^\top(K+\sigma^2I)^{-1}K_\star$, from which joint samples are drawn by Cholesky exactly as prior samples were.

### 3.2 Reading the mean

Using $\alpha = (K+\sigma^2I)^{-1}y$ stored at training time,

$$\mu^\star = \sum_{n=1}^{N}\alpha_n\,k(x_n,x^\star).$$

**The prediction is a weighted sum of kernel functions, one centered at each training input**, with weights fixed during training. The model began with infinitely many features, and its predictions live in an $N$-dimensional space determined by the data. Models whose effective size grows with the data are called **nonparametric**.

With the linear kernel this equals Lecture 7's posterior mean $m_N^\top\phi^\star$, which equals the ridge estimate of Lecture 5. The same prediction has now appeared three times under three names.

### 3.3 Reading the variance

$$v^\star = \underbrace{k^{\star\star}}_{\text{prior variance}}\;-\;\underbrace{k^{\star\top}\big(K+\sigma^2I\big)^{-1}k^\star}_{\text{removed by the data}\ \ge\ 0}.$$

Three consequences, all of which matter for the next three lectures.

**It reverts to the prior far from the data.** With a stationary kernel, $k(x_n,x^\star)\to0$ as $x^\star$ moves away from every training input, so $k^\star\to0$ and

$$\mu^\star\to0, \qquad v^\star\to k^{\star\star} = \sigma_f^2 .$$

Compare the failure that opened Lecture 8. The mean still returns to zero — the prior mean, which is fine. What changed is the variance: the model now reports the full prior uncertainty rather than claiming certainty. It says *I don't know*. That repair came from the prior, not from the data.

**It does not depend on $y$.** Only input locations appear in $v^\star$. You can compute where the model will be uncertain **before taking any measurements** — which is exactly what Bayesian optimization and experimental design exploit.

**It splits the two uncertainties.** For a new observation,

$$\operatorname{Var}[y^\star\mid\mathcal{D}] = \underbrace{\sigma^2}_{\text{aleatoric}}+\underbrace{v^\star}_{\text{epistemic}} .$$

The aleatoric part is constant and irreducible; the epistemic part is small near data, grows between and beyond it, and tends to $\sigma_f^2$ far away. Where data are dense, $v^\star\to0$ and only $\sigma^2$ remains: **you can learn the function exactly and still not predict the next measurement exactly.**

### 3.4 Computing it, and plotting it

> ### Algorithm: GP inference
>
> **Input:** stored $\hat\theta$, $L$, $\alpha$; a test input $x^\star$.
>
> 1. Compute $k^\star$ and $k^{\star\star}$.
> 2. **Mean:** $\mu^\star = k^{\star\top}\alpha$.
> 3. Solve $v = L^{-1}k^\star$. **Variance:** $v^\star = k^{\star\star}-v^\top v$.
> 4. For a noisy observation, add $\hat\sigma^2$.

After training, each mean costs $O(N)$ and each variance $O(N^2)$; the $O(N^3)$ was paid once.

**Plot functions, not only bands.** A band $\mu^\star\pm2\sqrt{v^\star}$ shows the uncertainty at each input separately and conceals how values at different inputs move together. Two models with nearly identical bands — a Matérn-1/2 and a squared-exponential GP on the same data — believe in completely different functions, one jagged and one smooth. Always overlay a handful of joint posterior samples using the multi-point form of §3.1.

---

## 4. Kernel design in practice

The kernel is the model. Everything you believe about the function is expressed there, so it is worth knowing how to build one for a problem with structure.

### 4.1 Composition: adding and multiplying kernels

Sums and products of valid kernels are valid (Lecture 8 §4.1), and each has a reading.

**A sum is an "or".** If $f_1\sim\mathcal{GP}(0,k_1)$ and $f_2\sim\mathcal{GP}(0,k_2)$ independently, then $f_1+f_2\sim\mathcal{GP}(0,k_1+k_2)$. Use a sum when the function is a superposition of distinct behaviors on different scales: a slow trend plus short-term fluctuation is $k_{\mathrm{SE}}(\ell_{\text{long}})+k_{\mathrm{SE}}(\ell_{\text{short}})$, and each component gets its own amplitude and length scale, all fitted by the marginal likelihood.

**A product is an "and".** $k_1k_2$ is large only where *both* factors are large, so it expresses a conjunction of conditions. The standard use is to localize a global structure: $k_{\mathrm{per}}\times k_{\mathrm{SE}}$ is *locally* periodic — a cycle whose shape drifts slowly over a length scale set by the SE factor, rather than repeating exactly forever.

The canonical worked example is atmospheric CO$_2$: a long-term trend ($k_{\mathrm{SE}}$, large $\ell$), plus a seasonal cycle that slowly changes shape ($k_{\mathrm{per}}\times k_{\mathrm{SE}}$), plus medium-scale irregularities, plus noise. Each term is a sentence about the physics, and each of its hyperparameters is a quantity you can read off after fitting and check against what you know.

**This is the payoff of the function-space view.** Model building becomes a matter of composing statements about covariance, rather than guessing a basis. And every constituent is fitted by the same Step 2.

### 4.2 Several input dimensions: ARD

For $x\in\mathbb{R}^d$, the natural generalization gives each input dimension its own length scale:

$$k(x,x') = \sigma_f^2\exp\left(-\frac12\sum_{i=1}^{d}\frac{(x_i-x_i')^2}{\ell_i^2}\right).$$

Fitting the $\ell_i$ by marginal likelihood does something useful automatically. If the function barely varies along dimension $i$, the evidence is maximized by making $\ell_i$ large — so large that the kernel becomes effectively constant in that coordinate and the dimension is ignored. Small $\ell_i$ means that input matters a great deal.

> **Automatic relevance determination (ARD).** Fitted inverse length scales $1/\ell_i$ rank the input dimensions by how much the function depends on them.

Two practical notes. **Standardize your inputs** before fitting, so that the $\ell_i$ start on a comparable footing and the optimization is well behaved; standardize or center $y$ too, which is part of what justifies the zero mean function. And ARD adds $d$ hyperparameters, so in high dimensions the caution of §2.7 about overfitting the evidence becomes real.

### 4.3 Mean functions

Nothing required $m(x) = 0$. With a general mean function the predictive becomes

$$f^\star\mid\mathcal{D}\sim\mathcal{N}\Big(m(x^\star)+k^{\star\top}\big(K+\sigma^2I\big)^{-1}\big(y-m(X)\big),\ \ v^\star\Big),$$

with the variance unchanged. Since a stationary GP reverts to its mean far from data (§3.3), the mean function is exactly the knob controlling extrapolation. If you know the function should grow linearly, or should approach a known physical asymptote, put that in $m$ and let the GP model the deviation from it. A zero mean is a claim, not a default — it is only innocuous after you have centered $y$.

### 4.4 Noise: the other half of the model

$\sigma^2$ is a modeling assumption like everything else.

**Deterministic simulators.** If the data come from a code that returns the same output for the same input, there is no observation noise: take $\sigma^2\to0$, so that the posterior mean interpolates the data exactly and $v^\star = 0$ at every training input. You will then certainly need jitter. This is the standard setting for surrogate modeling, and it is where the next lecture begins.

**Input-dependent noise.** If the noise level varies with $x$ and you know the variances, replace $\sigma^2 I$ by $\operatorname{diag}(\sigma_1^2,\dots,\sigma_N^2)$; every formula above is unchanged. Learning an unknown noise function is harder and takes us outside the exact GP.

---

## 5. What goes wrong, and how to recognize it

The fitted hyperparameters are interpretable, so read them before looking at any plot.

**$\hat\sigma^2$ has absorbed everything.** The fitted noise is close to the variance of $y$, and $\hat\ell$ is huge. The model has concluded the data is pure noise around a constant. Either the kernel cannot express the structure that is there, or the optimizer landed in the long-length-scale local maximum (§2.7). Restart, and check whether a different kernel is needed.

**$\hat\ell$ is tiny and $\hat\sigma^2$ near zero.** The posterior interpolates every point and reverts to the prior between them: the model has memorized rather than generalized. This is the other local maximum. It is also what you get if the data really is that rough, so compare marginal likelihoods and look at the samples before deciding.

**The Cholesky factorization keeps failing.** Add jitter first. If it persists, look for duplicated or nearly duplicated inputs, which make rows of $K$ identical.

**Extrapolation is useless.** Away from the data a stationary GP returns the mean function with full prior variance. That is honest, and if you need something better the fix is §4.3's mean function or a non-stationary kernel — not more data of the same kind.

**High-dimensional inputs.** In high dimensions, distances between random points concentrate: everything is roughly equidistant from everything else, so $k^\star$ is nearly constant and the posterior barely differs from the prior anywhere. Stationary GPs with ARD work well up to tens of input dimensions, not thousands.

**$N$ is too large.** The $O(N^3)$ factorization is the binding constraint past roughly $10^4$ points, at which stage you need sparse or approximate methods.

---

## 6. The GP in three steps, and where this goes

| | Gaussian process regression |
|---|---|
| **Step 1 — Model** | $f\sim\mathcal{GP}(m,k_\theta)$, $\ y_n = f(x_n)+\epsilon_n$, $\ \epsilon_n\sim\mathcal{N}(0,\sigma^2)$ |
| **Step 2 — Training** | $\hat\theta = \arg\max_\theta\log\mathcal{N}\big(y\mid0,\ K_\theta+\sigma^2I\big)$; store $L$ and $\alpha$ |
| **Step 3 — Inference** | $\mu^\star = k^{\star\top}\alpha$, $\ v^\star = k^{\star\star}-\|L^{-1}k^\star\|^2$; add $\sigma^2$ for a new observation |

Set this beside Lecture 7. The model moved from a prior on weights to a prior on functions. Training moved from "compute the posterior, assuming $\sigma^2$ and $\tau^2$ are known" to "fit the hyperparameters by maximum marginal likelihood." And inference now produces uncertainty that is honest everywhere, because the prior never claimed certainty anywhere.

**Where this goes.** The next three lectures consume exactly the two numbers $\mu^\star$ and $v^\star$.

- **Multi-fidelity modeling.** When a cheap approximate simulator and an expensive accurate one are both available, a GP can be built whose kernel couples them, so that many cheap runs sharpen predictions about the expensive one.
- **Bayesian optimization.** To minimize an expensive function, choose each next evaluation by trading a promising $\mu^\star$ against a large $v^\star$.
- **Active learning and experimental design.** Choose measurements to reduce $v^\star$ where it matters — possible precisely because $v^\star$ does not depend on $y$.

---

## Checkpoint E (take-home)

This is the complete version of the assignment begun in the Lecture 8 notes; items 6 and 7 are new. `numpy` only, plus `scipy.optimize` for the optimizer in item 3. Report fitted numbers, not pictures that look about right.

1. **Warm-up: the failure.** Fit Bayesian linear regression with $D = 15$ RBF features of width $\lambda = 0.2$ centered on a grid over $[-1,1]$, to 20 noisy observations of a smooth function on that interval. At $x^\star\in\{0,2,5\}$ report the predictive mean, the epistemic variance $\phi^{\star\top}S_N\phi^\star$, and the prior variance $\tau^2\|\phi^\star\|^2$. Confirm the epistemic variance never exceeds the prior variance, and that both vanish at $x^\star = 5$.
2. **Step 1 — the model.** Draw five prior samples on a 400-point grid for the squared-exponential, Matérn-1/2, and Matérn-5/2 kernels, each at $\ell\in\{0.1,0.5,2\}$. State the jitter each needed and why the squared-exponential needed the most.
3. **Step 2 — training.** For the data of item 1 with a Matérn-5/2 kernel: (a) fix $\sigma_f^2$, evaluate the log marginal likelihood on a grid over $(\log\ell,\log\sigma^2)$, plot its contours, and report the maximizer; (b) implement the gradient of §2.7 and check it against finite differences; (c) optimize all three log-hyperparameters from ten random starts, reporting each distinct local maximum, its value, and in one sentence what explanation of the data it represents.
4. **Step 3 — inference.** Implement §3.4 with the trained model. At $x^\star\in\{0,2,5\}$ report $\mu^\star$, $v^\star$, and $v^\star+\hat\sigma^2$; confirm $v^\star\to\hat\sigma_f^2$ at $x^\star = 5$. Plot the mean, both bands, and five posterior samples.
5. **GPs contain Lecture 7.** With the linear kernel $k = \tau^2\phi^\top\phi'$ using item 1's features and $\sigma^2,\tau^2$, confirm your GP predictions match item 1's Bayesian linear regression to machine precision.
6. **Composition.** Generate data from a linear trend plus a periodic component plus noise. Fit three GPs — $k_{\mathrm{SE}}$ alone, $k_{\mathrm{per}}$ alone, and $k_{\mathrm{lin}}+k_{\mathrm{per}}$ — and report the optimized log marginal likelihood of each along with the fitted hyperparameters. Do the fitted period and trend match the values you generated? Plot the extrapolation of each model one full period beyond the data and comment.
7. **ARD.** Generate data on $x\in\mathbb{R}^4$ from a function depending strongly on $x_1$, weakly on $x_2$, and not at all on $x_3,x_4$. Fit an ARD squared-exponential and report the four fitted length scales. Do they rank the inputs correctly? Repeat with $N$ reduced by a factor of four and report how the ranking degrades.

---

## Quiz-eligible facts

1. Maximizing the likelihood jointly over the function and the noise fails: set $f(x_n) = y_n$, let $\sigma^2\to0$, and the likelihood diverges. A GP has too many effective parameters to fit by maximization.
2. The marginal likelihood $p(y\mid\theta) = \int p(y\mid f)p(f\mid\theta)df$ averages the fit over prior functions rather than maximizing it. It is an ordinary likelihood for $\theta$: Bayes for the function, maximum likelihood for the hyperparameters.
3. $p(y\mid\theta) = \mathcal{N}(0,K_\theta+\sigma^2I)$, because $y = f+\epsilon$ is a sum of independent Gaussians. Its log is a data-fit term $-\frac12y^\top C^{-1}y$ plus a complexity term $-\frac12\log|C|$.
4. The marginal likelihood prefers the simplest model that explains the data, because it is normalized over all possible datasets: a flexible model spreads probability thinly and gives little to the one observed.
5. The same objective applied to Bayesian linear regression chooses $\sigma^2$ and $\tau^2$, hence the ridge penalty $\lambda = \sigma^2/\tau^2$, without cross-validation.
6. Optimize log-hyperparameters with the analytic gradient $\frac12\operatorname{tr}\big((\alpha\alpha^\top-C^{-1})\partial C/\partial\theta_j\big)$, and restart: short-$\ell$/low-noise and long-$\ell$/high-noise are competing local maxima.
7. The evidence does not replace a test set: it ranks the kernels offered, and cannot detect that all are wrong.
8. GP predictive: $\mu^\star = k^{\star\top}(K+\sigma^2I)^{-1}y$ and $v^\star = k^{\star\star}-k^{\star\top}(K+\sigma^2I)^{-1}k^\star$, by conditioning the joint Gaussian of $(y,f^\star)$; add $\sigma^2$ for a noisy observation.
9. The predictive mean is a weighted sum of kernel functions centered at the training inputs — a nonparametric model whose effective size grows with $N$.
10. For a stationary kernel, far from the data $\mu^\star$ returns to the mean function and $v^\star\to\sigma_f^2$: the model reports full prior uncertainty.
11. The predictive variance does not depend on $y$, so uncertainty can be computed before measuring — the basis of experimental design and Bayesian optimization.
12. Train and predict by Cholesky factorization of $K+\sigma^2I$, never by explicit inversion; $\log|C| = 2\sum_n\log L_{nn}$. Training is $O(N^3)$; afterwards each mean is $O(N)$ and each variance $O(N^2)$. Add jitter when $K$ is near-singular.
13. A sum of kernels is an "or" — independent additive components; a product is an "and" — both factors must be large. Locally periodic behavior is $k_{\mathrm{per}}\times k_{\mathrm{SE}}$.
14. ARD gives each input dimension its own length scale; large fitted $\ell_i$ means dimension $i$ is irrelevant. Standardize inputs before fitting.
15. A mean function controls extrapolation, since a stationary GP reverts to it away from data; the predictive variance is unchanged by the choice of mean.
16. For a deterministic simulator take $\sigma^2\to0$: the posterior interpolates and jitter becomes necessary.
17. Diagnostics: $\hat\sigma^2$ near $\operatorname{Var}[y]$ with huge $\hat\ell$ means the model found only noise; tiny $\hat\ell$ with tiny $\hat\sigma^2$ means it memorized. In high dimensions distances concentrate and the posterior barely departs from the prior.

---

## Practice problems

*Five questions in the style of the quizzes: each should take two or three minutes, with no computer. The answer follows each one — cover it and try first. Derivations and implementation are in Checkpoint E.*

**P1.** A classmate proposes: "To train a GP, first find the function that best fits the data, then pick the $\sigma^2$ that makes that fit most likely." What goes wrong, in one sentence?

*Answer.* Setting $f(x_n)=y_n$ makes every residual zero, so the likelihood $(2\pi\sigma^2)^{-N/2}$ diverges as $\sigma^2\to0$ and no maximum exists — the function is too flexible to be chosen by maximization, which is why training averages over $f$ instead. (§2.2–2.3)

**P2.** Two GPs are fitted to data from a smooth function: one with $\ell = 0.05$, one with $\ell = 1$. Both can pass through every data point. Which has the higher marginal likelihood, and why?

*Answer.* The one with $\ell = 1$. The marginal likelihood is a probability distribution over all possible datasets, so the short-length-scale model must spread its probability across an enormous variety of wiggly datasets and assigns little to the smooth one observed. Being *able* to fit the data is not the criterion; being *likely* to produce it is. (§2.5)

**P3.** (a) You keep the same training inputs but replace every target $y_n$ with zero, and predict using the same hyperparameters. Does the predictive variance $v^\star$ change? (b) A test point $x^\star$ lies far from all training data, with a stationary kernel. What are $\mu^\star$, $v^\star$, and $\operatorname{Var}[y^\star]$?

*Answer.* (a) No — $v^\star = k^{\star\star}-k^{\star\top}(K+\sigma^2I)^{-1}k^\star$ involves only input locations. (Retraining would change the fitted hyperparameters, which is a different matter.) (b) $k^\star\to0$, so $\mu^\star\to m(x^\star)$, which is $0$ after centering; $v^\star\to\sigma_f^2$; and $\operatorname{Var}[y^\star]\to\sigma_f^2+\sigma^2$. The model reverts to its prior and says *I don't know* — unlike the fixed RBF basis of Lecture 8, where both the mean and the variance collapsed to zero. (§3.3)

**P4.** You believe your signal is a slow trend with fast fluctuations riding on it. Do you use $k_{\mathrm{SE}}(\ell=5)+k_{\mathrm{SE}}(\ell=0.2)$ or the product of the two? What would the other choice actually model?

*Answer.* The sum, which corresponds to adding two independent GPs — exactly a slow component plus a fast one. The product of two squared-exponentials is again a squared-exponential, with $\ell_{\text{eff}}^{-2} = \ell_1^{-2}+\ell_2^{-2}$, so $\ell_{\text{eff}}<0.2$: the product gives fast fluctuations only, and the trend disappears. A sum is an "or"; a product is an "and". (§4.1)

**P5.** You fit an ARD squared-exponential to data with inputs $x\in\mathbb{R}^3$ and obtain $\ell_1 = 0.4$, $\ell_2 = 0.9$, $\ell_3 = 800$. What does this tell you, and what should you have done to the inputs before fitting?

*Answer.* The function depends most strongly on $x_1$, somewhat on $x_2$, and effectively not at all on $x_3$: a very large length scale makes the kernel nearly constant along that coordinate. Inputs should be standardized first, otherwise the length scales are reported in whatever units each column happened to use and cannot be compared. (§4.2)
