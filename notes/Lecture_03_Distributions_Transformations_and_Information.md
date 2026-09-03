# ENM 5310 — Lecture 3
## Common Distributions, the Gaussian Family, Transformations, and Information

**Reading:** Murphy, *PML: An Introduction*, §2.4–2.8, §3.1–3.2; §6.1–6.3 for the information-theoretic material.
**Builds on:** Lectures 1–2 (laws, densities, expectation, covariance, conditioning, Bayes).

**Notation for today.** $q(x)$ denotes the distribution that actually generated the data (written $p^\star$ in Lecture 1), $\hat q_N$ its empirical version, and $p_\theta(x)$ our model. The goal of the second half of the lecture is to say what $p_\theta \approx q$ should mean.

**Learning objectives.** You should be able to (i) pick an output distribution from the type and support of the data, and name the link function that maps unconstrained network outputs to its parameters; (ii) write numerically stable forms of the associated log-likelihoods; (iii) state the closure properties of the Gaussian family and use them; (iv) apply the change-of-variables formula in $\mathbb{R}^d$ and explain why the log-determinant appears; (v) explain why comparing distributions calls for a divergence rather than a distance, and define entropy, conditional entropy, cross-entropy, KL, and mutual information with correct intuitions attached.

---

## 1. Choosing a distribution means choosing a data type

Before any architecture question, a modeling question: what kind of object is the thing you are predicting? Binary or not; one of $C$ labels; a count; a real number; a positive real; a vector; a probability vector; an angle. Each type comes with a **support**, and the support rules out most distributions immediately.

The remaining choice is about **tails and shape**, and that is where the real modeling content lives. A Gaussian and a Student-$t$ have the same support and the same first two moments; they differ only in how much mass they place far from the center, and that difference decides whether one bad measurement moves your fit.

Throughout, the pattern for supervised models is the same:

$$p(y\mid x,\theta) = \mathcal{D}\big(y \;\big|\; f(x;\theta)\big),$$

a distribution family $\mathcal{D}$ chosen for the type of $y$, whose *parameters* are predicted by a function $f$. Since networks emit arbitrary reals and distribution parameters are constrained, each family needs a **link function** to bridge them. The rest of this lecture's first half is that table, filled in.

---

## 2. Bernoulli and binomial

**The distribution.** For $y\in\{0,1\}$ with success probability $\theta$:

$$\mathrm{Ber}(y\mid\theta) = \theta^{y}(1-\theta)^{1-y}, \qquad \mathbb{E}[y] = \theta, \qquad \mathbb{V}[y] = \theta(1-\theta).$$

Note the variance is *determined by* the mean and is maximal at $\theta = 1/2$ — a first instance of a general fact: outside the Gaussian, mean and variance are usually tied together, not free.

Summing $N$ independent trials with the same $\theta$ gives the **binomial**, $\mathrm{Bin}(s\mid N,\theta) = \binom{N}{s}\theta^s(1-\theta)^{N-s}$, with mean $N\theta$ and variance $N\theta(1-\theta)$. Use Bernoulli when you model each trial, binomial when you only observe the count out of a known number of trials (12 of 40 specimens failed).

**The link.** We need a parameter in $[0,1]$ from an unconstrained model output $a = f(x;\theta)$, the **logit**:

$$\sigma(a) \triangleq \frac{1}{1+e^{-a}} = \frac{e^a}{1+e^a}, \qquad p(y\mid x,\theta) = \mathrm{Ber}\big(y\mid\sigma(a)\big).$$

Properties to know without lookup (derive $\sigma'$ on the board; it is one line and it explains why the algebra is so pleasant):

$$\sigma'(a) = \sigma(a)\big(1-\sigma(a)\big), \qquad 1-\sigma(a) = \sigma(-a), \qquad \sigma^{-1}(p) = \log\frac{p}{1-p} \triangleq \mathrm{logit}(p),$$
$$\mathrm{softplus}(a) \triangleq \log(1+e^{a}), \qquad \frac{d}{da}\mathrm{softplus}(a) = \sigma(a).$$

**Why the logit is the natural coordinate.** The quantity $a$ *is* the log-odds:

$$\log\frac{p}{1-p} = \log\left(\frac{e^a}{1+e^a}\cdot\frac{1+e^a}{1}\right) = a.$$

Probabilities live on a bounded interval and combine multiplicatively; log-odds live on the whole real line and combine additively. That is why models are built in logit space. A linear model $a = w^\top x + b$ says the log-odds are linear in the features, so $w_j$ is the change in log-odds per unit $x_j$ — that is **logistic regression**, and the decision boundary is the level set $a(x)=0$.

**The log-likelihood, written so it does not overflow.** For one example,

$$\ell(a,y) = -\big[y\log\sigma(a) + (1-y)\log(1-\sigma(a))\big].$$

Computed naively — form $\sigma(a)$, then take its log — this returns `inf` for $|a| \gtrsim 40$ in float32. Use $-\log\sigma(a) = \mathrm{softplus}(-a)$ and $-\log(1-\sigma(a)) = \mathrm{softplus}(a)$:

$$\boxed{\ \ell(a,y) = \mathrm{softplus}(a) - y\,a\ }\qquad\text{with}\qquad \mathrm{softplus}(a) = \max(a,0) + \log\big(1+e^{-|a|}\big).$$

This is what `binary_cross_entropy_with_logits` does internally, and it is why you pass logits, never probabilities, to a loss function. **You will implement this from these equations rather than import it.** Deriving stable forms of standard losses recurs all semester, because the gap between a model that trains and one that produces NaNs at epoch 3 is usually two lines of this kind.

---

## 3. Categorical and multinomial

**The distribution.** For $y\in\{1,\dots,C\}$, with one-hot encoding $y_c$:

$$\mathrm{Cat}(y\mid\theta) = \prod_{c=1}^C \theta_c^{\,y_c}, \qquad \theta_c \ge 0, \quad \sum_c \theta_c = 1,$$

so there are $C-1$ free parameters. Counting outcomes over $N$ independent trials gives the **multinomial** with coefficient $N!/(N_1!\cdots N_C!)$. Bernoulli is the $C=2$ case; binomial is the $C=2$ multinomial.

**The link.** From logits $a\in\mathbb{R}^C$,

$$\mathrm{softmax}(a)_c = \frac{e^{a_c}}{\sum_{c'}e^{a_{c'}}}.$$

Three properties worth stating explicitly:

- **Shift invariance:** $\mathrm{softmax}(a+\kappa\mathbf{1}) = \mathrm{softmax}(a)$. The parameterization has one redundant degree of freedom per example, so logits are not identifiable and their absolute magnitudes mean nothing.
- **Temperature:** $\mathrm{softmax}(a/T)$ interpolates between uniform ($T\to\infty$) and one-hot argmax ($T\to0$). The same knob reappears as sampling temperature in generative models, in distillation, and as a post-hoc calibration parameter.
- **Reduction to the sigmoid:** at $C=2$, $\mathrm{softmax}(a)_0 = \sigma(a_0-a_1)$. Only differences matter, consistent with shift invariance.

**Log-sum-exp.** With $\mathrm{lse}(a) \triangleq \log\sum_c e^{a_c}$ we have $\log p(y=c\mid x) = a_c - \mathrm{lse}(a)$. Naively, $a = (1000,1001,1000)$ overflows and $a = (-1000,-999,-1000)$ underflows to zero. For any $m$,

$$\mathrm{lse}(a) = m + \log\sum_c e^{a_c-m}, \qquad \text{take } m = \max_c a_c,$$

which forces the largest exponent to $e^0=1$: no overflow, and any underflow is harmless. **The identity holds for every $m$; choosing the max is what makes it safe.** The same trick appears in forward–backward recursions, in importance weights, and in every ELBO you will compute.

Two gradient facts, cheap to derive and used constantly: $\nabla_a\,\mathrm{lse}(a) = \mathrm{softmax}(a)$ — the gradient of a log-partition function is the expected sufficient statistic, the organizing identity of exponential families — and therefore the categorical NLL has gradient $\nabla_a \ell = \mathrm{softmax}(a) - y_{\text{onehot}}$, *predicted minus observed*. Derive it once; you will recognize it in backprop forever.

---

## 4. Poisson: counts and rare events

**The distribution.** For $y\in\{0,1,2,\dots\}$ with rate $\lambda>0$:

$$\mathrm{Poi}(y\mid\lambda) = \frac{\lambda^y e^{-\lambda}}{y!}, \qquad \mathbb{E}[y] = \mathbb{V}[y] = \lambda.$$

**Where it comes from.** Take $\mathrm{Bin}(N,\theta)$ and let $N\to\infty$, $\theta\to0$ with $N\theta = \lambda$ fixed: many opportunities, each individually unlikely, with a stable expected count. That limit is exactly the situation in defect counts on a wafer, photon or particle arrivals in a detector window, failures in a fleet-hour, mutations per genome, or requests per second. If events occur independently at a constant rate, counts in any window are Poisson and the waiting times between them are exponential — the two facts are the same statement about a Poisson process.

**Two structural facts.** Independent Poissons add: $\mathrm{Poi}(\lambda_1) + \mathrm{Poi}(\lambda_2) = \mathrm{Poi}(\lambda_1+\lambda_2)$, so counting over a longer window or a larger area just scales $\lambda$. And the mean equals the variance, which means **the Poisson has no free noise parameter** — telling me the expected count tells me the scatter. That is a strong claim, and it is the source of the model's most common failure.

**The link, and the conditional version.** Rates must be positive, so let the model emit $a = f(x;\theta)$ and set $\lambda(x) = e^{a}$ (or $\mathrm{softplus}(a)$). Then

$$\ell(a,y) = -\log \mathrm{Poi}(y\mid e^a) = e^{a} - y\,a + \log y!,$$

and since $\log y!$ does not depend on the parameters, the training loss is the first two terms — again computed from the unconstrained $a$, never from $\lambda$. That model is **Poisson regression**, and it is the right default whenever your target is a nonnegative integer count. Using MSE on counts instead implicitly claims the noise is Gaussian, symmetric, and of constant width, all three of which are false near $\lambda \approx 0$ where the target cannot go negative.

**The failure mode: overdispersion.** Real count data are usually *more* variable than Poisson, because the rate itself fluctuates between units (different wafers, different patients, different days). The diagnostic is immediate — compute the ratio of sample variance to sample mean within groups of similar predicted rate; Poisson says one. When it comes back at 4, the fix is to give the rate its own randomness: mix $\lambda$ over a Gamma distribution and you get the **negative binomial**, which has a free dispersion parameter. This is a small, concrete example of a pattern that recurs for the rest of the course: *when a model's noise assumption is too rigid, make one of its parameters a random variable and marginalize.*

---

## 5. The Gaussian

### 5.1 Univariate

$$\mathcal{N}(y\mid\mu,\sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}}\exp\left(-\frac{(y-\mu)^2}{2\sigma^2}\right),$$

with mean $=$ mode $= \mu$, variance $\sigma^2$, precision $\lambda = 1/\sigma^2$, cdf $\Phi$ (no closed form; implemented through `erf`), and central 95% interval $\mu\pm1.96\sigma$.

**Why it dominates.** Four separate reasons, worth separating because only some apply in any given situation:

1. Two interpretable parameters that *are* the first two moments, and — uniquely among the families today — they are **free of each other**. Bernoulli and Poisson tie variance to mean; the Gaussian does not.
2. The **central limit theorem**: for iid $X_n$ with finite mean and variance, $(\bar X - \mu)/(\sigma/\sqrt N) \to \mathcal{N}(0,1)$. Errors that are aggregates of many small independent contributions are approximately Gaussian, which is the honest justification for Gaussian noise models. It also supplies the error bar on every Monte Carlo estimate from Lecture 1: $\bar X \pm 1.96\,\hat\sigma/\sqrt N$. *Caveats:* finite variance is required (Cauchy sums stay Cauchy), independence is required, and convergence in the tails is far slower than in the bulk — so rare-event probabilities are exactly where the approximation fails.
3. It is the **maximum-entropy** distribution given a mean and a variance (§9), i.e. the least committal choice when those are all you know.
4. It is analytically closed under nearly every operation we care about (§6). This is convenience, not evidence, and you should notice when you are invoking it.

**★ When the Gaussian is the wrong tail.** Exponentially decaying tails make the fit fragile: a point at $10\sigma$ contributes $50\times$ the loss of one at $\sqrt2\sigma$, so the mean is dragged toward it.

| Family | Density kernel | NLL in $r = y-\mu$ | Behavior |
|---|---|---|---|
| Gaussian | $e^{-r^2/2\sigma^2}$ | $r^2$ | fits the mean; outlier-sensitive |
| Student-$t$, $\nu$ dof | $\big(1+\tfrac{1}{\nu}(r/\sigma)^2\big)^{-(\nu+1)/2}$ | $\tfrac{\nu+1}{2}\log(1+r^2/\nu\sigma^2)$ | polynomial tails; robust |
| Laplace | $e^{-\|r\|/b}$ | $\|r\|/b$ | fits the median; $L_1$ loss |

Because the Student's log-density grows logarithmically rather than quadratically, a far-out point has bounded influence on the gradient. Robustness is a property of the *tail of your log-likelihood*, not a preprocessing step. At $\nu=1$ the Student is the **Cauchy**, which has no mean at all: 95% of standard normal mass lies within $\pm1.96$, but for a standard Cauchy the corresponding interval is about $\pm12.7$. By $\nu\gtrsim5$ the robustness is gone; $\nu=4$ is a common default.

### 5.2 Multivariate

$$\mathcal{N}(x\mid\mu,\Sigma) = \frac{1}{(2\pi)^{d/2}|\Sigma|^{1/2}}\exp\left(-\tfrac12(x-\mu)^\top\Sigma^{-1}(x-\mu)\right), \qquad x\in\mathbb{R}^d,$$

with $\Sigma \succ 0$ the covariance matrix of Lecture 2 §3.

**The geometry, which is the whole thing.** The exponent depends on $x$ only through the **Mahalanobis distance** $\Delta^2 = (x-\mu)^\top\Sigma^{-1}(x-\mu)$, so level sets of the density are ellipsoids $\Delta^2 = \text{const}$. Diagonalize $\Sigma = U\Lambda U^\top$: the columns of $U$ are the ellipsoid's axes and the standard deviations along them are $\sqrt{\lambda_i}$. A Gaussian is a ball that has been stretched and rotated, and $\Sigma^{-1}$ (the **precision matrix**) measures distance in units of the local spread — one standard deviation in every direction is one unit of Mahalanobis distance.

Two practical consequences. **Sampling** is a change of variables: take $\Sigma = LL^\top$ (Cholesky), draw $z\sim\mathcal{N}(0,I)$, and set $x = \mu + Lz$. **Never form $\Sigma^{-1}$**; to evaluate the log-density, solve $L u = (x-\mu)$ by forward substitution and use $\Delta^2 = \|u\|^2$ and $\log|\Sigma| = 2\sum_i \log L_{ii}$. Explicit inversion is slower and numerically worse, and this idiom is exactly what you will need for GP regression.

---

## 6. Closure: what makes the Gaussian family special

Three properties, all consequences of the quadratic exponent. Together they are why linear-Gaussian models admit closed-form inference while almost nothing else does.

**Affine maps.** If $x\sim\mathcal{N}(\mu,\Sigma)$ and $y = Ax+b$, then

$$y \sim \mathcal{N}\big(A\mu + b,\ A\Sigma A^\top\big).$$

Lecture 2 §3 gave the moments for *any* distribution; the content here is that for a Gaussian the moments are the entire answer, because the family is closed. Sums of independent Gaussians are the special case $A = (1\ \ 1)$: $\mathcal{N}(\mu_1,\sigma_1^2) + \mathcal{N}(\mu_2,\sigma_2^2) = \mathcal{N}(\mu_1+\mu_2,\ \sigma_1^2+\sigma_2^2)$, which is also the convolution of the two densities.

**Marginals.** Partition $x = (x_1, x_2)$. Then $p(x_1) = \mathcal{N}(x_1\mid\mu_1,\Sigma_{11})$ — you *delete* the rows and columns of the discarded block. Marginalizing a Gaussian, the operation Lecture 2 warned was generally intractable, is here free.

**Conditionals** (Lecture 2 §7, now in its proper home):

$$p(x_1\mid x_2) = \mathcal{N}\big(x_1 \mid \mu_1 + \Sigma_{12}\Sigma_{22}^{-1}(x_2-\mu_2),\ \ \Sigma_{11}-\Sigma_{12}\Sigma_{22}^{-1}\Sigma_{21}\big).$$

Recall the three readings: the conditional mean is *linear* in the observation, so for jointly Gaussian variables the MSE-optimal regressor is exactly linear; the conditional covariance does not depend on the observed value, so you can predict an experiment's informativeness before running it; and the Schur complement guarantees uncertainty only shrinks, by an amount set entirely by $\Sigma_{12}$.

> **The pattern to carry forward.** Marginalize, condition, or apply a linear map to a Gaussian and you get a Gaussian. That is why Gaussian process regression, the Kalman filter, and Bayesian linear regression have closed forms — and why the moment you introduce a nonlinearity or a non-Gaussian likelihood, you need the approximate inference machinery that occupies the second half of this course.

---

## 7. General transformations of random variables

If $x\sim p_x$ and $y = f(x)$, what is $p_y$?

**Discrete:** accumulate mass, $p_y(y) = \sum_{x:f(x)=y} p_x(x)$.

**Continuous:** you cannot sum densities, so work with the cdf. For monotone (hence invertible) $f$ with $g = f^{-1}$,

$$P_y(y) = \Pr(f(X)\le y) = \Pr(X\le g(y)) = P_x(g(y)) \ \Longrightarrow\ p_y(y) = p_x(g(y))\left|\frac{dg}{dy}\right|.$$

The version to remember is the conservation-of-mass argument: probability in corresponding infinitesimal intervals must agree, $p_x(x)\,|dx| = p_y(y)\,|dy|$, hence $p_y = p_x\,|dx/dy|$. **A density is not a scalar you can transport; it is mass per unit volume, so it must pick up the volume distortion.**

**Multivariate.** For invertible $f:\mathbb{R}^n\to\mathbb{R}^n$ with inverse $g$ and Jacobian $J_g = \partial g/\partial y^\top$:

$$\boxed{\ p_y(y) = p_x\big(g(y)\big)\,\big|\det J_g(y)\big| \qquad\Longleftrightarrow\qquad \log p_y(y) = \log p_x(g(y)) + \log\big|\det J_g(y)\big|.\ }$$

$|\det J|$ is the local volume-change factor: an affine map $f(x)=Ax+b$ sends the unit cube to a parallelepiped of volume $|\det A|$. The board example is Cartesian to polar, $g(r,\phi) = (r\cos\phi,\ r\sin\phi)$:

$$J_g = \begin{pmatrix}\cos\phi & -r\sin\phi\\ \sin\phi & r\cos\phi\end{pmatrix}, \qquad |\det J_g| = r \qquad\Longrightarrow\qquad p_{r,\phi} = p_{x_1,x_2}(r\cos\phi, r\sin\phi)\,r,$$

recovering the area element $r\,dr\,d\phi$. Geometry and probability agree, as they must.

**Four places this is load-bearing:**

1. **Sampling.** $X = P^{-1}(U)$ with $U\sim\mathrm{Unif}(0,1)$ has cdf $P$: the base case of every sampler. The Cholesky construction $x = \mu + Lz$ of §5.2 is the multivariate affine case.
2. **Normalizing flows.** Push a simple density through an invertible network and read off the exact log-likelihood with the boxed formula. The entire design space of flows is *architectures whose $\log|\det J|$ is cheap* — triangular, coupling, or low-rank Jacobians.
3. **The reparameterization trick.** Writing $y = \mu + \sigma\varepsilon$ with $\varepsilon\sim\mathcal{N}(0,1)$ keeps $\mu,\sigma$ differentiable while the randomness sits in a parameter-free variable. This is what makes VAE and variational gradients low-variance.
4. **Units and preprocessing.** Rescaling your data changes densities and log-likelihoods by $\log|\det A|$. Two models trained on differently normalized data cannot have their log-likelihoods compared without this correction — a mistake that shows up in published tables.

---

## 8. Fitting a model: what should $p_\theta \approx q$ mean?

We now have families and a way to transform them. The remaining question is the one the rest of the course answers: given data from an unknown $q$, how do we choose $\theta$ so that $p_\theta \approx q$?

The optimization is only well posed once "$\approx$" is a number. So: **what should we use to measure the discrepancy between two distributions?** Take the question seriously for a few minutes, because the candidates fail in instructive ways.

- **Match a few moments.** Choose $\theta$ so that $p_\theta$ reproduces the sample mean and covariance. Cheap and sometimes sufficient — but Lecture 1's Anscombe and Datasaurus examples were built precisely to show how much freedom that leaves. Moment matching cannot see anything the chosen moments do not encode.
- **Integrate the squared difference,** $\int (p_\theta - q)^2dx$. Well defined, but it weights regions by density rather than by relative error: being wrong by $10^{-3}$ where $q \approx 10^{-3}$ counts the same as being wrong by $10^{-3}$ where $q\approx1$, even though the first is a factor-of-two error and the second is a rounding error. It is also not invariant to reparameterizing $x$ — by §7, both densities pick up Jacobian factors and the "distance" changes when you switch from meters to millimeters.
- **Compare the *ratio* $p_\theta(x)/q(x)$.** Ratios have the right sensitivity — factor-of-two errors count the same everywhere — and Jacobian factors cancel between numerator and denominator, so the comparison is reparameterization-invariant. This is the right object.

Two more requirements, both from how we will use it. We only ever have samples from $q$, never $q$ itself, so the discrepancy must be expressible as an **expectation over data**: $\mathbb{E}_{q}[\,\text{something}(x)\,]$, which we can then estimate by a sample average with the Monte Carlo machinery of Lecture 1. And for independent observations the total discrepancy should **add up** across data points, so that more data means a proportionally larger objective rather than a wildly different one. Ratios multiply over independent samples; taking a logarithm turns that product into a sum.

Put the three together — a ratio, a logarithm, an expectation under $q$ — and there is essentially one candidate:

$$\mathbb{E}_{q}\!\left[\log\frac{q(x)}{p_\theta(x)}\right].$$

That object is the **Kullback–Leibler divergence**, and it is not an arbitrary choice: it is what you are forced into by the requirements above. It also turns out to have a clean interpretation in terms of information, which is what §9 develops. And in the next lecture we will show that minimizing it against the *empirical* distribution $\hat q_N$ is exactly maximum likelihood — so the estimator you already half-know is the solution to the problem we just posed.

---

## 9. Surprise, entropy, and divergence

**Surprisal.** Observing an outcome of probability $p$ carries information $-\log p$. Three requirements force that form: surprise decreases with probability, a certain event ($p=1$) is unsurprising, and independent events have **additive** surprise. Only the logarithm turns the multiplicativity of independent probabilities into additivity. The base sets the units — bits (base 2) or nats (base $e$); we use nats.

> **Entropy.** $\displaystyle \mathbb{H}[p] \triangleq \mathbb{E}_p[-\log p(X)] = -\sum_x p(x)\log p(x)$ — the *average* surprise, i.e. how uncertain the distribution is.

Anchor it with cases: a fair coin has $\mathbb{H} = \log 2$ (one bit); a two-headed coin has $\mathbb{H}=0$; a uniform distribution over $K$ outcomes has $\log K$ and is the maximum for that support. For a Bernoulli, $\mathbb{H}(\theta) = -\theta\log\theta - (1-\theta)\log(1-\theta)$, peaking at $\theta=1/2$ and falling off toward either certainty. Operationally (Shannon's source coding theorem) entropy is the minimum average number of nats needed to encode draws from $p$: **uncertainty and description length are the same quantity.** That is why generative image models report bits per dimension.

> **Conditional entropy.** $\displaystyle \mathbb{H}[X\mid Y] \triangleq \mathbb{E}_{p(y)}\big[\mathbb{H}[p(x\mid y)]\big]$ — the uncertainty about $X$ that *survives* learning $Y$, averaged over what you might learn.

This is the information-theoretic twin of Lecture 2's law of total variance: conditional entropy plays the role of the within-group term, "what is left after you know the group." The between-group term has a name too, below.

> **Cross-entropy.** $\displaystyle \mathbb{H}(q,p) \triangleq -\mathbb{E}_{q}[\log p(X)]$ — the average surprise you experience when the world is $q$ but you are predicting with $p$. In coding terms: the average code length when your code was optimized for the wrong distribution.

> **KL divergence.** $\displaystyle \mathrm{KL}(q\,\|\,p) \triangleq \mathbb{E}_{q}\!\left[\log\frac{q(X)}{p(X)}\right] = \mathbb{H}(q,p) - \mathbb{H}[q] \;\ge\; 0,$ with equality iff $p = q$.

**The one-sentence reading: KL is the *excess* surprise you suffer from using the wrong model — the extra nats per observation, over and above the irreducible uncertainty $\mathbb{H}[q]$ that no model can remove.** That framing explains everything about it. It is zero only for a perfect model. It is measured per observation, so it scales with dataset size. And it is not symmetric, because it asks about surprise experienced *by someone living in $q$*.

*Proof that $\mathrm{KL}\ge0$ (Gibbs' inequality), worth the board space.* Since $-\log$ is convex, Jensen gives

$$\mathrm{KL}(q\|p) = \mathbb{E}_q\!\left[-\log\frac{p(X)}{q(X)}\right] \ \ge\ -\log\mathbb{E}_q\!\left[\frac{p(X)}{q(X)}\right] = -\log\int q(x)\frac{p(x)}{q(x)}dx = -\log 1 = 0.\qquad\blacksquare$$

**KL is a divergence, not a distance.** It is asymmetric and violates the triangle inequality, so do not call it a metric. The asymmetry is a feature, and it matters in practice:

- $\mathrm{KL}(q\|p)$ — expectation under the *data* — blows up wherever $q$ has mass and $p$ does not. It is **mode-covering**: an under-expressive $p$ is forced to spread out and cover everything, filling in valleys where $q$ has none. This is the direction maximum likelihood minimizes.
- $\mathrm{KL}(p\|q)$ — expectation under the *model* — only penalizes $p$ where $p$ itself puts mass, so $p$ can safely ignore modes of $q$ and lock onto one. It is **mode-seeking**, the direction variational inference minimizes, and the reason mean-field VI systematically underestimates posterior variance.

Neither is right; they answer different questions. When we reach variational inference, the identity $\log p(x) = \text{ELBO} + \mathrm{KL}(q(z)\|p(z|x))$ will be readable with nothing but today's definitions.

> **Mutual information.** $\displaystyle \mathbb{I}(X;Y) \triangleq \mathrm{KL}\big(p(x,y)\,\|\,p(x)p(y)\big) = \mathbb{H}[X] - \mathbb{H}[X\mid Y] \ \ge 0.$

Read it as: how far the joint sits from the independent factorization, equivalently the expected reduction in uncertainty about $X$ from learning $Y$. It is zero **iff** $X\perp Y$ — and that is the contrast with correlation from Lecture 2 §3, which is zero for many dependent pairs. Mutual information detects *any* dependence, linear or not; correlation detects only the linear part. In the total-variance analogy, mutual information is the between-group term: the uncertainty that learning $Y$ removes.

Two uses to keep in view. Setting $X = \theta$ and $Y = y_\star$ gives

$$\underbrace{\mathbb{I}(\theta;y_\star\mid x_\star,\mathcal{D})}_{\text{epistemic}} = \underbrace{\mathbb{H}\big[p(y_\star\mid x_\star,\mathcal{D})\big]}_{\text{total predictive}} - \underbrace{\mathbb{E}_{p(\theta|\mathcal{D})}\big[\mathbb{H}[p(y_\star\mid x_\star,\theta)]\big]}_{\text{expected aleatoric}},$$

the clean formalization of the aleatoric/epistemic split, and the acquisition function behind Bayesian active learning and optimal experimental design: *run the experiment whose outcome would teach you the most about the parameters.* ★ Second, be skeptical of reported MI values in high dimensions — sample-based variational estimators such as InfoNCE saturate at $\log(\text{batch size})$, so treat them as loose lower bounds.

**★ Differential entropy and its trap.** For continuous $p$, $h[p] = -\int p\log p\,dx$. It is *not* the limit of the discrete entropy, it can be **negative** ($\mathcal{N}(0,\sigma^2)$ has $h = \tfrac12\log(2\pi e\sigma^2)$, negative for small $\sigma$), and by §7 it is **not invariant under reparameterization** — rescale $x$ and $h$ shifts by $\log|\det A|$. So the sign or magnitude of a differential entropy is a units-dependent statement. KL and mutual information, built from ratios, are immune. Prefer them.

**Maximum entropy, and a promise kept.** Among distributions satisfying given constraints, the maximum-entropy one is the least committal — it assumes nothing beyond the constraints. On a finite support with no constraints: uniform. With a fixed mean on $[0,\infty)$: exponential. **With a fixed mean and variance on $\mathbb{R}$: Gaussian**, which is reason 3 from §5.1, now with content. Choosing a Gaussian when a scale is all you know is formally minimal; choosing one when you know more (positivity, bounded support, skew, count data) discards information you had.

**Next lecture.** We take the objective from §8, replace $q$ by the empirical distribution $\hat q_N$ of Lecture 1, and find that minimizing $\mathrm{KL}(\hat q_N\|p_\theta)$ is exactly maximum likelihood — so cross-entropy losses, squared error, and Poisson regression are all one estimator wearing different distributional clothes.

---

## 10. Code checkpoint

Implement from the equations; `scipy.stats` remains off-limits.

1. **Stable BCE.** Implement `bce_with_logits(a,y)` as $\mathrm{softplus}(a) - ya$ with the $\max(a,0)+\log(1+e^{-|a|})$ form. Compare against the naive version at $a = \pm10,\pm50,\pm500$ in float32 and report where it breaks.
2. **Stable LSE and softmax gradients.** Implement `lse` with the max shift. Verify $\nabla_a\mathrm{lse}(a) = \mathrm{softmax}(a)$ by finite differences, and $\nabla_a\ell = \mathrm{softmax}(a)-y_{\text{onehot}}$ for the categorical NLL.
3. **Poisson from the binomial.** Plot $\mathrm{Bin}(N,\lambda/N)$ against $\mathrm{Poi}(\lambda)$ for $\lambda=3$ and $N = 10,100,10^4$; report the maximum absolute pmf difference at each $N$ and fit its decay rate in $N$.
4. **Overdispersion diagnostic.** Simulate counts with a fixed rate and with a Gamma-mixed rate. For each, compute the variance-to-mean ratio and state which dataset a Poisson model would misfit, and how you would see it in residuals.
5. **Sampling a multivariate Gaussian.** For a $\Sigma$ of your choice in $d=2$, sample via Cholesky ($x = \mu + Lz$), then verify $\hat\Sigma \approx \Sigma$ and overlay the 1-, 2-, and 3-$\sigma$ ellipses computed from the eigendecomposition. Implement the log-density with a triangular solve rather than an explicit inverse and check it against a direct computation.
6. **Closure.** Numerically confirm all three properties of §6 on a $d=3$ Gaussian: sample and check $A\Sigma A^\top$; drop a coordinate and check the marginal; condition on a coordinate within a thin slice and check the conditional mean and covariance against the block formula.
7. **Change of variables.** Sample $x\sim\mathcal{N}(0,1)$ and set $y = \tanh(x)$. Derive $p_y$ analytically, overlay it on a histogram of transformed samples, and verify $\int p_y = 1$ numerically — the check that catches a missing Jacobian.
8. **Entropy and KL by hand.** Implement $\mathbb{H}$, $\mathbb{H}(q,p)$, and $\mathrm{KL}(q\|p)$ for discrete distributions with the $0\log0 = 0$ convention. Verify $\mathrm{KL} = \mathbb{H}(q,p)-\mathbb{H}[q]$, that $\mathrm{KL}\ge0$ on random pairs, and that $\mathrm{KL}(q\|p)\ne\mathrm{KL}(p\|q)$.
9. **The two KLs.** Let $q$ be a two-component Gaussian mixture with well-separated modes and $p$ a single Gaussian. Numerically minimize $\mathrm{KL}(q\|p)$ and $\mathrm{KL}(p\|q)$ over $(\mu,\sigma)$, plot both fits against $q$, and explain each in one sentence.
10. **MI without correlation.** Construct $(X,Y)$ with $\hat\rho\approx0$ but clearly positive $\mathbb{I}(X;Y)$ (e.g. $Y = X^2+\varepsilon$ with symmetric $X$). Estimate both by binning and comment on sensitivity to the bin count.

---

## 11. What you must know cold (quiz-eligible)

- Which family goes with which data type, and the link function for each: $\sigma$, softmax, $\exp$/softplus.
- Bernoulli/binomial pmf and moments; $\sigma$ and its derivative; $1-\sigma(a) = \sigma(-a)$; logit as log-odds; the stable BCE $\mathrm{softplus}(a) - ya$ and why the naive form fails.
- Categorical/multinomial; softmax shift invariance, temperature, and reduction to $\sigma$; the log-sum-exp identity and why $m = \max_c a_c$; $\nabla_a\mathrm{lse} = \mathrm{softmax}$.
- Poisson pmf, $\mathbb{E} = \mathbb{V} = \lambda$, its binomial limit, additivity, the Poisson-regression NLL $e^a - ya$, and what overdispersion is.
- Gaussian pdf in 1-D and $d$-D; Mahalanobis distance and the ellipse picture; sampling via Cholesky; the four reasons the Gaussian is used, kept separate.
- The three closure properties: affine, marginal, conditional — with the conditional formula and its three readings.
- Change of variables in 1-D and $\mathbb{R}^d$, and why $|\det J|$ appears.
- CLT statement with its hypotheses; $\bar X\pm1.96\hat\sigma/\sqrt N$.
- Definitions of surprisal, entropy, conditional entropy, cross-entropy, KL, mutual information; $\mathrm{KL}\ge0$ with the Jensen proof; KL as excess surprise; asymmetry and its mode-covering/mode-seeking consequence.
- $\mathbb{I}(X;Y)=0$ iff independent, versus $\rho = 0$, which is not.
- Gaussian as maximum entropy given mean and variance.

---

## 12. Practice Set 0.3 (ungraded; quiz-eligible)

1. Derive the mean and variance of the Poisson from its pmf. Then obtain $\mathrm{Poi}(\lambda)$ as the $N\to\infty$, $N\theta = \lambda$ limit of $\mathrm{Bin}(N,\theta)$.
2. Show that the sum of two independent Poissons is Poisson. (Convolve the pmfs and recognize a binomial theorem.)
3. Show that the convolution of two Gaussians is Gaussian with means and variances adding. *Murphy Ex. 2.4.*
4. Derive the normalization constant of the 1-D Gaussian by squaring the integral and changing to polar coordinates; identify exactly where the Jacobian factor $r$ enters. *Murphy Ex. 2.12.*
5. Show that the level sets of a multivariate Gaussian are ellipsoids whose axes are the eigenvectors of $\Sigma$ with semi-axis lengths proportional to $\sqrt{\lambda_i}$.
6. Derive the gradient of the logistic-regression NLL with $a = w^\top x$ and show it equals $\sum_n(\sigma(a_n)-y_n)x_n$. Compare with the linear-regression gradient and explain the resemblance.
7. Let $X\sim\mathrm{Ga}(a,b)$ and $Y = 1/X$. Derive the density of $Y$ using §7. *Murphy Ex. 2.7.*
8. A colleague standardizes their inputs, retrains, and reports a much better test log-likelihood than the previous model. Using §7, explain what has to be checked before the comparison means anything.
9. Prove $\mathbb{I}(X;Y) = \mathbb{H}[X]-\mathbb{H}[X\mid Y]$ directly from the definition of KL.
10. Show that among densities on $\mathbb{R}$ with fixed mean $\mu$ and variance $\sigma^2$, the Gaussian maximizes differential entropy. (Lagrange multipliers, or more slickly: expand $\mathrm{KL}(p\|\mathcal{N})\ge0$.)
11. Compute $\mathrm{KL}\big(\mathcal{N}(\mu_1,\sigma_1^2)\,\|\,\mathcal{N}(\mu_2,\sigma_2^2)\big)$ in closed form. Verify it is zero iff the parameters match, and give a pair of Gaussians for which the two orderings differ by more than a factor of ten.
12. Section 8 rejected $\int(p_\theta-q)^2$ partly because it is not reparameterization-invariant. Make that precise: take $q$ and $p_\theta$ on $\mathbb{R}$, rescale $x\mapsto ax$, and show how the integral transforms while $\mathrm{KL}$ does not.
