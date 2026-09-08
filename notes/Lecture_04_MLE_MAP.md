# Lecture 4 — Maximum Likelihood, the Sampling Distribution, and MAP

**ENM 5310 — Data-driven Modeling and Probabilistic Scientific Computing**

**Builds on:** Lecture 1 (the empirical distribution, Monte Carlo, the $N^{-1/2}$ rate), Lecture 2 (conditioning, Bayes' rule, regression as a conditional mean), Lecture 3 (the likelihood recipe, Gaussian and Bernoulli likelihoods, change of variables, entropy and KL divergence).

**Reading:** Murphy I §4.1–4.2, §4.5, §4.7.1–4.7.2, §11.2, §11.3.

**Learning objectives.** After this lecture you should be able to:

1. Derive, line by line, that minimizing $\mathrm{KL}(q\,\|\,p_\theta)$ over $\theta$ is the same problem as maximizing $\sum_n \log p_\theta(x_n)$, and state precisely where that equivalence needs care when $x$ is continuous.
2. State what the MLE converges to when the model family does not contain the truth, and name which of the two KL divergences is being minimized.
3. Treat an estimator as a random variable: write down its sampling distribution, decompose its mean squared error into bias and variance, and define consistency and Fisher information.
4. Derive the MLE for the univariate Gaussian and for linear regression, and give the geometric reading of the normal equations.
5. Write down $\operatorname{Cov}[\hat w]$ for least squares and explain which directions of parameter space the data has failed to determine.
6. Exhibit three distinct ways the MLE can fail to exist, identify their common cause, and repair each with a prior.
7. Derive ridge and lasso as MAP estimates, and interpret $\lambda$ in terms of two variances.
8. State two respects in which the MAP estimate is a defective summary of the posterior.

### Notation used throughout

| Symbol | Meaning |
|---|---|
| $p^\star(x)$ | the true, unknown distribution the data came from |
| $\mathcal{D} = \{x_n\}_{n=1}^N$ | the observed data, drawn i.i.d. from $p^\star$ |
| $q(x)$ | the **empirical distribution** of the data (defined in §1.1) |
| $p_\theta(x)$ | the model, indexed by parameters $\theta\in\Theta$ |
| $\mathcal{P} = \{p_\theta:\theta\in\Theta\}$ | the model family |
| $\ell(\theta)$ | the log-likelihood, $\sum_n \log p_\theta(x_n)$ |
| $\mathrm{NLL}(\theta)$ | the per-datum negative log-likelihood, $-\tfrac1N\ell(\theta)$ |

---

## 1. From KL divergence to the log-likelihood

Lecture 3 posed the question *what should "$p_\theta \approx q$" mean?* and answered it with the KL divergence. This section carries that answer all the way to a concrete optimization problem, one step at a time.

### 1.1 The empirical distribution, restated

Given data $\mathcal{D} = \{x_1,\dots,x_N\}$, the **empirical distribution** is the probability distribution placing mass $1/N$ on each observed point and no mass anywhere else:

$$\boxed{\ q(x) \;\triangleq\; \frac{1}{N}\sum_{n=1}^{N}\delta(x-x_n).\ }$$

Here $\delta(\cdot)$ is the Dirac delta, defined for our purposes entirely by how it behaves inside an integral: $\int g(x)\,\delta(x-x_n)\,dx = g(x_n)$. Everything we need about $q$ follows from a single property, worth its own line because the next page uses it three times:

$$\mathbb{E}_{q}[g(X)] \;=\; \int g(x)\,q(x)\,dx \;=\; \frac{1}{N}\sum_{n=1}^{N}g(x_n). \tag{1.1}$$

**Expectations under $q$ are sample averages.** That is the entire content of the empirical distribution. It is the object Lecture 1 introduced as the thing whose expectations Monte Carlo computes, and it is the only formal representation of "the data" we will need.

If $X$ is discrete, $q$ is an ordinary probability mass function, $q(x) = \frac1N\#\{n: x_n = x\}$ — the histogram of the data, normalized to sum to one. Read $\delta$ as the Kronecker delta and every formula below holds verbatim.

### 1.2 The three information-theoretic quantities, restated

From Lecture 3, for distributions $p$ and $r$ on the same space:

- **Entropy:** $\mathbb{H}[p] \triangleq -\mathbb{E}_p[\log p(X)]$ — the average surprisal of $p$ under itself.
- **Cross-entropy:** $\mathbb{H}(p,r) \triangleq -\mathbb{E}_p[\log r(X)]$ — the average surprisal of a model $r$ when the world is actually $p$.
- **KL divergence:** $\mathrm{KL}(p\,\|\,r) \triangleq \mathbb{E}_p\!\left[\log\frac{p(X)}{r(X)}\right] = \mathbb{H}(p,r) - \mathbb{H}[p] \ \ge\ 0$, with equality if and only if $p = r$ almost everywhere.

The three are tied together by the single identity $\mathrm{KL} = \text{cross-entropy} - \text{entropy}$, which we now exploit.

### 1.3 The derivation

We seek the model closest to the data in KL divergence:

$$\hat\theta \;=\; \arg\min_{\theta\in\Theta}\ \mathrm{KL}\big(q\,\|\,p_\theta\big).$$

**Step 1 — expand the definition.**

$$\mathrm{KL}(q\|p_\theta) \;=\; \mathbb{E}_{q}\!\left[\log\frac{q(X)}{p_\theta(X)}\right].$$

**Step 2 — split the logarithm.** Since $\log(a/b) = \log a - \log b$,

$$\mathrm{KL}(q\|p_\theta) \;=\; \mathbb{E}_{q}\big[\log q(X) - \log p_\theta(X)\big].$$

**Step 3 — use linearity of expectation.** The expectation of a difference is the difference of expectations:

$$\mathrm{KL}(q\|p_\theta) \;=\; \underbrace{\mathbb{E}_{q}[\log q(X)]}_{\textstyle = -\,\mathbb{H}[q]} \;-\; \mathbb{E}_{q}[\log p_\theta(X)].$$

**Step 4 — discard the term free of $\theta$.** The first term depends only on the data. Adding a constant to an objective does not move its minimizer, so

$$\arg\min_\theta\ \mathrm{KL}(q\|p_\theta) \;=\; \arg\min_\theta\ \Big\{-\mathbb{E}_{q}[\log p_\theta(X)]\Big\} \;=\; \arg\min_\theta\ \mathbb{H}(q,p_\theta).$$

The middle expression is exactly the cross-entropy between the data and the model. **Minimizing KL and minimizing cross-entropy are the same problem** whenever the first argument is held fixed.

**Step 5 — evaluate the expectation with property (1.1).** Taking $g = \log p_\theta$,

$$-\,\mathbb{E}_{q}[\log p_\theta(X)] \;=\; -\frac{1}{N}\sum_{n=1}^{N}\log p_\theta(x_n) \;=\; \mathrm{NLL}(\theta).$$

This is the step where an abstract divergence between distributions becomes a finite sum over data points. Nothing was approximated: the empirical distribution was *constructed* so that this expectation is exactly a sample average.

**Step 6 — turn the minimization into a maximization.** Multiplying by the positive constant $N$ and negating converts $\arg\min$ to $\arg\max$:

$$\arg\min_\theta\ \left\{-\frac1N\sum_n \log p_\theta(x_n)\right\} \;=\; \arg\max_\theta\ \sum_{n=1}^{N}\log p_\theta(x_n).$$

**Step 7 — recognize the result.** Because the data are i.i.d., the joint density factorizes, $p_\theta(\mathcal{D}) = \prod_n p_\theta(x_n)$, so $\sum_n \log p_\theta(x_n) = \log p_\theta(\mathcal{D})$; and taking a logarithm does not move an argmax, since $\log$ is strictly increasing. Assembling everything:

$$\boxed{\ \arg\min_\theta \mathrm{KL}(q\|p_\theta) \;=\; \arg\min_\theta \mathbb{H}(q,p_\theta) \;=\; \arg\max_\theta \sum_{n=1}^N \log p_\theta(x_n) \;=\; \arg\max_\theta\, p_\theta(\mathcal{D}) \;=\; \hat\theta_{\mathrm{MLE}}.\ }$$

Maximum likelihood is not a separate principle that happens to work well. It is what "make the model close to the data" *evaluates to* when closeness is measured by KL divergence and the data is represented by its empirical distribution.

Two consequences worth carrying forward:

- The **cross-entropy loss** of classification is literally $\mathbb{H}(q,p_\theta)$. The name is not an analogy.
- **Squared error is a cross-entropy in disguise:** under a Gaussian likelihood the negative log-likelihood is a scaled sum of squared residuals plus a constant, as §6 makes explicit.

### 1.4 A technical caveat, stated honestly

When $X$ is continuous, $q$ places positive probability on finitely many points and therefore has no density with respect to Lebesgue measure. In that case $\mathbb{H}[q]$ and $\mathrm{KL}(q\|p_\theta)$ are not finite numbers, and Steps 1–3 are formal manipulations rather than statements about real quantities.

Everything from Step 4 onward survives intact. The cross-entropy $\mathbb{H}(q,p_\theta) = -\frac1N\sum_n\log p_\theta(x_n)$ is a perfectly well-defined finite number for any density $p_\theta$, and it is the only term carrying $\theta$. So the precise version of the boxed statement is:

> **Maximum likelihood minimizes the cross-entropy between the empirical distribution and the model.** For discrete $X$ it additionally minimizes $\mathrm{KL}(q\|p_\theta)$ exactly. For continuous $X$, the corresponding KL statement is a statement about the $N\to\infty$ limit, which §2 makes rigorous.

Use "MLE minimizes $\mathrm{KL}(q\|p_\theta)$" as a mnemonic, since the intuition it carries is correct. Do not mistake it for a theorem in the continuous case.

---

## 2. What maximum likelihood converges to

Section 1 characterized the finite-$N$ objective. This section characterizes what its minimizer approaches as data accumulate, and the answer is more interesting than "the truth."

By the law of large numbers (Lecture 1), for each fixed $\theta$ the sample average converges to the population expectation:

$$\mathrm{NLL}(\theta) = -\frac1N\sum_{n=1}^N \log p_\theta(x_n) \;\longrightarrow\; -\mathbb{E}_{p^\star}\big[\log p_\theta(X)\big] \;=\; \mathbb{H}(p^\star,p_\theta) \;=\; \mathrm{KL}(p^\star\|p_\theta) + \mathbb{H}[p^\star].$$

The final term does not involve $\theta$. Under conditions making this convergence uniform over $\Theta$ — enough that the *location* of the minimum converges and not merely its value — the minimizers converge as well:

$$\boxed{\ \hat\theta_{\mathrm{MLE}} \;\longrightarrow\; \theta^\star \;\triangleq\; \arg\min_{\theta\in\Theta}\ \mathrm{KL}\big(p^\star\,\|\,p_\theta\big).\ }$$

Note the structure. At finite $N$ we minimize a divergence to the *empirical* distribution; in the limit, a divergence to the *true* distribution. The gap between these two targets is where overfitting lives, and it is the same gap Lecture 1 identified in distinguishing the empirical distribution from the one that generated it.

**The well-specified case.** If the truth lies in the family, $p^\star = p_{\theta_0}$ for some $\theta_0\in\Theta$, then $\mathrm{KL}(p^\star\|p_{\theta_0}) = 0$ attains the minimum, so $\theta^\star = \theta_0$: the MLE recovers the truth. This is the case classical theory analyzes.

**The misspecified case.** If $p^\star\notin\mathcal{P}$ — the normal situation for any model of a real system — the MLE still converges, but to the projection of $p^\star$ onto $\mathcal{P}$ in the sense of forward KL. That projection has a specific and predictable character. The divergence

$$\mathrm{KL}(p^\star\|p_\theta) = \int p^\star(x)\,\log\frac{p^\star(x)}{p_\theta(x)}\,dx$$

is an average weighted by $p^\star$. Wherever $p^\star$ has mass and $p_\theta$ has none, the integrand is $+\infty$; wherever $p_\theta$ has mass and $p^\star$ has none, the integrand contributes nothing at all. The asymmetry is total: the fitted model is punished for *missing* regions where data occurred, and not punished for *inventing* regions where none did.

The consequence is **mass covering**. Fit a single Gaussian by maximum likelihood to bimodal data and the result is a wide Gaussian straddling both modes, placing most of its probability in the valley between them where nothing was ever observed. This is not a failure of the optimizer; it is the exact minimizer of the divergence that was chosen.

Lecture 3 §8 distinguished the two directions of KL. Variational inference, later in the course, minimizes the *reverse* divergence $\mathrm{KL}(p_\theta\|p^\star)$, whose characteristic failure is the opposite one — it locks onto a single mode and reports too little uncertainty. The theme is worth carrying: **every approximate inference method in this course is characterized by which direction of KL it minimizes, and therefore by the specific way in which it misleads you.**

---

## 3. The estimator is a random variable

This section makes the conceptual move on which the rest of the lecture depends.

The estimator $\hat\theta = \hat\theta(\mathcal{D})$ is a *function of the data*. The data are random. A function of a random variable is a random variable. Therefore $\hat\theta$ has a distribution.

> **Sampling distribution.** For an estimator $\hat\theta(\mathcal{D})$ computed from data $\mathcal{D}$ drawn from $p^\star$, the **sampling distribution** of $\hat\theta$ is the distribution of $\hat\theta(\mathcal{D})$ induced by the randomness of $\mathcal{D}$. It describes how the estimate would vary across hypothetical repetitions of the entire experiment.

You have one draw from this distribution. Every statement you can make about the reliability of a fit is a statement about a distribution you have sampled exactly once. This is why the closed-form expressions in §4.1 and §6.1 are valuable: they describe a distribution you cannot otherwise observe.

### 3.1 Three summaries

Let $\theta^\star$ be the target from §2. Define

$$\operatorname{bias}(\hat\theta) \triangleq \mathbb{E}[\hat\theta] - \theta^\star, \qquad \operatorname{Var}(\hat\theta) \triangleq \mathbb{E}\big[(\hat\theta - \mathbb{E}[\hat\theta])^2\big], \qquad \mathrm{MSE}(\hat\theta) \triangleq \mathbb{E}\big[(\hat\theta-\theta^\star)^2\big],$$

where every expectation is over the sampling distribution — that is, over datasets, not over data points. These three obey a decomposition worth deriving, since it is one line and recurs constantly:

$$
\begin{aligned}
\mathrm{MSE}(\hat\theta)
&= \mathbb{E}\big[\big(\hat\theta - \mathbb{E}[\hat\theta] + \mathbb{E}[\hat\theta] - \theta^\star\big)^2\big] \\[2pt]
&= \mathbb{E}\big[(\hat\theta-\mathbb{E}[\hat\theta])^2\big] \;+\; 2\,\mathbb{E}\big[\hat\theta - \mathbb{E}[\hat\theta]\big]\big(\mathbb{E}[\hat\theta]-\theta^\star\big) \;+\; \big(\mathbb{E}[\hat\theta]-\theta^\star\big)^2 \\[2pt]
&= \operatorname{Var}(\hat\theta) + \operatorname{bias}(\hat\theta)^2,
\end{aligned}
$$

since the cross term contains $\mathbb{E}[\hat\theta - \mathbb{E}[\hat\theta]] = 0$. The squared error splits into a spread term and an offset term which do not interact — structurally the same orthogonal decomposition as the law of total variance in Lecture 2.

The practical reading: **mean squared error is the quantity of interest; bias and variance are two ways of spending it.** An estimator that is systematically off but stable can beat one that is centered but erratic. Section 9 turns this observation into the justification for regularization.

### 3.2 Consistency

> **Convergence in probability.** $\hat\theta_N\xrightarrow{p}\theta^\star$ means: for every $\varepsilon>0$, $\Pr\big[\|\hat\theta_N-\theta^\star\|>\varepsilon\big]\to0$ as $N\to\infty$.
>
> **Consistency.** An estimator is **consistent** for $\theta^\star$ if $\hat\theta_N\xrightarrow{p}\theta^\star$.

Under regularity conditions the MLE is consistent for the $\theta^\star$ of §2. Note carefully what this does and does not say: it is a statement about a limit, not about your dataset. A consistent estimator can be arbitrarily poor at the sample size you actually possess.

### 3.3 Fisher information and the shape of the sampling distribution

Consistency says the estimate eventually lands on target. The natural follow-up is *how tightly*, and the answer is governed by the curvature of the log-likelihood.

> **Score.** The **score** is the gradient of the log-density with respect to the parameters, $s(\theta;x) \triangleq \nabla_\theta \log p_\theta(x)$.

The score has mean zero when expectations are taken under the model itself. Assuming differentiation and integration may be exchanged,

$$\mathbb{E}_{p_\theta}\big[s(\theta;X)\big] = \int p_\theta(x)\,\frac{\nabla_\theta p_\theta(x)}{p_\theta(x)}\,dx = \nabla_\theta\!\int p_\theta(x)\,dx = \nabla_\theta 1 = 0. \tag{3.1}$$

> **Fisher information.** The **Fisher information matrix** at $\theta$ is the covariance of the score,
> $$\mathbf{F}(\theta) \;\triangleq\; \mathbb{E}_{p_\theta}\!\left[s(\theta;X)\,s(\theta;X)^\top\right] \;=\; -\,\mathbb{E}_{p_\theta}\!\left[\nabla^2_\theta \log p_\theta(X)\right].$$

The equality of the two expressions costs one line. Writing $s = \nabla p_\theta / p_\theta$ and differentiating a second time,

$$\nabla^2\log p_\theta \;=\; \frac{\nabla^2 p_\theta}{p_\theta} \;-\; \frac{\nabla p_\theta\,\nabla p_\theta^\top}{p_\theta^2} \;=\; \frac{\nabla^2 p_\theta}{p_\theta} - s\,s^\top .$$

Taking expectations under $p_\theta$, the first term becomes $\int \nabla^2 p_\theta\,dx = \nabla^2\!\int p_\theta\,dx = 0$, leaving $\mathbb{E}[\nabla^2\log p_\theta] = -\mathbb{E}[ss^\top]$.

**Interpretation.** $\mathbf{F}$ is the *expected curvature of the negative log-likelihood* at $\theta$. Large $\mathbf{F}$ means a sharply peaked likelihood, so the data strongly constrains $\theta$. Small $\mathbf{F}$ means a flat likelihood, so the data barely constrains $\theta$ at all.

This is made precise by the following result, stated without proof and revisited in Lecture 7:

> **Asymptotic normality of the MLE.** Under regularity conditions, in the well-specified case,
> $$\sqrt N\,\big(\hat\theta_{\mathrm{MLE}} - \theta^\star\big)\ \xrightarrow{d}\ \mathcal{N}\big(0,\ \mathbf{F}(\theta^\star)^{-1}\big).$$

Read it as a single sentence: **the sampling covariance of the MLE is the inverse curvature of the expected log-likelihood, divided by $N$.** Flat likelihood, uncertain estimate. In Lecture 7 we will approximate a posterior distribution by a Gaussian whose covariance is the inverse Hessian of the log-posterior at its mode, and it will be the same sentence with a prior term added.

*(Under misspecification the limiting covariance becomes a "sandwich," $\mathbf F^{-1}\mathbf G\,\mathbf F^{-1}$, with $\mathbf G$ the score covariance taken under $p^\star$ rather than under $p_\theta$. The two coincide when the model is correct — which is precisely the identity derived above.)*

---

## 4. The Gaussian, in closed form

Take $p_\theta(x) = \mathcal{N}(x\mid\mu,\sigma^2)$ with $\theta = (\mu,\sigma^2)$. The log-likelihood is

$$\ell(\mu,\sigma^2) = -\frac{N}{2}\log(2\pi) - \frac{N}{2}\log\sigma^2 - \frac{1}{2\sigma^2}\sum_{n=1}^{N}(x_n-\mu)^2 .$$

**Stationarity in $\mu$.**

$$\frac{\partial\ell}{\partial\mu} = \frac{1}{\sigma^2}\sum_n (x_n-\mu) = 0 \quad\Longrightarrow\quad \hat\mu = \bar x \triangleq \frac1N\sum_n x_n .$$

The factor $1/\sigma^2$ cancels, so the location estimate does not depend on the scale estimate; the joint optimization decouples in one direction and may be solved sequentially.

**Stationarity in $\sigma^2$.** Treating $v = \sigma^2$ as the variable,

$$\frac{\partial\ell}{\partial v} = -\frac{N}{2v} + \frac{1}{2v^2}\sum_n (x_n-\hat\mu)^2 = 0 \quad\Longrightarrow\quad \hat\sigma^2 = \frac1N\sum_n (x_n-\bar x)^2 .$$

Both estimates are **moment matching**: the fitted model's mean and variance equal the data's sample mean and sample variance. That the MLE does the obvious thing here is reassuring but not informative. What follows is informative.

### 4.1 The sampling distribution of $\hat\mu$

$\hat\mu$ is a linear combination of independent Gaussians, and Lecture 3 established that such combinations are Gaussian, with mean and variance obtained by linearity:

$$\mathbb{E}[\hat\mu] = \frac1N\sum_n\mathbb{E}[X_n] = \mu, \qquad \operatorname{Var}[\hat\mu] = \frac{1}{N^2}\sum_n\operatorname{Var}[X_n] = \frac{\sigma^2}{N},$$

so that

$$\boxed{\ \hat\mu \sim \mathcal{N}\!\left(\mu,\ \frac{\sigma^2}{N}\right).\ }$$

The estimate has standard deviation $\sigma/\sqrt N$. This is the $N^{-1/2}$ rate from the Monte Carlo section of Lecture 1, reappearing as the error of a fitted parameter — and for the same reason, since $\hat\mu$ *is* a Monte Carlo estimate of $\mathbb{E}_{p^\star}[X]$. Point estimation and Monte Carlo integration are the same activity in different notation.

Distinguish two quantities that are easy to conflate. $\sigma^2$ is the spread of the *data*; $\sigma^2/N$ is the spread of the *estimate*. Collecting more data leaves the first unchanged and shrinks the second.

### 4.2 The sampling distribution of $\hat\sigma^2$

Use the identity

$$\sum_n (x_n-\bar x)^2 = \sum_n (x_n-\mu)^2 - N(\bar x-\mu)^2,$$

obtained by writing $x_n - \bar x = (x_n - \mu) - (\bar x - \mu)$, expanding, and observing that the cross term collapses to $-2N(\bar x-\mu)^2$. Taking expectations and using $\operatorname{Var}[\bar x] = \sigma^2/N$ from §4.1,

$$\mathbb{E}\Big[\sum_n(x_n-\bar x)^2\Big] = N\sigma^2 - N\cdot\frac{\sigma^2}{N} = (N-1)\sigma^2 \quad\Longrightarrow\quad \mathbb{E}[\hat\sigma^2] = \frac{N-1}{N}\,\sigma^2 .$$

The MLE systematically **underestimates** the variance. The mechanism is not mysterious: the data spreads around $\bar x$, and $\bar x$ was itself chosen to sit at the center of that very data, so the observed spread is slightly smaller than the spread around the true $\mu$. One degree of freedom was consumed in estimating the mean.

The bias is $O(1/N)$ and vanishes asymptotically, consistent with §3.2. It is recorded here as a concrete instance of the general point: an estimator has a distribution, and that distribution need not be centered on its target.

### 4.3 ★ The multivariate case

For $x_n\in\mathbb{R}^d$ with $p_\theta = \mathcal{N}(\mu,\Sigma)$, the same procedure — using $\nabla_\Sigma\log|\Sigma| = \Sigma^{-1}$ and the trace identity $a^\top A a = \operatorname{tr}(Aaa^\top)$, worked out in Murphy I §4.2.6 — gives

$$\hat\mu = \bar x, \qquad \hat\Sigma = \frac1N\sum_n (x_n-\bar x)(x_n-\bar x)^\top .$$

One structural fact matters more than the derivation. $\hat\Sigma$ is a sum of $N$ rank-one matrices subject to one linear constraint, so $\operatorname{rank}(\hat\Sigma)\le\min(N-1,d)$. **When $N\le d$, $\hat\Sigma$ is singular**, the Gaussian density is undefined, and the likelihood can be driven to $+\infty$ by collapsing the model onto the subspace spanned by the data. Hold this until §7.

---

## 5. Checkpoint A — the sampling distribution, empirically

Rules as usual: `numpy` only, `scipy.stats` off-limits, and results reported as fitted numbers rather than pictures that look about right.

1. Fix $\mu=2$, $\sigma^2=3$. For each $N\in\{5,10,20,50,200\}$, draw $R = 20{,}000$ independent datasets and compute $\hat\mu$ and $\hat\sigma^2$ for each. You now hold $R$ draws from each sampling distribution.
2. Histogram the $R$ values of $\hat\mu$ at $N=20$, normalized to a density. Overlay $\mathcal{N}(\mu,\sigma^2/N)$, coded from the formula rather than imported. Report the empirical mean and variance of $\hat\mu$ against the predicted $\mu$ and $\sigma^2/N$ to three significant figures.
3. Plot the empirical standard deviation of $\hat\mu$ against $N$ on log–log axes and fit a slope. State the value obtained and the value §4.1 predicts.
4. Plot $\mathbb{E}[\hat\sigma^2]/\sigma^2$ against $N$ with the curve $(N-1)/N$ overlaid. The deviation from $1$ should be plain at $N=5$ and invisible by eye at $N=200$. Quantify "invisible": compare the bias against the Monte Carlo error in your estimate of it, given $R$ replicates.
5. Generate a linear regression problem with $N=100$, $D=3$, known $\sigma^2$, and a design matrix of your choosing. Over $R=5{,}000$ replicates of the noise, compute $\hat w$ and compare the empirical covariance of $\hat w$ entry by entry against $\sigma^2(\Phi^\top\Phi)^{-1}$ from §6.1.
6. Repeat with a nearly collinear design, making the third column almost a linear combination of the first two. Report the condition number of $\Phi$, the empirical variance of each coefficient, and the empirical variance of the *predictions* $\Phi\hat w$ at the training inputs. Explain what you observe in one sentence.

---

## 6. Linear regression

Lecture 3 §4 constructed this model; here we solve it. Collect the feature vectors into a **design matrix** $\Phi\in\mathbb{R}^{N\times D}$ whose $n$-th row is $\phi(x_n)^\top$, and the targets into $y\in\mathbb{R}^N$. The model is

$$p(y\mid X,w,\sigma^2) = \prod_{n=1}^{N}\mathcal{N}\big(y_n \,\big|\, w^\top\phi(x_n),\ \sigma^2\big).$$

This is a claim about the conditional mean, $\mathbb{E}[y\mid x] = w^\top\phi(x)$, with everything unexplained declared to be independent Gaussian noise of constant variance. The map $\phi$ may be any fixed nonlinear transformation — polynomials, Fourier features, radial basis functions. The model is nonlinear in $x$ and linear in $w$, and only the latter matters for the algebra below. This is the hinge on which kernel methods will later turn.

The negative log-likelihood is

$$\mathrm{NLL}(w,\sigma^2) = \frac{N}{2}\log(2\pi\sigma^2) + \frac{1}{2\sigma^2}\|y-\Phi w\|_2^2 .$$

Since $\sigma^2>0$ multiplies the $w$-dependent term by a positive constant, the minimizing $w$ does not depend on it. **Least squares is not an approximation to maximum likelihood; it is exactly maximum likelihood under a homoscedastic Gaussian noise model.** The squared-error loss was chosen the moment the noise model was chosen, not separately.

**Normal equations.** Using $\nabla_w\|y-\Phi w\|_2^2 = -2\Phi^\top(y-\Phi w)$,

$$\boxed{\ \Phi^\top\Phi\,\hat w = \Phi^\top y \qquad\Longrightarrow\qquad \hat w = (\Phi^\top\Phi)^{-1}\Phi^\top y \quad\text{whenever } \Phi^\top\Phi \text{ is invertible.}\ }$$

**Geometry.** The condition $\Phi^\top(y-\Phi\hat w)=0$ says the residual vector is orthogonal to every column of $\Phi$. Consequently the fitted values

$$\hat y = \Phi\hat w = \underbrace{\Phi(\Phi^\top\Phi)^{-1}\Phi^\top}_{\textstyle H}\;y$$

are the orthogonal projection of $y$ onto the **column space** of $\Phi$, the subspace of $\mathbb{R}^N$ spanned by the feature columns. The matrix $H$ is symmetric and idempotent ($H^2 = H$) — the defining properties of an orthogonal projection — with $\operatorname{tr}H = D$. Least squares is a projection; the rest is bookkeeping.

**A numerical note.** Do not form $\Phi^\top\Phi$ and invert it. The **condition number** $\kappa(\Phi) = \sigma_{\max}(\Phi)/\sigma_{\min}(\Phi)$, the ratio of largest to smallest singular value, governs how errors in the inputs are amplified into errors in the solution, and $\kappa(\Phi^\top\Phi) = \kappa(\Phi)^2$. Forming the Gram matrix squares the conditioning problem before anything has been solved. Use the QR factorization of $\Phi$, or the SVD if you also want the spectrum — which §9 shows you do.

**Noise variance.** With $\hat w$ fixed, setting $\partial\,\mathrm{NLL}/\partial\sigma^2 = 0$ gives $\hat\sigma^2 = \frac1N\|y-\Phi\hat w\|_2^2$, with $\mathbb{E}[\hat\sigma^2] = \frac{N-D}{N}\sigma^2$ — the mechanism of §4.2, with $D$ degrees of freedom consumed instead of one.

### 6.1 The sampling distribution of $\hat w$

Suppose the model is **well specified**: the data really were generated as $y = \Phi w^\star + \epsilon$ with $\epsilon\sim\mathcal{N}(0,\sigma^2 I_N)$. Substituting,

$$\hat w = (\Phi^\top\Phi)^{-1}\Phi^\top\big(\Phi w^\star + \epsilon\big) = w^\star + (\Phi^\top\Phi)^{-1}\Phi^\top\epsilon .$$

This is a fixed vector plus a linear map applied to a Gaussian, hence Gaussian, with mean $w^\star$ and covariance $A(\sigma^2 I)A^\top$ where $A = (\Phi^\top\Phi)^{-1}\Phi^\top$. Expanding, and using symmetry of $(\Phi^\top\Phi)^{-1}$,

$$AA^\top = (\Phi^\top\Phi)^{-1}\Phi^\top\Phi(\Phi^\top\Phi)^{-1} = (\Phi^\top\Phi)^{-1},$$

so

$$\boxed{\ \hat w \sim \mathcal{N}\big(w^\star,\ \sigma^2(\Phi^\top\Phi)^{-1}\big).\ }$$

This is the most useful equation in the lecture. Three readings:

**1. Error bars, at no cost.** The square roots of the diagonal entries are the standard errors of the individual coefficients. One can now say "this coefficient is $0.4\pm0.9$" rather than "this coefficient is $0.4$" — and notice that the second statement was never a statement about anything.

**2. Conditioning is a statistical fact, not merely a numerical one.** Small eigenvalues of $\Phi^\top\Phi$ produce large variances along the corresponding directions of parameter space. Collinear features are not just an inconvenience for the linear solver; they are a substantive statement that *the data does not determine those combinations of parameters*. Section 9 identifies exactly which directions, and by how much.

**3. A forward reference.** In Lecture 6 we will place a prior $w\sim\mathcal{N}(0,\tau^2 I)$ on the coefficients and derive the posterior covariance

$$S_N = \big(\sigma^{-2}\Phi^\top\Phi + \tau^{-2}I\big)^{-1} .$$

As $\tau\to\infty$ — a prior carrying no information — $S_N\to\sigma^2(\Phi^\top\Phi)^{-1}$, the sampling covariance above. The two coincide in the flat-prior limit despite answering genuinely different questions: one asks how the estimate would move were the experiment repeated, the other asks what one should believe about $w$ given the data actually observed. Their agreement here is a special property of the Gaussian linear model, not a general theorem.

---

## 7. When the MLE fails: three degeneracies, one cause

Everything so far has assumed the maximizer exists and is unique. Frequently it does not.

**(a) The collapsing variance.** With $N=1$, the Gaussian MLE gives $\hat\mu = x_1$ and $\hat\sigma^2 = 0$, and the likelihood is *unbounded*: as $\sigma\to0$ with $\mu = x_1$, the density at $x_1$ diverges. The same pathology appears at scale in mixture models, where the likelihood is infinite whenever a component collapses onto a single data point. Any EM implementation written without a variance floor will eventually find it.

**(b) The underdetermined regression.** If $D>N$, or if two feature columns are linearly dependent, $\Phi^\top\Phi$ is singular. An entire affine subspace of $w$ then achieves the same maximum likelihood, and $\hat w$ is not well defined. Equivalently, by §6.1, the sampling variance along those directions is infinite.

**(c) Separability.** For logistic regression on linearly separable data — data that some hyperplane divides perfectly — replacing $w$ by $\alpha w$ and letting $\alpha\to\infty$ drives every predicted probability to $0$ or $1$ and pushes the likelihood monotonically toward a supremum that is never attained. The MLE does not exist: $\|\hat w\| = \infty$. You will watch an optimizer march off toward infinity in the next lecture, and should recognize it as this rather than as a bug.

**The common cause.** In every case the log-likelihood is *flat, or unbounded, along some direction of parameter space*, because the data carries no information about that direction. The likelihood is a function of the data alone. Where the data is silent, the likelihood is silent, and the maximizer is undefined or meaningless.

It follows that any repair must contribute information that is *not* a function of the data. That is exactly what a prior is.

---

## 8. MAP: the prior as curvature

Bayes' rule from Lecture 2 gives the posterior $p(\theta\mid\mathcal{D})\propto p(\mathcal{D}\mid\theta)\,p(\theta)$. The **maximum a posteriori** estimate is its mode:

$$\boxed{\ \hat\theta_{\mathrm{MAP}} \;\triangleq\; \arg\max_\theta\ \big[\log p(\mathcal{D}\mid\theta) + \log p(\theta)\big].\ }$$

The evidence $p(\mathcal{D})$ is constant in $\theta$ and drops out of the argmax. This is the cheapest possible use of Bayes' rule: it requires only the *shape* of the posterior near its peak, never its normalizing constant. That is why MAP survives into large-scale practice while the full posterior does not — and also why Lecture 7 must spend a lecture on the normalizer being dodged here.

The immediate consequence is a dictionary:

$$\underbrace{\arg\min_\theta\ \mathrm{NLL}(\theta) + \lambda R(\theta)}_{\text{"loss plus penalty"}} \quad\Longleftrightarrow\quad \underbrace{\arg\max_\theta\ \log p(\mathcal{D}\mid\theta) + \log p(\theta)}_{\text{MAP}}, \qquad \log p(\theta) = -\lambda R(\theta) + \text{const}.$$

**Every regularizer is a log-prior** — not by analogy but by definition, provided $\exp(-\lambda R(\theta))$ is normalizable.

### 8.1 Ridge regression

Take the prior $w\sim\mathcal{N}(0,\tau^2 I_D)$, so $\log p(w) = -\frac{1}{2\tau^2}\|w\|_2^2 + \text{const}$. The MAP objective is

$$\frac{1}{2\sigma^2}\|y-\Phi w\|_2^2 + \frac{1}{2\tau^2}\|w\|_2^2 ,$$

and multiplying through by $2\sigma^2$, which changes nothing, gives the familiar form

$$\|y-\Phi w\|_2^2 + \lambda\|w\|_2^2, \qquad \boxed{\ \lambda = \frac{\sigma^2}{\tau^2}.\ }$$

Setting the gradient to zero,

$$\hat w_{\mathrm{ridge}} = \big(\Phi^\top\Phi + \lambda I\big)^{-1}\Phi^\top y .$$

Two points.

**The matrix is invertible for every $\lambda>0$**, regardless of $\Phi$: adding $\lambda I$ adds $\lambda$ to each eigenvalue of the positive semi-definite matrix $\Phi^\top\Phi$. The prior has supplied curvature exactly where the likelihood had none. Degeneracy (b) of §7 is thereby repaired — as is (a), by a prior on $\sigma^2$ that vanishes as $\sigma\to0$, and (c), since adding $\frac{1}{2\tau^2}\|w\|_2^2$ to the logistic objective makes it grow without bound as $\|w\|\to\infty$, so a finite minimizer must exist. One repair, three diseases.

**$\lambda$ is a ratio of variances, not a tuning knob.** The identity $\lambda = \sigma^2/\tau^2$ says: regularize in proportion to how noisy the data is believed to be, and in inverse proportion to how large the coefficients are believed to be. Both are modeling statements that can be right or wrong. In Lecture 7 we will *estimate* $\lambda$ by maximizing the marginal likelihood rather than searching over it by cross-validation.

### 8.2 Lasso, and one warning

Take instead an independent Laplace prior on each coefficient, $p(w_d) = \frac{1}{2b}\exp(-|w_d|/b)$, so that $\log p(w) = -\frac1b\|w\|_1 + \text{const}$. The MAP objective becomes

$$\|y-\Phi w\|_2^2 + \lambda\|w\|_1 ,$$

which has no closed-form solution: the objective is convex but not differentiable at $w_d = 0$, requiring proximal or coordinate-descent methods, covered with the optimization material. Its characteristic behavior is **sparsity** — many coefficients set exactly to zero — arising geometrically because the $\ell_1$ ball has corners on the coordinate axes, and the elliptical level sets of the likelihood generically first touch it at a corner.

The warning, and the reason lasso appears in this lecture rather than a later one: **the sparsity is a property of the mode, not of the posterior.** Under a Laplace prior the posterior assigns zero probability to the event $w_d = 0$ exactly, and the posterior *mean* is sparse in no coordinate whatsoever. When a lasso fit reports that seven features were "selected," that is a statement about one particular summary statistic of the posterior, not about the posterior. Section 10 generalizes the complaint.

---

## 9. ★ Shrinkage in the SVD basis

This section fuses §6.1 and §8.1 into a single statement.

> **Thin singular value decomposition.** Any $\Phi\in\mathbb{R}^{N\times D}$ with $N\ge D$ can be written $\Phi = UDV^\top$, where $U\in\mathbb{R}^{N\times D}$ has orthonormal columns $u_1,\dots,u_D$, the matrix $V\in\mathbb{R}^{D\times D}$ is orthogonal with columns $v_1,\dots,v_D$, and $D = \operatorname{diag}(d_1\ge\cdots\ge d_D\ge0)$ holds the singular values. The $v_j$ are orthogonal directions in feature (and hence parameter) space, and $d_j$ measures how much the data varies along $v_j$.

Note that $\Phi^\top\Phi = VD^2V^\top$, so the eigenvalues of the Gram matrix are the $d_j^2$.

**Least squares.** Substituting, $\hat w_{\mathrm{LS}} = VD^{-1}U^\top y$, with sampling covariance from §6.1

$$\operatorname{Cov}[\hat w_{\mathrm{LS}}] = \sigma^2(\Phi^\top\Phi)^{-1} = \sigma^2 VD^{-2}V^\top .$$

Along the direction $v_j$ the sampling variance is $\sigma^2/d_j^2$. A direction along which the data barely varies yields a wildly uncertain coefficient.

**Ridge.** Since $\Phi^\top\Phi + \lambda I = V(D^2+\lambda I)V^\top$,

$$\hat w_{\mathrm{ridge}} = V(D^2+\lambda I)^{-1}DU^\top y, \qquad \hat y_{\mathrm{ridge}} = \Phi\hat w_{\mathrm{ridge}} = \sum_{j=1}^{D}u_j\,\frac{d_j^2}{d_j^2+\lambda}\,\big(u_j^\top y\big).$$

Compare with least squares, where the identical expression appears with every factor equal to one. Ridge retains the $j$-th direction with a **shrinkage factor** $d_j^2/(d_j^2+\lambda)\in(0,1)$, close to $1$ when $d_j^2\gg\lambda$ and close to $0$ when $d_j^2\ll\lambda$. Hence:

$$\boxed{\ \text{Ridge shrinks hardest along exactly those directions in which least squares is least determined.}\ }$$

It is not shrinking indiscriminately toward zero. It is suppressing the directions in which $\hat w_{\mathrm{LS}}$ was mostly noise, using a smooth threshold rather than a hard one.

> **Effective degrees of freedom.** $\ \mathrm{df}(\lambda) \triangleq \operatorname{tr}\big[\Phi(\Phi^\top\Phi+\lambda I)^{-1}\Phi^\top\big] = \sum_{j=1}^{D}\dfrac{d_j^2}{d_j^2+\lambda}.$

At $\lambda = 0$ this recovers $\operatorname{tr}H = D$ from §6. For $\lambda>0$ it is strictly smaller: the penalty buys a model with less than $D$ parameters' worth of flexibility, indexed continuously by $\lambda$.

Finally, the cost. Taking expectations, $\mathbb{E}[\hat w_{\mathrm{ridge}}] = V(D^2+\lambda I)^{-1}D^2V^\top w^\star \ne w^\star$, so ridge is **biased**, and biased toward zero most strongly in the noisiest directions. In the language of §3.1 it deliberately accepts bias in exchange for a larger reduction in variance. Whether the trade is favorable depends on $\lambda$; a classical result of Hoerl and Kennard guarantees that for any design and any true $w^\star$ there exists some $\lambda>0$ whose ridge estimate has strictly smaller mean squared error than least squares.

---

## 10. Two things the MAP estimate is not

These two facts are the reason the course does not stop at MAP.

### 10.1 MAP is not invariant to reparameterization

The MLE is. If $\eta = g(\theta)$ for an invertible, differentiable $g$, then

$$\hat\eta_{\mathrm{MLE}} = g\big(\hat\theta_{\mathrm{MLE}}\big),$$

because the likelihood is a *function of* $\theta$, not a *density in* $\theta$: its value at a point is unchanged by relabeling the points, so the location of the maximum simply transports.

A prior, however, *is* a density in $\theta$. By the change-of-variables formula from Lecture 3, the induced prior on $\eta$ is

$$p_\eta(\eta) = p_\theta\big(g^{-1}(\eta)\big)\,\left|\det\frac{\partial g^{-1}}{\partial\eta}\right| ,$$

and the Jacobian factor varies with $\eta$, so it moves the location of the maximum. Concretely: fit a scale parameter by MAP in the coordinate $\sigma$, then fit it by MAP in the coordinate $\log\sigma$ under the correspondingly transformed prior, and after mapping one answer back through $g$ the two will disagree. **The MAP estimate depends on the coordinate system in which the model happened to be written.**

Every other summary of the posterior — the mean, the quantiles, the predictive distribution, any posterior expectation — transforms correctly, because integration against a density is invariant while maximization of a density is not. The mode is the odd one out.

### 10.2 The mode is not where the mass is

Consider $\theta\sim\mathcal{N}(0,I_d)$ in $d$ dimensions. The density is maximized at the origin. Now ask where samples actually fall, by studying $\|\theta\|^2 = \sum_{j=1}^{d}\theta_j^2$:

$$\mathbb{E}\big[\|\theta\|^2\big] = \sum_{j=1}^{d}\mathbb{E}[\theta_j^2] = d, \qquad \operatorname{Var}\big[\|\theta\|^2\big] = \sum_{j=1}^{d}\operatorname{Var}[\theta_j^2] = \sum_{j=1}^{d}\big(\mathbb{E}[\theta_j^4]-1\big) = 2d,$$

using $\mathbb{E}[\theta_j^4] = 3$ for a standard Gaussian. So $\|\theta\|^2 = d \pm O(\sqrt d)$, meaning $\|\theta\|\approx\sqrt d$ with fluctuations of order one. Samples concentrate on a thin spherical shell of radius $\sqrt d$.

> **Typical set (informal).** The region containing essentially all of a distribution's probability mass. For a high-dimensional Gaussian it is a thin shell, not a ball around the mode.

The ratio of the density on that shell to the density at the mode is $e^{-d/2}$ — astronomically small — and yet the shell is where all the samples are, because the *volume* available grows faster than the density decays. In $d=1000$, essentially no sample from the distribution lands anywhere near its own mode.

The consequence: a MAP estimate can be an atypical, unrepresentative point of the very distribution it summarizes, and substituting it into a prediction discards all uncertainty in the process. This is not a small correction to be patched later. It is the reason the remaining Bayesian material in this course exists.

---

## 11. Where this leaves us

We now have a principle (maximum likelihood, derived from KL), a characterization of what it converges to (§2), a description of how reliable its answers are (§3, §4.1, §6.1), an account of when it breaks (§7), and a repair (§8). Two things are missing.

**Computation.** Both closed-form solutions came from setting a gradient to zero and being fortunate enough to solve the resulting equation. Logistic regression — a model already written down in Lecture 3 — admits no such solution. The next lecture derives its gradient and Hessian, then takes up the general question of what to do when $\nabla\ell(\theta)=0$ cannot be solved in closed form. That is the entry point to every optimizer used in the second half of the course.

**Uncertainty.** Section 10 argues that a point estimate is a poor summary of what the data supports. Lecture 6 keeps the entire posterior instead.

---

## Quiz-eligible facts

1. The empirical distribution is $q(x) = \frac1N\sum_n\delta(x-x_n)$; its defining property is that expectations under it are sample averages.
2. Minimizing $\mathrm{KL}(q\|p_\theta)$, minimizing the cross-entropy $\mathbb{H}(q,p_\theta)$, and maximizing $\sum_n\log p_\theta(x_n)$ are the same optimization problem; the entropy term $\mathbb{H}[q]$ is discarded because it does not involve $\theta$.
3. For continuous $x$, the exact statement concerns cross-entropy; the KL statement holds in the $N\to\infty$ limit with $p^\star$ in place of $q$.
4. The MLE converges to $\arg\min_\theta\mathrm{KL}(p^\star\|p_\theta)$. Under misspecification this projection is mass-covering, so the fitted model over-disperses.
5. An estimator is a random variable; the distribution induced by the randomness of the data is its sampling distribution.
6. $\mathrm{MSE} = \operatorname{Var} + \operatorname{bias}^2$, the cross term vanishing because $\mathbb{E}[\hat\theta-\mathbb{E}\hat\theta] = 0$.
7. The score is $\nabla_\theta\log p_\theta(x)$ and has zero mean under $p_\theta$. Fisher information is its covariance, equal to the expected negative Hessian of $\log p_\theta$.
8. Asymptotically, $\sqrt N(\hat\theta_{\mathrm{MLE}}-\theta^\star)\to\mathcal{N}(0,\mathbf F^{-1})$: flat likelihood means uncertain estimate.
9. $\hat\mu = \bar x\sim\mathcal{N}(\mu,\sigma^2/N)$; the estimate's error decays at the Monte Carlo rate $N^{-1/2}$.
10. $\hat\sigma^2_{\mathrm{MLE}} = \frac1N\sum_n(x_n-\bar x)^2$ with $\mathbb{E}[\hat\sigma^2] = \frac{N-1}{N}\sigma^2$; it is biased low because $\bar x$ was fitted to the same data.
11. Least squares is exactly MLE under an i.i.d. homoscedastic Gaussian noise model; the normal equations say the residual is orthogonal to the column space of $\Phi$, so $\hat y = Hy$ is a projection.
12. $\hat w_{\mathrm{LS}}\sim\mathcal{N}\big(w^\star,\sigma^2(\Phi^\top\Phi)^{-1}\big)$ when the model is well specified.
13. Solve least squares by QR or SVD; forming $\Phi^\top\Phi$ squares the condition number.
14. The MLE fails to exist in three ways — collapsing variance, singular $\Phi^\top\Phi$, logistic separability — all instances of a likelihood flat or unbounded along some direction.
15. Ridge is MAP under $w\sim\mathcal{N}(0,\tau^2 I)$ with $\lambda = \sigma^2/\tau^2$; lasso is MAP under a Laplace prior.
16. Ridge retains singular direction $j$ with factor $d_j^2/(d_j^2+\lambda)$; effective degrees of freedom is $\sum_j d_j^2/(d_j^2+\lambda)$.
17. Lasso's sparsity is a property of the posterior mode; the posterior mean is not sparse.
18. MAP is not invariant under reparameterization, because the prior acquires a Jacobian factor; the MLE is invariant.
19. In high dimensions the mode of a distribution lies outside its typical set.

---

## Practice problems

*Ungraded; these feed the quizzes.*

**P1.** Prove $\sum_n(x_n-\bar x)^2 = \sum_n(x_n-\mu)^2 - N(\bar x-\mu)^2$ and use it to derive $\mathbb{E}[\hat\sigma^2]$. Identify the single place where independence of the $x_n$ was used.

**P2.** Verify (3.1) numerically for a Gaussian model: draw samples from $p_\theta$, evaluate the score at the true $\theta$, and check that the sample average of the score approaches zero at the expected rate. Then compute the Fisher information of $\mathcal{N}(\mu,\sigma^2)$ with respect to $\mu$ analytically, and confirm that $\mathbf F^{-1}/N$ reproduces $\operatorname{Var}[\hat\mu]$ from §4.1.

**P3.** Two datasets are each fitted with a single Gaussian. Dataset A is unimodal; dataset B is a well-separated 50/50 mixture of two narrow modes. Predict, before computing, where the fitted mean and variance will land in each case, then verify numerically. Explain the outcome for B in terms of the asymmetry of $\mathrm{KL}(p^\star\|p_\theta)$.

**P4 (weighted least squares).** For heteroscedastic noise $y_n\sim\mathcal{N}(w^\top\phi(x_n),\sigma_n^2)$ with known $\sigma_n^2$, derive the MLE and its sampling covariance. Show that it reduces to §6 when all $\sigma_n$ are equal, and explain in one sentence why the weights are $1/\sigma_n^2$ rather than $1/\sigma_n$.

**P5 (influence).** For the Gaussian, Laplace, and Student-$t$ likelihoods of Lecture 3 §4, compute $\partial(-\log p)/\partial f$ as a function of the residual $r = y-f$ and plot all three. Which are bounded? Which tends to zero for large $|r|$? Using your plot, explain precisely what is meant by the claim that a single outlier can move a least-squares fit arbitrarily far. *No optimization required.*

**P6 (degeneracy).** Construct a two-feature regression dataset with $N=50$ on which $\hat w_{\mathrm{LS}}$ is numerically meaningless while the predictions $\Phi\hat w$ remain accurate. Report $\kappa(\Phi)$, both singular values, and the ridge shrinkage factors at $\lambda\in\{10^{-6},10^{-2},1\}$. Explain why predictions can be well determined when parameters are not.

**P7 (reparameterization).** Let $\theta>0$ carry a Gamma$(a,b)$ prior, with a single Gaussian observation. Compute $\hat\theta_{\mathrm{MAP}}$. Then compute $\hat\eta_{\mathrm{MAP}}$ under the prior induced on $\eta = \log\theta$, and show that $e^{\hat\eta_{\mathrm{MAP}}}\ne\hat\theta_{\mathrm{MAP}}$. Verify numerically that the posterior *mean* of $\theta$ is unaffected by the choice of coordinate.

**P8 (typical sets).** For $\theta\sim\mathcal{N}(0,I_d)$, estimate $\Pr[\|\theta\|<1]$ by Monte Carlo for $d = 1, 10, 100$. Plot the histogram of $\|\theta\|$ for $d = 100$ and mark the location of the mode of the density of $\theta$. Write two sentences on what this implies for prediction at a MAP estimate.
