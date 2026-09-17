# Lecture 7 — Bayesian Linear Regression and the Two Kinds of Uncertainty

**ENM 5310 — Data-driven Modeling and Probabilistic Scientific Computing**

**Builds on:** Lecture 2 (Bayes' rule, conditioning, Gaussian conditionals), Lecture 4 (maximum likelihood, the normal equations, the sampling distribution of $\hat w$, the SVD and conditioning), Lecture 5 (MAP, ridge as a Gaussian prior, why the mode is a defective summary), Lecture 6 (optimization).

**Reading:** Murphy I §3.3 (joint Gaussians and linear Gaussian systems), §4.6 (Bayesian statistics), §11.7 (Bayesian linear regression). Preview for next week: §17.1–17.2.

**Purpose.** Everything so far has produced a *point*: $\hat\theta_{\mathrm{MLE}}$, then $\hat\theta_{\mathrm{MAP}}$, then algorithms for finding them. This lecture keeps the whole posterior instead, works it out in closed form for linear regression, and uses it to separate the two kinds of predictive uncertainty. It ends in function space, which is where Gaussian processes begin.

---

### Notation

Data $\mathcal{D} = \{(x_n,y_n)\}_{n=1}^N$. Design matrix $\Phi\in\mathbb{R}^{N\times D}$ with rows $\phi(x_n)^\top$; targets $y\in\mathbb{R}^N$. Weights $w\in\mathbb{R}^D$. Noise variance $\sigma^2$; prior mean and covariance $m_0, S_0$; posterior mean and covariance $m_N, S_N$. A test input is $x^\star$ with feature vector $\phi^\star = \phi(x^\star)$, latent function value $f^\star = w^\top\phi^\star$, and observation $y^\star$.

---

## 1. Why a point estimate is not enough

Lectures 4 and 5 produced two estimators and Lecture 6 produced algorithms for computing them. All of it returns a single vector $\hat w$. Three things that vector cannot do.

**It cannot say how much the data constrained it.** Lecture 4 §6.1 did give a covariance, $\hat w\sim\mathcal{N}(w^\star,\sigma^2(\Phi^\top\Phi)^{-1})$, but read what that statement is *about*: it describes how $\hat w$ would vary across hypothetical repetitions of the experiment. It is a statement about a procedure, not about the dataset in front of you, and it requires the model to be well specified and $w^\star$ to exist. We would like to say something about $w$ given *this* data.

**It cannot propagate uncertainty into predictions.** Plugging $\hat w$ into $p(y^\star\mid x^\star,\hat w) = \mathcal{N}(\hat w^\top\phi^\star,\sigma^2)$ reports the same predictive width everywhere — right next to a dense cluster of training points and a thousand units outside the data range. That is obviously wrong, and it is wrong in the direction that causes harm: overconfidence exactly where the model knows least.

**It cannot compare models.** Choosing between a cubic and a degree-ten polynomial by comparing maximized likelihoods always selects the larger model. Something else is needed, and §7 supplies it.

Lecture 5 §4 added two more objections specific to the mode: MAP is not invariant to reparameterization, and in high dimensions the mode lies outside the typical set, so it can be an unrepresentative point of the very distribution it summarizes.

**The alternative is not a better point estimate. It is to stop producing a point.** Treat $w$ as a random variable, put a distribution on it before seeing data, and condition on the data. This changes the character of the problem in a way worth stating explicitly, because Lecture 6 was entirely about the opposite:

> **Learning is conditioning, not optimizing.** There is no objective to minimize and no step size to choose. The computational difficulty moves from optimization to **integration** — and the reason most of this course is not Bayesian is that the integrals are usually intractable. Linear regression with a Gaussian prior is one of the rare cases where they are not, which is why we do it in full.

---

## 2. The Bayesian machinery

Four objects. Fix them now; every Bayesian method in the course is an attempt to compute one of them.

**The posterior.** By Bayes' rule from Lecture 2,

$$p(w\mid\mathcal{D}) = \frac{p(\mathcal{D}\mid w)\,p(w)}{p(\mathcal{D})}, \qquad p(\mathcal{D}) = \int p(\mathcal{D}\mid w)\,p(w)\,dw .$$

The numerator is the two ingredients we already have: the likelihood from Lecture 3 and the prior from Lecture 5. The denominator is an integral over all of parameter space, and it is the source of essentially every computational difficulty in Bayesian inference.

**The posterior predictive.** What we actually want is a distribution over a new observation, not over parameters:

$$\boxed{\ p(y^\star\mid x^\star,\mathcal{D}) = \int p(y^\star\mid x^\star,w)\,p(w\mid\mathcal{D})\,dw .\ }$$

Read it as an **average of predictions, weighted by posterior plausibility**. Every parameter value gets a vote, weighted by how well it explains the data. Contrast the **plug-in approximation** $p(y^\star\mid x^\star,\hat w)$, which gives one parameter value all the votes. §6 quantifies exactly what that discards.

**The marginal likelihood (evidence).** The normalizer $p(\mathcal{D})$, viewed as a function of modeling choices rather than a constant, scores a whole model against the data. §7.

**Credible intervals.** A $95\%$ credible interval is any region $C$ with $\int_C p(w\mid\mathcal{D})\,dw = 0.95$ — a direct probability statement about $w$ given the data observed. This is not what a confidence interval says. A confidence interval is a statement about a procedure: intervals constructed this way cover the truth $95\%$ of the time across hypothetical repetitions. The two coincide numerically for the Gaussian linear model under a flat prior (§5), which is a coincidence of that model, not a general fact, and it is the source of a great deal of confusion.

---

## 3. Two Gaussian facts

Both were established in Lecture 2. They are restated here because §§4–8 are nothing but repeated application of them.

> **Fact 1 (marginals and conditionals).** If $\begin{pmatrix}z_1\\z_2\end{pmatrix}\sim\mathcal{N}\!\left(\begin{pmatrix}\mu_1\\\mu_2\end{pmatrix},\begin{pmatrix}\Sigma_{11}&\Sigma_{12}\\\Sigma_{21}&\Sigma_{22}\end{pmatrix}\right)$, then both the marginal $p(z_1)$ and the conditional $p(z_1\mid z_2)$ are Gaussian, with
> $$p(z_1) = \mathcal{N}(\mu_1,\Sigma_{11}), \qquad p(z_1\mid z_2) = \mathcal{N}\big(\mu_1+\Sigma_{12}\Sigma_{22}^{-1}(z_2-\mu_2),\ \ \Sigma_{11}-\Sigma_{12}\Sigma_{22}^{-1}\Sigma_{21}\big).$$
> Note the conditional covariance does not depend on the *value* of $z_2$ — only on which variable was observed.

> **Fact 2 (linear Gaussian systems).** If $p(z) = \mathcal{N}(z\mid\mu_z,\Sigma_z)$ and $p(y\mid z) = \mathcal{N}(y\mid Az+b,\Sigma_y)$, then
> $$p(z\mid y) = \mathcal{N}(z\mid\mu_{\mathrm{post}},\Sigma_{\mathrm{post}}), \qquad \Sigma_{\mathrm{post}}^{-1} = \Sigma_z^{-1}+A^\top\Sigma_y^{-1}A, \quad \mu_{\mathrm{post}} = \Sigma_{\mathrm{post}}\big[A^\top\Sigma_y^{-1}(y-b)+\Sigma_z^{-1}\mu_z\big],$$
> and the marginal is $\ p(y) = \mathcal{N}\big(y\mid A\mu_z+b,\ \Sigma_y+A\Sigma_z A^\top\big)$.

**The Gaussian family is closed under conditioning and marginalization**, which is the entire reason closed-form Bayesian inference exists for this model. Linear regression is a linear Gaussian system with $z = w$, $A = \Phi$, $b = 0$, $\Sigma_y = \sigma^2I$ — so Fact 2 already contains everything in §4. We derive it directly anyway, because the completing-the-square manoeuvre is one you will need again.

---

## 4. The posterior for Bayesian linear regression

**The model.** From Lecture 3, with a Gaussian prior from Lecture 5:

$$p(y\mid X,w) = \mathcal{N}\big(y\mid\Phi w,\ \sigma^2 I_N\big), \qquad p(w) = \mathcal{N}\big(w\mid m_0, S_0\big).$$

Treat $\sigma^2$ as known for now; §7 estimates it. The usual choice is $m_0 = 0$, $S_0 = \tau^2 I$, meaning "I expect the coefficients to be of order $\tau$, centered on zero, with no preferred direction."

**The derivation.** Work with the log posterior and drop every term free of $w$, since the posterior must normalize and the normalizer is therefore determined by the $w$-dependent part:

$$\log p(w\mid\mathcal{D}) = -\frac{1}{2\sigma^2}(y-\Phi w)^\top(y-\Phi w) - \frac12(w-m_0)^\top S_0^{-1}(w-m_0) + \text{const}.$$

Expand both quadratics and collect powers of $w$:

$$
\begin{aligned}
-\frac{1}{2\sigma^2}\big[\,y^\top y - 2w^\top\Phi^\top y + w^\top\Phi^\top\Phi w\,\big] &\ -\ \frac12\big[\,w^\top S_0^{-1}w - 2w^\top S_0^{-1}m_0 + m_0^\top S_0^{-1}m_0\,\big] \\[4pt]
= \ -\tfrac12\,w^\top\underbrace{\big[S_0^{-1}+\sigma^{-2}\Phi^\top\Phi\big]}_{\text{quadratic}}w \ &+\ w^\top\underbrace{\big[S_0^{-1}m_0+\sigma^{-2}\Phi^\top y\big]}_{\text{linear}} \ +\ \text{const}.
\end{aligned}
$$

Now compare against a generic Gaussian $\mathcal{N}(w\mid m_N,S_N)$, whose log density is

$$-\tfrac12(w-m_N)^\top S_N^{-1}(w-m_N) + \text{const} = -\tfrac12 w^\top S_N^{-1}w + w^\top S_N^{-1}m_N + \text{const}.$$

The log posterior is a quadratic in $w$, so the posterior **is** Gaussian, and matching coefficients term by term gives the answer with no further work:

$$\boxed{\ S_N^{-1} = S_0^{-1} + \sigma^{-2}\Phi^\top\Phi, \qquad m_N = S_N\big(S_0^{-1}m_0 + \sigma^{-2}\Phi^\top y\big).\ }$$

**Read the first equation.** $S^{-1}$ is the **precision** — the inverse covariance — and the rule says

$$\text{posterior precision} = \text{prior precision} + \text{data precision}.$$

**Information is additive in precision, not in variance.** Each datum contributes $\sigma^{-2}\phi_n\phi_n^\top$, a rank-one term pointing along its own feature direction. A direction of weight space that no data point has a component along receives no contribution at all and retains exactly its prior precision — which is the formal version of Lecture 5 §1's statement that the likelihood is silent where the data is silent. Only now the prior is doing something visible rather than merely repairing a degeneracy.

**Read the second equation.** $m_N$ is a precision-weighted average of the prior mean and the data. With $m_0 = 0$ and $S_0 = \tau^2 I$:

$$S_N = \big(\tau^{-2}I + \sigma^{-2}\Phi^\top\Phi\big)^{-1}, \qquad m_N = \sigma^{-2}S_N\Phi^\top y = \big(\Phi^\top\Phi + \tfrac{\sigma^2}{\tau^2}I\big)^{-1}\Phi^\top y .$$

That last expression is the **ridge estimate** of Lecture 5 §2.1, with $\lambda = \sigma^2/\tau^2$. And because a Gaussian's mean equals its mode, the posterior mean *is* the MAP estimate. So ridge regression, MAP estimation, and the Bayesian posterior mean are three names for one vector.

**The new content is $S_N$.** MAP computed $m_N$ and discarded everything else. The covariance is what the rest of the lecture is about.

---

## 5. Limits, connections, and sequential updating

### 5.1 Four limits, each worth checking

**Flat prior, $\tau\to\infty$.** Then $S_0^{-1}\to0$ and

$$S_N \to \sigma^2(\Phi^\top\Phi)^{-1}, \qquad m_N\to(\Phi^\top\Phi)^{-1}\Phi^\top y = \hat w_{\mathrm{MLE}} .$$

Compare Lecture 4 §6.1: the frequentist **sampling covariance** of $\hat w$ is the same matrix. The two are answers to genuinely different questions — "how would my estimate move if I re-ran the experiment" versus "what should I believe about $w$ given this data" — and they agree here as a special property of the Gaussian linear model, not as a general theorem. This is also why credible and confidence intervals coincide for this model and generally do not.

**No data, $N = 0$.** $S_N = S_0$, $m_N = m_0$: the posterior is the prior. The machinery degrades gracefully.

**Much data, $N\to\infty$.** Since $\Phi^\top\Phi = \sum_n\phi_n\phi_n^\top\approx N\,\mathbb{E}[\phi\phi^\top]$,

$$S_N \approx \frac{\sigma^2}{N}\big(\mathbb{E}[\phi\phi^\top]\big)^{-1} = O(1/N).$$

The posterior contracts at rate $1/N$ — the $N^{-1/2}$ rate of Lecture 1 once more, now describing the width of a belief. The prior's contribution, fixed at $S_0^{-1}$, is swamped by the growing $\sigma^{-2}\Phi^\top\Phi$. **The prior matters when data is scarce and washes out when it is plentiful**, which is the correct behavior and is worth verifying rather than asserting.

**Noiseless limit, $\sigma^2\to0$.** The data precision diverges and the posterior concentrates on weights that interpolate the observations exactly. This is the limit in which GP regression becomes exact interpolation next week.

### 5.2 Sequential updating

The posterior precision is a sum over data points:

$$S_N^{-1} = S_0^{-1} + \sigma^{-2}\sum_{n=1}^{N}\phi_n\phi_n^\top .$$

So the data can be absorbed in any order and in any grouping. Split $\mathcal{D}$ into $\mathcal{D}_1$ and $\mathcal{D}_2$; then the posterior after $\mathcal{D}_1$, used as the prior for $\mathcal{D}_2$, gives exactly the posterior after all of $\mathcal{D}$. In particular data can be absorbed **one point at a time**, each contributing a rank-one update to the precision.

This is more than a computational convenience — it is the clearest possible statement of what the Bayesian framework means by learning:

> **Learning is the repeated transformation of one distribution into another by conditioning.** The prior for the next observation is the posterior from the last.

The picture to hold: with a two-parameter model ($y = w_0+w_1x$), plot the posterior over $(w_0,w_1)$ as contours for $N = 0,1,2,5,20$, alongside samples of the corresponding lines drawn from each posterior. At $N=0$ the lines are a random scatter; after one point they all pass near it while pivoting freely; after twenty they are a tight bundle. Checkpoint D asks you to produce this.

The same structure is the Kalman filter, which is Bayesian linear regression with the parameters allowed to drift between observations. If you have met the Kalman filter as a separate subject, it is this.

---

## 6. The posterior predictive and the two kinds of uncertainty

### 6.1 Derivation

We want $p(y^\star\mid x^\star,\mathcal{D}) = \int\mathcal{N}(y^\star\mid w^\top\phi^\star,\sigma^2)\,\mathcal{N}(w\mid m_N,S_N)\,dw$. Rather than grinding through the integral, note that under the posterior,

$$y^\star = \underbrace{w^\top\phi^\star}_{\text{linear in }w\sim\mathcal{N}(m_N,S_N)} + \underbrace{\epsilon}_{\mathcal{N}(0,\sigma^2),\ \text{independent}},$$

which is a linear function of a Gaussian plus an independent Gaussian, hence Gaussian, with mean and variance obtained by linearity:

$$\boxed{\ p(y^\star\mid x^\star,\mathcal{D}) = \mathcal{N}\Big(y^\star \,\Big|\, \underbrace{m_N^\top\phi^\star}_{\text{prediction}},\ \ \underbrace{\sigma^2}_{\text{aleatoric}} + \underbrace{\phi^{\star\top}S_N\phi^\star}_{\text{epistemic}}\Big).\ }$$

**Distinguish two predictive distributions**, because the difference is a constant source of confusion and it will matter enormously for GPs:

$$p(f^\star\mid x^\star,\mathcal{D}) = \mathcal{N}\big(m_N^\top\phi^\star,\ \phi^{\star\top}S_N\phi^\star\big) \qquad\text{(the latent function)},$$
$$p(y^\star\mid x^\star,\mathcal{D}) = \mathcal{N}\big(m_N^\top\phi^\star,\ \sigma^2+\phi^{\star\top}S_N\phi^\star\big) \qquad\text{(a new observation)}.$$

The first answers "what is the underlying trend here"; the second answers "what will I measure if I go and measure". Plot the wrong one and your error bands will be wrong by exactly $\sigma^2$.

### 6.2 The decomposition

> **Aleatoric uncertainty** — $\sigma^2$ — is the irreducible scatter the model attributes to the observation process itself. It does not depend on $x^\star$; it does not shrink as $N$ grows; collecting more data will never reduce it. It is a *modeling assumption*, imported when Lecture 3 chose a Gaussian noise term.
>
> **Epistemic uncertainty** — $\phi^{\star\top}S_N\phi^\star$ — is uncertainty about $w$, pushed through the model to the prediction. It depends strongly on $x^\star$; it shrinks like $O(1/N)$ by §5.1; and it is *reducible*, by collecting data in the right places.

The contrast is the point. One term is a statement about the world and is fixed; the other is a statement about our knowledge and is a function of what we have looked at. Under the plug-in approximation only the first survives — which is why plug-in predictions have constant-width error bars and are equally confident everywhere.

**Why the epistemic term grows away from the data.** Diagonalize as in Lecture 4 §9: $S_N$ is large along directions of weight space that the data has failed to constrain, and small along well-constrained ones. Then $\phi^{\star\top}S_N\phi^\star$ is large exactly when $\phi(x^\star)$ has a substantial component along a poorly-constrained direction — i.e. when predicting at $x^\star$ requires knowing something the data never told us. Near the training inputs, $\phi(x^\star)$ lies mostly in the span the data has pinned down, and the term is small. This is the same spectrum $\{d_j\}$ that set ridge shrinkage in Lecture 4 and the condition number in Lecture 6, now controlling error bars.

**The rates, stated together.** As $N\to\infty$ with data spread over the input domain, the epistemic term vanishes as $O(1/N)$ and the total predictive variance tends to $\sigma^2$. In the limit of infinite data you learn the function exactly and remain permanently uncertain about the next measurement. If your predictive intervals do not show that behavior, something is wrong.

### 6.3 An honest caveat

The epistemic variance answers a narrower question than students usually assume. It answers: *given this fixed basis $\phi$, how much do the data constrain $w$?* It says nothing about whether the basis was right.

The failure is easy to exhibit and worth carrying around. Take a basis of localized bumps — radial basis functions centered on the training inputs — and evaluate far outside the data. Then $\phi(x^\star)\to0$, so the predictive mean $m_N^\top\phi^\star\to0$ *and* the epistemic variance $\phi^{\star\top}S_N\phi^\star\to0$. The model returns "the answer is zero, and I am certain." That is the worst possible output, and it is produced by correct application of the formulas.

**Model misspecification is not captured by the posterior.** The posterior is a distribution over $w$ *within* the assumed family, and it cannot express doubt about the family itself. Section 7's marginal likelihood is the beginning of an answer; honest extrapolation behavior — GP priors whose variance reverts to the prior away from data — is a large part of why the next section of the course exists.

### 6.4 What the two terms are for

Both feed directly into the next few weeks. **Decision-making under squared loss** uses the predictive mean, which is optimal regardless of the variance. **Active learning and experimental design** use the epistemic term alone: you gain nothing by sampling where uncertainty is purely aleatoric, so the acquisition functions of Bayesian optimization are built from $\phi^{\star\top}S_N\phi^\star$, not from the total. **Detecting distribution shift** uses epistemic variance as the signal that a query is unlike anything seen.

One plotting instruction. When you display a posterior, draw **samples of functions** from it, not only the mean and a $\pm2\sigma$ band. The band shows the marginal variance at each $x$ separately and hides all correlation structure; two posteriors with identical bands can have completely different sets of plausible functions. Function samples show what the model actually believes.

---

## 7. The marginal likelihood and empirical Bayes

We have been treating $\sigma^2$ and $\tau^2$ as known. They are not. The Bayesian answer is to integrate the parameters out and score the remaining choices.

By Fact 2's marginal, or directly from $y = \Phi w+\epsilon$ with $w\sim\mathcal{N}(m_0,S_0)$ and $\epsilon\sim\mathcal{N}(0,\sigma^2I)$ independent,

$$\boxed{\ p(y\mid X,\sigma^2,\tau^2) = \mathcal{N}\big(y \,\big|\, \Phi m_0,\ \underbrace{\sigma^2 I + \Phi S_0\Phi^\top}_{\textstyle C}\big),\ }$$

which for $m_0 = 0$, $S_0 = \tau^2 I$ is $\mathcal{N}(y\mid 0,\ \sigma^2I+\tau^2\Phi\Phi^\top)$. Note what has happened: **$w$ has been integrated away entirely.** What remains is a single Gaussian over the observed targets, with a covariance built from the features. Remember that matrix $\tau^2\Phi\Phi^\top$ — §8 gives it a name.

The log marginal likelihood is

$$\log p(y\mid X) = \underbrace{-\tfrac12 y^\top C^{-1}y}_{\text{data fit}} \;\underbrace{-\;\tfrac12\log|C|}_{\text{complexity penalty}} \;-\;\tfrac N2\log2\pi .$$

**The two terms trade off automatically.** A model with a broad prior (large $\tau^2$) can fit many datasets, so it spreads its probability mass thinly; that shows up as a large $\log|C|$ and is penalized. A model with a narrow prior concentrates mass but fits the observed $y$ poorly, inflating $y^\top C^{-1}y$. The maximum sits between. This is **Occam's razor arising from normalization**, not from an added penalty — the marginal likelihood is a probability distribution over datasets, and a model that hedges across many possible datasets necessarily assigns less to the one observed.

**Empirical Bayes (type-II maximum likelihood).** Choose the hyperparameters by maximizing this quantity:

$$(\hat\sigma^2,\hat\tau^2) = \arg\max_{\sigma^2,\tau^2}\ \log p(y\mid X,\sigma^2,\tau^2).$$

Since $\lambda = \sigma^2/\tau^2$ is the ridge penalty of Lecture 5, this is an alternative to cross-validating for $\lambda$ — one that uses no held-out data and requires no data splitting. Two cautions. It is maximum likelihood at a higher level, so it inherits the usual overfitting risk when there are many hyperparameters. And it is not a substitute for a test set: the marginal likelihood is computed under the assumed model, so it can rank two wrong models against each other without noticing that both are wrong.

Next week this exact objective, with $C$ built from a kernel instead of a feature matrix, is how every GP hyperparameter gets learned. Getting comfortable with it now is the single best preparation for that material.

---

## 8. From weight space to function space

This section is the bridge, and it is the reason the lecture is structured as it is.

**A prior on weights is a prior on functions.** Take $w\sim\mathcal{N}(0,\tau^2 I)$ and look at the vector of function values at the training inputs, $f = \Phi w$. A linear map of a Gaussian is Gaussian:

$$\mathbb{E}[f] = 0, \qquad \operatorname{Cov}[f] = \Phi\,(\tau^2 I)\,\Phi^\top = \tau^2\Phi\Phi^\top \qquad\Longrightarrow\qquad f\sim\mathcal{N}\big(0,\ \tau^2\Phi\Phi^\top\big).$$

Look at the entries of that covariance matrix:

$$\big[\tau^2\Phi\Phi^\top\big]_{ij} = \tau^2\,\phi(x_i)^\top\phi(x_j) \;\triangleq\; k(x_i,x_j).$$

> **Kernel.** $k(x,x') = \tau^2\,\phi(x)^\top\phi(x')$ — the prior covariance between the function's values at two inputs.

So: **a Gaussian prior over weights, together with a model linear in those weights, is exactly a Gaussian prior over function values whose covariance is an inner product of feature vectors.** The covariance between $f(x)$ and $f(x')$ depends only on how similar $x$ and $x'$ look through $\phi$ — nearby inputs get correlated function values, distant ones do not. That is a statement about smoothness, expressed without ever mentioning weights.

**The predictive distribution can be written the same way.** Applying the matrix inversion lemma to §6's expressions, with $K_{ij} = k(x_i,x_j)$, $[k^\star]_n = k(x_n,x^\star)$, and $k^{\star\star} = k(x^\star,x^\star)$:

$$\mathbb{E}[f^\star\mid\mathcal{D}] = k^{\star\top}\big(K+\sigma^2I\big)^{-1}y, \qquad \operatorname{Var}[f^\star\mid\mathcal{D}] = k^{\star\star} - k^{\star\top}\big(K+\sigma^2I\big)^{-1}k^\star .$$

These are algebraically identical to $m_N^\top\phi^\star$ and $\phi^{\star\top}S_N\phi^\star$ — the same predictions, rewritten. **But $w$, $\Phi$, and $D$ have all disappeared.** Everything is expressed through evaluations of $k$.

Three consequences, which together are the case for Gaussian processes.

**Cost changes from $O(D^3)$ to $O(N^3)$.** The weight-space form inverts a $D\times D$ matrix; the function-space form inverts an $N\times N$ one. When $D\gg N$ — many basis functions, modest data — the second is far cheaper. This is the same primal/dual switch that makes kernel methods work generally.

**$D$ may be infinite.** Nothing above requires $\phi$ to have finitely many components, only that the inner product $\phi(x)^\top\phi(x')$ exists and is computable. Some kernels correspond to infinite-dimensional feature maps, and you get the benefits of an infinite basis at finite cost. The squared-exponential kernel is the standard example.

**Specifying $k$ is more natural than specifying $\phi$.** Choosing a basis means answering "which functions should I add up"; choosing a kernel means answering "how similar should the function values at these two inputs be" — which is usually the question you can actually answer about a physical system. Smoothness, length scales, periodicity, and amplitude all become explicit kernel hyperparameters, fitted by the marginal likelihood of §7.

**Where this goes.** A Gaussian process is the object obtained by taking this construction seriously for *all* inputs at once: a distribution over functions such that any finite collection of function values is jointly Gaussian with covariance given by $k$. Everything in §§4–7 carries over — the same conditioning, the same aleatoric/epistemic split, the same marginal likelihood. What changes is that the epistemic variance now reverts to the prior away from the data instead of collapsing to zero, which repairs the failure of §6.3.

---

## 9. Where this leaves us

The posterior was computable here because both prior and likelihood were Gaussian and the model was linear in the parameters, so the log posterior was quadratic and completing the square finished the job. That is a narrow set of conditions. Change the likelihood to Bernoulli — logistic regression, which we have already built — and the posterior has no closed form, because the sigmoid is not conjugate to anything.

Three responses to that, and they organize the rest of the Bayesian material in this course: **approximate** the posterior with something tractable (the Laplace approximation, variational inference), **sample** from it without ever normalizing it (MCMC), or **change the model** so that tractability is preserved. Next week takes the third route, and the result is Gaussian processes.

---

## Checkpoint D (take-home, due before the next lecture)

`numpy` only. Report fitted numbers, not pictures that look about right.

1. **Implement.** Code $S_N$ and $m_N$ from §4 for $m_0 = 0$, $S_0 = \tau^2I$. Verify to machine precision that $m_N$ equals the ridge solution with $\lambda = \sigma^2/\tau^2$ computed by an independent route, and that $S_N\to\sigma^2(\Phi^\top\Phi)^{-1}$ as $\tau^2$ grows — report the relative Frobenius error at $\tau^2\in\{10^2,10^4,10^6\}$.
2. **Sequential updating.** For $y = w_0+w_1x+\epsilon$ with known $\sigma^2$, plot posterior contours over $(w_0,w_1)$ for $N = 0,1,2,5,20$, and beside each panel plot six lines drawn from that posterior together with the data seen so far. Confirm numerically that absorbing the data in two batches gives the same $(m_N,S_N)$ as absorbing it all at once.
3. **The decomposition.** With a polynomial or Fourier basis and data on $x\in[-1,1]$, plot the predictive mean with two bands: $\pm2\sqrt{\phi^{\star\top}S_N\phi^\star}$ and $\pm2\sqrt{\sigma^2+\phi^{\star\top}S_N\phi^\star}$. Report the epistemic variance at a point inside the data and at $x = 2$, for $N\in\{5,20,100\}$. Verify the $O(1/N)$ rate by fitting a slope on log–log axes, and confirm the aleatoric part is constant.
4. **Function samples versus bands.** For one fitted model, overlay 20 functions sampled from the posterior on the $\pm2\sigma$ band. Then construct a second posterior with a visibly different correlation structure but a similar band, and show both. One sentence on what the band fails to convey.
5. **The failure of §6.3.** Repeat item 3 with a basis of RBFs centered on the training inputs. Report the predictive mean and epistemic variance at $x = 10$. Explain in two sentences why the model is confident, and what would have to change for it to be appropriately uncertain.
6. **Marginal likelihood.** Evaluate $\log p(y\mid X,\sigma^2,\tau^2)$ on a grid and report the maximizing pair. Compare the implied $\lambda = \hat\sigma^2/\hat\tau^2$ against the $\lambda$ selected by 5-fold cross-validation, and against the true noise variance you used to generate the data. Then fix the noise and use the evidence to select polynomial degree; report whether it agrees with held-out test error and, if not, where they diverge.

---

## Quiz-eligible facts

1. The posterior predictive $\int p(y^\star\mid x^\star,w)p(w\mid\mathcal{D})dw$ averages predictions weighted by posterior plausibility; the plug-in approximation gives one parameter value all the weight.
2. A credible interval is a probability statement about $w$ given the data; a confidence interval is a statement about a procedure across hypothetical repetitions. They coincide for the Gaussian linear model under a flat prior, and generally do not.
3. Gaussians are closed under conditioning and marginalization; for a linear Gaussian system, $\Sigma_{\mathrm{post}}^{-1} = \Sigma_z^{-1}+A^\top\Sigma_y^{-1}A$ and $p(y) = \mathcal{N}(A\mu_z+b,\ \Sigma_y+A\Sigma_zA^\top)$.
4. For Bayesian linear regression, $S_N^{-1} = S_0^{-1}+\sigma^{-2}\Phi^\top\Phi$ and $m_N = S_N(S_0^{-1}m_0+\sigma^{-2}\Phi^\top y)$: posterior precision is prior precision plus data precision. Information is additive in precision, not variance.
5. With $m_0 = 0$ and $S_0 = \tau^2I$, $m_N$ is the ridge estimate with $\lambda = \sigma^2/\tau^2$; since a Gaussian's mean is its mode, posterior mean, MAP, and ridge coincide. The new content is $S_N$.
6. As $\tau\to\infty$, $S_N\to\sigma^2(\Phi^\top\Phi)^{-1}$ — the same matrix as the MLE's sampling covariance, answering a different question.
7. $S_N = O(1/N)$: the posterior contracts at the Monte Carlo rate and the prior washes out.
8. The posterior precision is a sum over data points, so data can be absorbed sequentially in any order; the posterior after one batch is the prior for the next. This structure is the Kalman filter.
9. $p(y^\star\mid x^\star,\mathcal{D}) = \mathcal{N}(m_N^\top\phi^\star,\ \sigma^2+\phi^{\star\top}S_N\phi^\star)$; for the latent function $f^\star$ the $\sigma^2$ is absent.
10. Aleatoric variance $\sigma^2$ is irreducible, independent of $x^\star$, and a modeling assumption; epistemic variance $\phi^{\star\top}S_N\phi^\star$ depends on $x^\star$, shrinks as $O(1/N)$, and is reducible by collecting data. Total predictive variance tends to $\sigma^2$, not to zero.
11. Epistemic variance is large when $\phi(x^\star)$ has a large component along a direction of weight space the data has not constrained — the same spectrum that governs ridge shrinkage and the condition number.
12. The posterior cannot express doubt about the model class. With a localized basis, extrapolation drives both the mean and the epistemic variance to zero: confidently wrong.
13. Active learning and Bayesian optimization use the epistemic term alone, since sampling where uncertainty is purely aleatoric gains nothing.
14. Plot function samples, not only $\pm2\sigma$ bands: the band shows marginal variance per point and hides correlation structure.
15. $p(y\mid X) = \mathcal{N}(\Phi m_0,\ \sigma^2I+\Phi S_0\Phi^\top)$; its log splits into a data-fit term $-\frac12 y^\top C^{-1}y$ and a complexity penalty $-\frac12\log|C|$. Occam's razor arises from normalization, not from an added penalty.
16. Empirical Bayes maximizes the marginal likelihood over hyperparameters — an alternative to cross-validating for $\lambda$, needing no held-out data, but not a substitute for a test set.
17. A Gaussian prior on weights induces a Gaussian prior on function values with covariance $k(x,x') = \tau^2\phi(x)^\top\phi(x')$.
18. The predictive can be written entirely in terms of $k$: mean $k^{\star\top}(K+\sigma^2I)^{-1}y$, variance $k^{\star\star}-k^{\star\top}(K+\sigma^2I)^{-1}k^\star$. Cost moves from $O(D^3)$ to $O(N^3)$, and $D$ may be infinite.

---

## Practice problems

*Ungraded; these feed the quizzes.*

**P1.** Derive $S_N$ and $m_N$ a second way, by applying Fact 2 directly with $z = w$, $A = \Phi$, $\Sigma_y = \sigma^2I$. Confirm the two derivations agree, and say which you would rather repeat for a model with correlated noise $\Sigma_y\ne\sigma^2I$.

**P2.** Show that absorbing $\mathcal{D}_1$ then $\mathcal{D}_2$ gives the same posterior as absorbing $\mathcal{D} = \mathcal{D}_1\cup\mathcal{D}_2$ at once. Then write the rank-one update for absorbing a single datum $(\phi_n,y_n)$, using the Sherman–Morrison formula to update $S_N$ directly rather than $S_N^{-1}$. What is the cost per datum?

**P3.** Prove that the posterior predictive variance $\sigma^2+\phi^{\star\top}S_N\phi^\star$ is never smaller than $\sigma^2$, and that it is monotonically non-increasing as data is added — i.e. adding a datum can never increase epistemic uncertainty anywhere. Does the same hold for the predictive *mean*?

**P4.** Take $D = 2$ with $\phi(x) = (1,x)^\top$ and a single observation at $x_1 = 0$. Compute $S_N$ explicitly and show that the posterior variance of the slope is unchanged from the prior. Explain the result in one sentence using the rank-one structure of §4.

**P5.** For $S_0 = \tau^2I$, expand $\phi^{\star\top}S_N\phi^\star$ in the eigenbasis of $\Phi^\top\Phi$ using the SVD $\Phi = UDV^\top$. Show the epistemic variance along $v_j$ is $(\tau^{-2}+\sigma^{-2}d_j^2)^{-1}$, and interpret both limits $d_j\to0$ and $d_j\to\infty$.

**P6.** Derive the marginal likelihood by completing the square in $w$ and performing the Gaussian integral explicitly, rather than by citing Fact 2. Identify where the $\log|C|$ term comes from.

**P7.** Verify the kernel form of the predictive numerically: compute $m_N^\top\phi^\star$ and $\phi^{\star\top}S_N\phi^\star$ from §6, then $k^{\star\top}(K+\sigma^2I)^{-1}y$ and $k^{\star\star}-k^{\star\top}(K+\sigma^2I)^{-1}k^\star$ from §8, and confirm agreement to machine precision. Time both for $(N,D) = (50,500)$ and $(500,50)$ and comment.

**P8.** Show that as $\sigma^2\to0$ the posterior mean interpolates the training data exactly when $D\ge N$ and $\Phi$ has full row rank. What happens to $S_N$ in the directions spanned by the data, and in the directions orthogonal to it?
