# Lecture 8 — Gaussian Processes

**ENM 5310 — Data-driven Modeling and Probabilistic Scientific Computing**

**Builds on:** Lecture 2 (Gaussian conditioning), Lecture 4 (maximum likelihood), Lecture 5 (MAP, and the ways maximum likelihood can fail), Lecture 6 (L-BFGS), Lecture 7 §§1–4, 5.1, 6.1–6.2 (Bayesian linear regression: posterior, predictive, aleatoric and epistemic uncertainty).

**Reading:** Rasmussen & Williams, *Gaussian Processes for Machine Learning*, §§2.1–2.3 and §5.4 (freely available at gaussianprocess.org/gpml). Companion: Murphy I §17.1–17.2.

**Note.** The Lecture 7 notes contain sections we did not reach in class. Everything from them that this lecture needs is developed again here. You do not need to have read them.

**Learning objectives.** After this lecture you should be able to:

1. Describe any probabilistic ML method in three steps — model, training, inference — and place MLE, MAP, Bayesian linear regression, and GPs in that frame.
2. Explain why Bayesian linear regression with a localized basis is confidently wrong far from data, and show that the cause is the prior.
3. Show that a Gaussian prior on weights is a Gaussian prior on function values, define a kernel, and explain via Mercer's theorem how a feature map and a kernel each determine the other.
4. State the GP model and interpret the hyperparameters of common kernels.
5. Explain why maximizing the likelihood over the function fails, why integrating the function out fixes it, and why the result — the marginal likelihood — is an ordinary likelihood for the hyperparameters.
6. Explain why maximizing the marginal likelihood prefers the simplest model that fits the data.
7. Derive the GP predictive mean and variance by Gaussian conditioning, and explain why the variance reverts to the prior away from data.
8. Implement GP training and inference with a Cholesky factorization.

### Notation

Training inputs $x_1,\dots,x_N$, targets $y\in\mathbb{R}^N$, noise variance $\sigma^2$. For Bayesian linear regression: basis $\phi(x)\in\mathbb{R}^D$, design matrix $\Phi\in\mathbb{R}^{N\times D}$, weights $w\sim\mathcal{N}(0,\tau^2I)$. For GPs: kernel $k(x,x')$; Gram matrix $K\in\mathbb{R}^{N\times N}$ with $K_{nm} = k(x_n,x_m)$; at a test input $x^\star$, the vector $k^\star\in\mathbb{R}^N$ with entries $k(x_n,x^\star)$ and the scalar $k^{\star\star} = k(x^\star,x^\star)$. Hyperparameters $\theta = (\sigma_f^2,\ell,\sigma^2)$: signal amplitude, length scale, noise variance.

---

## 1. Where we are: Bayesian linear regression in three steps

Every method in this course can be described in three steps.

> **Step 1 — Model.** Write down a probabilistic model of how the data were generated: what is observed, what is unknown, and what distributions connect them.
>
> **Step 2 — Training.** Use the observed data to determine the unknowns.
>
> **Step 3 — Inference.** Use the trained model to make predictions, with uncertainty, at new inputs.

Here is Bayesian linear regression from Lecture 7 in that form.

**Step 1 — Model.** $\ y_n = w^\top\phi(x_n)+\epsilon_n$ with $\epsilon_n\sim\mathcal{N}(0,\sigma^2)$, and prior $w\sim\mathcal{N}(0,\tau^2I)$. The unknowns are the weights $w$ — and also the two numbers $\sigma^2$ and $\tau^2$, which we call **hyperparameters** because they describe distributions rather than the function itself.

**Step 2 — Training.** Compute the posterior over the weights, $w\mid\mathcal{D}\sim\mathcal{N}(m_N,S_N)$, with

$$S_N = \big(\tau^{-2}I+\sigma^{-2}\Phi^\top\Phi\big)^{-1}, \qquad m_N = \sigma^{-2}S_N\Phi^\top y .$$

**Step 3 — Inference.** At a test input $x^\star$ with $\phi^\star = \phi(x^\star)$,

$$p(y^\star\mid x^\star,\mathcal{D}) = \mathcal{N}\Big(y^\star\,\Big|\,m_N^\top\phi^\star,\ \ \underbrace{\sigma^2}_{\text{aleatoric}}+\underbrace{\phi^{\star\top}S_N\phi^\star}_{\text{epistemic}}\Big).$$

Compare the approach of Lectures 4 and 5:

| | MLE / MAP | Bayesian |
|---|---|---|
| **Training** | Find one best $w$ by optimization | Compute a distribution over $w$ |
| **Inference** | Plug in that single $\hat w$ | Average predictions over the posterior |

Two loose ends remain, and this lecture resolves both.

**(a)** Step 2 assumed $\sigma^2$ and $\tau^2$ were known. They never are. How should we choose them? (Answered in §6.)

**(b)** Step 3 claimed the epistemic term grows where the data has not constrained the model. That claim can fail badly. (§2.)

---

## 2. A failure: confidently wrong far from the data

Take a natural basis for smooth one-dimensional regression: **radial basis functions** (RBFs), Gaussian bumps of width $\lambda$ centered at points $c_1,\dots,c_D$,

$$\phi_d(x) = \exp\!\left(-\frac{(x-c_d)^2}{2\lambda^2}\right).$$

Place the centers on a grid covering the training inputs, say $[-1,1]$, fit Bayesian linear regression, and plot the predictive distribution over $[-5,5]$.

Inside $[-1,1]$ the fit looks excellent. The mean follows the data, and the epistemic band is narrow near observations and wider between them.

Now move to $x^\star = 5$. Every basis function is centered far away, so every entry of $\phi(x^\star)$ is essentially zero. Then

$$m_N^\top\phi^\star\approx0 \qquad\text{and}\qquad \phi^{\star\top}S_N\phi^\star\approx0 .$$

The mean falls to zero. **So does the epistemic variance.** The only uncertainty left is the noise $\sigma^2$ — the same width the model reports in the middle of the densest data. The model is claiming that the function equals zero at $x = 5$, and that it is as sure of this as of anything it has seen.

Nothing went wrong in the calculation. The formulas were applied correctly. So the right question is not "where is the bug in the posterior?" but:

> **What did we assume about the function before we saw any data?**

A prior over weights does not answer this directly. We need to translate it into a statement about functions.

---

## 3. From weights to functions

### 3.1 A prior on weights is a prior on functions

Pick any inputs $x_1,\dots,x_M$ and collect the function values into a vector $f = (f(x_1),\dots,f(x_M))^\top$. With $f(x) = w^\top\phi(x)$ this is $f = \Phi w$. A linear function of a Gaussian is Gaussian, so

$$\mathbb{E}[f] = 0, \qquad \operatorname{Cov}[f] = \Phi\,(\tau^2I)\,\Phi^\top = \tau^2\Phi\Phi^\top .$$

Reading off the individual entries:

$$\boxed{\ \operatorname{Cov}\big[f(x),f(x')\big] = \tau^2\,\phi(x)^\top\phi(x'), \qquad \operatorname{Var}\big[f(x)\big] = \tau^2\,\|\phi(x)\|^2 .\ }$$

The prior on $w$ is now a direct statement about the function: how much its value at each input varies, and how its values at different inputs move together.

### 3.2 The failure, diagnosed

For the RBF basis, $\|\phi(x)\|^2 = \sum_d\exp\big(-(x-c_d)^2/\lambda^2\big)$, which goes to zero far from every center. So:

> **Before seeing any data, the prior already said that $f(x)$ is zero, with near-certainty, far from the centers.**

Data could not have fixed this. Recall the conditioning formula from Lecture 2: for jointly Gaussian $(z_1,z_2)$,

$$\operatorname{Cov}[z_1\mid z_2] = \Sigma_{11}-\Sigma_{12}\Sigma_{22}^{-1}\Sigma_{21}.$$

The subtracted term is always positive semi-definite, so the conditional variance is never larger than the unconditional one.

> **Conditioning never increases variance.** Data can only remove uncertainty that the prior contained. Where the prior contained none, the posterior has none either.

The failure in §2 was a failure of the prior. It is hard to see in weight space and obvious in function space. That is the reason to think in terms of functions from here on: it is the view in which our real assumptions — how the function should behave, and how unsure we are, at each input — are written down explicitly.

---

## 4. From features to kernels

### 4.1 Only inner products matter

Look at §3.1 again. The prior over function values depends on the features only through the inner product $\phi(x)^\top\phi(x')$. The individual features never appear alone. (We will see in §7 that the same is true of the predictions.) That suggests skipping the features and specifying the inner product directly.

> **Kernel.** A function $k(x,x')$ giving the prior covariance between function values: $k(x,x') = \operatorname{Cov}[f(x),f(x')]$. From features, $k(x,x') = \tau^2\phi(x)^\top\phi(x')$.
>
> **Gram matrix.** For inputs $x_1,\dots,x_M$, the matrix $K$ with entries $K_{ij} = k(x_i,x_j)$.

Not every function of two inputs can be a kernel. Every Gram matrix must be a valid covariance matrix:

> **Valid kernel.** $k$ is symmetric, and every Gram matrix it produces is positive semi-definite.

Any kernel built from features is valid, and §4.4 shows the converse: every valid kernel comes from some feature map. Sums, products, and positive multiples of valid kernels are also valid, which is how kernels get combined.

### 4.2 The fix, by design

The failure came from the prior variance $k(x,x) = \tau^2\|\phi(x)\|^2$ falling to zero. So ask for a kernel where that cannot happen:

> **Stationary kernel.** $k(x,x')$ depends only on the difference $x-x'$. Then the prior variance $k(x,x)$ is the same at every input, so it never collapses — and by §3.2, neither does the posterior's.

Is there a stationary kernel that corresponds to a sensible feature model? Yes, and it is the RBF model that just failed.

### 4.3 ★ Infinitely many features: the squared-exponential kernel

The RBF model failed because its bumps covered only $[-1,1]$. The obvious fix is to place bumps *everywhere*. In weight space that is impossible — infinitely many weights. But the kernel only needs the sum $\sum_d\phi_d(x)\phi_d(x')$, and that sum has a closed form.

Place centers $c_d = d\Delta$ on a grid of spacing $\Delta$ covering the whole real line. The kernel is

$$k(x,x') = \tau^2\sum_d\exp\!\left(-\frac{(x-c_d)^2}{2\lambda^2}\right)\exp\!\left(-\frac{(x'-c_d)^2}{2\lambda^2}\right).$$

*Combine the exponents.* With $\bar x = (x+x')/2$, one checks by expanding that $(x-c)^2+(x'-c)^2 = 2(c-\bar x)^2+\tfrac12(x-x')^2$. So each term is

$$\exp\!\left(-\frac{(x-x')^2}{4\lambda^2}\right)\exp\!\left(-\frac{(c_d-\bar x)^2}{\lambda^2}\right),$$

and the first factor comes out of the sum.

*Turn the sum into an integral.* For a fine grid, $\Delta\ll\lambda$, the remaining sum is approximately $\frac1\Delta\int\exp\big(-(c-\bar x)^2/\lambda^2\big)dc = \lambda\sqrt\pi/\Delta$.

*Keep the variance finite.* As $\Delta\to0$ this would blow up, so each weight's prior variance must shrink as the grid gets finer: $\tau^2 = \sigma_f^2\Delta/(\lambda\sqrt\pi)$. Writing $\ell = \sqrt2\lambda$, the limit is

$$\boxed{\ k_{\mathrm{SE}}(x,x') = \sigma_f^2\exp\!\left(-\frac{(x-x')^2}{2\ell^2}\right).\ }$$

This is the **squared-exponential kernel**. Three things to notice:

- **It is stationary.** $k(x,x) = \sigma_f^2$ everywhere. The prior is never certain anywhere.
- **It is the failed model, completed.** Nothing new was assumed — just more of the same basis functions, everywhere.
- **The cost moved.** Weight space would need an infinite matrix. As §7 shows, the kernel view needs only an $N\times N$ matrix, however many features there are.

### 4.4 ★ Features and kernels determine each other: Mercer's theorem

So far we have gone in one direction. Given features, the kernel is their inner product (absorbing $\tau^2$ into the features):

$$k(x,x') = \sum_d\phi_d(x)\,\phi_d(x').$$

This always produces a valid kernel. The reverse direction — given a kernel, find features — is Mercer's theorem, and it is easiest to see first for matrices.

**The matrix version, which you already know.** A Gram matrix $K$ is symmetric and positive semi-definite, so it has an eigendecomposition with non-negative eigenvalues:

$$K = \sum_j\lambda_j\,u_ju_j^\top \qquad\Longrightarrow\qquad K_{nm} = \sum_j\big(\sqrt{\lambda_j}\,u_{jn}\big)\big(\sqrt{\lambda_j}\,u_{jm}\big).$$

Read the right-hand side: input $x_n$ has a feature vector with components $\sqrt{\lambda_j}\,u_{jn}$, and $K$ is exactly the matrix of inner products of these feature vectors. **Every valid Gram matrix is secretly a feature model.**

**The function version.** Mercer's theorem says the same thing for the kernel itself, with eigenvectors replaced by eigenfunctions and the matrix product replaced by an integral.

> **Mercer's theorem.** Let $k$ be a continuous valid kernel on a bounded input domain, with inputs weighted by a density $p(x)$. Then there are orthonormal functions $\psi_1,\psi_2,\dots$ and numbers $\lambda_1\ge\lambda_2\ge\dots\ge0$ satisfying
> $$\int k(x,x')\,\psi_j(x')\,p(x')\,dx' = \lambda_j\,\psi_j(x),$$
> such that
> $$k(x,x') = \sum_{j=1}^\infty\lambda_j\,\psi_j(x)\,\psi_j(x').$$

So every valid kernel is an inner product of features, $\phi_j(x) = \sqrt{\lambda_j}\,\psi_j(x)$ — possibly infinitely many of them. The two directions together:

$$\underbrace{\text{features }\phi}_{\text{weight space}}\ \xrightarrow{\ \ \text{inner product}\ \ }\ \underbrace{\text{kernel }k}_{\text{function space}}\ \xrightarrow{\ \ \text{Mercer eigendecomposition}\ \ }\ \text{features }\sqrt{\lambda_j}\,\psi_j .$$

Three consequences.

**Every GP is Bayesian linear regression, always.** The random function
$$f(x) = \sum_j w_j\sqrt{\lambda_j}\,\psi_j(x), \qquad w_j\sim\mathcal{N}(0,1)\ \text{independent},$$
has covariance exactly $k$. The weight-space and function-space views of §3 are not just related for special kernels; they are equivalent for every valid kernel.

**The eigenvalues reveal the assumptions.** $\lambda_j$ is how much prior variance the kernel puts on its $j$-th feature, and for the stationary kernels in this lecture, later features oscillate faster. For the squared-exponential the $\lambda_j$ decay extremely fast, so almost all prior variance sits on a few smooth features — hence very smooth samples. For Matérn kernels they decay more slowly, leaving more weight on oscillatory features — hence rougher samples. The table in §5.3 is really a statement about how fast the eigenvalues decay.

**Truncation gives a cheaper model.** Keeping only the first $M$ features, $k(x,x')\approx\sum_{j\le M}\lambda_j\psi_j(x)\psi_j(x')$, turns the GP back into Bayesian linear regression with $M$ features, whose training cost is $O(NM^2+M^3)$ instead of $O(N^3)$. In practice the $\psi_j$ are approximated from the eigenvectors of a Gram matrix on a sample of inputs — the matrix version above — and this is one of the main routes to scalable GPs.

One caution: the feature map is not unique. Replacing $\phi$ by $Q\phi$ for any orthogonal matrix $Q$ leaves every inner product, and so the kernel, unchanged. The kernel is the object the model actually depends on — one more reason to specify it directly.

---

## 5. Step 1 — The model: a Gaussian process prior

### 5.1 What a Gaussian process is

> **Gaussian process.** A random function $f$ such that, for any finite set of inputs, the vector of function values is jointly Gaussian. It is specified by a mean function $m(x)$ and a kernel $k(x,x')$, and we write $f\sim\mathcal{GP}(m,k)$.

In practice you never handle a whole function. You only evaluate it at finitely many inputs, and there it is an ordinary multivariate Gaussian: $\big(f(x_1),\dots,f(x_M)\big)\sim\mathcal{N}(0,K)$ when $m = 0$. Everything in this lecture is Lecture 2's Gaussian algebra applied to such a vector. We use $m\equiv0$ throughout.

### 5.2 The model

$$\boxed{\ \textbf{Step 1:}\qquad f\sim\mathcal{GP}(0,k_\theta), \qquad y_n = f(x_n)+\epsilon_n, \qquad \epsilon_n\sim\mathcal{N}(0,\sigma^2).\ }$$

At the training inputs this says two simple things. Write $f = (f(x_1),\dots,f(x_N))^\top$. Then

$$\text{prior:}\quad f\sim\mathcal{N}(0,K_\theta), \qquad\qquad \text{likelihood:}\quad y\mid f\sim\mathcal{N}(f,\sigma^2I).$$

**The unknowns** are the function $f$ and the hyperparameters $\theta = (\sigma_f^2,\ell,\sigma^2)$. There are no weights. The function itself plays the role that $w$ played in Lecture 7.

### 5.3 Choosing a kernel is choosing assumptions

Write $r = |x-x'|$. Each kernel below has an **amplitude** $\sigma_f^2$ — how large the function typically is — and a **length scale** $\ell$ — roughly, how far you must move before the function's value changes substantially.

| Kernel | $k(x,x')$ | What prior samples look like |
|---|---|---|
| Squared-exponential | $\sigma_f^2\exp\!\big(-\frac{r^2}{2\ell^2}\big)$ | Extremely smooth |
| Matérn-1/2 | $\sigma_f^2\exp\!\big(-\frac r\ell\big)$ | Rough and jagged |
| Matérn-5/2 | $\sigma_f^2\big(1+\frac{\sqrt5r}{\ell}+\frac{5r^2}{3\ell^2}\big)\exp\!\big(-\frac{\sqrt5r}{\ell}\big)$ | Smooth, twice differentiable |
| Linear | $\tau^2\phi(x)^\top\phi(x')$ | Exactly Bayesian linear regression |

Three remarks.

**Always look at prior samples before fitting.** Build $K$ on a fine grid of inputs, factor it as $K = LL^\top$ (Cholesky), and draw $f = Lz$ with $z\sim\mathcal{N}(0,I)$. A handful of draws shows exactly what the kernel assumes.

**Smoothness is an assumption.** The squared-exponential assumes the function is infinitely smooth, which is stronger than most physical quantities justify. **Matérn-5/2 is a better default for physical data.**

**The linear kernel recovers Lecture 7 exactly.** A GP with $k = \tau^2\phi^\top\phi'$ makes the same predictions as Bayesian linear regression with basis $\phi$. GPs do not replace what we did; they contain it.

---

## 6. Step 2 — Training: fitting the hyperparameters

### 6.1 What is there to learn?

Two kinds of unknowns came out of Step 1, and they are treated differently.

- **The function $f$.** Infinitely many values, and the data tells us about only a few of them. Following Lecture 7, we keep a *full distribution* over it and never commit to a single function. That happens in Step 3.
- **The hyperparameters $\theta$.** A handful of numbers — three, here — describing the *kind* of function. The data typically pins these down well, so we pick single best values, in the spirit of Lecture 4.

So Step 2 is: **find good values of $\theta$.** The natural first idea is the one from Lecture 4 — maximize the likelihood.

### 6.2 Why not maximize the likelihood over everything?

Try maximizing the likelihood jointly over the function and the noise:

$$p(y\mid f,\sigma^2) = \prod_{n=1}^N\mathcal{N}\big(y_n\,\big|\,f(x_n),\,\sigma^2\big).$$

The best function is obvious: set $f(x_n) = y_n$ at every training input, so every residual is zero. The likelihood is then

$$p(y\mid f,\sigma^2) = (2\pi\sigma^2)^{-N/2},$$

which grows **without bound** as $\sigma^2\to0$. The maximum does not exist.

This is the collapsing-variance failure from Lecture 5: the model "explains" the data by memorizing it and declaring there is no noise at all. The function is simply too flexible to be chosen by maximization. Any hyperparameters fitted this way are meaningless.

### 6.3 The fix: average over the function instead of maximizing over it

Instead of asking

> *How well does the **best** function explain the data?*

ask

> *How well does a **typical** function from my prior explain the data, on average?*

Mathematically, average the likelihood over the prior on $f$:

$$\boxed{\ p(y\mid\theta) = \int p(y\mid f,\sigma^2)\;p(f\mid\theta)\;df .\ }$$

This is the **marginal likelihood**. "Marginal" because $f$ has been integrated out — marginalized — rather than optimized.

Two things make it the right objective:

- **It cannot memorize.** A single perfect-fitting function no longer wins, because it is just one function among all those the prior allows, and it is weighted by how plausible the prior finds it.
- **It is an ordinary likelihood.** It is the probability of the observed data as a function of the remaining unknowns, $\theta$ — exactly the kind of object Lecture 4 maximized. So maximizing it is simply **maximum likelihood for the hyperparameters**. This is called **type-II maximum likelihood** or **empirical Bayes**.

In short: **Bayes for the function, maximum likelihood for the hyperparameters.**

### 6.4 Computing it

The integral looks daunting, but Step 1 makes it easy. At the training inputs,

$$y = f+\epsilon, \qquad f\sim\mathcal{N}(0,K_\theta), \qquad \epsilon\sim\mathcal{N}(0,\sigma^2I), \qquad f\text{ and }\epsilon\text{ independent}.$$

A sum of independent Gaussians is Gaussian, with the covariances added. So without evaluating any integral,

$$\boxed{\ p(y\mid\theta) = \mathcal{N}\big(y\,\big|\,0,\ C_\theta\big), \qquad C_\theta = K_\theta+\sigma^2I .\ }$$

Taking logs,

$$\log p(y\mid\theta) = \underbrace{-\tfrac12\,y^\top C_\theta^{-1}y}_{\text{data fit}}\ \underbrace{-\ \tfrac12\log\big|C_\theta\big|}_{\text{complexity penalty}}\ -\ \tfrac N2\log2\pi .$$

### 6.5 Why this picks sensible hyperparameters

Return to the question in §6.3 — how well does a *typical* prior function explain the data? — and try it for three length scales.

- **Length scale too short.** Prior functions wiggle wildly. Almost any particular draw wiggles the wrong way at most data points. On average, poor fit.
- **Length scale too long.** Prior functions are nearly flat. If the data vary, almost every draw misses them. On average, poor fit.
- **Length scale about right.** Many prior draws follow the general shape of the data. On average, good fit.

The marginal likelihood is highest in the third case. It prefers **the simplest model that still explains the data** — Occam's razor — without anyone adding a penalty for complexity. The penalty comes from the averaging.

The same idea explains the two terms. $p(y\mid\theta)$ is a probability distribution over *all possible datasets*, so it must add up to one. A very flexible model can produce many different datasets, so it spreads its probability thinly and gives only a little to the one we observed. That is the $-\frac12\log|C_\theta|$ term, which is large and negative when the model is flexible. A rigid model concentrates its probability on a few smooth datasets; if ours is not among them, the data-fit term $-\frac12y^\top C_\theta^{-1}y$ is large and negative. The maximum balances the two.

### 6.6 The same recipe answers loose end (a)

Lecture 7 left $\sigma^2$ and $\tau^2$ unexplained. Bayesian linear regression is a GP with kernel $k = \tau^2\phi^\top\phi'$, whose Gram matrix is $K = \tau^2\Phi\Phi^\top$. So its marginal likelihood is

$$p(y\mid\sigma^2,\tau^2) = \mathcal{N}\big(y\,\big|\,0,\ \tau^2\Phi\Phi^\top+\sigma^2I\big),$$

and maximizing it over $(\sigma^2,\tau^2)$ chooses them. Since the ridge penalty is $\lambda = \sigma^2/\tau^2$, this chooses the ridge penalty too — with no cross-validation and no held-out data.

### 6.7 Doing the optimization

Training is now an optimization problem, and Lecture 6 supplies the tools.

**Optimize the logarithms.** All hyperparameters must be positive. Optimize $(\log\sigma_f^2,\log\ell,\log\sigma^2)$ instead: the problem becomes unconstrained and better conditioned.

**Use L-BFGS with the exact gradient.** With $\alpha = C_\theta^{-1}y$,

$$\frac{\partial\log p(y\mid\theta)}{\partial\theta_j} = \frac12\operatorname{tr}\!\Big(\big(\alpha\alpha^\top-C_\theta^{-1}\big)\,\frac{\partial C_\theta}{\partial\theta_j}\Big).$$

Only the derivative of the kernel is needed. Check it against finite differences before trusting it.

**Restart from several initializations.** The marginal likelihood can have more than one local maximum. A common pair: a *short length scale with small noise*, where the function wiggles through every point; and a *long length scale with large noise*, where a smooth trend is seen through heavy scatter. Compare the marginal likelihood at each, and look at both fits.

**Two cautions.** This is still maximum likelihood, so with many hyperparameters it can overfit. And it only compares the kernels you offered it; it cannot tell you they are all wrong. It does not replace a held-out test set.

### 6.8 What training leaves behind

Never compute $C_\theta^{-1}$ explicitly. Factor it once, and keep the pieces for Step 3.

> ### Algorithm: GP training
>
> **Input:** inputs $x_{1:N}$, targets $y$, kernel family, initial $\theta$.
>
> 1. Maximize $\log p(y\mid\theta)$ over $\log\theta$ with L-BFGS, evaluating it as below.
> 2. At the final $\hat\theta$: form $C = K_{\hat\theta}+\hat\sigma^2I$ and its **Cholesky factor** $L$ (lower triangular, $C = LL^\top$).
> 3. Solve $\alpha = L^{-\top}(L^{-1}y)$ by two triangular solves.
> 4. **Store $\hat\theta$, $L$, and $\alpha$.**
>
> The log marginal likelihood at any $\theta$ is $-\tfrac12y^\top\alpha-\sum_n\log L_{nn}-\tfrac N2\log2\pi$, using $\log|C| = 2\sum_n\log L_{nn}$.

**Cost.** The Cholesky factorization costs $O(N^3)$ and dominates. This limits exact GPs to roughly tens of thousands of data points.

**Jitter.** If $\sigma^2$ is tiny, $K$ is often numerically singular and the Cholesky factorization fails. Add a small constant, around $10^{-6}\sigma_f^2$, to the diagonal. This is a numerical fix, not a modeling choice, and should be reported.

---

## 7. Step 3 — Inference: predictions at test points

### 7.1 The predictive distribution

We want the distribution of $f^\star = f(x^\star)$ given the data. Under the model, the training targets and the test value are jointly Gaussian. Their covariances follow from the kernel, plus noise on the training targets only:

$$\begin{pmatrix}y\\ f^\star\end{pmatrix}\sim\mathcal{N}\left(0,\ \begin{pmatrix}K+\sigma^2I & k^\star\\ k^{\star\top} & k^{\star\star}\end{pmatrix}\right).$$

Apply the Gaussian conditioning formula from Lecture 2 — for zero-mean $(z_1,z_2)$, $\ z_1\mid z_2\sim\mathcal{N}\big(\Sigma_{12}\Sigma_{22}^{-1}z_2,\ \Sigma_{11}-\Sigma_{12}\Sigma_{22}^{-1}\Sigma_{21}\big)$ — with $z_1 = f^\star$ and $z_2 = y$:

$$\boxed{\ f^\star\mid\mathcal{D}\sim\mathcal{N}(\mu^\star,v^\star), \qquad \mu^\star = k^{\star\top}\big(K+\sigma^2I\big)^{-1}y, \qquad v^\star = k^{\star\star}-k^{\star\top}\big(K+\sigma^2I\big)^{-1}k^\star .\ }$$

For a new **noisy observation**, add the noise back: $\ y^\star\mid\mathcal{D}\sim\mathcal{N}(\mu^\star,\ v^\star+\sigma^2)$.

That is all of GP prediction. No weights, no optimization, no step size — just conditioning a Gaussian.

### 7.2 The predictive mean

Using $\alpha = (K+\sigma^2I)^{-1}y$ from training,

$$\mu^\star = \sum_{n=1}^N\alpha_n\,k(x_n,x^\star).$$

**The prediction is a weighted sum of kernel bumps, one centered at each training input**, with weights $\alpha$ fixed during training. The model started with infinitely many features (§4.3), but its predictions live in an $N$-dimensional space set by the data. Models like this, whose effective size grows with the amount of data, are called **nonparametric**.

With the linear kernel, $\mu^\star$ equals Lecture 7's posterior mean $m_N^\top\phi^\star$, which equals the ridge estimate of Lecture 5. The same prediction has now appeared three times.

### 7.3 The predictive variance

$$v^\star = \underbrace{k^{\star\star}}_{\text{prior variance}}-\underbrace{k^{\star\top}\big(K+\sigma^2I\big)^{-1}k^\star}_{\text{uncertainty removed by the data}} .$$

The variance is the prior variance, minus what the data has explained. Three consequences.

**It returns to the prior far from the data.** For a stationary kernel, $k(x_n,x^\star)\to0$ as $x^\star$ moves away from every training input. So $k^\star\to0$, and

$$\mu^\star\to0, \qquad v^\star\to k^{\star\star} = \sigma_f^2 .$$

Compare §2 carefully. The mean still returns to zero, the prior mean — that is fine. What changed is the variance: the model now reports the full prior uncertainty $\sigma_f^2$ instead of claiming certainty. It says *I don't know*. The failure is repaired, and it was repaired by the prior.

**It does not depend on $y$.** Only the input locations appear in $v^\star$. You can compute where the model will be uncertain *before taking any measurements*. The next topic, Bayesian optimization, is built on exactly this.

**The two kinds of uncertainty.** For a new observation,

$$\operatorname{Var}[y^\star\mid\mathcal{D}] = \underbrace{\sigma^2}_{\text{aleatoric}}+\underbrace{v^\star}_{\text{epistemic}} .$$

The aleatoric part is constant and irreducible. The epistemic part is small near data, larger between and beyond it, and approaches $\sigma_f^2$ far away. With plentiful data, $v^\star\to0$ and only $\sigma^2$ remains: you can learn the function exactly and still not predict the next measurement exactly.

### 7.4 Computing it

> ### Algorithm: GP inference
>
> **Input:** stored $\hat\theta$, $L$, $\alpha$ from training; a test input $x^\star$.
>
> 1. Compute $k^\star$ (entries $k(x_n,x^\star)$) and $k^{\star\star}$.
> 2. **Mean:** $\mu^\star = k^{\star\top}\alpha$.
> 3. Solve $v = L^{-1}k^\star$. **Variance:** $v^\star = k^{\star\star}-v^\top v$.
> 4. For a noisy observation, add $\hat\sigma^2$.

After training, each mean costs $O(N)$ and each variance $O(N^2)$. The expensive $O(N^3)$ step was done once.

### 7.5 Show functions, not just bands

A plot of $\mu^\star\pm2\sqrt{v^\star}$ shows the uncertainty at each input separately and hides how values at different inputs move together. Two models with similar bands — say Matérn-1/2 and squared-exponential fitted to the same data — can believe in very different functions, one jagged and one smooth. Always overlay a few **posterior samples**. To draw them at test inputs $x^\star_{1:M}$, use the same conditioning with vectors in place of scalars: mean $K_\star^\top\alpha$ and covariance $K_{\star\star}-K_\star^\top(K+\sigma^2I)^{-1}K_\star$, then sample as in §5.3.

---

## 8. The GP in three steps, and where this goes

| | Gaussian process regression |
|---|---|
| **Step 1 — Model** | $f\sim\mathcal{GP}(0,k_\theta)$, $\ y_n = f(x_n)+\epsilon_n$, $\ \epsilon_n\sim\mathcal{N}(0,\sigma^2)$ |
| **Step 2 — Training** | $\hat\theta = \arg\max_\theta\log\mathcal{N}\big(y\mid0,\,K_\theta+\sigma^2I\big)$ by L-BFGS; store $L$ and $\alpha$ |
| **Step 3 — Inference** | $\mu^\star = k^{\star\top}\alpha$, $\ v^\star = k^{\star\star}-\|L^{-1}k^\star\|^2$; add $\sigma^2$ for a new observation |

Set this beside Lecture 7. The model changed from a prior on weights to a prior on functions. Training changed from "compute the posterior, assuming $\sigma^2$ and $\tau^2$" to "fit the hyperparameters by maximum marginal likelihood." And inference now gives uncertainty that is honest everywhere, because the prior never claimed certainty anywhere.

**Where this goes.** Bayesian optimization chooses where to evaluate an expensive function by balancing a high predicted value $\mu^\star$ against high uncertainty $v^\star$. Experimental design and active learning choose measurements to reduce $v^\star$ where it matters. All of them are built on the two formulas of §7.

---

## Checkpoint E (take-home)

`numpy` only, plus `scipy.optimize` for the optimizer in item 3. Report fitted numbers, not pictures that look about right.

1. **Warm-up: the failure.** Fit Bayesian linear regression with $D = 15$ RBF features of width $\lambda = 0.2$ centered on a grid over $[-1,1]$, to 20 noisy observations of a smooth function on that interval. At $x^\star\in\{0,\,2,\,5\}$, report the predictive mean, the epistemic variance $\phi^{\star\top}S_N\phi^\star$, and the prior variance $\tau^2\|\phi^\star\|^2$. Confirm the epistemic variance never exceeds the prior variance, and that both vanish at $x^\star = 5$.
2. **Step 1 — the model.** Draw five prior samples on a 400-point grid for the squared-exponential, Matérn-1/2, and Matérn-5/2 kernels, each at $\ell\in\{0.1,\,0.5,\,2\}$. State what jitter each needed, and why the squared-exponential needed the most.
3. **Step 2 — training.** For the data of item 1 with a Matérn-5/2 kernel: (a) fix $\sigma_f^2$, evaluate the log marginal likelihood on a grid over $(\log\ell,\log\sigma^2)$, plot its contours, and report the maximizer; (b) implement the gradient of §6.7 and check it against finite differences; (c) optimize all three log-hyperparameters with L-BFGS from ten random starts, and report each distinct local maximum, its value, and in one sentence what explanation of the data it represents.
4. **Step 3 — inference.** Implement §7.4 using the trained model. At $x^\star\in\{0,\,2,\,5\}$ report $\mu^\star$, $v^\star$, and $v^\star+\hat\sigma^2$. Confirm $v^\star\to\hat\sigma_f^2$ at $x^\star = 5$. Plot the mean, both bands, and five posterior samples.
5. **GPs contain Lecture 7.** Using the linear kernel $k = \tau^2\phi^\top\phi'$ with item 1's RBF features and item 1's $\sigma^2,\tau^2$, confirm that your GP predictions match item 1's Bayesian linear regression to machine precision.

---

## Quiz-eligible facts

1. Every method in the course has three steps: model, training, inference. MLE/MAP train by optimizing a single $\hat w$ and infer by plugging it in; Bayesian methods train by computing a posterior and infer by averaging over it.
2. With a localized basis, Bayesian linear regression's predictive mean *and* epistemic variance go to zero far from the basis centers: confidently wrong.
3. A prior $w\sim\mathcal{N}(0,\tau^2I)$ induces a Gaussian prior on function values with $\operatorname{Cov}[f(x),f(x')] = \tau^2\phi(x)^\top\phi(x')$; the prior variance at $x$ is $\tau^2\|\phi(x)\|^2$.
4. Conditioning never increases variance. The posterior cannot be uncertain where the prior was certain, so the failure above is a failure of the prior.
5. A kernel gives the prior covariance between function values. It is valid when every Gram matrix it produces is positive semi-definite; sums and products of valid kernels are valid.
6. A stationary kernel depends only on $x-x'$, so its prior variance is the same everywhere and never collapses.
7. The squared-exponential kernel is the limit of Bayesian RBF regression with a bump at every point of the line.
7a. Mercer's theorem: every valid kernel can be written $k(x,x') = \sum_j\lambda_j\psi_j(x)\psi_j(x')$ with $\lambda_j\ge0$, so it is the inner product of features $\sqrt{\lambda_j}\psi_j$. Hence every GP is Bayesian linear regression with (possibly infinitely many) features. Fast eigenvalue decay means smooth samples; truncating to $M$ features gives a cheaper approximation. The feature map is not unique; the kernel is.
8. A GP is a random function whose values at any finite set of inputs are jointly Gaussian, specified by a mean function and a kernel. The linear kernel recovers Bayesian linear regression.
9. Matérn-5/2 is a better default than squared-exponential for physical data, because the squared-exponential assumes infinite smoothness.
10. Maximizing the likelihood jointly over the function and the noise fails: set $f(x_n) = y_n$ and let $\sigma^2\to0$, and the likelihood diverges — the collapsing-variance failure of Lecture 5.
11. The marginal likelihood $p(y\mid\theta) = \int p(y\mid f)\,p(f\mid\theta)\,df$ averages the fit over prior functions instead of maximizing it. It is an ordinary likelihood for $\theta$, so maximizing it is maximum likelihood for the hyperparameters: Bayes for the function, MLE for the hyperparameters.
12. For a GP, $p(y\mid\theta) = \mathcal{N}(0,K_\theta+\sigma^2I)$, because $y = f+\epsilon$ is a sum of independent Gaussians. Its log is a data-fit term $-\frac12y^\top C^{-1}y$ plus a complexity term $-\frac12\log|C|$.
13. The marginal likelihood prefers the simplest model that explains the data, because a flexible model spreads its probability over many possible datasets and gives little to the one observed.
14. Fit hyperparameters by optimizing their logarithms with L-BFGS and the exact gradient, from several starting points: local maxima can correspond to competing explanations of the data.
15. GP predictive: $\mu^\star = k^{\star\top}(K+\sigma^2I)^{-1}y$ and $v^\star = k^{\star\star}-k^{\star\top}(K+\sigma^2I)^{-1}k^\star$, by conditioning the joint Gaussian of $(y,f^\star)$. Add $\sigma^2$ for a noisy observation.
16. The predictive mean is a weighted sum of kernel bumps centered at the training inputs. For a stationary kernel, far from the data $\mu^\star\to0$ and $v^\star\to\sigma_f^2$: the model reports full prior uncertainty.
17. The predictive variance does not depend on $y$, so uncertainty can be computed before measuring.
18. Compute GPs with a Cholesky factorization of $K+\sigma^2I$, never an explicit inverse. Training costs $O(N^3)$ once; afterwards each mean costs $O(N)$ and each variance $O(N^2)$. Add jitter when $K$ is numerically singular.

---

## Practice problems

*Ungraded; these feed the quizzes.*

**P1.** Write ridge regression (Lecture 5) and Bayesian linear regression (Lecture 7) each in the three-step form of §1. What exactly differs in Step 2, and what differs in Step 3?

**P2.** For a polynomial basis $\phi(x) = (1,x,\dots,x^p)^\top$ with $w\sim\mathcal{N}(0,\tau^2I)$, write the induced kernel and its prior variance $k(x,x)$. Is it stationary? Describe how prior samples behave as $|x|$ grows.

**P3.** Verify the identity $(x-c)^2+(x'-c)^2 = 2(c-\bar x)^2+\frac12(x-x')^2$ used in §4.3. Why must the per-weight variance $\tau^2$ shrink as the grid spacing $\Delta$ shrinks?

**P4.** Show the divergence of §6.2 concretely. For $N = 3$ data points, set $f(x_n) = y_n$ and plot $\log p(y\mid f,\sigma^2)$ against $\log\sigma^2$. Then compute the marginal likelihood for a squared-exponential GP on the same data and plot it against $\log\sigma^2$ with $\ell$ and $\sigma_f^2$ fixed. Which one has a finite maximum?

**P5.** For a single training point $x_1$ with target $y_1$ and a squared-exponential kernel, write $\mu^\star$ and $v^\star$ explicitly as functions of $x^\star$. Sketch both. How far from $x_1$ must you go for the variance to recover half its prior value?

**P6.** Explain in words, without equations, why a GP with a very short length scale has a *low* marginal likelihood on smooth data, even though it can fit that data perfectly.

**P7.** Derive the gradient of §6.7 using the identities $\partial\log|C|/\partial\theta = \operatorname{tr}(C^{-1}\partial C/\partial\theta)$ and $\partial C^{-1}/\partial\theta = -C^{-1}(\partial C/\partial\theta)C^{-1}$.

**P8.** Construct a dataset of about 30 points on which the marginal likelihood has two distinct local maxima in $\ell$. Plot the posterior mean and samples at each. Which would you report, and what additional data would settle the question?

**P9 (seeing Mercer's features).** On a grid of 200 points in $[-3,3]$, build the Gram matrices of a squared-exponential and a Matérn-1/2 kernel with the same $\ell$ and $\sigma_f^2$, and compute their eigendecompositions. Plot the first five eigenvectors of each, and plot both eigenvalue sequences on a log scale. Then reconstruct each Gram matrix from its top $M$ eigenpairs and report the relative error for $M\in\{5,10,20,50\}$. Which kernel needs more features to represent accurately, and how does that match what its prior samples look like?
