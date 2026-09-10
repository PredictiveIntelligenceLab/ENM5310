# Lecture 5 — MAP, Logistic Regression, and Optimization by Necessity

**ENM 5310 — Data-driven Modeling and Probabilistic Scientific Computing**

**Builds on:** Lecture 3 (the Bernoulli likelihood, the sigmoid link, the stable form $\mathrm{softplus}(a) - ya$), Lecture 4 (maximum likelihood from KL, sampling distributions, the normal equations and their geometry, Fisher information).

**Reading:** Murphy I §4.5, §11.3, §10.1–10.2. Optional for the optimization material: Murphy I §8.1–8.3.

**Note on structure.** Part A completes the material of Lecture 4 (§§7–10 of those notes). It is written here in compressed form, stating results and their reasoning, with the full derivations left in the Lecture 4 file where they are already worked out. Parts B and C are new and are developed in full.

**Learning objectives.** After this lecture you should be able to:

1. Name three distinct ways the MLE can fail to exist, identify their common cause, and explain how a prior repairs all three.
2. Derive ridge and lasso as MAP estimates and interpret $\lambda$ as a ratio of variances.
3. State two respects in which the MAP estimate is a defective summary of the posterior.
4. Write the logistic negative log-likelihood in a numerically stable form and explain why the naive form fails.
5. Derive the gradient and Hessian of the logistic NLL, recognize the common $\Phi^\top(\text{prediction} - \text{target})$ structure, and explain precisely why no closed-form solution exists.
6. Prove the logistic NLL is convex and give an explicit step size guaranteeing gradient descent converges.
7. Derive the convergence rate of gradient descent on a quadratic and express it in terms of the condition number of the Hessian.
8. Explain how feature scaling changes the condition number, why Newton's method is unaffected, and why Newton's method is nonetheless abandoned at scale.

---

# Part A — Completing Lecture 4

## 1. Three degeneracies of the MLE

Lectures 3 and 4 assumed throughout that the maximizer of the likelihood exists and is unique. Frequently it does not.

**(a) The collapsing variance.** For a Gaussian with $N=1$, the MLE gives $\hat\mu = x_1$ and $\hat\sigma^2 = 0$, and the likelihood is *unbounded*: as $\sigma\to0$ with $\mu = x_1$, the density at $x_1$ diverges. The same pathology arises in mixture models, where the likelihood is infinite whenever one component collapses onto a single data point.

**(b) The underdetermined regression.** If $D > N$, or if two feature columns are linearly dependent, then $\Phi^\top\Phi$ is singular. An entire affine subspace of $w$ achieves the same maximum likelihood and $\hat w$ is not well defined. Equivalently, by Lecture 4 §6.1, the sampling variance along those directions is infinite.

**(c) Separability.** For logistic regression on data that some hyperplane divides perfectly, the likelihood approaches a supremum that is never attained, and $\|\hat w\| = \infty$. Section 8 works this out in detail now that the model is in front of us.

**The common cause.** In every case the log-likelihood is *flat, or unbounded, along some direction of parameter space*, because the data carries no information about that direction. The likelihood is a function of the data alone: where the data is silent, the likelihood is silent, and the maximizer is undefined or meaningless.

Any repair must therefore supply information that is *not* a function of the data. That is exactly what a prior is.

---

## 2. MAP: the prior as curvature

Bayes' rule gives $p(\theta\mid\mathcal{D})\propto p(\mathcal{D}\mid\theta)\,p(\theta)$. The **maximum a posteriori** estimate is its mode:

$$\boxed{\ \hat\theta_{\mathrm{MAP}} \;\triangleq\; \arg\max_\theta\ \big[\log p(\mathcal{D}\mid\theta) + \log p(\theta)\big].\ }$$

The evidence $p(\mathcal{D})$ is constant in $\theta$ and drops out. This is the cheapest possible use of Bayes' rule — it needs only the *shape* of the posterior near its peak, never its normalizing constant — which is why MAP survives into large-scale practice while the full posterior does not, and why Lecture 7 must spend a lecture on the normalizer being dodged here.

The consequence is a dictionary between two vocabularies:

$$\underbrace{\arg\min_\theta\ \mathrm{NLL}(\theta) + \lambda R(\theta)}_{\text{``loss plus penalty''}} \quad\Longleftrightarrow\quad \underbrace{\arg\max_\theta\ \log p(\mathcal{D}\mid\theta) + \log p(\theta)}_{\text{MAP}}, \qquad \log p(\theta) = -\lambda R(\theta) + \text{const}.$$

**Every regularizer is a log-prior**, provided $\exp(-\lambda R(\theta))$ is normalizable.

### 2.1 Ridge

With $w\sim\mathcal{N}(0,\tau^2 I_D)$ we have $\log p(w) = -\frac{1}{2\tau^2}\|w\|_2^2 + \text{const}$, so the MAP objective is $\frac{1}{2\sigma^2}\|y-\Phi w\|_2^2 + \frac{1}{2\tau^2}\|w\|_2^2$. Multiplying by $2\sigma^2$, which changes nothing,

$$\|y-\Phi w\|_2^2 + \lambda\|w\|_2^2, \qquad \boxed{\ \lambda = \frac{\sigma^2}{\tau^2}\ } \qquad\Longrightarrow\qquad \hat w_{\mathrm{ridge}} = (\Phi^\top\Phi + \lambda I)^{-1}\Phi^\top y.$$

**The matrix is invertible for every $\lambda>0$**, whatever $\Phi$ does, because adding $\lambda I$ adds $\lambda$ to each eigenvalue of the positive semi-definite $\Phi^\top\Phi$. The prior supplied curvature precisely where the likelihood had none. This repairs degeneracy (b) directly; it repairs (a) via a prior on $\sigma^2$ vanishing as $\sigma\to0$; and §8 shows it repairs (c). One repair, three diseases.

**$\lambda$ is a ratio of variances, not a tuning knob.** The identity $\lambda = \sigma^2/\tau^2$ says: regularize in proportion to how noisy the data is believed to be, and in inverse proportion to how large the coefficients are believed to be. Both are modeling claims that can be right or wrong. In Lecture 7 we will *estimate* $\lambda$ by maximizing the marginal likelihood rather than searching over it by cross-validation.

### 2.2 Lasso, and one warning

An independent Laplace prior $p(w_d) = \frac{1}{2b}e^{-|w_d|/b}$ gives $\log p(w) = -\frac1b\|w\|_1 + \text{const}$, hence the objective $\|y-\Phi w\|_2^2 + \lambda\|w\|_1$. There is no closed form: the objective is convex but not differentiable at $w_d = 0$, requiring proximal or coordinate methods. Sparsity arises geometrically, because the $\ell_1$ ball has corners on the coordinate axes and the elliptical level sets of the likelihood generically first touch it at a corner.

The warning: **the sparsity is a property of the mode, not of the posterior.** Under a Laplace prior the posterior assigns zero probability to the event $w_d = 0$ exactly, and the posterior *mean* is sparse in no coordinate at all. A lasso fit reporting that seven features were "selected" is making a statement about one summary statistic, not about the posterior. Section 4 generalizes the complaint.

---

## 3. ★ Shrinkage in the SVD basis

Full derivation in Lecture 4 §9; the results are what matter here. Write the thin SVD $\Phi = UDV^\top$ with singular values $d_1\ge\cdots\ge d_D$, so that $\Phi^\top\Phi = VD^2V^\top$ and the eigenvalues of the Gram matrix are the $d_j^2$.

Least squares has sampling covariance $\sigma^2VD^{-2}V^\top$, so along the direction $v_j$ its variance is $\sigma^2/d_j^2$: a direction along which the data barely varies gives a wildly uncertain coefficient. Ridge, meanwhile, produces fitted values

$$\hat y_{\mathrm{ridge}} = \sum_{j=1}^{D}u_j\,\frac{d_j^2}{d_j^2+\lambda}\,(u_j^\top y),$$

retaining direction $j$ with a **shrinkage factor** $d_j^2/(d_j^2+\lambda)\in(0,1)$ that is near $1$ when $d_j^2\gg\lambda$ and near $0$ when $d_j^2\ll\lambda$. Comparing the two statements:

$$\boxed{\ \text{Ridge shrinks hardest along exactly those directions in which least squares is least determined.}\ }$$

Ridge is not shrinking indiscriminately toward zero; it is suppressing the directions in which $\hat w_{\mathrm{LS}}$ was mostly noise, with a smooth threshold rather than a hard one. The **effective degrees of freedom** $\mathrm{df}(\lambda) = \sum_j d_j^2/(d_j^2+\lambda)$ falls from $D$ at $\lambda = 0$ toward $0$ as $\lambda$ grows, indexing the model's flexibility continuously.

The cost is bias: $\mathbb{E}[\hat w_{\mathrm{ridge}}]\ne w^\star$, with the bias toward zero largest in the noisiest directions. In the language of Lecture 4 §3.1, ridge deliberately accepts bias to purchase a larger reduction in variance.

Keep the $d_j^2$ in view — they will return in §11 as the source of the condition number that governs how fast gradient descent runs.

---

## 4. Two things the MAP estimate is not

These are the two facts that make Lecture 6 necessary.

### 4.1 MAP is not invariant to reparameterization

The MLE is. For invertible differentiable $g$ and $\eta = g(\theta)$, we have $\hat\eta_{\mathrm{MLE}} = g(\hat\theta_{\mathrm{MLE}})$, because the likelihood is a *function of* $\theta$, not a *density in* $\theta$: relabeling the points does not change the value attached to each, so the location of the maximum simply transports.

A prior, however, *is* a density in $\theta$. By the change-of-variables formula from Lecture 3,

$$p_\eta(\eta) = p_\theta\big(g^{-1}(\eta)\big)\left|\det\frac{\partial g^{-1}}{\partial\eta}\right|,$$

and the Jacobian factor varies with $\eta$, so it moves the maximum. Fit a scale parameter by MAP in the coordinate $\sigma$, then in the coordinate $\log\sigma$ under the correspondingly transformed prior, map one answer back through $g$, and the two disagree. **The MAP estimate depends on the coordinate system the model happened to be written in.** Every other summary of the posterior — mean, quantiles, predictive distribution, any posterior expectation — transforms correctly, because integration against a density is invariant while maximization of a density is not.

### 4.2 The mode is not where the mass is

For $\theta\sim\mathcal{N}(0,I_d)$ the density is maximized at the origin. But examining $\|\theta\|^2 = \sum_{j=1}^d\theta_j^2$,

$$\mathbb{E}\big[\|\theta\|^2\big] = d, \qquad \operatorname{Var}\big[\|\theta\|^2\big] = \sum_{j=1}^d\big(\mathbb{E}[\theta_j^4]-1\big) = 2d,$$

using $\mathbb{E}[\theta_j^4] = 3$. So $\|\theta\| \approx \sqrt d$ with $O(1)$ fluctuations: samples concentrate on a thin spherical shell of radius $\sqrt d$. The density on that shell is smaller than the density at the mode by a factor $e^{-d/2}$, and yet the shell is where every sample lands, because the available volume grows faster than the density decays. In $d = 1000$, essentially no sample from the distribution falls anywhere near its own mode.

> **Typical set (informal).** The region containing essentially all of a distribution's probability mass. For a high-dimensional Gaussian it is a thin shell, not a ball around the mode.

A MAP estimate can therefore be an atypical, unrepresentative point of the very distribution it summarizes, and substituting it into a prediction discards all uncertainty. This is not a correction to be patched later; it is why the remaining Bayesian material in this course exists.

---

# Part B — Logistic Regression

## 5. The likelihood and its stable negative log

Lecture 3 §2 constructed this model from the recipe $p(y\mid x,\theta) = \mathcal{D}(y\mid f(x;\theta))$, taking $\mathcal{D}$ Bernoulli and the link $\sigma$:

$$p(y\mid x,w) = \mathrm{Bern}\big(y \,\big|\, \mu\big), \qquad \mu = \sigma(a), \qquad a = w^\top\phi(x), \qquad \sigma(a) = \frac{1}{1+e^{-a}}.$$

The logit $a$ is the log-odds, $a = \log\frac{\mu}{1-\mu}$, which is the sense in which the model is "linear": it is linear in the log-odds, not in the probability.

Two derivative facts, both used repeatedly:

$$\sigma(-a) = 1-\sigma(a), \qquad \sigma'(a) = \sigma(a)\big(1-\sigma(a)\big). \tag{5.1}$$

The second follows by differentiating $(1+e^{-a})^{-1}$ and recognizing the pieces; verify it once by hand, because it is the reason every expression below is as clean as it is.

**The negative log-likelihood.** Since $p(y\mid\mu) = \mu^y(1-\mu)^{1-y}$,

$$\mathrm{NLL}(w) = -\sum_{n=1}^{N}\Big[y_n\log\mu_n + (1-y_n)\log(1-\mu_n)\Big]. \tag{5.2}$$

This is the cross-entropy $\mathbb{H}(q, p_w)$ of Lecture 4 §1, evaluated for a Bernoulli model. The name is literal.

**Never implement (5.2) as written.** When $a_n$ is large and positive, $\mu_n$ rounds to $1.0$ in floating point and $\log(1-\mu_n)$ returns $-\infty$; symmetrically for large negative $a_n$. The likelihood is perfectly finite at these points — the code is what fails. Substituting $\mu = \sigma(a)$ into (5.2) and simplifying term by term:

$$-\big[y\log\sigma(a) + (1-y)\log\sigma(-a)\big] = \log\big(1+e^{a}\big) - y\,a \;=\; \mathrm{softplus}(a) - y\,a, \tag{5.3}$$

which is the form Lecture 3 gave. It never forms $\mu$ at all. Evaluate $\mathrm{softplus}$ itself in the stable form

$$\mathrm{softplus}(a) = \log(1+e^{a}) = \max(a,0) + \log\big(1+e^{-|a|}\big), \tag{5.4}$$

in which the exponential argument is never positive and therefore never overflows. Expression (5.4) is exact, not an approximation — verify it by cases on the sign of $a$.

This is a recurring pattern worth naming: **the mathematically natural form of an objective and the numerically usable form of it are usually different expressions.** You will meet this again with log-sum-exp in softmax regression, with the log-determinant in Gaussian likelihoods, and with the reparameterized ELBO.

---

## 6. The gradient, and why there is no closed form

Differentiate the stable form (5.3), which is the easier route. For a single datum, using $\frac{d}{da}\mathrm{softplus}(a) = \sigma(a)$:

$$\frac{\partial}{\partial a_n}\Big[\mathrm{softplus}(a_n) - y_n a_n\Big] = \sigma(a_n) - y_n = \mu_n - y_n.$$

Then by the chain rule with $a_n = w^\top\phi(x_n)$, so $\partial a_n/\partial w = \phi(x_n)$:

$$\boxed{\ \nabla_w\,\mathrm{NLL}(w) \;=\; \sum_{n=1}^{N}\big(\mu_n - y_n\big)\,\phi(x_n) \;=\; \Phi^\top\big(\mu - y\big), \qquad \mu = \sigma(\Phi w).\ }$$

Stare at this next to the linear-regression gradient from Lecture 4 §6, $\nabla_w = \Phi^\top(\Phi w - y)$. Both have the form

$$\nabla_w = \Phi^\top\big(\text{prediction} - \text{target}\big).$$

This is not a coincidence of two examples. It holds for the Gaussian, the Bernoulli, the categorical, the Poisson — for every model built by the Lecture 3 recipe with the canonical link, where "prediction" means the model's mean $\mathbb{E}[y\mid x]$. The choice of output distribution changes what "prediction" denotes and changes nothing else about the shape of the gradient. When you write a backward pass for a network with a softmax head and find the output-layer gradient is $(\hat{y} - y)$, this is the reason.

**Interpretation.** Each datum contributes its feature vector, weighted by how wrong the prediction was and in which direction. A confidently correct point ($\mu_n\approx y_n$) contributes almost nothing; a confidently *incorrect* point contributes nearly the full $\pm\phi(x_n)$. The gradient is dominated by the points the model currently gets wrong — which is desirable, and is also why a single mislabeled outlier at large $\|\phi(x)\|$ can drag the fit.

**Why there is no closed form.** Setting the gradient to zero gives

$$\Phi^\top\big(\sigma(\Phi w) - y\big) = 0,$$

a system of $D$ equations in $D$ unknowns in which $w$ appears inside a transcendental function. Contrast the linear case: there, the prediction $\Phi w$ was *linear* in $w$, so the stationarity condition was a linear system and Gaussian elimination finished the job. Here $\sigma$ is nonlinear, no algebraic rearrangement isolates $w$, and there is no closed-form solution — not because we have not found one, but because none exists in elementary terms.

This is the moment the course's fourth question — *computation* — stops being a formality. Everything from here on is iterative.

---

## 7. The Hessian, convexity, and a step-size bound

Differentiate the gradient once more. Using (5.1), $\partial\mu_n/\partial a_n = \mu_n(1-\mu_n)$, so

$$\boxed{\ \mathbf{H}(w) \;=\; \nabla^2_w\,\mathrm{NLL}(w) \;=\; \sum_{n=1}^{N}\mu_n(1-\mu_n)\,\phi(x_n)\phi(x_n)^\top \;=\; \Phi^\top S\,\Phi, \qquad S = \operatorname{diag}\big(\mu_n(1-\mu_n)\big).\ }$$

**Convexity.** Each $\mu_n(1-\mu_n) > 0$, since $\mu_n\in(0,1)$. So for any $v\in\mathbb{R}^D$,

$$v^\top\mathbf{H}v = \sum_n \mu_n(1-\mu_n)\big(\phi(x_n)^\top v\big)^2 \;\ge\; 0,$$

with equality only if $\Phi v = 0$. Hence $\mathbf{H}\succeq0$ always, and $\mathbf{H}\succ0$ whenever $\Phi$ has full column rank. **The logistic NLL is convex**, so every stationary point is a global minimum and there are no local minima to be trapped in.

Two remarks, in this order.

*First:* note that $\mathbf{H}$ is the empirical analogue of the Fisher information from Lecture 4 §3.3. For this model the observed Hessian happens not to depend on $y$ at all — only on the predictions — which is a special feature of canonical-link models and the reason Newton's method and Fisher scoring coincide here.

*Second, and more important:* **this is the last convex objective in this course.** Every neural network loss is nonconvex, and the deep learning module will be about behavior in a landscape with saddle points, plateaus, and no global guarantees. Convexity is being used here as a scaffold: it lets us isolate the effects of *step size* and *conditioning* from the effects of a bad landscape. If gradient descent struggles on a convex problem — and §10 shows it does — then the difficulty is not the landscape, and diagnosing it in a setting where the landscape is exonerated is the only way to learn to tell the two apart.

**A concrete step-size bound.** Since $\mu(1-\mu)\le\frac14$ for all $\mu\in(0,1)$, we have $S\preceq\frac14 I$ and therefore

$$\mathbf{H} = \Phi^\top S\Phi \;\preceq\; \tfrac14\Phi^\top\Phi \qquad\Longrightarrow\qquad \lambda_{\max}(\mathbf{H}) \;\le\; \tfrac14\lambda_{\max}(\Phi^\top\Phi) = \tfrac14 d_1^2,$$

with $d_1$ the largest singular value of $\Phi$ from §3. Section 10 shows gradient descent on a function with $\lambda_{\max}(\mathbf H)\le L$ converges for any fixed step size $\eta < 2/L$. So

$$\eta < \frac{8}{d_1^{2}} \qquad\text{guarantees convergence, computable before running anything.}$$

You are expected to compute this bound and check it against the divergence threshold you observe empirically. "I tried some step sizes and $0.01$ seemed to work" is not an acceptable account of a training run in this course.

---

## 8. Separability revisited

Return to degeneracy (c) of §1, now with the model in hand. Suppose the data is **linearly separable**: there is a $w_0$ with $\phi(x_n)^\top w_0 > 0$ for every $n$ with $y_n = 1$ and $< 0$ for every $n$ with $y_n = 0$.

Consider the ray $w = \alpha w_0$ for $\alpha > 0$. Every logit $a_n = \alpha\,\phi(x_n)^\top w_0$ has the correct sign and grows in magnitude linearly in $\alpha$, so $\mu_n\to y_n$ and, by (5.3), each term of the NLL tends to zero. Therefore

$$\mathrm{NLL}(\alpha w_0)\ \downarrow\ 0 \quad\text{as}\quad \alpha\to\infty,$$

and since $\mathrm{NLL} > 0$ always, the infimum is $0$ and is **never attained**. The MLE does not exist; $\|\hat w\| = \infty$.

Watch the Hessian while this happens. As $\mu_n\to0$ or $1$, the weights $\mu_n(1-\mu_n)\to0$, so $S\to0$ and $\mathbf{H}\to0$. The objective flattens as the optimizer travels along the ray: each step buys less reduction than the last while $\|w\|$ keeps growing. An optimizer run on separable data does not crash — it converges in objective value and diverges in parameter value, reporting probabilities of $1.000$ with unbounded confidence. Recognize this behavior when you see it in Checkpoint B.

**The repair.** Add a Gaussian prior, $w\sim\mathcal{N}(0,\tau^2I)$, giving the MAP objective

$$J(w) = \mathrm{NLL}(w) + \frac{1}{2\tau^2}\|w\|_2^2, \qquad \nabla^2 J = \Phi^\top S\Phi + \frac{1}{\tau^2}I \succ 0.$$

The Hessian is now strictly positive definite everywhere, so $J$ is *strictly* convex; and $J(w)\ge\frac{1}{2\tau^2}\|w\|^2\to\infty$ as $\|w\|\to\infty$, so the sublevel sets are bounded and a minimizer must exist. It is unique. The prior did not merely improve the estimate — it made the estimation problem well posed, exactly as it did for ridge in §2.1, and for the same reason: it supplied curvature where the likelihood had none.

*Deferred:* the derivation of the sigmoid from shared-covariance Gaussian class conditionals, which exhibits logistic regression as the discriminative counterpart of a generative model. It belongs with the generative-modeling material and is not needed for anything before then.

---

# Part C — Optimization by Necessity

We need this machinery because §6 says we do. The scope here is deliberately narrow: five questions, answered on a convex problem where the landscape cannot be blamed. Section 13 lists what is being left out and where it is picked up.

## 9. Gradient descent, and why the gradient

**The update.** $\ w_{k+1} = w_k - \eta\,\nabla f(w_k)$, with step size $\eta > 0$.

Why the gradient direction? Not because it is "downhill" — infinitely many directions are. Build a local model. Taylor expansion gives $f(w+\delta)\approx f(w) + \nabla f(w)^\top\delta$ for small $\|\delta\|$, and this linear model is unbounded below, so it must be minimized over a region small enough for the approximation to hold:

$$\min_{\|\delta\|_2 \le r}\ \nabla f(w)^\top\delta.$$

By Cauchy–Schwarz the minimum is attained at $\delta = -r\,\nabla f(w)/\|\nabla f(w)\|_2$. So the gradient direction is the **steepest descent direction with respect to the Euclidean norm**, and the step size $\eta$ is standing in for the trust radius $r$.

Note the qualifier. The steepest direction depends on which norm measures "small." Choose the norm $\|\delta\|_A^2 = \delta^\top A\delta$ instead and you get the direction $-A^{-1}\nabla f$. Gradient descent is not the canonical first-order method; it is the one obtained by declaring that all coordinates of $w$ are measured in comparable units. Section 11 shows what happens when that declaration is false, and §12 shows what happens when $A$ is chosen well.

## 10. Convergence on a quadratic: the condition number

Analyze the exact quadratic

$$f(w) = \tfrac12 (w-w^\star)^\top \mathbf{H}\,(w-w^\star), \qquad \mathbf{H}\succ0,$$

which is what any smooth objective looks like near a minimum — and near its minimum the logistic NLL is exactly this with the $\mathbf{H}$ of §7. So conclusions drawn here predict the behavior actually observed on the real problem, at least in the endgame.

Let $e_k = w_k - w^\star$. Since $\nabla f(w) = \mathbf{H}(w-w^\star)$,

$$e_{k+1} = e_k - \eta\mathbf{H}e_k = (I - \eta\mathbf{H})\,e_k.$$

Diagonalize: $\mathbf{H} = Q\Lambda Q^\top$ with eigenvalues $\lambda_1\ge\cdots\ge\lambda_D>0$ and orthonormal $Q$. Writing $\tilde e_k = Q^\top e_k$ for the error in the eigenbasis, the update **decouples completely**:

$$\tilde e_{k+1,j} = (1-\eta\lambda_j)\,\tilde e_{k,j} \qquad\Longrightarrow\qquad \tilde e_{k,j} = (1-\eta\lambda_j)^k\,\tilde e_{0,j}. \tag{10.1}$$

Each eigendirection contracts geometrically at its own rate $|1-\eta\lambda_j|$, independently of the others. This one equation contains everything.

**Stability.** Convergence requires $|1-\eta\lambda_j| < 1$ for every $j$, i.e.

$$0 < \eta < \frac{2}{\lambda_{\max}}.$$

Exceed this and the component along $v_1$ — the *stiffest* direction — grows in magnitude and alternates in sign each step. That is the signature of an oscillating, diverging loss curve, and it tells you the step size violated a bound set by the largest eigenvalue.

**Rate.** The slowest component governs the overall rate, $\rho(\eta) = \max_j|1-\eta\lambda_j|$, whose maximum is attained at $\lambda_{\max}$ or $\lambda_{\min}$. Minimizing over $\eta$: the two are balanced when $1-\eta\lambda_{\min} = -(1-\eta\lambda_{\max})$, giving

$$\eta^\star = \frac{2}{\lambda_{\min}+\lambda_{\max}}, \qquad \rho^\star = \frac{\lambda_{\max}-\lambda_{\min}}{\lambda_{\max}+\lambda_{\min}} = \boxed{\ \frac{\kappa-1}{\kappa+1}\ }, \qquad \kappa \triangleq \frac{\lambda_{\max}}{\lambda_{\min}}.$$

> **Condition number of the Hessian.** $\kappa = \lambda_{\max}(\mathbf{H})/\lambda_{\min}(\mathbf{H})$, the ratio of the steepest to the shallowest curvature. Geometrically, $\sqrt\kappa$ is the aspect ratio of the elliptical level sets.

**Read the result, because it is the whole lesson.** The number of iterations to reduce the error by a factor $\varepsilon$ is $k \approx \log(1/\varepsilon)/\log(1/\rho^\star)$, and for large $\kappa$, $\log(1/\rho^\star)\approx 2/\kappa$, so

$$k \;\approx\; \frac{\kappa}{2}\,\log\frac{1}{\varepsilon}: \qquad \textbf{iteration count is proportional to } \kappa.$$

Concretely, for a factor of $10^{-6}$:

| $\kappa$ | $\rho^\star$ | iterations |
|---|---|---|
| $1$ | $0$ | $1$ |
| $10$ | $0.818$ | $\approx 70$ |
| $10^2$ | $0.980$ | $\approx 690$ |
| $10^4$ | $0.9998$ | $\approx 69{,}000$ |

**The mechanism, in one sentence:** the largest eigenvalue sets a ceiling on the step size, the smallest eigenvalue determines how much progress that step makes, and $\kappa$ is the ratio — so a single stiff direction throttles every other direction in the problem. Nothing about this is a defect of the implementation. It is what (10.1) says.

## 11. Where $\kappa$ comes from: feature scaling

For logistic regression, $\mathbf{H} = \Phi^\top S\Phi$, so $\kappa$ is a property of the *design matrix*, which is a property of *how you chose to write down your features*.

Rescale the $j$-th feature column by a constant $c$ — measure a length in nanometers instead of metres, say. Then $\Phi\to\Phi C$ with $C = \operatorname{diag}(\dots,c,\dots)$, and

$$\mathbf{H} \to C\,\Phi^\top S\Phi\,C,$$

whose $j$-th row and column are scaled by $c$ and whose $(j,j)$ entry by $c^2$. The eigenvalues move, and $\kappa$ can be changed by many orders of magnitude by a change of units alone. Note also, from §3, that $\lambda_{\max}(\mathbf{H})\le\frac14 d_1^2$: the singular value spectrum that governed ridge shrinkage is the same spectrum that governs optimization speed. Conditioning is one phenomenon appearing in two places.

**This is the failure mode to internalize.** Take one dataset, one solver, one initialization, and two feature scalings. The optimization is the *same problem* in the sense that the two runs describe identical models and reach identical predictions — the minimizers are related by $w\to C^{-1}w$ — yet one converges in tens of iterations and the other in tens of thousands. A practitioner who does not know this diagnoses it as "the model won't train," changes the architecture, and never finds out.

Standardizing feature columns to zero mean and unit variance is therefore not cosmetic tidying. It is an intervention on $\kappa$, and it is the cheapest optimization improvement available in most problems. It also does not fix the general case: correlated features produce a poorly conditioned $\Phi^\top\Phi$ regardless of per-column scaling, since standardization only equalizes the diagonal.

## 12. Newton's method: minimizing a local quadratic model

### 12.1 The step, from a second-order Taylor expansion

Section 9 built a *linear* model of the objective and immediately hit a problem: a nonconstant linear function has no minimum, so the model could not be minimized. A trust radius had to be imposed from outside, and it survived into the algorithm as the step size $\eta$ — a parameter the model itself never supplied.

The natural repair is a model that *does* have a minimum. Expand $f$ to second order about the current iterate $w_k$:

$$f(w_k+\delta) \;=\; f(w_k) + \nabla f(w_k)^\top\delta + \tfrac12\,\delta^\top\mathbf{H}(w_k)\,\delta \;+\; O\big(\|\delta\|^3\big).$$

Discard the remainder and give the truncation a name. Writing $g_k \triangleq \nabla f(w_k)$ and $\mathbf{H}_k\triangleq\mathbf{H}(w_k)$, the **local quadratic model** is

$$m_k(\delta) \;\triangleq\; f(w_k) + g_k^\top\delta + \tfrac12\,\delta^\top\mathbf{H}_k\,\delta. \tag{12.1}$$

Now minimize the model *exactly*. It is a quadratic in $\delta$, so set its gradient to zero:

$$\nabla_\delta m_k(\delta) = g_k + \mathbf{H}_k\,\delta = 0.$$

If $\mathbf{H}_k \succ 0$ then $m_k$ is strictly convex and this stationary point is its unique minimizer, giving the **Newton step**

$$\boxed{\ \mathbf{H}_k\,\delta_k = -\,g_k, \qquad w_{k+1} = w_k + \delta_k = w_k - \mathbf{H}_k^{-1}\nabla f(w_k).\ }$$

Three observations, in order of importance.

**No step size was invented.** Gradient descent minimizes an *approximate* model over an *artificial* region and needs $\eta$ to say how big that region is. Newton minimizes an approximate model over the whole space, because the quadratic model is bounded below. The step length is an output of the curvature rather than an input to the algorithm. That is the structural difference between the two methods, and everything else in this section follows from it.

**The step is scale-aware, direction by direction.** Compare $-\eta g$ against $-\mathbf{H}^{-1}g$. The first applies a single scalar to every direction. The second applies $\mathbf{H}^{-1}$, which in the Hessian eigenbasis multiplies the $j$-th component by $1/\lambda_j$: long steps along flat directions, short steps along steeply curved ones. Recall from §10 that gradient descent's difficulty was precisely that one scalar $\eta$ had to serve both $\lambda_{\max}$ and $\lambda_{\min}$ at once. Newton's step is what you get by refusing to make that compromise.

**Solve the system; do not invert.** The boxed form is written with $\mathbf{H}_k^{-1}$ for readability, but implementations solve $\mathbf{H}_k\delta_k = -g_k$ by Cholesky factorization, which is both cheaper and better conditioned than forming an inverse. This is the same reflex as §6 of Lecture 4 regarding the normal equations.

### 12.2 What the model being exact buys you

**On a quadratic, one step suffices.** If $f$ is exactly quadratic, the $O(\|\delta\|^3)$ remainder vanishes, so $m_k = f$ identically and minimizing the model *is* minimizing the objective. From any starting point, $w_1 = w^\star$.

Set that against §10, where the same quadratic cost gradient descent $\Theta(\kappa)$ iterations. **The condition number does not appear in Newton's iteration count at all.** It has moved: $\kappa$ now governs the difficulty of *solving* the linear system $\mathbf{H}\delta = -g$, which is a numerical-linear-algebra cost paid once per step, rather than an iteration count paid over and over.

**Near a minimum, digits double.** For general smooth $f$ with $\mathbf{H}(w^\star)\succ0$ and a Lipschitz Hessian, there are constants such that, once $\|e_k\| = \|w_k-w^\star\|$ is small enough,

$$\|e_{k+1}\| \;\le\; C\,\|e_k\|^2 .$$

This is **quadratic convergence**, and it is qualitatively unlike the geometric $\rho^k$ decay of §10: the number of correct digits roughly doubles per iteration, so an error of $10^{-2}$ becomes $10^{-4}$, then $10^{-8}$, then $10^{-16}$, and the run terminates. The reason is exactly that the discarded remainder is $O(\|\delta\|^3)$, so the model's error is second order in the distance to the solution.

Note the qualifier "once $\|e_k\|$ is small enough." Far from the optimum the quadratic model may be a poor description of $f$, and the undamped step can overshoot or increase the objective. The standard fix is a **damped** Newton iteration, $w_{k+1} = w_k + \eta_k\delta_k$ with $\eta_k\in(0,1]$ chosen by a line search, which recovers global convergence while leaving $\eta_k = 1$ near the optimum so the quadratic rate survives.

### 12.3 Affine invariance

This is the property that connects the method back to §11.

Reparameterize the problem as $w = A\tilde w$ for invertible $A$, and let $\tilde f(\tilde w) = f(A\tilde w)$. By the chain rule,

$$\tilde g = A^\top g, \qquad \tilde{\mathbf{H}} = A^\top\mathbf{H}A .$$

The Newton step computed in the new coordinates is therefore

$$\tilde\delta = -\tilde{\mathbf{H}}^{-1}\tilde g = -\big(A^\top\mathbf{H}A\big)^{-1}A^\top g = -A^{-1}\mathbf{H}^{-1}A^{-\top}A^\top g = -A^{-1}\mathbf{H}^{-1}g = A^{-1}\delta,$$

which is *exactly* the original step expressed in the new coordinates. **Newton's method takes the same sequence of iterates no matter how the features are scaled or linearly recombined.** The pathology of §11 — same problem, same solver, two unit systems, iteration counts differing by orders of magnitude — vanishes identically.

Return to the framing of §9: the steepest descent direction depends on which norm measures "small," and $-A^{-1}\nabla f$ is steepest descent in the norm $\|\delta\|_A^2 = \delta^\top A\delta$. Newton is the choice $A = \mathbf{H}$. It measures distance using the curvature of the function itself, which is the only choice available that does not import an arbitrary convention about the units of $w$.

### 12.4 When the Hessian is not positive definite

Everything above assumed $\mathbf{H}_k\succ0$. If $\mathbf{H}_k$ is singular, the model (12.1) is flat along some direction and $\delta_k$ is undefined; if $\mathbf{H}_k$ is indefinite, the model is *unbounded below* and its stationary point is a saddle, so "minimizing the model" returns a direction that may increase $f$.

For the logistic NLL, §7 guarantees $\mathbf{H}\succeq0$, and $\succ0$ when $\Phi$ has full column rank — so the difficulty here is near-singularity rather than indefiniteness, and it arrives via separability, where §8 showed $S\to0$ and hence $\mathbf{H}\to0$.

Two standard repairs, both of which you have already seen:

- **Damping / Levenberg–Marquardt:** solve $(\mathbf{H}_k + \gamma I)\delta_k = -g_k$ for some $\gamma>0$. As $\gamma\to0$ this is Newton; as $\gamma\to\infty$ it is gradient descent with step $1/\gamma$. Note that this is precisely the modification a Gaussian prior makes on its own: from §8, the MAP Hessian is $\Phi^\top S\Phi + \tau^{-2}I$, so **the prior damps Newton for free**, with $\gamma = \tau^{-2}$.
- **Trust region:** minimize $m_k(\delta)$ subject to $\|\delta\|\le r_k$, which is well posed even when $\mathbf{H}_k$ is indefinite, and adjust $r_k$ according to how well the model predicted the observed decrease. This is the honest version of §9's trust-radius argument, now applied to a model that deserves it.

Nonconvex objectives — every neural network — make indefinite Hessians the normal case rather than the exception, which is one of several reasons the deep learning module does not use Newton's method.

### 12.5 Iteratively reweighted least squares

For logistic regression the Newton step has a closed-form interpretation worth deriving, because it means no new machinery is required: each step is a weighted linear regression of the kind already solved in Lecture 4.

Substitute $g = \Phi^\top(\mu-y)$ from §6 and $\mathbf{H} = \Phi^\top S\Phi$ from §7, and rearrange, factoring $\Phi^\top S$ out of the bracket:

$$
\begin{aligned}
w^{+} &= w - \big(\Phi^\top S\Phi\big)^{-1}\Phi^\top(\mu - y) \\[2pt]
&= \big(\Phi^\top S\Phi\big)^{-1}\Big[\big(\Phi^\top S\Phi\big)w - \Phi^\top(\mu-y)\Big] \\[2pt]
&= \big(\Phi^\top S\Phi\big)^{-1}\Phi^\top S\Big[\Phi w + S^{-1}(y - \mu)\Big] \\[2pt]
&= \big(\Phi^\top S\Phi\big)^{-1}\Phi^\top S\,z, \qquad\quad \boxed{\ z \;\triangleq\; \Phi w + S^{-1}(y-\mu).\ }
\end{aligned}
$$

Compare with the weighted least squares estimator (Lecture 4, practice problem P4): $\hat w = (\Phi^\top W\Phi)^{-1}\Phi^\top W y$ for weight matrix $W$. The two expressions are identical, with $W = S$ and the targets replaced by $z$. So:

> **Each Newton step for logistic regression is a weighted least squares fit** to a manufactured target $z$, with weights $S$, both recomputed at the current $w$. Iterating gives **iteratively reweighted least squares (IRLS)**.

**What $z$ and $S$ actually are.** Neither is arbitrary, and the interpretation is worth the two minutes.

The **working response** $z_n = a_n + (y_n-\mu_n)/s_n$, where $s_n = \mu_n(1-\mu_n) = \sigma'(a_n)$, is a linearization of the link. We observe $y_n$ but the model is linear in the logit, not in the probability; so we ask what logit *would* have produced $y_n$, to first order. Inverting the map $a\mapsto\mu$ about the current $a_n$ gives $a_n + (y_n-\mu_n)/\sigma'(a_n)$, which is $z_n$. The working response is the observation, transported to the scale on which the model is linear.

The **weights** are the precisions of that transported observation. Treating $a_n$, $\mu_n$, $s_n$ as fixed at the current iterate,

$$\operatorname{Var}[z_n] = \frac{\operatorname{Var}[y_n]}{s_n^2} = \frac{\mu_n(1-\mu_n)}{s_n^2} = \frac{s_n}{s_n^2} = \frac{1}{s_n},$$

so $s_n = 1/\operatorname{Var}[z_n]$. Points whose predicted probability is near $0$ or $1$ have small $s_n$: linearizing the link there amplifies noise enormously, and the algorithm correctly declines to trust them. Points with $\mu_n\approx\tfrac12$ get the maximum weight $\tfrac14$. **IRLS is weighted least squares with exactly the weights the noise model dictates** — which it must be, since it was derived from that noise model.

---

> ### Algorithm: IRLS for logistic regression (with optional Gaussian prior)
>
> **Input:** design matrix $\Phi\in\mathbb{R}^{N\times D}$, labels $y\in\{0,1\}^N$, tolerance $\texttt{tol}$, prior variance $\tau^2$ (set $\tau^2=\infty$ for no prior), weight floor $\varepsilon$ (e.g. $10^{-10}$), maximum iterations $K$.
>
> **Initialize:** $w_0 = 0$.
>
> For $k = 0,1,2,\dots,K-1$:
>
> 1. **Logits.** $a \leftarrow \Phi w_k$.
> 2. **Probabilities.** $\mu \leftarrow \sigma(a)$, evaluated in the stable form of §5.
> 3. **Weights.** $s_n \leftarrow \max\big(\mu_n(1-\mu_n),\ \varepsilon\big)$ for each $n$; set $S = \operatorname{diag}(s)$.
> 4. **Working response.** $z \leftarrow a + (y-\mu)\,/\,s$ (elementwise division).
> 5. **Weighted least squares solve.** Set $w_{k+1}$ to the minimizer of
>    $$\sum_{n=1}^{N} s_n\big(z_n - \phi(x_n)^\top w\big)^2 \;+\; \frac{1}{\tau^2}\|w\|_2^2,$$
>    i.e. solve $\big(\Phi^\top S\Phi + \tau^{-2}I\big)\,w_{k+1} = \Phi^\top S\,z$.
> 6. **Convergence test.** Stop if $\big\|\Phi^\top(\mu-y) + \tau^{-2}w_{k+1}\big\|_\infty < \texttt{tol}$, or if $\|w_{k+1}-w_k\|_2 \le \texttt{tol}\,(1+\|w_k\|_2)$.
>
> **Return** $w_{k+1}$.

**Three implementation notes, all of which matter.**

*Step 5 must not form the normal equations.* Row-scale instead: define $\tilde\Phi = S^{1/2}\Phi$ and $\tilde z = S^{1/2}z$, then solve the ordinary least squares problem $\min_w\|\tilde z - \tilde\Phi w\|_2^2$ by QR. Forming $\Phi^\top S\Phi$ squares the condition number, exactly as in Lecture 4 §6, and the resulting loss of accuracy is worst in precisely the ill-conditioned problems where you needed Newton's method in the first place. (With a prior, append $\tau^{-1}I_D$ as $D$ extra rows of $\tilde\Phi$ and $D$ zeros to $\tilde z$; the augmented least squares problem is algebraically identical to step 5 and inherits the same numerical advantage.)

*The weight floor $\varepsilon$ in step 3 is not cosmetic.* Under separability, §8 showed $s_n\to0$, which makes the division in step 4 blow up and $\Phi^\top S\Phi$ collapse toward singularity. With no prior, IRLS on separable data will fail — and should be allowed to fail loudly rather than silently return a large arbitrary $w$. With a prior, $\tau^{-2}I$ keeps step 5 well posed regardless.

*Step 6 tests the gradient, not the objective.* The objective changes by less and less near the optimum in a way that says nothing about proximity to the solution; the gradient norm is the quantity that actually certifies stationarity. This is the convergence criterion expected in this course.

### 12.6 Why Newton is nonetheless abandoned

Forming $\mathbf{H}$ costs $O(ND^2)$, storing it $O(D^2)$, and factoring it $O(D^3)$ per iteration. At $D = 10$ this is free; at $D = 10^4$ it is painful; at $D = 10^9$ the Hessian cannot be written down, let alone factored.

Modern models live in the last regime, which is why the second half of this course uses first-order methods almost exclusively — and why much of the practical craft there consists of *approximating* curvature cheaply, recovering some of Newton's affine invariance without paying $O(D^3)$. Quasi-Newton methods (BFGS, L-BFGS) build a low-rank approximation to $\mathbf{H}^{-1}$ from successive gradients; the adaptive optimizers of §13 keep only its diagonal. Both are best understood as answers to the question this section poses: *how much of $\mathbf{H}^{-1}$ can be afforded?*

Note the shape of the trade, because it recurs throughout the course: gradient descent is cheap per iteration and needs $O(\kappa)$ iterations; Newton is expensive per iteration and needs a handful. Which wins is a question about $D$ and $\kappa$, not a question about which method is better.

## 13. What is deferred, and to where

Stated explicitly so that nobody leaves believing this is the whole subject. Not covered today, and picked up in the optimization and deep learning material:

- **Momentum and Nesterov acceleration**, which improve the $O(\kappa)$ iteration count to $O(\sqrt\kappa)$ — a real change in the exponent, not a tweak.
- **Adaptive methods** — AdaGrad, RMSProp, Adam — which build a cheap diagonal approximation to curvature and are the practical descendants of §12's argument.
- **Line search and trust regions**, which choose $\eta$ per step instead of fixing it.
- **Stochastic gradients and minibatching**, without which none of this scales to large $N$, and which change the analysis of §10 substantially since the gradient is then a random variable.
- **Learning-rate schedules**, and why a decaying $\eta$ is necessary once gradients are stochastic.
- **Nonconvexity**: saddle points, plateaus, and the loss of every global guarantee assumed in Part C.

---

## Checkpoint B (take-home, due before Lecture 6)

`numpy` only; `scipy.stats` and any library optimizer are off-limits. Report fitted numbers, not pictures that look about right.

1. **Implement and verify.** Code $\mathrm{NLL}$, $\nabla\mathrm{NLL}$, and $\mathbf{H}$ from (5.3), §6, and §7. Use the stable forms (5.3)–(5.4). Verify the gradient against central finite differences and report the maximum relative error over 20 random $w$; verify the Hessian the same way against finite differences of the gradient. A gradient check is the single highest-value habit in this course — do it before every experiment for the rest of the semester.
2. **Overflow.** Construct a $w$ for which the naive form (5.2) returns `nan` or `inf` while (5.3) returns a correct finite value. Report both outputs and the value of $\max_n|a_n|$ at which the naive form first fails.
3. **Step size.** On a well-conditioned separable-free dataset ($N = 500$, $D = 5$), run gradient descent from $w = 0$ for $\eta$ spanning several orders of magnitude. Report iterations to $\|\nabla\|_2 < 10^{-6}$ against $\eta$, identify the largest $\eta$ that converges, and compare it to the bound $2/\lambda_{\max}(\mathbf{H})$ evaluated at the optimum and to the a-priori bound $8/d_1^2$ of §7.
4. **Conditioning.** Take the same dataset and rescale one feature column by $10^3$. Report $\kappa(\mathbf{H})$ at the optimum for both scalings, the iteration count to the same tolerance for both, and the ratio of iteration counts. Compare that ratio against the $\kappa/2\cdot\log(1/\varepsilon)$ prediction of §10. Then standardize all columns and report $\kappa$ and the iteration count a third time.
5. **Newton.** Implement Newton's method via the IRLS form of §12. Report iteration counts for *both* scalings of item 4 and confirm they are identical up to floating point. State in one sentence which equation in §12 predicts this.
6. **Separability.** Construct a separable dataset. Run gradient descent for $10^4$ iterations and plot $\mathrm{NLL}$ and $\|w\|_2$ against iteration count. Confirm $\mathrm{NLL}\to0$ and $\|w\|_2\to\infty$, and fit the growth of $\|w\|_2$ against iteration on suitable axes — report the fitted exponent or rate. Then add a Gaussian prior with $\tau^2 = 1$ and show $\|w\|_2$ converges; report the limiting value for $\tau^2\in\{0.1, 1, 10\}$ and the trend.

---

## Quiz-eligible facts

1. The MLE fails to exist in three ways — collapsing variance, singular $\Phi^\top\Phi$, logistic separability — all instances of a log-likelihood flat or unbounded along some direction of parameter space.
2. A prior repairs all three because it supplies information that is not a function of the data.
3. Ridge is MAP under $w\sim\mathcal{N}(0,\tau^2I)$ with $\lambda = \sigma^2/\tau^2$; lasso is MAP under a Laplace prior. Every regularizer is a log-prior.
4. Ridge retains singular direction $j$ with factor $d_j^2/(d_j^2+\lambda)$, shrinking hardest where least squares is least determined; $\mathrm{df}(\lambda) = \sum_j d_j^2/(d_j^2+\lambda)$.
5. Lasso's sparsity is a property of the posterior mode, not the posterior; the posterior mean is not sparse.
6. MAP is not invariant under reparameterization, because the prior acquires a Jacobian factor; the MLE is invariant. In high dimensions the mode lies outside the typical set.
7. $\sigma'(a) = \sigma(a)(1-\sigma(a))$ and $1-\sigma(a) = \sigma(-a)$.
8. The stable logistic NLL per datum is $\mathrm{softplus}(a) - ya$, with $\mathrm{softplus}(a) = \max(a,0) + \log(1+e^{-|a|})$ exactly.
9. $\nabla_w\mathrm{NLL} = \Phi^\top(\mu - y)$ with $\mu = \sigma(\Phi w)$ — the same $\Phi^\top(\text{prediction}-\text{target})$ form as linear regression, and as every canonical-link model.
10. No closed form exists because $\Phi^\top(\sigma(\Phi w)-y) = 0$ is transcendental in $w$, whereas the linear-regression stationarity condition is linear.
11. $\mathbf{H} = \Phi^\top S\Phi$ with $S = \operatorname{diag}(\mu_n(1-\mu_n))$; hence $\mathbf{H}\succeq0$ and the logistic NLL is convex, strictly so if $\Phi$ has full column rank.
12. $\mu(1-\mu)\le\frac14$, so $\lambda_{\max}(\mathbf{H})\le\frac14 d_1^2$ and $\eta < 8/d_1^2$ guarantees gradient descent converges.
13. On separable data the NLL infimum is $0$ and is unattained, $\|\hat w\|\to\infty$, and $\mathbf{H}\to0$; a Gaussian prior makes the objective strictly convex with bounded sublevel sets, so a unique minimizer exists.
14. Gradient descent is steepest descent in the Euclidean norm, obtained by minimizing a linear model over a ball.
15. On a quadratic, the error decouples in the Hessian eigenbasis: $\tilde e_{k,j} = (1-\eta\lambda_j)^k\tilde e_{0,j}$.
16. Gradient descent converges iff $0<\eta<2/\lambda_{\max}$; the optimal fixed step is $2/(\lambda_{\min}+\lambda_{\max})$ with rate $(\kappa-1)/(\kappa+1)$, so iteration count grows proportionally to $\kappa$.
17. Divergence appears as an oscillating, growing loss along the stiffest eigendirection, and indicates $\eta$ exceeded a bound set by $\lambda_{\max}$.
18. Rescaling feature columns changes $\kappa$ and therefore the iteration count, without changing the model or its predictions. Standardization equalizes the diagonal only; correlated features remain ill-conditioned.
19. The Newton step is the exact minimizer of the second-order Taylor model $m_k(\delta) = f + g^\top\delta + \frac12\delta^\top\mathbf{H}\delta$, obtained by solving $\mathbf{H}\delta = -g$. No step size is introduced, because the quadratic model is bounded below whereas a linear model is not.
20. Newton is exact in one step on a quadratic, and converges quadratically near a minimum — the number of correct digits roughly doubles per iteration — because the discarded Taylor remainder is $O(\|\delta\|^3)$.
21. Newton is affine invariant: under $w = A\tilde w$, $\tilde\delta = A^{-1}\delta$. Hence feature scaling does not change its iterates, and $\kappa$ does not enter its iteration count.
22. Newton is steepest descent in the norm $\|\delta\|_{\mathbf H}^2 = \delta^\top\mathbf{H}\delta$; gradient descent is the special case $A = I$.
23. If $\mathbf{H}$ is indefinite the quadratic model is unbounded below and the step is meaningless; damping solves $(\mathbf{H}+\gamma I)\delta = -g$, interpolating between Newton ($\gamma\to0$) and gradient descent ($\gamma\to\infty$). A Gaussian prior supplies this damping with $\gamma = \tau^{-2}$.
24. For logistic regression the Newton step equals a weighted least squares solve with weights $S$ and working response $z = \Phi w + S^{-1}(y-\mu)$ — iteratively reweighted least squares.
25. The working response $z_n = a_n + (y_n-\mu_n)/\sigma'(a_n)$ is the observation linearized onto the logit scale; the weight $s_n = \mu_n(1-\mu_n)$ equals $1/\operatorname{Var}[z_n]$, so IRLS uses exactly the precisions the Bernoulli noise model dictates.
26. Implement the IRLS solve by QR on the row-scaled $S^{1/2}\Phi$ against $S^{1/2}z$, never by forming $\Phi^\top S\Phi$; and terminate on the gradient norm, not on the change in objective.
27. Newton costs $O(D^3)$ per iteration and $O(D^2)$ memory, which is why large-scale training uses first-order methods; quasi-Newton and adaptive methods are cheap approximations to $\mathbf{H}^{-1}$.

---

## Practice problems

*Ungraded; these feed the quizzes.*

**P1.** Prove (5.4) exactly, by cases on the sign of $a$. Then show that the naive evaluation of $\log(1+e^a)$ overflows for $a\gtrsim 710$ in double precision, while (5.4) does not, and explain what happens to the *gradient* $\sigma(a)$ in the same regime.

**P2.** Derive $\nabla\mathrm{NLL}$ and $\mathbf{H}$ starting from form (5.2) rather than (5.3), and confirm you obtain the same expressions. Which derivation would you rather do again, and what does that suggest about which form to differentiate in general?

**P3.** Show that the softmax/categorical model of Lecture 3 §3 has gradient $\Phi^\top(\hat Y - Y)$ in the appropriate matrix notation. Identify what plays the role of $S$ in its Hessian and verify that the Hessian is positive semi-definite.

**P4.** For the quadratic of §10 with $D = 2$, $\lambda_1 = 100$, $\lambda_2 = 1$, and $\eta = \eta^\star$: compute the number of iterations required to reduce $\|e\|$ by $10^{-6}$, then plot the trajectory $w_k$ over the level sets of $f$. Explain the characteristic zig-zag in terms of (10.1).

**P5.** Show that gradient descent with $\eta = 2/\lambda_{\max}$ exactly does not converge but does not diverge either. What does the iterate sequence do along $v_1$? Along $v_D$?

**P6.** Prove that preconditioned gradient descent, $w_{k+1} = w_k - \eta A^{-1}\nabla f(w_k)$, converges at a rate governed by $\kappa(A^{-1}\mathbf{H})$ rather than $\kappa(\mathbf{H})$. What choice of $A$ makes this $1$, and what does the resulting method reduce to?

**P7.** For the separable case of §8, take $D=1$, $\phi(x) = x$, and the two data points $(x,y) = (1,1)$ and $(-1,0)$. Write $\mathrm{NLL}(w)$ explicitly, show it is strictly decreasing in $w$, and compute $\lim_{w\to\infty}\mathrm{NLL}(w)$. Then add the prior term and solve for $\hat w_{\mathrm{MAP}}$ numerically for $\tau^2\in\{0.1,1,10\}$. Show $\hat w_{\mathrm{MAP}}\to\infty$ as $\tau^2\to\infty$ and explain why that is the correct behavior.

**P8.** Show that Fisher scoring — Newton's method with $\mathbf{H}$ replaced by the Fisher information $\mathbf{F}$ of Lecture 4 §3.3 — coincides with Newton's method for logistic regression, and give a model with a non-canonical link for which the two differ.

**P9 (the working response).** Derive $z_n = a_n + (y_n-\mu_n)/\sigma'(a_n)$ directly, by linearizing the map $a\mapsto\mu = \sigma(a)$ about the current $a_n$ and solving for the logit that would have produced $y_n$. Then verify $\operatorname{Var}[z_n] = 1/s_n$ and explain, in one sentence each, what happens to $z_n$ and to $s_n$ for a point the model predicts with near-certainty.

**P10 (damping).** Implement damped Newton, $w_{k+1} = w_k + \eta_k\delta_k$, with $\eta_k$ chosen by backtracking: start at $\eta_k = 1$ and halve until $f(w_k+\eta_k\delta_k) < f(w_k)$. Construct a logistic problem and an initialization far from the optimum at which the undamped step *increases* the objective, and report the objective values before and after. Then confirm that $\eta_k = 1$ is accepted on every iteration once the iterates are near the optimum, and say why that must be so.

**P11 (rewriting the solve).** Show that the augmented least squares problem $\min_w \big\|\begin{bmatrix}S^{1/2}z\\ 0\end{bmatrix} - \begin{bmatrix}S^{1/2}\Phi\\ \tau^{-1}I\end{bmatrix}w\big\|_2^2$ has the same solution as step 5 of the IRLS algorithm. Compare the condition numbers of the augmented matrix and of $\Phi^\top S\Phi + \tau^{-2}I$ on a deliberately ill-conditioned design, and report both.
