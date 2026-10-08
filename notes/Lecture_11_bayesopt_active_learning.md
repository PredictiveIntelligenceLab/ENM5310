# Lecture 11 — Bayesian Optimization and Active Learning

**ENM 5310 — Data-driven Modeling and Probabilistic Scientific Computing**

**Builds on:** Lecture 7 (aleatoric and epistemic uncertainty), Lecture 9 (GP training and inference; the predictive mean $\mu(x)$ and variance $v(x)$, and the fact that $v$ does not depend on the observed targets), Lecture 10 (multi-fidelity GPs).

**Reading:** Shahriari et al. (2016), *Taking the human out of the loop: a review of Bayesian optimization* — §§I–IV. Frazier (2018), *A tutorial on Bayesian optimization*. Settles (2009), *Active learning literature survey* — §§1–3. Companion: Murphy II, the Bayesian optimization section of the optimization chapter.

**What this lecture adds.** Lectures 8–10 built a model that reports $\mu(x)$ and $v(x)$ at any input. This lecture puts a **decision layer** on top of it: given that model, where should I spend my next expensive evaluation? Two versions of the question — *to find an optimum* (Bayesian optimization) and *to learn a function or a boundary* (active learning) — turn out to be the same construction with a different objective.



**Learning objectives.** After this lecture you should be able to:

1. Explain why "where is the minimum of $f$?" has no answer from data alone, and what a probabilistic model replaces it with.
2. Define the posterior distribution over the minimizer, say why it is intractable, and describe how to sample from it.
3. Derive the probability of improvement and the expected improvement in closed form for a Gaussian predictive, and identify the exploitation and exploration terms in EI.
4. State the lower confidence bound rule and explain what $\beta$ controls.
5. Write down the Bayesian optimization loop and name the practical choices it hides.
6. Explain uncertainty sampling for classification, why predictive entropy is the wrong criterion, and how BALD fixes it.
7. Say when these methods are appropriate for a problem of your own, and what baseline to compare against.

### Notation

Objective $f:\mathcal{X}\to\mathbb{R}$ to be **minimized** over a bounded domain $\mathcal{X}\subset\mathbb{R}^d$ (for maximization, flip signs). Data $\mathcal{D}_n = \{(x_i,y_i)\}_{i=1}^n$ with $y_i = f(x_i)+\epsilon_i$. GP posterior mean and standard deviation $\mu(x)$, $\sigma(x) = \sqrt{v(x)}$. Incumbent $\tau = \min_i y_i$. Standard normal density and CDF $\varphi$, $\Phi$. Acquisition function $\alpha(x)$.

---

## 1. Where is the minimum?

Here are eight noisy observations of some function on $[0,1]$. **Where is the minimum of $f$?**

The instinctive answer — point at the lowest data point — is wrong in three separate ways, and each one is worth naming.

**It answers a different question.** The lowest observation is the minimum of the *data*, not of the function. Those are the same thing only if you happen to have sampled at the minimizer.

**The observations are noisy.** The lowest $y_i$ may be a point with ordinary $f$ and a lucky draw of $\epsilon$. Taking the minimum over $n$ noisy values systematically selects for negative noise, so $\min_i y_i$ is biased low.

**There is nothing between the points.** Any function consistent with the data could dip sharply in a gap you did not sample.

In fact the question as posed has **no answer from data alone**. It presupposes knowledge of $f$ that we do not have. The honest reformulation is:

> Given the data, what do I **believe** about the location of the minimum — and where should I spend my next expensive evaluation to sharpen that belief?

That is a question a probabilistic model can answer, and the answer will not be a point but a distribution.

**The setting this matters in.** Everything below assumes evaluations of $f$ are **expensive** — hours of simulation, a day of lab work, a wind-tunnel run — so you will get tens, not millions, of them; that you have **no gradients**; and that $f$ may be **noisy** and is certainly **non-convex**. Under those conditions the methods of Lecture 6 are unusable: gradient descent needs cheap gradients, and a grid search in $d$ dimensions needs exponentially many points. Typical instances: tuning the design parameters of a device, choosing operating conditions for an experiment, calibrating a simulator, selecting hyperparameters of an expensive model.

---

## 2. The posterior over the minimizer, and why we cannot use it

The GP of Lectures 8–9 gives a posterior over *functions*. The minimizer is a functional of the function,

$$x^\star = \arg\min_{x\in\mathcal{X}} f(x),$$

so the posterior over $f$ induces a posterior over $x^\star$ — the pushforward of $p(f\mid\mathcal{D})$ through $\arg\min$:

$$\boxed{\ p\big(x^\star\mid\mathcal{D}\big).\ }$$

**This is the object we actually want.** It is the complete answer to the opening question, and it immediately suggests the right way to choose the next evaluation: pick the $x$ that will teach us the most about $x^\star$, for instance by maximally reducing the entropy of $p(x^\star\mid\mathcal{D})$.

**You can look at it, by sampling.** Draw a function from the GP posterior on a fine grid (Lecture 9 §3.1, the joint multi-point predictive), take its $\arg\min$, and record it; repeat. The histogram of those arg-mins is a Monte Carlo estimate of $p(x^\star\mid\mathcal{D})$. It is worth drawing once in class, because it makes several things obvious at once: the distribution is typically multimodal, it concentrates near low posterior mean *and* near high uncertainty, and it bears little resemblance to a point estimate.

**But you cannot compute with it.** There is no closed form, even for a GP. The $\arg\min$ depends on the entire sample path simultaneously, not on the marginal at any one $x$, so none of Lecture 9's per-point formulas apply. Using it to *choose* the next point is worse still: that would require, for each candidate $x$, an expectation over the hypothetical observation $y$ of the entropy of a distribution you can only estimate by sampling — a nested Monte Carlo computation, for every candidate, at every iteration. (Methods that do approximately this exist and are called *entropy search*; they are accurate and slow.)

**So we do what one always does with an intractable objective: replace it with a tractable proxy.**

> **Acquisition function.** A cheap function $\alpha(x)$, computed from the posterior at $x$ alone — in practice from $\mu(x)$ and $\sigma(x)$ — whose maximizer is taken as the next evaluation point:
> $$x_{n+1} = \arg\max_{x\in\mathcal{X}} \alpha(x).$$

Three things make this work. $\alpha$ is cheap, so we may optimize it hard — it is evaluated on the surrogate, not on $f$. It is available in closed form, so we have gradients and can use Lecture 6's machinery. And the next three sections are nothing more than three different, defensible choices of $\alpha$.

The pattern is worth naming, because it recurs: **we have an expensive, intractable decision problem, and we solve a cheap, explicit surrogate of it instead.**

---

## 3. Improvement-based acquisitions

The first family asks a concrete question: *how much better than what I already have might this point be?*

Define the **improvement** at $x$, relative to the incumbent $\tau = \min_i y_i$:

$$I(x) = \max\big(\tau - f(x),\ 0\big),$$

a random variable, because $f(x)$ is. It is zero when the point turns out no better than what we have, and equal to the margin when it is better. Under the GP posterior, $f(x)\mid\mathcal{D}\sim\mathcal{N}\big(\mu(x),\sigma^2(x)\big)$, so both acquisitions below follow from one-dimensional Gaussian integrals.

Throughout, write

$$z(x) = \frac{\tau-\mu(x)}{\sigma(x)},$$

the number of posterior standard deviations by which the incumbent beats the predicted mean. Large positive $z$ means "this point is predicted to be much better than what I have, relative to my uncertainty."

### 3.1 Probability of improvement

Ask only whether there is *any* improvement:

$$\alpha_{\mathrm{PI}}(x) = \Pr\big[f(x)<\tau\big] = \Phi\!\left(\frac{\tau-\mu(x)}{\sigma(x)}\right) = \Phi\big(z(x)\big).$$

One line, directly from the Gaussian CDF. It is intuitive and it was the first acquisition proposed (Kushner, 1964).

**It also behaves badly, for an instructive reason.** PI counts probability and ignores magnitude. A point almost certain to improve by $10^{-6}$ scores higher than one with a $40\%$ chance of improving by a factor of ten. So PI is **relentlessly exploitative**: it clusters new evaluations immediately around the current best point, refining a local basin while never looking elsewhere.

The standard patch is a margin $\xi>0$,

$$\alpha_{\mathrm{PI}}(x) = \Phi\!\left(\frac{\tau-\xi-\mu(x)}{\sigma(x)}\right),$$

demanding improvement by at least $\xi$. This helps, but $\xi$ is a free parameter with no natural scale, and performance is sensitive to it. The lesson generalizes: **an acquisition that ignores the size of the prize will over-exploit.**

### 3.2 Expected improvement

Take the expectation of the improvement itself rather than the probability that it is positive:

$$\alpha_{\mathrm{EI}}(x) = \mathbb{E}\big[I(x)\big] = \int_{-\infty}^{\tau}(\tau-f)\,\mathcal{N}\big(f\mid\mu,\sigma^2\big)\,df .$$

**The integral is elementary.** Substitute $u = (f-\mu)/\sigma$, so $f = \mu+\sigma u$ and the upper limit becomes $z$:

$$\alpha_{\mathrm{EI}} = \int_{-\infty}^{z}\big(\tau-\mu-\sigma u\big)\varphi(u)\,du = (\tau-\mu)\Phi(z)-\sigma\int_{-\infty}^{z}u\,\varphi(u)\,du .$$

The last integral is $-\varphi(z)$, since $u\varphi(u) = -\varphi'(u)$. Hence

$$\boxed{\ \alpha_{\mathrm{EI}}(x) = \big(\tau-\mu(x)\big)\,\Phi\big(z(x)\big)\;+\;\sigma(x)\,\varphi\big(z(x)\big) = \sigma(x)\Big[z\Phi(z)+\varphi(z)\Big].\ }$$

**Read the two terms. This is the heart of the lecture.**

- $(\tau-\mu)\Phi(z)$ is **exploitation**: large where the posterior mean is low, i.e. where the model already believes the function is small.
- $\sigma\varphi(z)$ is **exploration**: large where the posterior standard deviation is big, i.e. where the model does not know.

Nobody put those two terms there by hand, and there is no knob trading them off. They fall out of taking an expectation of a quantity that rewards being better than the incumbent, under a posterior that admits it might be wrong. **The exploration–exploitation balance is a consequence of honest uncertainty, not an addition to it.** This is the single best argument in the course for carrying a predictive distribution rather than a point prediction.

Two properties worth checking on the formula. At an already-observed noiseless point, $\sigma = 0$ and $\alpha_{\mathrm{EI}} = 0$: EI never wastes an evaluation re-measuring something known. And as $\sigma\to\infty$ at fixed $\mu$, EI grows — a completely unexplored region is always worth something.

EI is the default acquisition function, and has been since Jones, Schonlau and Welch popularized it in 1998 (their method is called EGO). Start here.

---

## 4. Optimism-based acquisitions

A different principle, with a different pedigree: **optimism in the face of uncertainty.** Rather than averaging over what might happen, act as if the plausible best case were true.

$$\boxed{\ \alpha_{\mathrm{LCB}}(x) = \mu(x)-\sqrt{\beta}\,\sigma(x), \qquad x_{n+1} = \arg\min_x\ \alpha_{\mathrm{LCB}}(x).\ }$$

This is the **lower confidence bound** — for maximization it is the upper confidence bound, UCB, which is the name you will more often meet. It evaluates wherever the optimistic estimate of the function is smallest: either because the mean is low, or because the uncertainty is large.

**$\beta$ is an explicit exploration knob**, which is the main practical difference from EI. At $\beta = 0$ the rule minimizes the posterior mean, which is pure exploitation and will stall in the first basin it finds. Large $\beta$ sends it to the most uncertain region regardless of the mean, which is pure exploration. Typical values are $\beta$ of order a few. Having a knob is a liability when you do not know what to set it to, and an asset when you *do* know that your problem needs more exploration than usual.

**It also comes with theory.** For GP priors and a schedule $\beta_n$ growing logarithmically in $n$, GP-UCB achieves sublinear cumulative regret (Srinivas et al., 2010) — the average quality of evaluated points approaches the optimum. EI has guarantees too, but UCB's are cleaner, and this is the main reason UCB dominates the theoretical literature while EI dominates practice.

**Thompson sampling** deserves a mention, because it closes the loop back to §2. Draw *one* function from the GP posterior, minimize that draw, and evaluate there:

$$f_s\sim p(f\mid\mathcal{D}_n), \qquad x_{n+1} = \arg\min_x f_s(x).$$

By construction, $x_{n+1}$ is a sample from $p(x^\star\mid\mathcal{D}_n)$ — the intractable posterior of §2. We cannot compute that distribution, but we can draw from it, and drawing from it is a sensible randomized rule. It needs no knob, it explores automatically because different draws have different minimizers, and it parallelizes trivially: for a batch of $q$ evaluations, take $q$ independent draws.

| | Rule | Knob | Behavior |
|---|---|---|---|
| PI | $\Phi(z)$ | $\xi$ | Over-exploits; ignores the size of the improvement |
| EI | $\sigma[z\Phi(z)+\varphi(z)]$ | none | Balanced; the sensible default |
| LCB | $\mu-\sqrt\beta\,\sigma$ | $\beta$ | Explicit trade-off; strong theory |
| Thompson | $\arg\min$ of one posterior draw | none | Randomized; batches for free |

---

## 5. The Bayesian optimization loop in practice

> ### Algorithm: Bayesian optimization
>
> 1. **Initial design.** Evaluate $f$ at $n_0$ space-filling points — a Latin hypercube or Sobol sequence, roughly $10d$ if you can afford it, fewer if not.
> 2. **Fit the surrogate.** Train a GP on $\mathcal{D}_n$ by maximizing the marginal likelihood (Lecture 9 §2).
> 3. **Optimize the acquisition.** $x_{n+1} = \arg\max_x\alpha(x)$ over $\mathcal{X}$.
> 4. **Evaluate.** Run the expensive $f$ at $x_{n+1}$, obtain $y_{n+1}$.
> 5. **Augment and repeat.** $\mathcal{D}_{n+1} = \mathcal{D}_n\cup\{(x_{n+1},y_{n+1})\}$; return to step 2 until the budget is spent.
> 6. **Report.** The best observed point, or — if observations are noisy — the minimizer of the posterior mean.

Five practical points that the six lines above conceal.

**Step 3 is itself a global optimization.** The acquisition is multimodal and often nearly flat between sharp peaks. But it is cheap and differentiable, so the standard approach is many random restarts of L-BFGS with analytic gradients, or a coarse random or Sobol search followed by local refinement. Use enough restarts: a BO run that under-optimizes its acquisition is behaving randomly without telling you.

**Refit the hyperparameters every iteration**, and expect trouble early on. With $n_0 = 10$ points the marginal likelihood is poorly determined, and the local optimum with a huge length scale and large noise (Lecture 9 §5) is a real hazard — it makes the surrogate nearly flat, which makes the acquisition nearly flat, which makes BO degenerate into random sampling. Priors on the hyperparameters help considerably here.

**Standardize everything.** Inputs scaled to $[0,1]^d$, outputs centered and scaled to unit variance. All the defaults in the literature — $\beta$ of order a few, jitter of $10^{-6}$ — assume you have done this.

**Noise changes the incumbent.** With noisy observations, $\tau = \min_i y_i$ is biased low and can be an outlier, which makes EI chase a value the function never achieved. Use the minimum of the posterior mean at the observed points instead.

**Know the regime.** BO works well up to roughly $10$–$20$ dimensions. Beyond that, the GP needs help in the form of structure — additive kernels, a learned low-dimensional embedding, or trust regions. And **always run random search as a baseline**: it is embarrassingly strong in higher dimensions, it costs nothing to implement, and a BO run that does not beat it is not working.

**Two extensions you can reach immediately.** Constraints: if a separate expensive function $c(x)\le0$ must hold, model $c$ with its own GP and multiply, $\alpha(x)\cdot\Pr[c(x)\le0]$. Multi-fidelity: with Lecture 10's model you choose not only *where* but at *which fidelity*, by dividing the acquisition by the cost of an evaluation at that level and picking the best improvement per unit cost.

---

## 6. Active learning: where should I measure?

Change the goal. Suppose you do not want the minimum — you want to **learn the function**, or a boundary in it, from as few expensive measurements as possible. Calibrating a model across an operating envelope, mapping a stability boundary, labeling data when labels require an expert: the budget is the same, the question is different.

The machinery is identical. A probabilistic model supplies a predictive distribution; an acquisition function scores candidate inputs; you evaluate the best one and repeat. **Only the acquisition changes, because only the objective changed.**

### 6.1 Regression: uncertainty sampling, and its catch

The obvious rule is to measure where you are least sure:

$$\alpha(x) = \sigma(x) \qquad\text{or}\qquad \alpha(x) = v(x).$$

Lecture 9 §3.3 noted that for a GP with fixed hyperparameters $v(x)$ does not depend on the observed targets $y$. That has a striking consequence here:

> **You can plan the entire experimental campaign before running a single experiment.** Choose $x_1$, pretend you ran it, update $v$ (which needs no $y$), choose $x_2$, and so on.

That is genuinely useful when experiments are run in batches or must be scheduled in advance. But notice what it also means: a criterion that never looks at the data is a **geometric design rule**. For a stationary kernel, maximizing $v$ just places points far from existing points, which reproduces space-filling design. The data enters only through the fitted hyperparameters.

So the honest statement is: for a GP with known hyperparameters, active learning in regression is design of experiments, and the classical answer — fill the space — is close to optimal. It becomes interesting when the kernel is non-stationary, when hyperparameters are being learned, or when you care about a *functional* of $f$ rather than $f$ itself (a boundary, an integral, a failure probability), because then uniform coverage is wasteful.

### 6.2 Classification: measuring near the boundary

Now the interesting case, and the one that motivates the whole subject. The model outputs a class probability $\pi(x) = p(y=1\mid x,\mathcal{D})$, and the thing you want to learn well is the **decision boundary**, the set where $\pi = 1/2$. Labels far inside a class are nearly worthless; labels near the boundary are what locate it.

The natural criterion is **uncertainty sampling**: pick $x$ where the predicted label is most uncertain, measured by the predictive entropy

$$\mathbb{H}\big[\pi(x)\big] = -\pi\log\pi-(1-\pi)\log(1-\pi),$$

maximized at $\pi = 1/2$. This points exactly at the current estimate of the boundary, and it works — a handful of labels placed along a boundary beats many times as many placed at random.

**But predictive entropy is the wrong quantity, and Lecture 7 already told us why.** It is large in two very different situations:

- the model **does not know** the label because it has seen nothing nearby — *epistemic* uncertainty, which a label would resolve; and
- the label is **genuinely ambiguous**, a region where the classes truly overlap and repeated measurements would come back $50/50$ — *aleatoric* uncertainty, which no label will ever resolve.

Uncertainty sampling cannot tell these apart and will happily spend your entire budget re-measuring an inherently noisy region. This is the single most common failure of naive active learning.

**The fix is to score the epistemic part only.** Ask how much a label at $x$ would tell you about the model parameters $\theta$ — the mutual information between the label and the parameters:

$$\boxed{\ \alpha_{\mathrm{BALD}}(x) = \mathbb{I}\big[y;\theta\mid x,\mathcal{D}\big] = \underbrace{\mathbb{H}\Big[\mathbb{E}_{p(\theta\mid\mathcal{D})}\big[p(y\mid x,\theta)\big]\Big]}_{\text{total predictive entropy}} \;-\; \underbrace{\mathbb{E}_{p(\theta\mid\mathcal{D})}\Big[\mathbb{H}\big[p(y\mid x,\theta)\big]\Big]}_{\text{expected aleatoric entropy}} .\ }$$

The first term is the uncertainty-sampling criterion. The second subtracts the uncertainty that remains *even when $\theta$ is known* — the irreducible part. What survives is exactly the disagreement among plausible models. This is **BALD** (Bayesian Active Learning by Disagreement, Houlsby et al., 2011), and it is the aleatoric/epistemic decomposition of Lecture 7 used as a decision rule.

A picture that makes it stick: BALD is large where plausible models **disagree** — some say class 1, some say class 0 — and small both where they agree confidently and where they all agree the answer is a coin flip.

**Computing it.** Draw $S$ samples $\theta_s$ from the posterior (for a GP classifier, draw latent functions; for a Bayesian neural network, use whatever posterior approximation you have), and estimate both terms by Monte Carlo:

$$\bar\pi = \frac1S\sum_s\pi_s(x), \qquad \alpha_{\mathrm{BALD}}\approx\mathbb{H}[\bar\pi]-\frac1S\sum_s\mathbb{H}\big[\pi_s(x)\big].$$

### 6.3 Two cautions

**Batches need diversity.** Taking the top $q$ points by any acquisition gives $q$ nearly identical points, since the acquisition surface is smooth and its peak is a region, not a point. Real batch methods either penalize proximity to already-selected points, or use the randomization of Thompson sampling, or maximize a joint information gain.

**Active learning can be worse than random.** Every selected point changes the model that selects the next one, so a misspecified model can steer itself into a region, confirm its own beliefs, and never discover that it is wrong elsewhere. There is no equivalent of a held-out set to warn you, because your data is no longer a random sample of anything. Compare against random selection, on a properly held-out set, before trusting a gain.

---

## 7. What to take away

**Everything in this lecture is a decision layer on top of the model you already had.** The model supplies $\mu(x)$ and $\sigma(x)$; the acquisition turns those two numbers into a choice; the choice produces a datum; the datum updates the model. Nothing about the GP changed.

**The intractable objective is the honest one, and the acquisition is the proxy.** The thing we actually want — the posterior over the minimizer, or the expected information gain about the boundary — is well defined and out of reach. PI, EI, LCB, Thompson sampling and BALD are all cheap stand-ins. Knowing what each one approximates is what lets you tell when it will fail.

**Exploration versus exploitation is not a knob you must tune; it is what a posterior gives you for free.** The two terms of EI are the cleanest demonstration in this course of why uncertainty is worth carrying.

**For your own problems.** Reach for these methods when evaluations cost minutes or more, you have no gradients, and your budget is in the tens or hundreds: optimizing a design, calibrating a model against data, choosing experimental conditions, mapping a feasibility or failure boundary. Do not reach for them when $f$ is cheap — use Lecture 6 — or when you have thousands of free evaluations and a hundred dimensions, where the surrogate is the bottleneck. And measure against random search every time.

---

## Checkpoint G (take-home)

`numpy` and your GP code from Checkpoint E. Report fitted numbers, not pictures that look about right.

1. **The opening question.** Take $f(x) = \sin(3x)+x^2/2$ on $[-2,2]$ with noise $\sigma = 0.05$, and $8$ random observations. Fit a GP, draw $2000$ joint posterior samples on a fine grid, take the $\arg\min$ of each, and plot the histogram — your estimate of $p(x^\star\mid\mathcal{D})$. Is it unimodal? Compare its mode with the location of the lowest observation and with the true minimizer.
2. **The two terms.** On the same fit, plot $\alpha_{\mathrm{EI}}$ together with its exploitation term $(\tau-\mu)\Phi(z)$ and its exploration term $\sigma\varphi(z)$ separately. Identify a region where each dominates, and report the $x$ that maximizes each of the three.
3. **A loop that works.** Implement the algorithm of §5 with EI, starting from $5$ points and running $25$ iterations. Plot the simple regret $\min_i f(x_i)-f(x^\star)$ against iteration, averaged over $20$ random seeds, alongside the same curve for random search and for LCB with $\beta\in\{1,4,16\}$. Report the mean and standard error at iteration $25$ for each method, and say which differences are real rather than seed noise.
4. **Breaking PI.** Run the same loop with probability of improvement at $\xi = 0$. Plot the sequence of evaluated points against iteration for PI and for EI on the same axes. Describe in two sentences what PI does differently and relate it to §3.1.
5. **Active learning on a boundary.** Generate a 2-D binary classification problem whose true boundary is a circle, with labels flipped with probability $0.3$ inside a small disc elsewhere in the domain (an irreducibly ambiguous region). Using any Bayesian classifier you can sample from, select $30$ points by (a) random sampling, (b) maximum predictive entropy, and (c) BALD. Plot the selected points for each, and report test accuracy against a held-out grid. Where does uncertainty sampling spend its budget, and why does BALD avoid it?

---

## Quiz-eligible facts

1. "Where is the minimum?" has no answer from data alone: the lowest observation is the minimum of the data, is biased low under noise, and says nothing about the gaps between samples.
2. The posterior over the minimizer, $p(x^\star\mid\mathcal{D})$, is the pushforward of the GP posterior through $\arg\min$. It can be sampled — draw a posterior function, take its $\arg\min$ — but has no closed form, because $\arg\min$ depends on the whole sample path.
3. An acquisition function is a cheap proxy for an intractable decision objective, computed from $\mu(x)$ and $\sigma(x)$ and optimized on the surrogate rather than on $f$.
4. $\alpha_{\mathrm{PI}} = \Phi(z)$ with $z = (\tau-\mu)/\sigma$. It ignores the size of the improvement and therefore over-exploits, clustering around the incumbent.
5. $\alpha_{\mathrm{EI}} = (\tau-\mu)\Phi(z)+\sigma\varphi(z) = \sigma[z\Phi(z)+\varphi(z)]$, obtained by integrating $\max(\tau-f,0)$ against the Gaussian predictive. The first term is exploitation, the second exploration; no knob trades them off.
6. EI is zero at a noiselessly observed point, since $\sigma = 0$ there.
7. $\alpha_{\mathrm{LCB}} = \mu-\sqrt\beta\,\sigma$, minimized: optimism in the face of uncertainty. $\beta$ is an explicit exploration knob, and GP-UCB with a growing $\beta_n$ has sublinear regret.
8. Thompson sampling minimizes a single posterior draw; the resulting point is a sample from $p(x^\star\mid\mathcal{D})$, and $q$ draws give a batch for free.
9. The BO loop is: space-filling initial design, fit the GP, maximize the acquisition, evaluate, repeat. The acquisition optimization needs many restarts, and the hyperparameter fit is fragile at small $n$.
10. With noisy observations, use the minimum of the posterior mean as the incumbent, not $\min_i y_i$.
11. BO suits $d\lesssim10$–$20$ and budgets of tens to hundreds; always compare against random search.
12. For a GP with fixed hyperparameters, the predictive variance does not depend on $y$, so a whole active-learning campaign can be planned in advance — and pure variance sampling therefore reduces to space-filling design.
13. In classification, uncertainty sampling maximizes the predictive entropy, which is maximal at $\pi = 1/2$ and so targets the decision boundary.
14. Predictive entropy conflates epistemic and aleatoric uncertainty, so uncertainty sampling wastes labels in irreducibly ambiguous regions.
15. BALD scores the mutual information $\mathbb{I}[y;\theta\mid x,\mathcal{D}]$ = total predictive entropy minus expected aleatoric entropy, which is large where plausible models disagree. It is estimated by sampling from the posterior.
16. Greedy batches lack diversity, and active learning can underperform random selection when the model is misspecified, because the data is no longer a random sample.

---

## Practice problems

*Five questions in the style of the quizzes: each should take two or three minutes, with no computer. The answer follows each one — cover it and try first. Implementation is in Checkpoint G.*

**P1.** Why can't we simply choose the next evaluation by maximizing the reduction in entropy of $p(x^\star\mid\mathcal{D})$, which is the objective we actually care about?

*Answer.* Because $p(x^\star\mid\mathcal{D})$ has no closed form — $\arg\min$ depends on the entire posterior sample path, not on any per-point marginal — so it can only be estimated by sampling. Scoring a candidate $x$ would require averaging that sampling-based entropy over the unknown outcome $y$, for every candidate, at every iteration. Acquisition functions replace this with something computable from $\mu(x)$ and $\sigma(x)$ alone. (§2)

**P2.** Point $A$ has $\mu = -1.0$, $\sigma = 0.01$. Point $B$ has $\mu = -0.5$, $\sigma = 1.0$. The incumbent is $\tau = -1.0$. Which does PI prefer, which does EI prefer, and what does the disagreement illustrate?

*Answer.* At $A$, $z = 0$, so PI $= \Phi(0) = 0.5$ and EI $= \sigma\varphi(0)\approx0.004$ — a near-certain coin flip on a negligible gain. At $B$, $z = -0.5$, so PI $= \Phi(-0.5)\approx0.31$ and EI $= \sigma[z\Phi(z)+\varphi(z)]\approx 0.198$. PI prefers $A$, EI prefers $B$ by a wide margin. PI counts only the probability of improving; EI weighs it by how much. (§3)

**P3.** Show that EI is exactly zero at a point already observed without noise, and explain why that is the behavior you want.

*Answer.* There $\sigma = 0$, and $\alpha_{\mathrm{EI}} = \sigma[z\Phi(z)+\varphi(z)] = 0$ — both terms carry a factor of $\sigma$. This is correct: re-evaluating a noiselessly known point yields no new information and cannot improve on a value you already hold. (With noisy observations $\sigma$ does not vanish at observed points, and repeated measurement can be worthwhile.) (§3.2)

**P4.** A classifier is being trained with uncertainty sampling. The domain contains a region where the two classes genuinely overlap, so labels there are close to a coin flip no matter how much data you collect. What will uncertainty sampling do, and what criterion fixes it?

*Answer.* It will keep selecting points in that region, because the predictive entropy there is maximal and never decreases — the uncertainty is aleatoric, so labels cannot reduce it. BALD fixes this by subtracting the expected aleatoric entropy, leaving the mutual information between the label and the parameters: it is large only where plausible models *disagree*, and in a truly ambiguous region they all agree the answer is a coin flip. (§6.2)

**P5.** For a GP with fixed hyperparameters, the predictive variance does not depend on $y$. Name one practical opportunity this creates for active learning, and one limitation it exposes.

*Answer.* The opportunity: an entire campaign of measurements can be chosen in advance, before any experiment is run, which matters when experiments are batched or must be scheduled. The limitation: a criterion that never looks at the measured values is a purely geometric design rule, so with a stationary kernel variance sampling just spreads points out and reproduces classical space-filling design. It becomes genuinely adaptive only when hyperparameters are learned, the kernel is non-stationary, or the target is a functional of $f$ such as a boundary. (§6.1)
