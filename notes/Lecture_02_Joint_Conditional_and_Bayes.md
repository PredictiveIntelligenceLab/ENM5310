# ENM 5310 — Lecture 2
## Several Random Variables: Conditioning, Bayes' Rule, and Why Regression Is a Conditional Expectation

**Reading:** Murphy, *PML: An Introduction*, §2.2.3–2.2.5, §2.3, §2.7.4.
**Builds on:** Lecture 1 (probability spaces, random variables, laws, expectation, empirical distribution).


**Learning objectives.** You should be able to (i) factor a joint distribution three ways and say what each factorization assumes; (ii) count parameters and explain why conditional independence is the only thing that makes high-dimensional modeling possible; (iii) prove that the conditional mean minimizes squared error, and identify the irreducible term; (iv) apply Bayes' rule to a diagnostic problem and explain base-rate neglect quantitatively; (v) carry out a full prior → likelihood → posterior → posterior-predictive calculation in a conjugate model.

---

## 1. Joints, marginals, conditionals

Everything interesting involves more than one quantity: input and output, latent state and observation, parameter and data. For two random variables $X, Y$, the **joint distribution** $p(x,y) = p(X=x, Y=y)$ specifies the whole story. Finite case: a table whose entries are non-negative and sum to one.

$$
\begin{array}{c|cc}
p(X,Y) & Y=0 & Y=1\\\hline
X=0 & 0.2 & 0.3\\
X=1 & 0.3 & 0.2
\end{array}
$$

**Marginalization (sum rule / rule of total probability).** To discard a variable, integrate it out:

$$p(x) = \sum_y p(x,y), \qquad p(x) = \int p(x,y)\,dy.$$

The name comes from accountants writing row and column sums in the margin of the table. The operation is trivial to write and often computationally brutal — most of the second half of this course exists because marginalization over latent variables or parameters is intractable.

**Conditioning.** From the definition of conditional probability applied to events $\{X=x\}$, $\{Y=y\}$:

$$p(y \mid x) = \frac{p(x,y)}{p(x)} \qquad \Longleftrightarrow \qquad p(x,y) = p(x)\,p(y\mid x) \quad \textbf{(product rule)}.$$

**Chain rule.** Iterate the product rule over $D$ variables:

$$p(x_{1:D}) = p(x_1)\,p(x_2\mid x_1)\,p(x_3\mid x_1,x_2)\cdots p(x_D \mid x_{1:D-1}).$$

This is an identity — no assumptions. It is also the entire architectural principle behind autoregressive models: a language model, a PixelCNN, a diffusion model's reverse chain, all are the chain rule with each conditional parameterized by a network. Write it on the board and note explicitly: *nothing has been assumed yet; the modeling begins when we choose to truncate or restrict these conditionals.*

---

## 2. Independence, conditional independence, and parameter counting

$$X \perp Y \iff p(x,y) = p(x)p(y); \qquad X \perp Y \mid Z \iff p(x,y\mid z) = p(x\mid z)\,p(y\mid z).$$

For a set $X_1,\dots,X_n$, **mutual independence** requires factorization for *every subset*, not just pairwise. Pairwise independence does not imply mutual independence — construct a counterexample (Practice Set 0.2, Q1); it takes three binary variables.

### Why this is the whole game

Take $X$ with 6 states and $Y$ with 5. A general joint needs $6\times5 - 1 = 29$ free parameters. Assume independence and you need $(6-1) + (5-1) = 9$. Now scale up: $D$ binary variables have a joint table with $2^D - 1$ free parameters. At $D = 100$ that is more parameters than there are atoms in the observable universe, and you have $N \sim 10^4$ data points.

> **The structural claim underlying all of probabilistic modeling:** high-dimensional joint distributions are learnable *only* because they factorize. Every model in this course is a choice of factorization, i.e. a set of conditional independence assumptions, plus a parameterization of the surviving conditionals.

Unconditional independence is almost always false in engineering systems — everything influences everything. Conditional independence is often nearly true, because the influence is *mediated*: $X \to Z \to Y$. Given the intermediate state $Z$, the endpoints decouple. That graph is the simplest **graphical model**; the general machinery (Murphy §3.6) is a language for writing factorizations as graphs, and we will use it informally when we get to latent-variable models.

Concrete instances you already know: the Markov property in a dynamical system ($x_{t+1} \perp x_{t-1} \mid x_t$); the naive Bayes assumption (features conditionally independent given the label); the mean-field family in variational inference (latents mutually independent given data). Each is a factorization choice made for tractability, and each is wrong in a way you must be able to diagnose.

---

## 3. Conditional expectation, the tower property, and total variance

Once we can condition, we can take expectations under conditionals.

$$\mathbb{E}[X \mid Y = y] = \int x\, p(x\mid y)\,dx.$$

This is a number for each fixed $y$; as a function of $Y$ it is itself a **random variable**, written $\mathbb{E}[X\mid Y]$. Getting comfortable with that type change is the main conceptual work of this segment.

**Law of iterated expectations (tower property).**

$$\mathbb{E}[X] = \mathbb{E}_Y\big[\,\mathbb{E}[X \mid Y]\,\big].$$

*Proof (discrete, one line worth doing on the board):*
$$\mathbb{E}_Y\big[\mathbb{E}[X|Y]\big] = \sum_y \Big(\sum_x x\,p(x\mid y)\Big) p(y) = \sum_{x,y} x\, p(x,y) = \mathbb{E}[X].\qquad\blacksquare$$

*Intuition.* Bulbs from factory 1 last 5000 h on average, from factory 2, 4000 h; factory 1 supplies 60%. Then $\mathbb{E}[\text{lifetime}] = 0.6(5000) + 0.4(4000) = 4600$. You average the group averages, weighted by group size. Every stratified estimator, every importance-weighted correction, and every minibatch gradient estimator you will write is an instance of this identity.

**Law of total variance (conditional variance formula).**

$$\boxed{\ \mathbb{V}[X] = \underbrace{\mathbb{E}_Y\big[\mathbb{V}[X\mid Y]\big]}_{\text{within-group}} + \underbrace{\mathbb{V}_Y\big[\mathbb{E}[X\mid Y]\big]}_{\text{between-group}}\ }$$

*Derivation.* Write $\mu_{X|Y} = \mathbb{E}[X|Y]$, $s_{X|Y} = \mathbb{E}[X^2|Y]$, so $\sigma^2_{X|Y} = s_{X|Y} - \mu_{X|Y}^2$. Then
$$\mathbb{V}[X] = \mathbb{E}[X^2] - (\mathbb{E}[X])^2 = \mathbb{E}_Y[s_{X|Y}] - \big(\mathbb{E}_Y[\mu_{X|Y}]\big)^2 = \mathbb{E}_Y[\sigma^2_{X|Y}] + \mathbb{E}_Y[\mu^2_{X|Y}] - \big(\mathbb{E}_Y[\mu_{X|Y}]\big)^2,$$
and the last two terms are exactly $\mathbb{V}_Y[\mu_{X|Y}]$. $\blacksquare$

*Check on a mixture.* For $p(x) = 0.5\,\mathcal{N}(x|0,0.5^2) + 0.5\,\mathcal{N}(x|2,0.5^2)$ with $Y$ the component indicator: within-group $= 0.5(0.25)+0.5(0.25) = 0.25$; between-group $= 0.5(0-1)^2 + 0.5(2-1)^2 = 1$. The spread is dominated by *which* mode you are in, not by the spread within a mode. That is the correct picture of a multimodal predictive distribution, and it is why reporting a single predictive standard deviation for a bimodal posterior is meaningless.

**Why we care.** This decomposition is the mathematical skeleton of:
- **aleatoric vs. epistemic uncertainty** — take $Y = \theta$ (parameters): $\mathbb{V}[y_\star] = \mathbb{E}_\theta[\mathbb{V}[y_\star|\theta]] + \mathbb{V}_\theta[\mathbb{E}[y_\star|\theta]]$, i.e. noise-in-the-data plus disagreement-among-plausible-models. Deep ensembles estimate exactly the second term by brute force.
- the **bias–variance decomposition**, which we will derive in the estimation lecture as a special case.

---

## 4. Regression is a conditional expectation

This is the theorem that tells you what supervised learning is actually trying to compute.

> **Theorem.** Among all measurable functions $f$, the minimizer of the mean squared error $\mathbb{E}\big[(Y - f(X))^2\big]$ is
> $$f^\star(x) = \mathbb{E}[Y \mid X = x].$$

*Proof.* Add and subtract $\mathbb{E}[Y|X]$:
$$\mathbb{E}\big[(Y - f(X))^2\big] = \mathbb{E}\Big[\big(Y - \mathbb{E}[Y|X]\big)^2\Big] + 2\,\mathbb{E}\Big[\big(Y-\mathbb{E}[Y|X]\big)\big(\mathbb{E}[Y|X] - f(X)\big)\Big] + \mathbb{E}\Big[\big(\mathbb{E}[Y|X]-f(X)\big)^2\Big].$$
For the cross term, condition on $X$ and use the tower property: the inner factor $\big(\mathbb{E}[Y|X]-f(X)\big)$ is $X$-measurable and comes out, while $\mathbb{E}\big[Y - \mathbb{E}[Y|X] \,\big|\, X\big] = 0$. So the cross term vanishes and
$$\mathbb{E}\big[(Y-f(X))^2\big] = \underbrace{\mathbb{E}\big[\mathbb{V}[Y|X]\big]}_{\text{irreducible}} + \underbrace{\mathbb{E}\big[(\mathbb{E}[Y|X]-f(X))^2\big]}_{\ge 0,\ =0 \text{ iff } f=f^\star}.\qquad\blacksquare$$

Three consequences you should internalize now, because they reappear in every experimental result you will produce this semester:

1. **Least squares does not target $Y$; it targets $\mathbb{E}[Y|X]$.** A "perfect" regressor still has nonzero test MSE equal to $\mathbb{E}[\mathbb{V}[Y|X]]$. If you do not know that floor for your dataset, you cannot tell a converged model from an underfit one — and you will waste weeks on the difference.
2. **The floor is aleatoric.** More data, bigger models, longer training do not reduce $\mathbb{E}[\mathbb{V}[Y|X]]$. Only better inputs (measuring more of what $Y$ depends on) reduce it. When a training curve plateaus, the first question is whether it plateaued at the noise floor.
3. **The conditional mean is a terrible summary of a multimodal conditional.** If $p(y|x)$ is bimodal, $\mathbb{E}[Y|X=x]$ sits in the valley between the modes and may be a physically impossible output. This is precisely why we build *generative* models of $p(y|x)$ rather than point predictors — and why an $L_2$-trained model produces blurry images and averaged-out turbulent fields. The blur is not a bug in the architecture; it is this theorem doing exactly what it says.

Different loss, different functional: minimizing $\mathbb{E}|Y-f(X)|$ gives the conditional **median**; the pinball loss at level $\tau$ gives the conditional $\tau$-**quantile**. Your loss function is a choice of which summary of $p(y|x)$ you want. Choose it deliberately.

---

## 5. Bayes' rule

Nothing new — a rearrangement of the product rule — but it changes what we can ask. For a hidden quantity $H$ and observed data $Y = y$:

$$\boxed{\ p(H = h \mid Y=y) = \frac{p(H=h)\,p(Y=y\mid H=h)}{p(Y=y)}, \qquad p(Y=y) = \sum_{h'} p(h')\,p(y\mid h')\ }$$

with the standard names:

| Term | Name | Read as |
|---|---|---|
| $p(h)$ | prior | belief before data |
| $p(y\mid h)$ as a function of $h$, with $y$ fixed | **likelihood** | how well $h$ explains what we saw |
| $p(y)$ | marginal likelihood / evidence | normalizer; later, a model-selection score |
| $p(h\mid y)$ | posterior | belief after data |

$$\text{posterior} \propto \text{prior} \times \text{likelihood}.$$

> **The single most common error in this material:** the likelihood $p(y\mid h)$, viewed as a function of $h$, is **not a probability distribution over $h$**. It does not integrate to 1 over $h$, and $p(y|h) = 0.9$ does not mean "$h$ is 90% likely." Maximum likelihood picks the $h$ under which the observed data are most probable — a completely different statement from picking the most probable $h$.

### 5.1 Worked example: diagnostic testing and base rates

Let $H=1$ denote infection, $Y=1$ a positive test. Test characteristics: **sensitivity** $p(Y=1|H=1) = 0.875$ (TPR), **specificity** $p(Y=0|H=0)=0.975$ (TNR), hence FPR $=0.025$, FNR $=0.125$. Prevalence $p(H=1)=0.1$.

$$p(H=1\mid Y=1) = \frac{\text{TPR}\cdot\pi}{\text{TPR}\cdot\pi + \text{FPR}\cdot(1-\pi)} = \frac{0.875 \times 0.1}{0.875\times 0.1 + 0.025 \times 0.9} = 0.795.$$

$$p(H=1 \mid Y=0) = \frac{\text{FNR}\cdot\pi}{\text{FNR}\cdot\pi + \text{TNR}\cdot(1-\pi)} = \frac{0.125\times0.1}{0.125\times0.1 + 0.975\times0.9} = 0.014.$$

Now drop the prevalence to $\pi = 0.01$ and redo it: the posteriors become **26%** and 0.13%. A positive result from a 97.5%-specific test leaves you more likely healthy than sick.

*Make it concrete with natural frequencies* — this is the presentation that actually lands. Out of 100,000 people at $\pi=0.01$: 1,000 infected, of whom $0.875\times1000 = 875$ test positive; 99,000 healthy, of whom $0.025 \times 99{,}000 = 2{,}475$ test positive. Total positives $= 3{,}350$, of which 875 are true. $875/3350 = 0.26$.

**The engineering lesson, stated generally:** when the base rate is small, the false positives from the large healthy population swamp the true positives from the small infected one. Replace "infected" with "defective weld," "anomalous sensor reading," "hit compound in a screening library," "flagged transaction." Any rare-event detector must be evaluated against its base rate, not against its sensitivity and specificity alone. This is also why accuracy is a useless metric on imbalanced data, and why we will use precision-recall and calibration curves instead.

### 5.2 Monty Hall, briefly

Three doors, prize behind one, you pick door 1, the host — who knows where the prize is and never opens it — opens door 3. With $p(H_i)=1/3$ and the host's policy $p(Y{=}3|H_1)=\tfrac12$, $p(Y{=}3|H_2)=1$, $p(Y{=}3|H_3)=0$:

$$p(Y=3) = \tfrac12\cdot\tfrac13 + 1\cdot\tfrac13 + 0 = \tfrac12 \ \Longrightarrow\ p(H_1|Y{=}3)=\tfrac13,\quad p(H_2|Y{=}3)=\tfrac23.$$

Switch. The point for us is not the puzzle; it is that **the likelihood encodes the data-generating protocol**, and changing the protocol changes the inference even though the observed data are identical. If the host opened a door at random and happened to reveal no prize, $p(Y=3|H_1) = p(Y=3|H_2) = 1/2$ and the posterior is $1/2$–$1/2$: switching gains nothing. Same observation, different answer.

Carry this into your own work: *how the data were collected is part of the model.* Nonrandom missingness, stopping rules, selection on the outcome, benchmark curation — each changes $p(y|h)$, and none of them is visible in the data file.

---

## 6. Inverse problems

Forward direction — given the state of the world $h$, predict outcomes $y$ — is what your simulator does. **Inverse probability** goes the other way: infer $h$ from observed $y$. This is where most of scientific ML lives: recover a permeability field from pressure measurements, a source term from downstream concentrations, a constitutive law from load–displacement curves, a 3-D shape from a 2-D image.

These problems are typically **ill-posed**: many $h$ are consistent with the same $y$ (Murphy Fig. 2.8 — a planar line drawing is consistent with infinitely many 3-D structures), and the map can be discontinuous in the data. Bayes' rule gives a principled response:

$$p(h\mid y) \propto \underbrace{p(y\mid h)}_{\text{forward model + noise}} \ \cdot \ \underbrace{p(h)}_{\text{prior}}.$$

The forward model is your physics; the prior is what rules out or downweights implausible worlds. Take negative logs with a Gaussian noise model $y = \mathcal{A}(h) + \varepsilon$, $\varepsilon \sim \mathcal{N}(0,\sigma^2 I)$:

$$-\log p(h\mid y) = \frac{1}{2\sigma^2}\|y - \mathcal{A}(h)\|^2 \;-\; \log p(h) \;+\; \text{const}.$$

Maximizing the posterior (MAP) is therefore **regularized least squares**, with the regularizer being the negative log prior. A Gaussian prior $\mathcal{N}(0,\tau^2 I)$ gives Tikhonov / ridge, $\lambda = \sigma^2/\tau^2$; a Laplace prior gives $L_1$ / LASSO. This is a correspondence to keep permanently in view: **every regularizer you have ever used is a prior, and every regularization weight is a ratio of noise to prior variance.** It also tells you what MAP throws away — the width of the posterior, i.e. all the epistemic uncertainty.

---

## 7. A complete Bayesian calculation: Beta–Bernoulli

**Model.** $\theta \in [0,1]$ is an unknown success probability (fraction of parts passing inspection, say). Data: $N$ independent trials $y_n \in \{0,1\}$ with $s = \sum_n y_n$ successes.

**Likelihood.** $p(\mathcal{D}\mid\theta) = \prod_n \theta^{y_n}(1-\theta)^{1-y_n} = \theta^{s}(1-\theta)^{N-s}$.

**Prior.** The **Beta distribution** on $[0,1]$,
$$\mathrm{Beta}(\theta\mid a,b) = \frac{1}{B(a,b)}\theta^{a-1}(1-\theta)^{b-1}, \qquad B(a,b) = \frac{\Gamma(a)\Gamma(b)}{\Gamma(a+b)},$$
with $\mathbb{E}[\theta] = \frac{a}{a+b}$, $\mathrm{mode} = \frac{a-1}{a+b-2}$ (for $a,b>1$), $\mathbb{V}[\theta] = \frac{ab}{(a+b)^2(a+b+1)}$. $a=b=1$ is uniform; $a,b<1$ puts spikes at the endpoints; $a,b>1$ is unimodal.

**Posterior.** Multiply and collect exponents:
$$p(\theta\mid\mathcal{D}) \propto \theta^{a+s-1}(1-\theta)^{b+N-s-1} \ \Longrightarrow\ \theta\mid\mathcal{D} \sim \mathrm{Beta}(a+s,\; b+N-s).$$

Prior and posterior are in the same family: the Beta is **conjugate** to the Bernoulli likelihood. Four things to extract from this, all of which generalize:

1. **Priors as pseudo-counts.** $a$ and $b$ act as $a-1$ prior successes and $b-1$ prior failures. Prior strength is measured in units of data.
2. **Shrinkage.** With $\hat\theta_{\mathrm{MLE}} = s/N$ and prior mean $m = a/(a+b)$,
 $$\mathbb{E}[\theta\mid\mathcal{D}] = \frac{a+s}{a+b+N} = \lambda\, m + (1-\lambda)\,\hat\theta_{\mathrm{MLE}}, \qquad \lambda = \frac{a+b}{a+b+N}.$$
 A convex combination of prior and data, with the data winning as $N$ grows. Nearly every Bayesian estimator you will meet has this form.
1. **Sequential updating.** Today's posterior is tomorrow's prior: updating on $\mathcal{D}_1$ then $\mathcal{D}_2$ gives the same answer as updating on $\mathcal{D}_1 \cup \mathcal{D}_2$. Exact online learning, for free, in this family only.
2. **Posterior predictive, and why MAP is not enough.** For the next trial,
 $$p(\tilde y = 1\mid \mathcal{D}) = \int_0^1 \theta\, p(\theta\mid\mathcal{D})\,d\theta = \mathbb{E}[\theta\mid\mathcal{D}] = \frac{a+s}{a+b+N}.$$
 With $a=b=1$ this is Laplace's rule of succession $(s+1)/(N+2)$. Note what happened: after 5 trials and 5 successes, the MLE says $\Pr(\text{next success}) = 1$ — a claim that no failure is possible, ever. The predictive says $6/7$. **Averaging over the posterior instead of plugging in a point estimate is what prevents overconfidence on small data**, and it is the reason we will fight so hard for approximate posteriors later in the semester.

Everything past this point in the course is the same four steps (potentially with an intractable integral in step 4, which we may have to approximate numerically).

---

## 8. Code checkpoint

1. **Total variance.** Sample $10^6$ points from the two-component Gaussian mixture of §3. Estimate $\mathbb{V}[X]$ directly, and separately estimate the within- and between-group terms. Verify they sum. Then plot the histogram and mark $\mathbb{E}[X]$ — observe that the mean lands where there is almost no probability mass.
2. **Conditional mean, empirically.** Generate $y = \sin(2\pi x) + \varepsilon$, $\varepsilon\sim\mathcal{N}(0,0.2^2)$, $x\sim\mathrm{Unif}(0,1)$, $N=10^4$. Bin $x$ and compute per-bin means; overlay $\sin(2\pi x)$. Compute the empirical $\mathbb{E}[\mathbb{V}[Y|X]]$ and confirm it recovers $0.04$. **This number is the test-MSE floor for any model of this dataset.**
3. **Bimodal target.** Repeat with $y = \pm\sqrt{x} + \varepsilon$ (sign chosen by a fair coin). Fit any least-squares regressor you like. Explain the result in one sentence using §4.
4. **Base rates.** Plot $p(H=1|Y=1)$ against prevalence $\pi \in [10^{-4}, 0.5]$ on a log-$x$ axis for the test in §5.1. Mark the prevalence at which the posterior crosses 0.5.
5. **Beta–Bernoulli.** Implement the update, and animate the posterior for $N = 1, 2, 5, 10, 100$ with $\theta_{\text{true}} = 0.7$ under priors $\mathrm{Beta}(1,1)$, $\mathrm{Beta}(5,5)$, and $\mathrm{Beta}(1,20)$. Report how many observations it takes for the three posteriors to become visually indistinguishable. That number is the honest answer to "how much does my prior matter?"

---

## 9. What you must know cold (quiz-eligible)

- Sum rule, product rule, chain rule; the fact that the chain rule assumes nothing.
- Definitions of $\perp$ and $\perp\mid Z$; that pairwise $\ne$ mutual.
- Parameter count of a general joint over $D$ binary variables, and why factorization is required.
- Tower property and law of total variance, including the one-line proofs.
- $\arg\min_f \mathbb{E}[(Y-f(X))^2] = \mathbb{E}[Y|X]$, with the irreducible term identified.
- Bayes' rule with all four terms named; likelihood is not a distribution over $h$.
- The base-rate computation, and the natural-frequency version of it.
- Beta–Bernoulli conjugate update; posterior mean as shrinkage; posterior predictive vs. plug-in.

---

## 10. Practice Set 0.2 (ungraded; quiz-eligible)

1. Give three binary random variables that are pairwise independent but not mutually independent. *Murphy Ex. 2.2.*
2. Prove: $X\perp Y\mid Z$ iff there exist $g,h$ with $p(x,y\mid z) = g(x,z)h(y,z)$ for all $z$ with $p(z)>0$. *Murphy Ex. 2.3.*
3. You test positive on a test that is "99% accurate" (both sensitivity and specificity $=0.99$) for a disease affecting 1 in 10,000. Compute $p(\text{disease}\mid +)$. *Murphy Ex. 2.9.*
4. **The prosecutor's fallacy.** Blood at a crime scene matches a type present in 1% of the population, and the defendant matches. The prosecutor argues there is a 99% chance of guilt. State precisely which conditional probability has been confused with which. Then diagnose the defense's counter-argument ("8,000 people in the city match, so this is 1-in-8,000 evidence and therefore irrelevant"), which is also wrong. *Murphy Ex. 2.10.*
5. Your neighbor has two children. (a) You ask whether he has any boys; he says yes. What is the probability the other is a girl? (b) Instead, you see one child run past and it is a boy. Now what is the probability the other is a girl? Explain why the answers differ, in the language of §5.2. *Murphy Ex. 2.11.*
6. Derive the mean, mode, and variance of $\mathrm{Beta}(a,b)$. *Murphy Ex. 2.8.*
7. Show that minimizing $\mathbb{E}|Y-f(X)|$ over all $f$ yields the conditional median. Then state what this implies about the robustness of $L_1$-trained regressors to heavy-tailed noise.
8. A colleague's regression model reaches test MSE $0.041$ and stops improving with more data, more parameters, and more training. Describe the experiment you would run to determine whether this is the noise floor or a modeling failure. Be specific about what you would compute.

---


