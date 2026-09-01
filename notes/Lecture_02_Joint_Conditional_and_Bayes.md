# ENM 5310 — Lecture 2
## Several Random Variables: Dependence, Conditioning, and Bayes' Rule

**Reading:** Murphy, *PML: An Introduction*, §2.2.3–2.2.5, §2.3; §3.1–3.2 for the multivariate Gaussian.
**Builds on:** Lecture 1 (probability spaces, random variables, laws, expectation, empirical distribution).


**Learning objectives.** You should be able to (i) factor a joint distribution and state what each factorization assumes; (ii) explain why conditional independence is what makes high-dimensional modeling possible; (iii) compute covariances and say precisely what correlation does and does not detect; (iv) prove that the conditional mean minimizes squared error and identify the irreducible term; (v) apply Bayes' rule to a diagnostic problem and explain base-rate neglect quantitatively; (vi) condition a joint Gaussian and recognize the formula as the engine behind GP regression and the Kalman filter.

---

## 1. Joints, marginals, conditionals — and Bayes

Everything interesting involves more than one quantity: input and output, latent state and observation, parameter and data. For two random variables $X, Y$, the **joint distribution** $p(x,y)$ specifies the whole story. In the finite case it is a table with non-negative entries summing to one.

$$
\begin{array}{c|cc}
p(X,Y) & Y=0 & Y=1\\\hline
X=0 & 0.2 & 0.3\\
X=1 & 0.3 & 0.2
\end{array}
$$

From the joint, three operations generate everything else.

**Marginalization (sum rule).** To discard a variable, integrate it out:

$$p(x) = \sum_y p(x,y), \qquad p(x) = \int p(x,y)\,dy.$$

The name comes from accountants writing row and column sums in the margin of the table. Trivial to write, often computationally brutal — much of the second half of this course exists because marginalizing over latent variables or parameters is intractable.

**Conditioning (product rule).**

$$p(y \mid x) = \frac{p(x,y)}{p(x)} \qquad \Longleftrightarrow \qquad p(x,y) = p(x)\,p(y\mid x).$$

Geometrically: pick out one row of the table and renormalize it so it sums to one. Conditioning is *restrict and rescale*, nothing more.

**Chain rule.** Iterating the product rule over $D$ variables:

$$p(x_{1:D}) = p(x_1)\,p(x_2\mid x_1)\,p(x_3\mid x_1,x_2)\cdots p(x_D \mid x_{1:D-1}).$$

This is an identity — no assumptions. It is also the architectural principle behind every autoregressive model: a language model, a PixelCNN, the reverse chain of a diffusion model. Each is the chain rule with the conditionals parameterized by a network. *Nothing has been assumed yet; modeling begins when we choose to truncate or restrict these conditionals.*

**Bayes' rule.** The product rule factors the same joint two ways, $p(x,y) = p(x)\,p(y|x) = p(y)\,p(x|y)$. Equate and divide:

$$\boxed{\ p(x \mid y) = \frac{p(y \mid x)\,p(x)}{p(y)}, \qquad p(y) = \sum_{x'} p(y\mid x')\,p(x')\ }$$

That is the entire derivation — one line, with no content beyond the product rule. What Bayes' rule buys is a **direction reversal**: it converts knowledge of $p(y\mid x)$, which is what a physical model or a measurement device gives you, into $p(x\mid y)$, which is what you want. We use it as an identity for today's mechanics and return in §6 to what it means when $x$ is an unknown parameter rather than another observable.

---

## 2. Independence, conditional independence, and parameter counting

$$X \perp Y \iff p(x,y) = p(x)p(y); \qquad X \perp Y \mid Z \iff p(x,y\mid z) = p(x\mid z)\,p(y\mid z).$$

For a set $X_1,\dots,X_n$, **mutual independence** requires factorization over *every subset*, not just pairs. Pairwise independence does not imply mutual independence (Practice Set 0.2, Q1; three binary variables suffice).

### Why factorization is the whole game

Take $X$ with 6 states and $Y$ with 5. A general joint needs $6\times5-1 = 29$ free parameters. Assume independence and you need $(6-1)+(5-1) = 9$. Now scale: $D$ binary variables give a joint table with $2^D-1$ free parameters. At $D=100$ that exceeds the number of atoms in the observable universe, and you have $N \sim 10^4$ data points.

> High-dimensional joint distributions are learnable **only** because they factorize. Every model in this course is a choice of factorization — a set of conditional independence assumptions — plus a parameterization of the surviving conditionals.

Unconditional independence is almost always false in engineering systems: everything influences everything. Conditional independence is often nearly true, because influence is *mediated*: $X \to Z \to Y$. Given the intermediate state $Z$, the endpoints decouple. That graph is the simplest **graphical model**.

Instances you already know: the Markov property in a dynamical system ($x_{t+1}\perp x_{t-1}\mid x_t$); naive Bayes (features conditionally independent given the label); the mean-field family in variational inference (latents independent given data). Each is a factorization chosen for tractability, and each is wrong in a way you must be able to diagnose.

---

## 3. Covariance and correlation: measuring linear dependence

Independence is a yes/no property of the entire joint. We also want a *number* summarizing how two variables move together.

> **Definition.** $\displaystyle \mathrm{Cov}[X,Y] \triangleq \mathbb{E}\big[(X-\mu_X)(Y-\mu_Y)\big] = \mathbb{E}[XY] - \mu_X\mu_Y.$

**Read the definition rather than memorizing it.** The product $(X-\mu_X)(Y-\mu_Y)$ is positive whenever both variables sit on the same side of their means, negative when they sit on opposite sides. Covariance is the average of that product: positive if the pair tends to deviate together, negative if oppositely, zero if the two tendencies cancel. On the board: draw the scatter cloud, put crosshairs at $(\mu_X,\mu_Y)$, label the quadrants $+,-,-,+$. Covariance is a mass-weighted vote among quadrants.

Properties to have at hand:

$$\mathrm{Cov}[X,X] = \mathbb{V}[X], \qquad \mathrm{Cov}[aX+b,\ cY+d] = ac\,\mathrm{Cov}[X,Y],$$
$$\mathbb{V}[X+Y] = \mathbb{V}[X] + \mathbb{V}[Y] + 2\,\mathrm{Cov}[X,Y].$$

The last line explains why "variances add" required independence in Lecture 1: the cross term vanishes only when the covariance does.

**Covariance matrix.** For a random vector $x\in\mathbb{R}^d$ with mean $\mu$,

$$\Sigma \triangleq \mathrm{Cov}[x] = \mathbb{E}\big[(x-\mu)(x-\mu)^\top\big], \qquad \Sigma_{ij} = \mathrm{Cov}[x_i,x_j].$$

$\Sigma$ is symmetric and positive semi-definite, and the proof is one line that also hands you the most useful identity in the section: for any fixed $a$,

$$\mathbb{V}[a^\top x] = a^\top \Sigma a \;\ge\; 0.$$

Every linear combination has non-negative variance, hence $\Sigma\succeq0$. More generally, for an affine map,

$$\mathbb{E}[Ax+b] = A\mu+b, \qquad \mathrm{Cov}[Ax+b] = A\Sigma A^\top.$$

Second moments transform by **sandwiching**. This one formula sits behind error propagation in measurement chains, the covariance update in a Kalman filter, and the whitening transform $\tilde x = \Lambda^{-1/2}U^\top(x-\mu)$ built from the eigendecomposition $\Sigma = U\Lambda U^\top$, which produces uncorrelated unit-variance coordinates. PCA is that eigendecomposition read as a modeling tool.

**Correlation.** Covariance carries units of $[X][Y]$, so its magnitude means nothing on its own. Normalize:

$$\rho_{XY} \triangleq \frac{\mathrm{Cov}[X,Y]}{\sigma_X\sigma_Y} \in [-1,1],$$

with the bound from Cauchy–Schwarz. The extremes are the informative part: $|\rho| = 1$ **iff** $Y = aX+b$ exactly (with probability one). So correlation measures *proximity to an exact linear relationship*, and that is all it measures.

### The two things people get wrong here

**(i) Independent $\Rightarrow$ uncorrelated, but uncorrelated $\not\Rightarrow$ independent.** Let $X$ be symmetric about zero and $Y = X^2$. Then

$$\mathrm{Cov}[X,Y] = \mathbb{E}[X^3] - \mathbb{E}[X]\,\mathbb{E}[X^2] = 0,$$

yet $Y$ is a deterministic function of $X$. Zero correlation, total dependence. Correlation is blind to any dependence that is not linear — symmetric, periodic, threshold, or heteroscedastic structure all pass undetected. This is the quantitative form of Lecture 1's Anscombe point: all four Anscombe datasets have $\rho=0.816$, including the one whose true relationship is a parabola and the one that is a vertical stack plus a single lever point.

The important exception: **if $(X,Y)$ are jointly Gaussian, uncorrelated does imply independent.** For Gaussians the covariance is the entire dependence structure. That is a property of one family, not a general fact, and assuming it elsewhere is a common and expensive error.

**(ii) Correlation is not causation, and it is also not prediction.** $\rho = 0.3$ sounds weak but can carry real predictive value; $\rho=0.95$ computed on a dataset with one leverage point can vanish when that point is removed. Report the scatter plot next to the number, always.

**From data.** The plug-in principle (Lecture 1) applied to $\hat p_N$ gives the empirical covariance

$$\hat\Sigma = \frac{1}{N-1}\sum_{n=1}^N (x_n-\bar x)(x_n-\bar x)^\top,$$

with $N-1$ for the unbiased version and $N$ for the plug-in/MLE version. Know which one your library uses. In $d$ dimensions with $N \lesssim d$ this matrix is singular, and every method that inverts it — Gaussian discriminant analysis, Mahalanobis distances, GP regression with a fitted covariance — fails in a way that regularization exists to fix.

---

## 4. Conditional expectation: your best guess given partial information

Once we can condition, we can take expectations under conditionals:

$$\mathbb{E}[X \mid Y=y] = \int x\,p(x\mid y)\,dx.$$

**The intuition to hold onto:** $\mathbb{E}[X\mid Y=y]$ is your best single guess about $X$ if someone tells you $Y=y$ and nothing else. Before the message arrives you do not know which $y$ you will hear, so the guess-you-will-make is itself uncertain. That is why $\mathbb{E}[X\mid Y]$, as a function of the random $Y$, is **a random variable**, whereas $\mathbb{E}[X\mid Y=y]$ is a number. Nearly all confusion in this material traces back to blurring those two.

**Law of iterated expectations (tower property).**

$$\mathbb{E}[X] = \mathbb{E}_Y\big[\,\mathbb{E}[X\mid Y]\,\big].$$

*In words:* if you average, over all the messages you might receive, the guess you would make after receiving each one, you recover the guess you would make with no message at all. Information changes your estimate, but it cannot change it *on average, in advance* — otherwise you should have moved your estimate already. (This is the seed of the martingale property, and it is why unbiased estimators stay unbiased under stratification.)

*Proof (discrete, one line for the board):*
$$\mathbb{E}_Y\big[\mathbb{E}[X|Y]\big] = \sum_y\Big(\sum_x x\,p(x\mid y)\Big)p(y) = \sum_{x,y} x\,p(x,y) = \mathbb{E}[X].\qquad\blacksquare$$

*Concrete version.* Bulbs from factory 1 last 5000 h on average, factory 2, 4000 h; factory 1 supplies 60%. Then $\mathbb{E}[\text{lifetime}] = 0.6(5000)+0.4(4000) = 4600$: average the group averages, weighted by group size. Every stratified estimator, importance-weighted correction, and minibatch gradient estimate is this identity.

Here's a drop-in replacement for the law-of-total-variance block in §4. The main change is proving it by splitting the deviation into two orthogonal pieces, which makes the result look inevitable rather than like an algebraic accident — and which sets up §5 so that the regression theorem costs almost nothing.

**Law of total variance.**

$$\mathbb{V}[X] \;=\; \underbrace{\mathbb{E}_Y\big[\mathbb{V}[X\mid Y]\big]}_{\text{uncertainty that survives learning } Y} \;+\; \underbrace{\mathbb{V}_Y\big[\mathbb{E}[X\mid Y]\big]}_{\text{how much learning } Y \text{ moves your guess}}$$

**Where it comes from.** Split the deviation of $X$ from its mean into two pieces by inserting the conditional mean:

$$X - \mathbb{E}[X] \;=\; \underbrace{\big(X - \mathbb{E}[X\mid Y]\big)}_{\text{how far } X \text{ falls from the guess}} \;+\; \underbrace{\big(\mathbb{E}[X\mid Y] - \mathbb{E}[X]\big)}_{\text{how far the guess moved}}$$

These two pieces are **uncorrelated**. The second is a function of $Y$ alone, and the first has conditional mean zero given $Y$, so conditioning on $Y$ and applying the tower property kills the cross term. Uncorrelated pieces have additive variances (§3), so take variances of both sides and the theorem falls out — it is the Pythagorean theorem, with the two legs being *prediction error* and *guess movement*.

**Two ways to read it.**

*As two stages of randomness.* Think of how a draw of $X$ is actually generated: first nature picks a group $Y$, then it draws $X$ from within that group. Two stages, two contributions to the spread. If you only counted the within-group scatter you would be pretending every group sits at the same place; if you only counted the between-group scatter you would be pretending each group is a single point.

*As the value of information.* The tower property said learning $Y$ cannot change your estimate *on average*. This says something sharper: **the amount your estimate moves around as $Y$ varies is exactly the amount of uncertainty $Y$ removes.** An observation is informative precisely to the extent that different outcomes of it would send you to different conclusions. If every possible value of $Y$ would leave your best guess where it was, $\mathbb{V}_Y[\mathbb{E}[X|Y]] = 0$ and the observation is worthless. This is the seed of expected information gain, and we will meet it again as an acquisition function.

**Numbers.** Yield strength of a bracket sourced equally from three suppliers, each with within-supplier standard deviation 10 MPa and means 300, 320, 340 MPa.

- Within: $\mathbb{E}_Y[\mathbb{V}[X|Y]] = 100$.
- Between: grand mean 320, so $\mathbb{V}_Y[\mathbb{E}[X|Y]] = \tfrac13\big[(-20)^2 + 0^2 + 20^2\big] = 266.7$.
- Total: $366.7$, i.e. $\sigma = 19.1$ MPa across the whole supply, versus $10$ MPa once you know the supplier.

Knowing which supplier shipped the part removes $266.7/366.7 = 73\%$ of the variance. That fraction is $R^2$, and §7 will show it equals $\rho^2$ in the Gaussian case.

The two extremes are worth stating out loud. If the suppliers had identical means, the between term vanishes and the supplier label is uninformative. If each supplier were perfectly consistent, the within term vanishes and the label tells you everything. Real cases sit between, and the ratio is what "informative" means quantitatively.

**The error this is designed to prevent.** Averaging the conditional variances — reporting $10$ MPa as the spread of the parts you receive — understates the true spread by nearly a factor of two, because it forgets that the groups sit in different places. The same mistake in a predictive model is reporting the average within-model uncertainty of an ensemble while ignoring how much the ensemble members disagree with each other. That disagreement *is* the second term.

**Why this decomposition earns its lecture time.** Set $Y = \theta$, the parameters:

$$\mathbb{V}[y_\star] = \underbrace{\mathbb{E}_\theta\big[\mathbb{V}[y_\star\mid\theta]\big]}_{\text{aleatoric}} + \underbrace{\mathbb{V}_\theta\big[\mathbb{E}[y_\star\mid\theta]\big]}_{\text{epistemic}}$$

Noise that persists even if you knew the model, plus disagreement among the models the data cannot rule out. This is Lecture 1's split with an equation attached, and a deep ensemble estimates the second term by brute force. The bias–variance decomposition is the same algebra with $Y$ = the training set.


---

## 5. Regression is a conditional expectation

This is the theorem that says what supervised learning is actually computing.

> **Theorem.** Among all measurable functions $f$, the minimizer of $\mathbb{E}\big[(Y-f(X))^2\big]$ is
> $$f^\star(x) = \mathbb{E}[Y\mid X=x].$$

**The intuitive argument.** Use the tower property to take the expectation over $X$ on the outside:

$$\mathbb{E}\big[(Y-f(X))^2\big] = \mathbb{E}_X\Big[\ \underbrace{\mathbb{E}\big[(Y-f(x))^2\ \big|\ X=x\big]}_{\text{an inner problem at fixed } x}\ \Big].$$

Look at the inner expectation. For a *fixed* $x$, the value $f(x)$ is just a constant $c$, so the question becomes: which constant best predicts the random variable $Y\mid X=x$ under squared error? You can already answer that — the best constant predictor of any random variable is its mean, because

$$\mathbb{E}\big[(Y-c)^2\mid x\big] = \mathbb{V}[Y\mid x] + \big(\mathbb{E}[Y\mid x]-c\big)^2,$$

minimized at $c=\mathbb{E}[Y\mid x]$ with residual $\mathbb{V}[Y\mid x]$. The outer average is a sum of non-negative terms, and $f$ is free to pick a different constant at each $x$, so **the problem decouples: solve it separately in every vertical slice.** Picture the scatter cloud; above each $x$ sits a slice with distribution $p(y\mid x)$; the optimal curve threads the mean of every slice. Averaging back over $x$:

$$\mathbb{E}\big[(Y-f(X))^2\big] = \underbrace{\mathbb{E}\big[\mathbb{V}[Y|X]\big]}_{\text{irreducible}} + \underbrace{\mathbb{E}\big[(\mathbb{E}[Y|X]-f(X))^2\big]}_{\ge0,\ =0\ \text{iff}\ f=f^\star}.\qquad\blacksquare$$

**The geometric version, in one sentence.** The identity $\mathbb{E}\big[(Y-\mathbb{E}[Y|X])\,g(X)\big]=0$ for every $g$ says the residual is orthogonal to every function of $X$. So $\mathbb{E}[Y|X]$ is the **orthogonal projection** of $Y$ onto the space of functions of $X$, and the decomposition above is the Pythagorean theorem in that space. If projections are intuitive to you, that is the fastest way to hold the whole result.

**Three consequences you will meet in your own experiments:**

1. **Least squares does not target $Y$; it targets $\mathbb{E}[Y|X]$.** A perfect regressor still has test MSE equal to $\mathbb{E}[\mathbb{V}[Y|X]]$. If you do not know that floor for your dataset, you cannot distinguish a converged model from an underfit one, and you will lose weeks to the difference.
2. **The floor is aleatoric.** More data, bigger models, and longer training do not reduce it. Only better inputs — measuring more of what $Y$ actually depends on — do. When a training curve plateaus, the first question is whether it plateaued at the noise floor.
3. **The conditional mean is a poor summary of a multimodal conditional.** If $p(y|x)$ is bimodal, $\mathbb{E}[Y|X=x]$ lands in the valley between the modes and may be a physically impossible output. This is why $L_2$-trained models produce blurry images and over-smoothed turbulent fields, and why we build generative models of $p(y|x)$ instead of point predictors. The blur is not an architecture bug; it is this theorem working correctly.

**Your loss chooses which summary you get.** Minimizing $\mathbb{E}|Y-f(X)|$ returns the conditional **median**; the pinball loss at level $\tau$ returns the conditional $\tau$-**quantile**; a full negative log-likelihood (Lecture 3) returns the whole conditional distribution. Choose deliberately.

---

## 6. Bayes as inference

§1 gave Bayes' rule as an identity relating two observables. It becomes *inference* when the conditioned variable is an unknown we want to learn — a parameter, a hypothesis, a hidden state:

$$p(h\mid y) = \frac{p(h)\,p(y\mid h)}{p(y)}, \qquad p(y) = \sum_{h'} p(h')\,p(y\mid h').$$

| Term | Name | Read as |
|---|---|---|
| $p(h)$ | prior | belief before data |
| $p(y\mid h)$ as a function of $h$, $y$ fixed | **likelihood** | how well $h$ explains what we saw |
| $p(y)$ | marginal likelihood / evidence | normalizer; later, a model-selection score |
| $p(h\mid y)$ | posterior | belief after data |

$$\text{posterior}\ \propto\ \text{prior}\times\text{likelihood}.$$

> **The most common error in this material:** the likelihood $p(y\mid h)$, read as a function of $h$, is **not a distribution over $h$**. It does not integrate to one over $h$, and $p(y|h)=0.9$ does not mean "$h$ is 90% likely." Maximum likelihood selects the $h$ under which the observed data are most probable — a different statement from selecting the most probable $h$. The two coincide only under a flat prior.

### 6.1 Diagnostic testing and base rates

Let $H=1$ denote infection and $Y=1$ a positive test. **Sensitivity** $p(Y{=}1|H{=}1) = 0.875$, **specificity** $p(Y{=}0|H{=}0)=0.975$ (so FPR $=0.025$, FNR $=0.125$), prevalence $\pi = 0.1$:

$$p(H{=}1\mid Y{=}1) = \frac{\text{TPR}\cdot\pi}{\text{TPR}\cdot\pi+\text{FPR}\cdot(1-\pi)} = \frac{0.875(0.1)}{0.875(0.1)+0.025(0.9)} = 0.795,$$

$$p(H{=}1\mid Y{=}0) = \frac{\text{FNR}\cdot\pi}{\text{FNR}\cdot\pi+\text{TNR}\cdot(1-\pi)} = \frac{0.125(0.1)}{0.125(0.1)+0.975(0.9)} = 0.014.$$

Now set $\pi = 0.01$ and redo it: the posteriors become **26%** and 0.13%. A positive result from a 97.5%-specific test leaves you more likely healthy than sick.

*Natural frequencies make it obvious.* Out of 100,000 people at $\pi=0.01$: 1,000 infected, of whom $0.875(1000) = 875$ test positive; 99,000 healthy, of whom $0.025(99{,}000) = 2{,}475$ test positive. Total positives $3{,}350$, of which 875 are true: $875/3350 = 0.26$.

**The general statement:** when the base rate is small, false positives drawn from the large negative population swamp true positives from the small positive one. Replace "infected" with defective weld, anomalous sensor reading, hit compound in a screening library, flagged transaction. Any rare-event detector must be judged against its base rate, not against sensitivity and specificity alone — which is also why accuracy is useless on imbalanced data and why we will use precision–recall and calibration curves instead.

### 6.2 Monty Hall

You pick door 1; the host — who knows the prize location and never reveals it — opens door 3. With $p(H_i)=1/3$ and host policy $p(Y{=}3|H_1)=\tfrac12$, $p(Y{=}3|H_2)=1$, $p(Y{=}3|H_3)=0$, we get $p(Y{=}3)=\tfrac12$, hence $p(H_1|Y{=}3)=\tfrac13$ and $p(H_2|Y{=}3)=\tfrac23$. Switch.

The lesson is not the puzzle. It is that **the likelihood encodes the data-collection protocol.** Had the host opened a door at random and happened to miss the prize, $p(Y{=}3|H_1)=p(Y{=}3|H_2)=\tfrac12$ and the posterior would be $\tfrac12$–$\tfrac12$: switching gains nothing. Identical observation, different answer. Carry this into your own work — nonrandom missingness, stopping rules, selection on the outcome, and benchmark curation all change $p(y|h)$, and none of them is visible in the data file.

---

## 7. Worked example: conditioning a joint Gaussian

We close with the one conditional distribution you will use more than any other. Let $x\in\mathbb{R}^d$ be Gaussian, $p(x) = \mathcal{N}(x\mid\mu,\Sigma)$, and split it into the part you observe and the part you want:

$$x = \begin{pmatrix}x_1\\x_2\end{pmatrix}, \qquad \mu = \begin{pmatrix}\mu_1\\\mu_2\end{pmatrix}, \qquad \Sigma = \begin{pmatrix}\Sigma_{11}&\Sigma_{12}\\ \Sigma_{21}&\Sigma_{22}\end{pmatrix}.$$

Both the marginal and the conditional are again Gaussian, and (Murphy §3.2.3; we derive it by completing the square in the linear-Gaussian lecture):

$$p(x_1) = \mathcal{N}(x_1\mid\mu_1,\Sigma_{11}), \qquad
\boxed{\ p(x_1\mid x_2) = \mathcal{N}\big(x_1\ \big|\ \mu_1 + \Sigma_{12}\Sigma_{22}^{-1}(x_2-\mu_2),\ \ \Sigma_{11}-\Sigma_{12}\Sigma_{22}^{-1}\Sigma_{21}\big).\ }$$

Take it apart; every piece is something we built today.

- **The conditional mean is linear in the observation.** By §5, $\mathbb{E}[x_1|x_2]$ is the MSE-optimal predictor of $x_1$ from $x_2$ — so for jointly Gaussian variables *the optimal regressor is exactly linear*, with coefficients $\Sigma_{12}\Sigma_{22}^{-1}$. Linear regression is not an approximation here; it is the truth. This is the precise sense in which "Gaussian" and "linear" are one assumption seen from two sides, and it is why §3 was worth the time.
- **The conditional covariance does not depend on the observed value $x_2$.** You can compute how much uncertainty an experiment will remove *before running it*. That is what makes Bayesian experimental design and Bayesian optimization tractable.
- **Uncertainty can only shrink.** $\Sigma_{11}-\Sigma_{12}\Sigma_{22}^{-1}\Sigma_{21}\preceq\Sigma_{11}$ — a Schur complement — and the shrinkage is governed entirely by the cross-covariance $\Sigma_{12}$. If $\Sigma_{12}=0$, uncorrelated and therefore (Gaussian!) independent, the conditional equals the marginal and the observation taught you nothing.

**Scalar case, worth writing out.** With $\rho = \mathrm{corr}(x_1,x_2)$:

$$\mathbb{E}[x_1\mid x_2] = \mu_1 + \rho\,\frac{\sigma_1}{\sigma_2}(x_2-\mu_2), \qquad \mathbb{V}[x_1\mid x_2] = \sigma_1^2\,(1-\rho^2).$$

The variance reduction is $\rho^2$ — the familiar "fraction of variance explained," now derived rather than asserted. At $\rho = 0.5$ you have removed only 25% of the variance.

**Where this reappears.** Gaussian process regression *is* this formula, with $\Sigma$ supplied by a kernel evaluated at training and test inputs: form the joint over function values, condition on the observed ones, read off the predictive mean and variance. The Kalman filter *is* this formula applied recursively, alternating conditioning on a new measurement with propagation through linear dynamics via $\mathrm{Cov}[Ax] = A\Sigma A^\top$ from §3. Bayesian linear regression is the same computation with parameters in place of function values. When we reach those topics the derivation will already be done; only the bookkeeping will be new.

---

## 8. Code checkpoint

1. **Correlation is blind to nonlinearity.** Sample $X\sim\mathcal{N}(0,1)$, set $Y=X^2$. Report $\hat\rho$ with an error bar over 200 replicates, and show the scatter plot. Confirm $\hat\rho\to0$ as $N$ grows while the dependence is total.
2. **Affine transformation of covariance.** Sample $x\sim\mathcal{N}(0,\Sigma)$ in $d=3$ for a $\Sigma$ of your choice. Pick a random $A\in\mathbb{R}^{2\times3}$, form $y=Ax$, and check $\hat\Sigma_y\approx A\Sigma A^\top$. Then whiten: compute $\Sigma = U\Lambda U^\top$, form $\tilde x = \Lambda^{-1/2}U^\top x$, and confirm $\hat\Sigma_{\tilde x}\approx I$.
3. **Singular covariance.** Generate $N = 50$ points in $d = 100$ dimensions. Compute $\hat\Sigma$, report its rank and smallest eigenvalue, and try to invert it. Then add $\lambda I$ and watch the condition number as $\lambda$ varies. This is the failure that regularization exists to fix.
4. **Total variance.** Sample $10^6$ points from the mixture in §4. Estimate $\mathbb{V}[X]$ directly and the two decomposition terms separately; verify they sum. Plot the histogram and mark $\mathbb{E}[X]$ — note that the mean sits where there is almost no mass.
5. **Conditional mean, empirically.** Generate $y = \sin(2\pi x)+\varepsilon$, $\varepsilon\sim\mathcal{N}(0,0.2^2)$, $x\sim\mathrm{Unif}(0,1)$, $N=10^4$. Bin $x$, compute per-bin means, overlay $\sin(2\pi x)$. Compute the empirical $\mathbb{E}[\mathbb{V}[Y|X]]$ and confirm it recovers $0.04$. **That number is the test-MSE floor for any model of this dataset.**
6. **Bimodal target.** Repeat with $y = \pm\sqrt{x}+\varepsilon$, sign chosen by a fair coin. Fit any least-squares regressor. Explain the result in one sentence using §5.
7. **Base rates.** Plot $p(H{=}1|Y{=}1)$ against prevalence $\pi\in[10^{-4},0.5]$ on a log-$x$ axis for the test in §6.1. Mark where the posterior crosses $0.5$.
8. **Gaussian conditioning.** Sample $10^5$ points from a 2-D Gaussian with $\rho=0.8$. Inside thin slices around several values of $c$, estimate $\mathbb{E}[x_1|x_2\approx c]$ and $\mathbb{V}[x_1|x_2\approx c]$ empirically and compare against the scalar formulas in §7. Confirm the empirical conditional variance is flat in $c$ — the property that makes the Gaussian special.
9. **A two-line Gaussian process.** Build a $5\times5$ covariance matrix from $k(x,x') = \exp\big(-\tfrac12(x-x')^2/\ell^2\big)$ at five input locations. Condition on the function values at three of them using the block formula and report the predictive mean and variance at the other two. You have now implemented GP regression; the rest of that lecture is choosing $k$ and handling noise.

---

## 9. What you must know (quiz-eligible)

- Sum rule, product rule, chain rule; that the chain rule assumes nothing; **Bayes' rule derived from the product rule in one line.**
- Definitions of $\perp$ and $\perp\mid Z$; pairwise $\ne$ mutual.
- Parameter count of a general joint over $D$ binary variables; why factorization is mandatory.
- $\mathrm{Cov}[X,Y]=\mathbb{E}[XY]-\mu_X\mu_Y$; $\mathbb{V}[X+Y]$ with the cross term; $\mathrm{Cov}[Ax+b]=A\Sigma A^\top$; $\Sigma\succeq0$ via $\mathbb{V}[a^\top x]=a^\top\Sigma a$.
- $\rho\in[-1,1]$; $|\rho|=1$ iff exactly linear; **independent $\Rightarrow$ uncorrelated, converse false in general and true for jointly Gaussian.**
- Tower property and law of total variance, with the one-line proofs and the within/between reading.
- $\arg\min_f\mathbb{E}[(Y-f(X))^2] = \mathbb{E}[Y|X]$, the slice-by-slice argument, and the irreducible term.
- Bayes' rule with all four terms named; the likelihood is not a distribution over $h$.
- The base-rate computation and its natural-frequency version.
- Gaussian conditioning: conditional mean linear in the observation, conditional covariance independent of it, and $\sigma_1^2(1-\rho^2)$ in the scalar case.

---

## 10. Practice Set 0.2 (ungraded; quiz-eligible)

1. Give three binary random variables that are pairwise independent but not mutually independent. *Murphy Ex. 2.2.*
2. Prove $\mathbb{V}[X+Y] = \mathbb{V}[X]+\mathbb{V}[Y]+2\mathrm{Cov}[X,Y]$ from the definitions, and state exactly where independence would be used. *Murphy Ex. 2.6.*
3. Let $X\sim\mathrm{Unif}(-1,1)$ and $Y=X^2$. Show $\mathrm{Cov}[X,Y]=0$ and that $X$ and $Y$ are dependent. Then construct a second example with $\rho=0$ whose dependence is *not* symmetric in $X$.
4. Show that $\Sigma$ is positive semi-definite. Then show that if $\Sigma$ is singular the data lie in a lower-dimensional affine subspace with probability one, and say what that implies for any method that inverts $\hat\Sigma$.
5. Prove: $X\perp Y\mid Z$ iff there exist $g,h$ with $p(x,y\mid z)=g(x,z)h(y,z)$ for all $z$ with $p(z)>0$. *Murphy Ex. 2.3.*
6. You test positive on a test that is "99% accurate" (sensitivity $=$ specificity $= 0.99$) for a disease affecting 1 in 10,000. Compute $p(\text{disease}\mid+)$. *Murphy Ex. 2.9.*
7. **The prosecutor's fallacy.** Blood at a scene matches a type present in 1% of the population, and the defendant matches; the prosecutor claims a 99% chance of guilt. State precisely which conditional has been confused with which. Then diagnose the defense's counter-argument ("8,000 people in the city match, so this is 1-in-8,000 evidence and therefore irrelevant"), which is also wrong. *Murphy Ex. 2.10.*
8. Your neighbor has two children. (a) You ask whether he has any boys; he says yes. What is the probability the other is a girl? (b) Instead, you see one child run past and it is a boy. Now what? Explain the difference using §6.2. *Murphy Ex. 2.11.*
9. Show that minimizing $\mathbb{E}|Y-f(X)|$ over all $f$ yields the conditional median, and state what that implies about $L_1$-trained regressors under heavy-tailed noise.
10. For a 2-D Gaussian, derive $p(x_1\mid x_2)$ directly: write out the joint density, treat $x_2$ as fixed, and complete the square in $x_1$. Verify you recover the boxed formula in §7. (Do this once by hand; you will not need to again.)
11. Using §7, show that observing a variable with $|\rho| = 0.9$ removes 81% of the variance while $|\rho|=0.3$ removes 9%. Comment on what this implies for feature selection by correlation ranking.
12. A colleague's regression model reaches test MSE $0.041$ and stops improving with more data, more parameters, and more training. Describe the experiment you would run to determine whether this is the noise floor or a modeling failure. Be specific about what you would compute.

---
