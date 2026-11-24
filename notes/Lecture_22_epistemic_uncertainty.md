# Lecture 22 — Epistemic Uncertainty in Deep Learning

**ENM 5310 — Data-driven Modeling and Probabilistic Scientific Computing**

**Builds on:** Lecture 3 (KL divergence), Lectures 4–5 (MLE, MAP, a Gaussian prior as weight decay), Lecture 7 (Bayesian linear regression; the predictive distribution; aleatoric vs. epistemic uncertainty), Lectures 8–9 (Gaussian processes), Lecture 11 (active learning and Bayesian optimization, where epistemic uncertainty drives decisions), Lecture 12 (MLPs as point estimates; permutation symmetry; infinitely wide networks as GPs), Lecture 13 (SGD, weight decay, dropout, multiple seeds).

**Reading:** Murphy II, Ch. 17 (Bayesian neural networks), §§17.1–17.3. Wilson & Izmailov (2020), *Bayesian Deep Learning and a Probabilistic Perspective of Generalization*, §§1–4. Lakshminarayanan, Pritzel & Blundell (2017), *Simple and Scalable Predictive Uncertainty Estimation Using Deep Ensembles*. Optional: Maddox et al. (2019), *SWAG*; Osband, Aslanides & Cassirer (2018), *Randomized Prior Functions*; Osband et al. (2023), *Epistemic Neural Networks*; Daxberger et al. (2021), *Laplace Redux*.

**This section of the course.** The next three lectures return to uncertainty. Today covers **epistemic** uncertainty — what the model does not know because it has seen finite data — for deep networks. The following two cover **aleatoric** uncertainty — irreducible randomness in the data, which for high-dimensional outputs means learning a full distribution — through generative models: VAEs (Lecture 23), then diffusion and flow matching (Lecture 24). Two tools introduced today come back in both: the evidence lower bound and the reparameterization trick (§7), and Langevin dynamics (§8).

**The point of this lecture.** Since Lecture 12, every network we trained returned a single parameter vector and a single prediction. Lecture 7 showed what a Bayesian treatment buys for linear models: a predictive distribution whose epistemic part grows away from the data. For networks with millions of parameters the exact posterior is out of reach. This lecture goes from practice to principle. First the methods people actually use: a Bayesian last layer, MC dropout, deep ensembles, SWAG, and two cheap additions to ensembles (randomized priors and epinets). Then the principled machinery that explains and extends them: variational inference and stochastic-gradient MCMC. One question organizes all of it: **what distribution over functions does each method produce, and at what cost?**

A note on notation: in this lecture $q$ denotes an approximating distribution over parameters, as in Murphy. The empirical distribution of the data, written $q(x)$ elsewhere in the course, does not appear; data averages are written as explicit sums.

---

### Notation

| Symbol | Meaning |
|---|---|
| $\mathcal D=\{(x_n,y_n)\}_{n=1}^N$ | training data |
| $\theta\in\mathbb{R}^P$ | network parameters; $f(x;\theta)$ the network |
| $p(\theta)$ | prior over parameters, usually $\mathcal N(0,\alpha^2I)$ |
| $p(\mathcal D\mid\theta)=\prod_np(y_n\mid x_n,\theta)$ | likelihood |
| $p(\theta\mid\mathcal D)$ | posterior |
| $p(y\mid x,\mathcal D)$ | posterior predictive distribution |
| $\phi(x)$ | penultimate-layer features; $w$ the last-layer weights |
| $q_\psi(\theta)$ | approximate posterior with variational parameters $\psi$ |
| $M$ | number of ensemble members or posterior samples |
| $z$ | epistemic index (§6) |

**Learning objectives.** After this lecture you should be able to:

1. Decompose predictive variance into aleatoric and epistemic parts, and explain why only the epistemic part should drive active learning and out-of-distribution detection.
2. Write the Bayesian predictive for a neural network and explain why the parameter posterior is hard to approximate.
3. Compute the predictive of a network with a Bayesian last layer, and explain when its uncertainty fails.
4. Explain MC dropout, deep ensembles and SWAG, what kind of diversity each captures, and why ensembles are the baseline to beat.
5. Explain what randomized prior functions and epinets add to an ensemble, and why joint predictions matter.
6. Derive the evidence lower bound, explain the reparameterization trick, and state the main failure modes of mean-field variational inference.
7. Write Langevin dynamics and SGLD, and explain why SGD plus calibrated noise samples the posterior.
8. Evaluate uncertainty with proper scoring rules, calibration, out-of-distribution tests and downstream decisions.

---

## 0. Why epistemic uncertainty, and how to measure it (8 min)

### 0.1 Two sources of uncertainty

Lecture 7 split predictive uncertainty into two parts. In general, if a model's parameters are uncertain, the **law of total variance** gives, for a scalar output,

$$
\operatorname{Var}[y\mid x,\mathcal D] =
\underbrace{\mathbb E_{\theta\mid\mathcal D}\big[\operatorname{Var}(y\mid x,\theta)\big]}_{\text{aleatoric}} +
\underbrace{\operatorname{Var}_{\theta\mid\mathcal D}\big[\mathbb E(y\mid x,\theta)\big]}_{\text{epistemic}} .
$$

- **Aleatoric** uncertainty is noise in the data-generating process: sensor noise, turbulent fluctuations, unresolved scales. More data from the same process does not reduce it.
- **Epistemic** uncertainty is disagreement among models that all fit the data. It is large where data are sparse and shrinks as data accumulate.

For classification the analogous split uses entropy: the total predictive entropy $\mathbb H[\mathbb E_\theta\,p(y\mid x,\theta)]$ equals the expected entropy $\mathbb E_\theta\,\mathbb H[p(y\mid x,\theta)]$ (aleatoric) plus the mutual information between $y$ and $\theta$ (epistemic).

### 0.2 Why it matters in science and engineering

- **Active learning and Bayesian optimization** (Lecture 11): the next experiment or simulation should go where the model is ignorant, not where the data are noisy. Only the epistemic term carries that information.
- **Out-of-distribution detection:** a neural operator queried far outside its training distribution should say so.
- **Trusting a surrogate:** replacing a solver with a network is defensible only if you know where the network is reliable.

### 0.3 The problem with a single network

A network trained by MLE or MAP is a point estimate (Lecture 12 §3.4). Its predictive distribution contains only the aleatoric noise model. Far from the data, a ReLU network extrapolates linearly with whatever confidence its outputs imply. The goal of this lecture is to put the epistemic term back.

**A running example.** Throughout, picture 1D regression with data on $[-1,-0.3]\cup[0.3,1]$ and a gap in between. A good method should be confident where there is data, *more* uncertain in the gap ("in-between" uncertainty), and increasingly uncertain outside $[-1,1]$. A Gaussian process (Lecture 8) does all three. Several popular approximations do not (Foong et al., 2020).

---

## 1. The Bayesian ideal for neural networks, and why it is hard (8 min)

### 1.1 The ideal

Exactly as for Bayesian linear regression:

1. **Model:** a prior $p(\theta)$, typically $\mathcal N(0,\alpha^2I)$, and a likelihood $p(y\mid x,\theta)$ built from the network.
2. **Inference:** the posterior $p(\theta\mid\mathcal D)\propto p(\mathcal D\mid\theta)\,p(\theta)$.
3. **Prediction:** the posterior predictive, a **Bayesian model average**:
$$
p(y\mid x,\mathcal D) = \int p(y\mid x,\theta)\,p(\theta\mid\mathcal D)\,d\theta\;\approx\;\frac1M\sum_{m=1}^{M}p(y\mid x,\theta_m),\qquad \theta_m\sim p(\theta\mid\mathcal D).
$$

Every method in this lecture produces a set of plausible networks $\{\theta_m\}$, or a distribution to sample them from, and averages their predictions.

### 1.2 Why it is hard

- **Dimension.** $P$ is in the millions. No grid, no exact normalization, and anything requiring a $P\times P$ covariance is out of the question.
- **No conjugacy.** The network is nonlinear in $\theta$, so the posterior is not Gaussian or any other closed form.
- **Multimodality.** The posterior has many separated modes. Some are trivially equivalent — permuting hidden units gives the same function (Lecture 12 §3.3) — but many correspond to genuinely different functions that fit the data equally well.
- **Priors on weights are opaque.** A Gaussian on weights induces *some* distribution over functions — for infinitely wide networks, a GP (Lecture 12 §6) — but it is hard to reason about what a weight prior means for a finite deep network.

**A useful shift in viewpoint.** We do not care about $\theta$ itself; we care about the distribution over **functions** it induces, and in particular about the spread of $f(x;\theta)$ at the inputs we will query. Two very different weight vectors that compute the same function are, for prediction, the same sample. Keep this in mind when judging each method: does it produce *functionally diverse* networks?

---

## 2. A Bayesian last layer (8 min)

The cheapest way to recover a posterior is to be Bayesian about only the part of the network where Bayes is easy. Write the network as

$$
f(x;\theta)=w^\top\phi(x),
$$

with $\phi$ the penultimate features (Lecture 12 §3.1). Train the network as usual, freeze $\phi$, and put a Gaussian prior $w\sim\mathcal N(0,\alpha^2I)$ on the last layer. For a Gaussian likelihood the posterior over $w$ is then *exactly* Gaussian, and the predictive is Lecture 7's formula with learned features:

$$
p(y\mid x,\mathcal D) = \mathcal N\Big(\mu_w^\top\phi(x),\ \underbrace{\phi(x)^\top\Sigma_w\,\phi(x)}_{\text{epistemic}}+\sigma^2\Big),\qquad \Sigma_w = \Big(\tfrac{1}{\sigma^2}\Phi^\top\Phi+\alpha^{-2}I\Big)^{-1},\quad \mu_w=\tfrac{1}{\sigma^2}\Sigma_w\Phi^\top y,
$$

where $\Phi$ stacks $\phi(x_n)^\top$ as rows. For non-Gaussian likelihoods (classification), the last-layer posterior is not Gaussian, and the standard fix is a Gaussian approximation at its mode (a last-layer Laplace approximation; Daxberger et al., 2021).

**Deep learning learns the features; Bayesian linear regression supplies the uncertainty.** It costs one training run plus a $d\times d$ matrix inverse, it is post hoc, and it is a strong baseline.

**When it fails.** All the epistemic uncertainty lives in feature space. If the learned features $\phi(x)$ collapse far from the data — mapping unfamiliar inputs onto familiar feature vectors — then $\phi(x)^\top\Sigma_w\phi(x)$ stays small and the model cannot know it is extrapolating. In the running example, a ReLU network's features often vary little in the gap, and the last layer reports confidence there. Methods that vary the *features* as well as the readout avoid this; that is the rest of the lecture.

> **Remember.** Freeze the trained features and do Bayesian linear regression on the last layer: exact for Gaussian likelihoods, one training run, epistemic variance $\phi(x)^\top\Sigma_w\phi(x)$. It is only as good as the features: if they collapse away from the data, so does the uncertainty.

---

## 3. MC dropout (5 min)

Lecture 13 used dropout as a regularizer and switched it off at test time. **MC dropout** (Gal & Ghahramani, 2016) leaves it on: run $M$ stochastic forward passes, each with a fresh dropout mask, and treat them as $M$ networks:

$$
p(y\mid x,\mathcal D)\approx\frac1M\sum_{m=1}^{M}p\big(y\mid x,\theta\odot\text{mask}_m\big),
$$

with the epistemic variance given by the spread of the $M$ predicted means.

It is attractive because it is free: any network trained with dropout already supports it, and the only cost is $M$ forward passes. Its weakness is that the amount of uncertainty is set by the dropout rate, a hyperparameter chosen for regularization, not inferred from data. In particular the spread does not shrink in the right way as data accumulate, as a posterior's should (Osband, 2016), and on the running example it tends to miss the in-between uncertainty (Foong et al., 2020). §7 shows it can be read as variational inference with a particular family, which explains both its appeal and these limits. Treat it as a cheap baseline, not a posterior.

---

## 4. Deep ensembles (10 min)

### 4.1 The method

Train $M$ networks independently, from different random initializations and data orders, and average their predictions (Lakshminarayanan et al., 2017). For regression, let each network output a mean and a variance, $\big(\mu_m(x),\sigma_m^2(x)\big)$, trained by the Gaussian negative log-likelihood. The ensemble predictive is the mixture, with moments

$$
\mu_*(x) = \frac1M\sum_m\mu_m(x),\qquad
\sigma_*^2(x) = \underbrace{\frac1M\sum_m\sigma_m^2(x)}_{\text{aleatoric}}+\underbrace{\frac1M\sum_m\big(\mu_m(x)-\mu_*(x)\big)^2}_{\text{epistemic}},
$$

the law of total variance of §0.1, with ensemble members as posterior samples.

### 4.2 Why it works

It was first proposed as a non-Bayesian method, but the best explanation is Bayesian. Different random initializations end in **different modes** of the loss landscape, and those modes compute genuinely different functions away from the data (Fort et al., 2019). Methods that explore around a single mode — the last layer of §2, SWAG in §5, the variational methods of §7 — capture variation within one basin; ensembles capture variation *between* basins, which is where most functional diversity lives. An ensemble is therefore a coarse but well-placed approximation to the Bayesian model average (Wilson & Izmailov, 2020). When Izmailov et al. (2021) computed a reference posterior for image classifiers with full-batch Hamiltonian Monte Carlo (§8), deep ensembles came closer to it than the standard approximate-inference methods.

Empirically, deep ensembles are among the most reliable methods under dataset shift (Ovadia et al., 2019), and they are the baseline every new method must beat. $M=5$ already captures much of the benefit.

### 4.3 Limitations

- **Cost:** $M$ times the training and inference.
- **Collapse where members agree for the wrong reason.** If all members share the same inductive bias, they can make the same confident extrapolation far from the data; nothing in training forces them to disagree there. §6 addresses this.
- **No explicit prior.** Uncertainty in unexplored regions depends on accidents of initialization rather than on a stated prior.

> **Remember.** Train $M$ networks from different initializations and average. Mixture variance = mean of member variances (aleatoric) + spread of member means (epistemic). They work because different initializations reach functionally different modes. Simple, strong, $M\times$ cost.

---

## 5. SWAG (6 min)

SWAG — stochastic weight averaging Gaussian (Maddox et al., 2019) — builds a posterior approximation from a single training run, using the SGD trajectory itself.

1. Train normally.
2. Continue SGD with a constant or cyclical, relatively high learning rate, so the iterates bounce around the basin. Record $K$ snapshots $\theta_1,\dots,\theta_K$, typically once per epoch.
3. Fit a Gaussian $\mathcal N(\bar\theta,\Sigma)$ to the snapshots: mean $\bar\theta=\frac1K\sum_k\theta_k$; covariance as the sum of a diagonal part and a rank-$R$ part built from the last $R$ deviations $D=[\theta_k-\bar\theta]$:
$$
\Sigma = \tfrac12\Big(\Sigma_{\text{diag}}+\tfrac{1}{R-1}DD^\top\Big).
$$
4. At test time, sample $\theta_m\sim\mathcal N(\bar\theta,\Sigma)$, recompute batch-normalization statistics if used, and average predictions.

Storage is about $(R+2)P$ numbers, with $R\approx20$. The rationale: SGD with a constant step size behaves like a noisy sampler whose spread reflects the local geometry of the loss — an idea made precise by SGLD in §8. SWAG covers one basin, so it captures within-mode diversity only. **MultiSWAG**, an ensemble of SWAG models, combines within-basin and between-basin diversity and often beats both components (Wilson & Izmailov, 2020).

---

## 6. Cheap additions to ensembles: randomized priors and epinets (7 min)

Two auxiliary ideas address the ensemble's weaknesses in §4.3: no explicit prior, and the $M\times$ cost.

**Randomized prior functions** (Osband et al., 2018). Give each ensemble member a fixed, randomly initialized **prior network** $p_m$ that is never trained:

$$
f_m(x) = g_{\theta_m}(x)+\beta\,p_m(x).
$$

Train $g_{\theta_m}$ as usual. Near the data, $g_{\theta_m}$ learns to cancel the prior function so that $f_m$ fits the data. Far from the data, nothing forces it to, and $f_m$ reverts toward its own random prior function. The members therefore **disagree by construction where there is no data**, and $\beta$ sets how much — an explicit knob for prior uncertainty, at no extra cost. The idea has an exact justification: for Bayesian linear regression, solving the regularized least-squares problem with perturbed targets and a regularizer anchored at a random prior draw, $\frac{1}{\sigma^2}\lVert y+\tilde\varepsilon-\Phi\theta\rVert^2+\frac{1}{\tau^2}\lVert\theta-\theta_0\rVert^2$, returns an **exact posterior sample** (practice problem P4; Code Checkpoint 22B). The prior network is $\theta_0$ expressed in function space.

**Epinets** (Osband et al., 2023). An ensemble indexes its beliefs by a member number $m\in\{1,\dots,M\}$. Generalize: an *epistemic neural network* $f_\theta(x,z)$ takes an extra **epistemic index** $z$, and varying $z$ with $x$ fixed generates the model's range of beliefs. An epinet implements this cheaply by adding a small network, with a learnable part and a fixed randomized-prior part, on top of the frozen features of a large base model, with $z\sim\mathcal N(0,I)$. Hundreds of index samples cost little more than one forward pass of the base network, and on image classification epinets match large ensembles at a fraction of the cost.

**Why joint predictions matter.** Osband et al. also argue that epistemic uncertainty should be judged on *joint* predictions over several inputs. A fair coin and a coin of unknown uniform bias both give $P(\text{heads})=\tfrac12$, but two heads in a row have probability $\tfrac14$ and $\mathbb E[p^2]=\tfrac13$ respectively. Epistemic uncertainty makes outcomes correlated: one observation is informative about the next. An active-learning or BO loop (Lecture 11) relies on exactly this, so draw one network (or one $z$) and use it for all candidate points.

> **Remember.** Randomized priors: add a fixed random network to each ensemble member so members disagree away from the data; exact for linear models. Epinets: index an ensemble with a continuous $z$ in a small add-on network. Judge epistemic uncertainty by joint predictions, not only marginal ones.

---

## 7. Variational inference (16 min)

The methods so far are practical, but none comes with a statement of *what* posterior it approximates. Variational inference supplies one.

### 7.1 Inference as optimization

Choose a tractable family of distributions $q_\psi(\theta)$ and find the member closest to the posterior:

$$
\psi^\star = \arg\min_\psi\ \mathrm{KL}\big(q_\psi(\theta)\,\big\|\,p(\theta\mid\mathcal D)\big).
$$

This KL involves the unknown normalizer $p(\mathcal D)$, so it cannot be computed directly. The way around it is the central identity of this section and of the next lecture.

### 7.2 The evidence lower bound

1. Write the KL and use Bayes' rule, $p(\theta\mid\mathcal D)=p(\mathcal D\mid\theta)p(\theta)/p(\mathcal D)$:
$$
\mathrm{KL}(q_\psi\,\|\,p(\cdot\mid\mathcal D)) = \mathbb E_{q_\psi}\big[\log q_\psi(\theta)-\log p(\mathcal D\mid\theta)-\log p(\theta)\big]+\log p(\mathcal D).
$$
2. Rearrange:
$$
\log p(\mathcal D) = \underbrace{\mathbb E_{q_\psi}\big[\log p(\mathcal D\mid\theta)\big]-\mathrm{KL}\big(q_\psi(\theta)\,\|\,p(\theta)\big)}_{\text{ELBO}(\psi)}+\mathrm{KL}\big(q_\psi\,\|\,p(\cdot\mid\mathcal D)\big).
$$
3. The last KL is non-negative, so the ELBO is a lower bound on the log evidence, and **maximizing the ELBO over $\psi$ is the same as minimizing the KL to the posterior** — because $\log p(\mathcal D)$ does not depend on $\psi$.

The two terms have clear roles. The expected log-likelihood rewards fitting the data; the KL to the prior penalizes distributions that are more concentrated than the data justify. With $q_\psi$ a point mass, the ELBO reduces (up to a constant) to the MAP objective: VI is MAP with a distribution instead of a point.

### 7.3 Mean-field Gaussian VI and the reparameterization trick

The standard choice for networks is a fully factorized Gaussian, $q_\psi(\theta)=\prod_i\mathcal N(\theta_i\mid\mu_i,\sigma_i^2)$, with $\psi=(\mu,\rho)$ and $\sigma=\log(1+e^\rho)$ to keep $\sigma$ positive. The KL to a Gaussian prior is available in closed form. The expected log-likelihood is not, and we need its gradient with respect to $\mu$ and $\sigma$.

**The reparameterization trick.** Write a sample from $q_\psi$ as a deterministic function of $\psi$ and fixed noise:

$$
\theta = \mu+\sigma\odot\varepsilon,\qquad \varepsilon\sim\mathcal N(0,I).
$$

Then $\mathbb E_{q_\psi}[g(\theta)] = \mathbb E_{\varepsilon}[g(\mu+\sigma\odot\varepsilon)]$, and since the distribution of $\varepsilon$ does not depend on $\psi$, the gradient moves inside the expectation:

$$
\nabla_\psi\,\mathbb E_{q_\psi}[g(\theta)] = \mathbb E_{\varepsilon}\big[\nabla_\psi\,g(\mu+\sigma\odot\varepsilon)\big]\approx\nabla_\psi\,g(\mu+\sigma\odot\varepsilon_1),
$$

a single-sample, unbiased gradient that backpropagation computes like any other (Lecture 13 §1). With minibatches, scale the log-likelihood by $N/B$. This algorithm is Bayes by Backprop (Blundell et al., 2015). The reparameterization trick is also what makes VAEs trainable, next lecture.

**MC dropout, revisited.** Multiplying each weight row by a Bernoulli mask is itself a sample from a distribution over weights. Gal & Ghahramani (2016) showed that training with dropout and weight decay approximately maximizes an ELBO for that Bernoulli family. That is why MC dropout (§3) behaves like a crude variational posterior, and why it shares the failure modes below.

### 7.4 What goes wrong

- **Mode-seeking.** $\mathrm{KL}(q\,\|\,p)$ heavily penalizes $q$ for placing mass where the posterior has little, and barely penalizes it for missing posterior mass. The optimal $q$ sits inside one mode and underestimates the variance.
- **Mean-field is too restrictive.** Posterior correlations between weights are strong, and a factorized Gaussian cannot represent them. For single-hidden-layer networks, mean-field Gaussian VI and MC dropout provably cannot represent increased uncertainty *between* separated data clusters (Foong et al., 2020) — the running example's gap.
- **Underfitting.** The KL term often dominates for large networks, so practitioners down-weight it or "temper" the posterior. That improves accuracy but departs from Bayesian inference, and the resulting uncertainties are harder to interpret (Wenzel et al., 2020).
- **Cost.** Twice the parameters ($\mu$ and $\sigma$), and noisy gradients.

In practice, mean-field VI for whole networks rarely beats a deep ensemble. Its lasting importance is the ELBO and the reparameterization trick, which become the engine of latent-variable generative models in Lecture 23.

> **Remember.** $\log p(\mathcal D) = \mathrm{ELBO}+\mathrm{KL}(q\,\|\,p(\theta\mid\mathcal D))$, so maximizing the ELBO fits $q$ to the posterior. The reparameterization trick $\theta=\mu+\sigma\odot\varepsilon$ makes ELBO gradients ordinary backpropagation. MC dropout is VI with a Bernoulli family. Mean-field VI is mode-seeking, underestimates variance, and can miss in-between uncertainty.

---

## 8. Sampling: Langevin dynamics, SGLD and HMC (12 min)

### 8.1 Langevin dynamics

Instead of fitting a distribution, simulate a process whose long-run distribution is the posterior. Write $\pi(\theta)\propto p(\mathcal D\mid\theta)p(\theta)$. The **Langevin stochastic differential equation**

$$
d\theta_t = \nabla_\theta\log\pi(\theta_t)\,dt+\sqrt2\,dW_t,
$$

where $W_t$ is Brownian motion, has $\pi$ as its stationary distribution: the gradient term pulls toward high posterior density, the noise spreads the samples, and the two balance exactly at $\pi$. Its Euler–Maruyama discretization with step $\epsilon$ is

$$
\theta_{t+1} = \theta_t+\epsilon\,\nabla_\theta\log\pi(\theta_t)+\sqrt{2\epsilon}\,\xi_t,\qquad\xi_t\sim\mathcal N(0,I).
$$

Without the noise, this is gradient ascent on the log posterior — MAP training. **Sampling is optimization plus the right amount of noise.** The same SDE, driven by a *learned* gradient of a log density, is how diffusion models generate samples (Lecture 24).

### 8.2 Stochastic-gradient Langevin dynamics

Replace the full gradient by a minibatch estimate and let the step size decay (Welling & Teh, 2011):

$$
\theta_{t+1}=\theta_t+\frac{\eta_t}{2}\Big(\nabla\log p(\theta_t)+\frac NB\sum_{n\in\mathcal B_t}\nabla\log p(y_n\mid x_n,\theta_t)\Big)+\mathcal N(0,\eta_tI).
$$

This is SGD with weight decay plus Gaussian noise of a prescribed size. As $\eta_t\to0$, the injected noise dominates the minibatch noise and the iterates sample the posterior. Compare SWAG (§5): there, the minibatch noise of constant-step SGD was used as an *uncalibrated* sampler; SGLD calibrates the noise so that the target is the posterior. In practice, cyclical step-size schedules help the chain move between modes (Zhang et al., 2020). Iterates are correlated, so keep every $k$-th one.

### 8.3 Hamiltonian Monte Carlo

HMC augments $\theta$ with a momentum variable, simulates Hamiltonian dynamics to propose distant moves, and accepts or rejects them with a Metropolis step so that the samples are exact up to Monte Carlo error. It is the gold standard for posterior sampling, and the reference against which the other methods are judged. For neural networks it needs full-batch gradients and very many of them: Izmailov et al. (2021) used hundreds of accelerators to run it on modern image models. Their findings close the loop with §4: the HMC model average beat standard training, and deep ensembles came close to it.

> **Remember.** Langevin dynamics $d\theta = \nabla\log\pi\,dt+\sqrt2\,dW$ samples $\pi$; discretized, it is gradient ascent plus noise. SGLD uses minibatch gradients and decaying steps, and calibrates the noise that SWAG uses informally. HMC is the accurate, expensive reference.

---

## 9. Evaluating uncertainty, and a map of the methods (10 min)

### 9.1 How to measure whether uncertainty is any good

Uncertainty estimates are predictions and must be tested like predictions (Lecture 13 §8). No single number suffices.

| Criterion | Measure | What it catches |
|---|---|---|
| Overall quality | **Test negative log-likelihood**, a proper scoring rule: minimized in expectation only by the true predictive distribution | both over- and under-confidence |
| Calibration (regression) | **Coverage**: fraction of test targets inside the central $\alpha$-interval, for several $\alpha$; plot observed vs nominal | systematic over- or under-confidence |
| Calibration (classification) | **Reliability diagram** and expected calibration error, $\mathrm{ECE}=\sum_b\frac{n_b}{N}\lvert\mathrm{acc}_b-\mathrm{conf}_b\rvert$ over confidence bins | confidence that does not match accuracy |
| Sharpness | average predictive width, *given* calibration | an uninformative but calibrated predictor |
| Epistemic signal | **out-of-distribution detection** (AUROC using epistemic variance or mutual information as the score) | uncertainty that does not grow away from the data |
| Usefulness | **downstream decisions**: active-learning error vs. number of labels, BO regret, joint log-loss | uncertainty that is calibrated but useless for decisions |

Two warnings. Calibration alone is easy: predicting the unconditional distribution of $y$ for every input is perfectly calibrated and useless. And evaluate under **dataset shift**, not only on held-out data from the training distribution: that is where epistemic uncertainty earns its keep and where many methods degrade (Ovadia et al., 2019). Report everything as mean ± standard deviation over seeds.

### 9.2 The map

| Method | Approximates the posterior by | Captures multiple modes? | Training cost | Test cost | Notes |
|---|---|---|---|---|---|
| Bayesian last layer | exact Gaussian over the last layer | no | ≈ 1× (post hoc) | ≈ 1× | Bayesian linear regression on learned features |
| MC dropout | Bernoulli masks (crude VI) | no | 1× | $M$ passes | heuristic; uncertainty set by the dropout rate |
| Deep ensembles | $M$ independently trained modes | yes | $M\times$ | $M\times$ | the baseline to beat |
| SWAG | Gaussian from the SGD trajectory | no (MultiSWAG: yes) | ≈ 1× | $M$ passes | low-rank + diagonal covariance |
| Randomized priors | ensemble with fixed prior networks | yes | $M\times$ | $M\times$ | exact for linear models; controllable prior scale |
| Epinets | small indexed network on a base model | via the index | ≈ 1–2× | cheap per sample | designed for joint predictions |
| Mean-field VI | factorized Gaussian | no | ≈ 2× | $M$ passes | underestimates variance; ELBO returns in Lecture 23 |
| SGLD / cyclical SG-MCMC | samples along a noisy SGD chain | partly | 1–few× | $M$ passes | tuning-sensitive |
| HMC | asymptotically exact samples | yes, slowly | very high | $M$ passes | reference, not routine |

### 9.3 Other approaches worth knowing

- **Distance-aware single models.** SNGP (Liu et al., 2020) constrains the network's Lipschitz constant with spectral normalization, so distances in feature space reflect distances in input space, and replaces the last layer by an approximate GP. This directly fixes the feature-collapse failure of §2, in one forward pass.
- **Evidential deep learning** predicts the parameters of a prior distribution over the likelihood's parameters in a single pass. It is fast and popular, but its uncertainties are trained by loss minimization without a posterior to anchor them, and they have been shown not to behave like epistemic uncertainty (Bengs et al., 2022; Meinert et al., 2023). Use with caution.
- **Conformal prediction** wraps any predictor and produces intervals with guaranteed coverage under exchangeability (Angelopoulos & Bates, 2023). It calibrates, but it does not by itself distinguish epistemic from aleatoric uncertainty, and the guarantee is marginal over inputs. It pairs well with the methods above.

### 9.4 Practical guidance

1. Start with a **deep ensemble** of 5, with randomized priors if out-of-distribution behavior matters.
2. If you can only afford one training run, try a **Bayesian last layer** post hoc, then **SWAG**; use MC dropout only as a baseline.
3. Use **joint predictions** — one sampled network or index across all candidate points — when the uncertainty drives sequential decisions.
4. Evaluate with NLL, coverage, out-of-distribution tests and the downstream task, under shift, over several seeds.

---

## Code Checkpoint 22

JAX permitted for gradients. Use the running example: $y=\sin(3x)+0.1\,\varepsilon$, with 40 training points on $[-1,-0.3]\cup[0.3,1]$; test on $[-2,2]$.

**A. Five approximations, one picture.** For a 2-hidden-layer MLP of width 50, implement:
1. a Bayesian last layer on one trained network (Lecture 7's formulas on the penultimate features);
2. MC dropout ($p=0.1$, $M=50$ passes);
3. a deep ensemble ($M=5$) with mean and variance outputs;
4. an ensemble with randomized prior functions ($\beta\in\{0,1,3\}$);
5. mean-field Gaussian VI with the reparameterization trick.

Plot each predictive mean ± 2 standard deviations, with epistemic and aleatoric parts shown separately. **Report the average epistemic standard deviation** in three regions — on the data, in the gap $[-0.3,0.3]$, and outside $[-1,1]$ — and the test NLL on $[-1,1]$. In three sentences, relate the gap results to §2, §6 and §7.4.

**B. Exact samples by optimization.** For Bayesian linear regression with polynomial features (Lecture 7), implement randomize-then-optimize (§6): perturb the targets with $\tilde\varepsilon\sim\mathcal N(0,\sigma^2I)$, draw $\theta_0\sim\mathcal N(0,\tau^2I)$, and solve the anchored least-squares problem. Draw 2000 samples and **report the maximum absolute deviation** of their empirical mean and covariance from the closed-form posterior.

**C. Does the uncertainty help?** Run an active-learning loop on the running example: start from 5 points and add the candidate with the largest epistemic variance, for 20 rounds, using (i) the deep ensemble and (ii) the randomized-prior ensemble, against (iii) random selection. **Report test RMSE versus number of points**, mean ± std over 5 seeds.

---

## Quiz-eligible facts

1. Law of total variance: $\operatorname{Var}[y\mid x,\mathcal D]=\mathbb E_\theta[\operatorname{Var}(y\mid x,\theta)]+\operatorname{Var}_\theta[\mathbb E(y\mid x,\theta)]$ — aleatoric plus epistemic. Only the epistemic part shrinks with more data and should drive active learning.
2. The Bayesian predictive is a model average, $\int p(y\mid x,\theta)p(\theta\mid\mathcal D)d\theta\approx\frac1M\sum_mp(y\mid x,\theta_m)$.
3. A Bayesian last layer is Bayesian linear regression on frozen learned features; its epistemic variance $\phi(x)^\top\Sigma_w\phi(x)$ fails if the features collapse away from the data.
4. MC dropout keeps dropout on at test time and averages $M$ passes; its uncertainty is set by the dropout rate and can be read as VI with a Bernoulli family.
5. Deep ensembles: mixture variance = mean of member variances (aleatoric) + variance of member means (epistemic); they work because different initializations reach functionally different modes.
6. SWAG fits a diagonal-plus-low-rank Gaussian to SGD iterates collected at a high constant learning rate; it captures one basin.
7. Randomized prior functions add a fixed random network to each ensemble member, so members disagree away from data; randomize-then-optimize is exact for Bayesian linear regression.
8. Epistemic uncertainty induces correlations between predictions; joint predictions can distinguish models that marginal predictions cannot.
9. $\log p(\mathcal D)=\mathrm{ELBO}+\mathrm{KL}(q\,\|\,p(\theta\mid\mathcal D))$, with $\mathrm{ELBO}=\mathbb E_q[\log p(\mathcal D\mid\theta)]-\mathrm{KL}(q\,\|\,p(\theta))$.
10. Reparameterization: $\theta=\mu+\sigma\odot\varepsilon$ makes $\nabla_\psi\mathbb E_{q_\psi}[g]$ an expectation of ordinary gradients.
11. $\mathrm{KL}(q\,\|\,p)$ is mode-seeking and underestimates posterior variance; mean-field VI and MC dropout can fail to represent in-between uncertainty.
12. Langevin dynamics $d\theta=\nabla\log\pi\,dt+\sqrt2\,dW$ has stationary distribution $\pi$; SGLD is SGD plus $\mathcal N(0,\eta_tI)$ noise with decaying steps.
13. NLL is a proper scoring rule; calibration alone is not enough, since a constant predictor can be perfectly calibrated.

---

## Practice problems

**P1.** A 4-member ensemble predicts means $1.0, 1.2, 0.8, 1.0$ at an input, each with aleatoric variance $0.04$. Give the predictive mean, the epistemic and aleatoric variances, and the total.

*Answer.* Mean $1.0$. Epistemic variance $\frac14(0+0.04+0.04+0)=0.02$. Aleatoric $0.04$. Total $0.06$.

**P2.** You fit mean-field Gaussian VI to a posterior that is a correlated 2D Gaussian with variances 1 and correlation 0.9. Will $q$ overestimate or underestimate the marginal variances? Why?

*Answer.* Underestimate. $\mathrm{KL}(q\,\|\,p)$ penalizes $q$ for placing mass where $p$ is small, so a factorized $q$ must fit inside the narrow diagonal ridge of $p$. The optimum has variances equal to the inverse diagonal of $p$'s precision matrix, $1-0.9^2=0.19$, far below the true marginal variance of 1.

**P3.** All five members of a deep ensemble predict nearly the same value far outside the training data, and that value is wrong. Name the failure and a fix that costs nothing extra.

*Answer.* Ensemble collapse: members share an inductive bias that extrapolates the same way, and training gives them no reason to disagree where there is no data. Add randomized prior functions: each member gets a fixed random prior network, so away from the data the members revert to different prior functions and disagree.

**P4.** For Bayesian linear regression $y=\Phi\theta+\varepsilon$, $\varepsilon\sim\mathcal N(0,\sigma^2I)$, $\theta\sim\mathcal N(0,\tau^2I)$, let $\hat\theta=\arg\min_\theta\frac{1}{\sigma^2}\lVert y+\tilde\varepsilon-\Phi\theta\rVert^2+\frac{1}{\tau^2}\lVert\theta-\theta_0\rVert^2$ with $\tilde\varepsilon\sim\mathcal N(0,\sigma^2I)$ and $\theta_0\sim\mathcal N(0,\tau^2I)$. Show that $\hat\theta$ has the posterior mean and covariance.

*Answer.* With $A=\Phi^\top\Phi/\sigma^2+I/\tau^2$, $\hat\theta=A^{-1}\big(\Phi^\top(y+\tilde\varepsilon)/\sigma^2+\theta_0/\tau^2\big)$. Its mean is $A^{-1}\Phi^\top y/\sigma^2$, the posterior mean. Its random part $A^{-1}(\Phi^\top\tilde\varepsilon/\sigma^2+\theta_0/\tau^2)$ has covariance $A^{-1}(\Phi^\top\Phi/\sigma^2+I/\tau^2)A^{-1}=A^{-1}$, the posterior covariance. Being Gaussian, $\hat\theta$ is an exact posterior sample.

**P5.** A fair coin and a coin of unknown uniform bias give the same probability of heads. Compute the probability of two heads in a row under each, and explain why this matters for active learning.

*Answer.* Fair: $\tfrac14$. Unknown bias: $\mathbb E[p^2]=\tfrac13$. Epistemic uncertainty makes outcomes correlated: each observation is informative about the next. An active-learning or BO agent must anticipate how a new observation will change its beliefs, which only joint predictions capture.

---

## References

- Angelopoulos, A. N., & Bates, S. (2023). Conformal prediction: A gentle introduction. *Foundations and Trends in Machine Learning*, 16(4), 494–591.
- Bengs, V., Hüllermeier, E., & Waegeman, W. (2022). Pitfalls of epistemic uncertainty quantification through loss minimisation. *NeurIPS*.
- Blundell, C., Cornebise, J., Kavukcuoglu, K., & Wierstra, D. (2015). Weight uncertainty in neural networks. *ICML*.
- Daxberger, E., Kristiadi, A., Immer, A., Eschenhagen, R., Bauer, M., & Hennig, P. (2021). Laplace redux — Effortless Bayesian deep learning. *NeurIPS*.
- Foong, A. Y. K., Burt, D. R., Li, Y., & Turner, R. E. (2020). On the expressiveness of approximate inference in Bayesian neural networks. *NeurIPS*.
- Fort, S., Hu, H., & Lakshminarayanan, B. (2019). Deep ensembles: A loss landscape perspective. arXiv:1912.02757.
- Gal, Y., & Ghahramani, Z. (2016). Dropout as a Bayesian approximation: Representing model uncertainty in deep learning. *ICML*.
- Izmailov, P., Vikram, S., Hoffman, M. D., & Wilson, A. G. (2021). What are Bayesian neural network posteriors really like? *ICML*.
- Lakshminarayanan, B., Pritzel, A., & Blundell, C. (2017). Simple and scalable predictive uncertainty estimation using deep ensembles. *NeurIPS*.
- Liu, J. Z., Lin, Z., Padhy, S., Tran, D., Bedrax-Weiss, T., & Lakshminarayanan, B. (2020). Simple and principled uncertainty estimation with deterministic deep learning via distance awareness. *NeurIPS*.
- Maddox, W. J., Izmailov, P., Garipov, T., Vetrov, D. P., & Wilson, A. G. (2019). A simple baseline for Bayesian uncertainty in deep learning. *NeurIPS*.
- Meinert, N., Gawlikowski, J., & Lavin, A. (2023). The unreasonable effectiveness of deep evidential regression. *AAAI*.
- Murphy, K. P. (2023). *Probabilistic Machine Learning: Advanced Topics*. MIT Press. Ch. 17.
- Osband, I. (2016). Risk versus uncertainty in deep learning: Bayes, bootstrap and the dangers of dropout. *NeurIPS Workshop on Bayesian Deep Learning*.
- Osband, I., Aslanides, J., & Cassirer, A. (2018). Randomized prior functions for deep reinforcement learning. *NeurIPS*.
- Osband, I., Wen, Z., Asghari, S. M., Dwaracherla, V., Ibrahimi, M., Lu, X., & Van Roy, B. (2023). Epistemic neural networks. *NeurIPS*.
- Ovadia, Y., et al. (2019). Can you trust your model's uncertainty? Evaluating predictive uncertainty under dataset shift. *NeurIPS*.
- Welling, M., & Teh, Y. W. (2011). Bayesian learning via stochastic gradient Langevin dynamics. *ICML*.
- Wenzel, F., et al. (2020). How good is the Bayes posterior in deep neural networks really? *ICML*.
- Wilson, A. G., & Izmailov, P. (2020). Bayesian deep learning and a probabilistic perspective of generalization. *NeurIPS*.
- Zhang, R., Li, C., Zhang, J., Chen, C., & Wilson, A. G. (2020). Cyclical stochastic gradient MCMC for Bayesian deep learning. *ICLR*.
