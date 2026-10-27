# Lecture 15 — Recurrent Neural Networks: Learning Dynamics from Sequences

**ENM 5310 — Data-driven Modeling and Probabilistic Scientific Computing**

**Builds on:** Lecture 3 (chain rule of probability), Lectures 4–5 (MLE; Gaussian and categorical likelihoods; ridge regression), the MCMC lectures (Markov chains and the Markov property), Lecture 12 (MLPs; random features), Lecture 13 (reverse-mode AD and VJPs; the backprop product of Jacobians; residual connections; gradient clipping), Lecture 14 (weight sharing; convolution as the unique translation-equivariant linear map).

**Reading:** Murphy I §15.1–15.2. Goodfellow, Bengio & Courville (2016), *Deep Learning*, Ch. 10. Olah (2015), *Understanding LSTM Networks* (colah.github.io). Pascanu, Mikolov & Bengio (2013), *On the difficulty of training recurrent neural networks*, §§1–3. Optional: Karpathy (2015), *The Unreasonable Effectiveness of Recurrent Neural Networks* (blog); Bahdanau, Cho & Bengio (2015), §§1–3.

**The point of this lecture.** CNNs share weights across *space* because the same local pattern means the same thing everywhere. Recurrent networks share weights across *time* because the same dynamics act at every step. An RNN is a learned discrete-time dynamical system, and nearly everything that makes RNNs useful — and hard to train — follows from that view. We end by locating the RNN's two structural limitations, an information bottleneck and strictly sequential computation, which are exactly what attention and the transformer remove.

---

### Notation

| Symbol | Meaning |
|---|---|
| $x_t\in\mathbb{R}^D$, $y_t$ | input and target at time step $t$; $x_{1:T}=(x_1,\dots,x_T)$ |
| $h_t\in\mathbb{R}^H$ | hidden state at step $t$; $h_0$ given (usually $0$) |
| $W\in\mathbb{R}^{H\times H}$, $U\in\mathbb{R}^{H\times D}$, $b$ | recurrent weights, input weights, bias |
| $V$, $c$ | readout weights and bias, $\hat y_t = Vh_t+c$ |
| $a_t = Wh_{t-1}+Ux_t+b$ | pre-activation, $h_t=\tanh(a_t)$ |
| $J_t = \partial h_t/\partial h_{t-1}$ | one-step state Jacobian |
| $\delta_t = \partial\mathcal L/\partial h_t$ | total cotangent of the hidden state |
| $\operatorname{sigm}$, $\odot$ | logistic sigmoid, elementwise product |

Vectors are columns, one sequence at a time.

**Learning objectives.** After this lecture you should be able to:

1. Write the autoregressive factorization of a sequence likelihood and explain how an RNN parameterizes it with a fixed-size state.
2. Write the vanilla RNN, count its parameters, unroll it, and choose its likelihood and loss for a given task.
3. Derive backpropagation through time as reverse-mode AD on the unrolled graph, and explain why parameter gradients sum over time.
4. Bound the gradient $\partial h_T/\partial h_k$ in terms of the recurrent weights, and connect vanishing and exploding gradients to the stability of a dynamical system, including the case of chaotic dynamics.
5. Write the LSTM and GRU, and explain how the cell state provides an additive gradient path, like a residual connection in time.
6. Explain teacher forcing, exposure bias and error accumulation in autoregressive rollout, and name remedies.
7. Describe the encoder–decoder bottleneck, attention as its fix, and the two properties of RNNs that motivate the transformer.

---

## 0. Why sequences, and why science cares (8 min)

Every model so far has mapped a fixed-size input to a fixed-size output. Many problems do not fit that mold: the input, the output, or both are **sequences** of variable length, and the order matters.

In machine learning the classic examples are language modeling (the distribution over text), speech recognition, machine translation, and captioning. Science and engineering supply at least as many:

- **Forecasting dynamical systems.** Predict the next state of a turbulent flow, a power grid, or a chaotic oscillator from its recent history. Echo state networks, a simple RNN variant (§5.4), forecast the chaotic Kuramoto–Sivashinsky equation several Lyapunov times ahead from data alone (Pathak et al., 2018).
- **Hydrology.** LSTMs trained on rainfall and meteorological series from 531 US catchments predicted river discharge better than calibrated conceptual hydrological models (Kratzert et al., 2019).
- **Path-dependent materials.** Stress in a plastic material depends on the entire loading history, not on the current strain. RNNs trained on simulated loading paths learn this history dependence and serve as surrogate constitutive laws (Mozaffar et al., 2019).
- **Monitoring and diagnosis.** Vibration signals for structural health monitoring, ECG and EEG traces, battery voltage curves for predicting remaining life.
- **Spatiotemporal fields.** Replacing the matrix products in an LSTM with convolutions gives the ConvLSTM, introduced for precipitation nowcasting from radar sequences (Shi et al., 2015).

What these share is that the past matters, and that how much of the past matters is not known in advance.

---

## 1. The probabilistic formulation: autoregressive models and state (10 min)

### 1.1 The autoregressive factorization

By the chain rule of probability (Lecture 3), any distribution over a sequence factorizes exactly as

$$
p(y_{1:T}) = \prod_{t=1}^{T} p(y_t\mid y_{1:t-1}).
$$

Modeling a sequence reduces to modeling the **next-step conditional**. If inputs $x_{1:T}$ are also present (forcing, controls, other sensors), condition on them as well: $p(y_t\mid y_{1:t-1},x_{1:t})$. The likelihood choice is the same as before: Gaussian for continuous signals (squared-error loss), categorical for tokens (cross-entropy), and maximum likelihood minimizes $-\sum_t\log p(y_t\mid y_{<t})$.

The difficulty is that the conditioning set grows with $t$. There are two ways to handle it.

### 1.2 Option 1: a fixed window

Assume the next step depends only on the last $k$ steps: $p(y_t\mid y_{t-k:t-1})$. This is a **$k$-th order Markov model** (the MCMC lectures used $k=1$). Any fixed-input model can implement it — an MLP on the concatenated window, or a 1D convolution over time (Lecture 14), which shares weights across time positions. Linear versions are the AR($k$) models of classical time-series analysis.

The limitation is that $k$ must be chosen in advance, cost grows with $k$, and nothing before the window can matter. A material that yielded 500 steps ago is invisible to a 100-step window.

### 1.3 Option 2: a state

Classical physics and control have long used a different idea: summarize the entire past in a **state** that is updated at each step,

$$
h_t = f(h_{t-1},\,x_t), \qquad \hat y_t = g(h_t).
$$

Given $h_t$, the past adds nothing — the state is a sufficient statistic for prediction. This is a discrete-time state-space model: a numerical integrator for an ODE, a Kalman filter, an internal-variable model of plasticity (where $h_t$ holds the plastic strain and hardening variables). A **recurrent neural network** learns $f$ and $g$ from data. Its memory is unbounded in principle, its cost per step is constant, and the same parameters handle sequences of any length.

---

## 2. The vanilla RNN (14 min)

### 2.1 The model

The simplest choice, due to Elman (1990), is an MLP layer that takes the previous state as an additional input:

$$
h_t = \tanh\big(Wh_{t-1}+Ux_t+b\big), \qquad \hat y_t = Vh_t + c, \qquad h_0 = 0.
$$

The parameters $\theta=\{W,U,b,V,c\}$ are **shared across all time steps**: $H^2+HD+H$ for the recurrence plus the readout. This is the temporal analogue of Lecture 14's weight sharing, with the same justification: the dynamics are the same at every step (time-translation invariance, or *autonomy*, in the language of dynamical systems).

### 2.2 Unrolling

For a sequence of length $T$, the recurrence unrolls into a feedforward network of depth $T$ in which every layer has the same weights:

$$
h_0 \xrightarrow{\;W,U\;} h_1 \xrightarrow{\;W,U\;} h_2 \xrightarrow{\;W,U\;} \cdots \xrightarrow{\;W,U\;} h_T,
\qquad x_t \text{ entering at every step}, \quad \hat y_t \text{ leaving at every step}.
$$

Two consequences carry through the rest of the lecture. Everything from Lecture 13 applies to the unrolled network, so training is reverse-mode AD on it (§3). And that network is *deep* — as deep as the sequence is long — which is where the difficulties come from (§4).

### 2.3 Input–output patterns

| Pattern | Example | Loss on |
|---|---|---|
| Many-to-one | Classify a vibration record as damaged/healthy; predict remaining battery life | $\hat y_T$ only |
| Many-to-many, aligned | Next-step forecasting; stress at every strain increment | every $\hat y_t$ |
| One-to-many | Generate a trajectory from an initial condition | every generated step |
| Many-to-many, unaligned | Translation; map a forcing history to a response of different length | decoder outputs (§7) |

Two common extensions: **stacked** RNNs, where the state sequence of one layer is the input sequence of the next, and **bidirectional** RNNs, which run a second RNN backward in time and concatenate the states. Bidirectional models see the future, so they suit offline tasks such as labeling a recorded signal, not forecasting.

### 2.4 Worked example: learning a sine wave

Take $y(t) = \sin(\pi t)$ sampled at spacing $\Delta t$, and train an RNN to predict $y_t$ from $y_{t-1}$ (so $x_t = y_{t-1}$), with a Gaussian likelihood:

$$
\mathcal L(\theta) = \frac{1}{T}\sum_{t=1}^{T}\big(y_t-\hat y_t\big)^2 .
$$

A first-order Markov model cannot solve this, because $y_{t-1}$ alone does not determine whether the wave is rising or falling. Two lags suffice: the linear recurrence $y_t = 2\cos(\pi\Delta t)\,y_{t-1} - y_{t-2}$ holds exactly for a sinusoid. An RNN with a 2D state can learn the equivalent rotation in its hidden space — the state encodes phase, which the input alone does not reveal.

If, instead of running the RNN over the whole sequence, you reset $h=0$ every two steps, you get a model that sees only two lags: a windowed model (§1.2) written in recurrent form. The power of the RNN comes from carrying the state forward across the entire sequence.

> **Remember.** An RNN is $h_t = \tanh(Wh_{t-1}+Ux_t+b)$ with the same weights at every step: a learned discrete dynamical system. Unrolled, it is a deep feedforward network with shared weights, and the likelihood determines the loss exactly as before.

---

## 3. Backpropagation through time (10 min)

The total loss is $\mathcal L = \sum_t\ell_t(\hat y_t, y_t)$. **Backpropagation through time** (BPTT) is reverse-mode AD on the unrolled graph — nothing more. Write $a_t = Wh_{t-1}+Ux_t+b$, so $h_t=\tanh(a_t)$, and let $\delta_t = \partial\mathcal L/\partial h_t$ be the *total* cotangent of $h_t$.

1. **Two routes into each state.** $h_t$ affects the loss directly, through $\hat y_t$, and indirectly, through $h_{t+1}$. So
$$
\delta_t = \underbrace{V^\top\frac{\partial\ell_t}{\partial\hat y_t}}_{\text{local}} + \underbrace{W^\top g_{t+1}}_{\text{from the future}}, \qquad \delta_T = V^\top\frac{\partial\ell_T}{\partial\hat y_T}.
$$
2. **Through the nonlinearity.** $g_t = \partial\mathcal L/\partial a_t = (1-h_t^2)\odot\delta_t$, the VJP of $\tanh$.
3. **Run backward** from $t=T$ to $t=1$, alternating steps 1 and 2.
4. **Parameter gradients sum over time**, because each weight is used at every step:
$$
\frac{\partial\mathcal L}{\partial W} = \sum_{t=1}^{T} g_t\,h_{t-1}^\top,\qquad
\frac{\partial\mathcal L}{\partial U} = \sum_{t=1}^{T} g_t\,x_t^\top,\qquad
\frac{\partial\mathcal L}{\partial b} = \sum_{t=1}^{T} g_t .
$$

This is the same accumulation as the convolution kernel gradient summing over positions (Lecture 14 §5), now over time steps.

**Cost and memory.** One forward and one backward sweep, each $O(T)$ steps, and every $h_t$ must be stored — memory grows linearly with sequence length. For long sequences, **truncated BPTT** splits the sequence into chunks of length $\tau$: the state is carried forward across chunk boundaries, but gradients are not propagated back past them. Memory becomes $O(\tau)$, at the price of never learning dependencies longer than $\tau$ steps from the gradient.

**The steps cannot be parallelized.** $h_t$ needs $h_{t-1}$, which needs $h_{t-2}$, and so on. A CNN processes all positions of an image at once; an RNN must take $T$ sequential steps, forward and backward. On GPUs built for large parallel matrix products, this is the RNN's most expensive property. It is the first of the two reasons we will eventually replace recurrence with attention.

---

## 4. Vanishing and exploding gradients: the dynamical-systems view (12 min)

### 4.1 The product of Jacobians

How much does the loss at step $T$ depend on the state at an earlier step $k$? By the chain rule,

$$
\frac{\partial h_T}{\partial h_k} = \prod_{t=k+1}^{T}J_t, \qquad J_t = \frac{\partial h_t}{\partial h_{t-1}} = \operatorname{diag}\big(1-h_t^2\big)\,W .
$$

This is Lecture 13 §2's product of layer Jacobians, with one crucial difference: **the same matrix $W$ appears in every factor**. In a feedforward network each layer has its own weights, and their effects can partly cancel. In an RNN, whatever $W$ does, it does $T-k$ times.

### 4.2 The bound

Let $\gamma = \sup|\tanh'| = 1$ and let $s_{\max}$ be the largest singular value of $W$. Then

$$
\Big\lVert\frac{\partial h_T}{\partial h_k}\Big\rVert \le \prod_{t=k+1}^{T}\lVert\operatorname{diag}(1-h_t^2)\rVert\,\lVert W\rVert \le \big(\gamma\,s_{\max}\big)^{T-k}.
$$

If $s_{\max}<1/\gamma$, the gradient **vanishes exponentially** in the time lag. Conversely, $s_{\max}>1/\gamma$ is necessary (though not sufficient, since saturation shrinks the $\tanh'$ factors) for it to **explode** (Pascanu et al., 2013).

The linear case makes this concrete. With $h_t = Wh_{t-1}$ and $W = Q\Lambda Q^{-1}$,

$$
\frac{\partial h_T}{\partial h_k} = W^{T-k} = Q\,\Lambda^{T-k}Q^{-1}:
$$

every eigendirection is multiplied by $\lambda_i^{T-k}$. Components with $|\lambda_i|<1$ are forgotten exponentially fast; components with $|\lambda_i|>1$ blow up. The window in which information survives without blowing up is $|\lambda_i|\approx1$.

### 4.3 What it means

**Vanishing gradients mean the network cannot learn long-range dependencies.** The gradient contribution from an event $T-k$ steps back decays like $(\gamma s_{\max})^{T-k}$. Short-range terms dominate the sum in §3, step 4, so the network learns short-term structure and fails to learn that a load cycle 200 steps ago determines today's stress. The information may be present in $h_t$; the gradient just never tells the network to keep it.

**Exploding gradients mean training is unstable.** A single long sequence can produce a gradient orders of magnitude larger than usual and destroy many steps of progress.

**This is a statement about stability.** View $h_t = f(h_{t-1},x_t)$ as a dynamical system. If it is contractive — nearby states converge — the system forgets its initial conditions, and the gradient with respect to them vanishes. If it is expansive, nearby states diverge and the gradient explodes. A trainable RNN must sit near the boundary, retaining information without amplifying it.

**For chaotic systems, exploding gradients are the truth, not a pathology.** If the RNN correctly reproduces chaotic dynamics with largest Lyapunov exponent $\lambda>0$, then perturbations really do grow like $e^{\lambda t}$, and so does $\partial h_T/\partial h_k$. No architecture removes this; the true sensitivity of a long trajectory to its initial state is exponential (Mikhaeil et al., 2022). The practical consequence: train chaotic forecasters on horizons of a few Lyapunov times, not on long rollouts, and expect predictability to be bounded by the physics, not by the model.

### 4.4 Remedies

| Problem | Remedy | Idea |
|---|---|---|
| Exploding | **Gradient clipping** (Lecture 13), $g\leftarrow g\min(1,c/\lVert g\rVert)$ | Keep the direction, cap the size |
| Exploding / memory | **Truncated BPTT** | Bound the product's length |
| Vanishing | **Orthogonal or identity initialization** of $W$ (e.g. Le et al., 2015) | Start with all $\lvert\lambda_i\rvert=1$ |
| Vanishing | **Gating** (LSTM, GRU; §5) | Replace the multiplicative path by an additive one |

Clipping fixes explosions cheaply and is always used. Vanishing requires a change of architecture.

> **Remember.** $\partial h_T/\partial h_k = \prod J_t$ with the same $W$ in every factor, so its norm behaves like $(\gamma s_{\max})^{T-k}$: exponential decay or growth in the time lag. This is a stability property of the learned dynamical system. Clip for explosions; gate for vanishing.

---

## 5. Gated RNNs: LSTM and GRU (14 min)

### 5.1 The idea

The vanilla RNN overwrites its whole state at every step: $h_t$ is a fresh nonlinear function of $h_{t-1}$. Lecture 13 showed how to keep gradients alive through depth: make each layer compute a correction, $h+F(h)$, so the backward product contains an identity path. Gated RNNs do the same through time, and they let the network *decide*, input by input, how much to keep.

### 5.2 Long short-term memory

The LSTM (Hochreiter & Schmidhuber, 1997, with the forget gate of Gers et al., 2000) carries two vectors: a **cell state** $c_t$, the long-term memory, and a hidden state $h_t$, the output. At each step it computes three gates, each a vector in $(0,1)^H$, plus a candidate update:

$$
\begin{aligned}
f_t &= \operatorname{sigm}\big(W_fh_{t-1}+U_fx_t+b_f\big) &&\text{forget gate: how much memory to keep}\\
i_t &= \operatorname{sigm}\big(W_ih_{t-1}+U_ix_t+b_i\big) &&\text{input gate: how much new content to write}\\
o_t &= \operatorname{sigm}\big(W_oh_{t-1}+U_ox_t+b_o\big) &&\text{output gate: how much memory to expose}\\
\tilde c_t &= \tanh\big(W_ch_{t-1}+U_cx_t+b_c\big) &&\text{candidate content}
\end{aligned}
$$

$$
c_t = f_t\odot c_{t-1} + i_t\odot\tilde c_t, \qquad h_t = o_t\odot\tanh(c_t).
$$

**Why it works.** The cell update is *additive*. Ignoring the dependence of the gates on $h_{t-1}$,

$$
\frac{\partial c_t}{\partial c_{t-1}} \approx \operatorname{diag}(f_t), \qquad \frac{\partial c_T}{\partial c_k}\approx\prod_{t=k+1}^{T}\operatorname{diag}(f_t).
$$

There is no $W$ in this product and no $\tanh'$ factor. When the network sets $f_t\approx1$, information and gradients pass through the cell unchanged for as long as the gate stays open — what Hochreiter and Schmidhuber called the "constant error carousel." With $f_t=1$ the update is $c_t = c_{t-1}+i_t\odot\tilde c_t$: a residual connection in time. And unlike a fixed $W$, the gates are functions of the input, so the network can hold a memory for 500 steps and then erase it in one.

**Parameters:** $4H(H+D+1)$, four times a vanilla RNN of the same width.

**A practical detail.** Initialize the forget-gate bias $b_f$ to about 1 rather than 0 (Jozefowicz et al., 2015). Then $f_t\approx\operatorname{sigm}(1)\approx0.73$ at the start of training rather than 0.5, and the memory decays far more slowly. Over 20 steps, $0.73^{20}\approx2\times10^{-3}$ against $0.5^{20}\approx10^{-6}$.

### 5.3 Gated recurrent unit

The GRU (Cho et al., 2014) merges the cell and hidden states and uses two gates:

$$
\begin{aligned}
z_t &= \operatorname{sigm}\big(W_zh_{t-1}+U_zx_t+b_z\big) &&\text{update gate}\\
r_t &= \operatorname{sigm}\big(W_rh_{t-1}+U_rx_t+b_r\big) &&\text{reset gate}\\
\tilde h_t &= \tanh\big(W_h(r_t\odot h_{t-1})+U_hx_t+b_h\big) &&\text{candidate state}\\
h_t &= z_t\odot h_{t-1} + (1-z_t)\odot\tilde h_t .
\end{aligned}
$$

(Some references swap the roles of $z_t$ and $1-z_t$; check before comparing code.) Parameters: $3H(H+D+1)$. On most tasks the GRU and LSTM perform similarly.

**An engineer's reading.** Rearrange the GRU update:

$$
h_t - h_{t-1} = (1-z_t)\odot\big(\tilde h_t - h_{t-1}\big).
$$

This is a forward-Euler step of the relaxation ODE $\tau\,\dot h = \tilde h - h$, with step size $\Delta t/\tau = 1-z_t$. The GRU is a leaky integrator whose **time constant is learned and input-dependent**: $z_t\to1$ freezes the state (long memory), $z_t\to0$ replaces it (fast response). The reset gate decides how much of the old state the candidate may use.

### 5.4 A gradient-free alternative: echo state networks

Vanishing and exploding gradients come from training $W$ through BPTT. **Reservoir computing** sidesteps both: draw $W$ and $U$ at random, scale $W$ to a spectral radius slightly below 1 so that the dynamics are stable but slowly forgetting (the "echo state property"), and train only the linear readout $V$ by ridge regression — the closed form of Lecture 5. It is Lecture 12's random-features model, in time. Echo state networks are surprisingly strong at forecasting chaotic systems (Pathak et al., 2018), and they make an excellent baseline (Code Checkpoint 15C).

> **Remember.** LSTM: $c_t = f_t\odot c_{t-1}+i_t\odot\tilde c_t$ — an additive, gated path whose Jacobian is $\operatorname{diag}(f_t)$, a residual connection in time. GRU: a leaky integrator with a learned, input-dependent time constant. Initialize the forget bias near 1.

---

## 6. Training for rollout: teacher forcing and its failure mode (8 min)

For forecasting, the model is trained to predict $y_t$ from the *true* past, but used to predict from *its own* past.

**Teacher forcing.** During training, feed the ground-truth $y_{t-1}$ as the input at step $t$. This is exactly maximum likelihood under the autoregressive factorization of §1.1, every step's loss can be computed in parallel given the true inputs, and training is stable.

**Free-running rollout.** At test time the true $y_{t-1}$ is not available, so the model feeds back its own prediction $\hat y_{t-1}$. Now the inputs come from a distribution the model never saw in training — its own slightly wrong outputs. This mismatch is called **exposure bias**. Errors compound: a model with 1% one-step error can be useless after 100 steps, and for chaotic dynamics the error grows like $e^{\lambda t}$ however accurate the one-step model is.

**Remedies**, all used in learned simulators:

- **Multi-step (rollout) loss.** Train on $k$-step free-running rollouts, backpropagating through the model's own predictions. This optimizes what you actually use, at the cost of longer, more sequential computation and the gradient issues of §4.
- **Scheduled sampling** (Bengio et al., 2015). Gradually replace true inputs with model predictions during training.
- **Noise injection.** Corrupt training inputs with noise comparable to the model's own errors, so that it learns to correct drift rather than amplify it (Sanchez-Gonzalez et al., 2020).

**Evaluate what you deploy.** Report rollout error against horizon, not just one-step error. For chaotic systems, report the **valid prediction time** — how long the rollout stays within a tolerance of the truth — in units of the Lyapunov time $1/\lambda$. A one-step metric alone hides the failure that matters.

---

## 7. Encoder–decoder models and the road to attention (12 min)

### 7.1 Sequence-to-sequence

When input and output sequences have different lengths and no alignment — translating a sentence, or mapping a forcing history to a response sampled differently — use two RNNs (Sutskever et al., 2014; Cho et al., 2014). An **encoder** reads $x_{1:S}$ and its final state becomes a context vector $m = h^{\text{enc}}_S$. A **decoder**, initialized from $m$, generates $y_1,y_2,\dots$ autoregressively, one step at a time.

### 7.2 The bottleneck

Everything the decoder knows about the input must pass through $m$, one fixed-size vector, however long the input. Cho et al. (2014) observed that translation quality fell sharply with sentence length. The second problem is §4 again: information about $x_1$ reaches the decoder only after passing through $S$ nonlinear recurrent steps.

### 7.3 Attention

Bahdanau et al. (2015) let the decoder look back at **every** encoder state, not just the last one. At each decoder step $t$, with decoder state $s_{t-1}$:

$$
e_{t,j} = \operatorname{score}\big(s_{t-1},\,h_j\big), \qquad
\alpha_{t,j} = \frac{\exp(e_{t,j})}{\sum_{j'}\exp(e_{t,j'})}, \qquad
m_t = \sum_{j=1}^{S}\alpha_{t,j}\,h_j .
$$

The weights $\alpha_{t,\cdot}$ form a probability distribution over input positions — a soft, learned alignment — and the context $m_t$ is recomputed at every output step. The bottleneck is gone, and every input position is one step away from every output step. In translation, the learned $\alpha$ recovered word alignments without supervision.

### 7.4 Why drop recurrence altogether

Attention was first added *on top of* recurrent networks. Vaswani et al. (2017) asked what remains if the recurrence is removed and attention is used everywhere, including within the input sequence itself. Compare three ways to build a layer over a sequence of length $n$ with width $d$:

| Layer | Cost per layer | Sequential operations | Longest path between two positions |
|---|---|---|---|
| Recurrent | $O(nd^2)$ | $O(n)$ | $O(n)$ |
| Convolutional (kernel $k$) | $O(knd^2)$ | $O(1)$ | $O(\log_k n)$ (dilated) or $O(n/k)$ |
| Self-attention | $O(n^2d)$ | $O(1)$ | $O(1)$ |

The two RNN limitations of this lecture appear in the last two columns. **Sequential computation**: an RNN cannot parallelize over time (§3). **Path length**: information between distant positions travels through $O(n)$ steps, each multiplying the gradient by a Jacobian (§4). Self-attention makes both $O(1)$, at the price of $O(n^2)$ cost in the sequence length. The transformer is built on this trade, and it is where the next lectures go.

### 7.5 Recurrence is not dead: linear state-space models

A footnote with a future. If the recurrence is *linear*, $h_t = Ah_{t-1}+Bx_t$, $y_t = Ch_t$ — the state-space model of control theory — then two things change. Stability is controlled directly by the eigenvalues of $A$, which can be parameterized to stay inside the unit circle, so vanishing and exploding gradients can be designed away. And the whole output sequence can be computed in parallel: unrolling gives $y_t = \sum_{k\ge0}CA^kB\,x_{t-k}$, a **convolution** with kernel $(CB, CAB, CA^2B,\dots)$ (Lecture 14), or alternatively a parallel scan with $O(\log T)$ depth. Structured state-space models (S4; Gu et al., 2022) and their input-dependent, gated successors (Mamba; Gu & Dao, 2023) build deep networks from such layers, interleaved with nonlinearities. They are competitive with transformers on long sequences — and the input-dependent gating that makes them work is the LSTM's idea, returning.

---

## 8. Summary (2 min)

An RNN models a sequence through the autoregressive factorization, using a fixed-size state updated by shared weights: a learned discrete dynamical system. Training is BPTT, reverse-mode AD on the unrolled network, with gradients summed over time. The gradient across a lag of $T-k$ steps is a product of $T-k$ Jacobians containing the same $W$, so it vanishes or explodes exponentially, which is a stability property of the learned dynamics (and, for chaotic systems, of the physics itself). LSTMs and GRUs add a gated additive path that acts like a residual connection in time. For forecasting, training on true inputs and testing on predicted ones causes error accumulation, which must be addressed in training and measured in evaluation. Finally, RNNs compute sequentially and connect distant positions through long paths; attention removes both limitations, which takes us to the transformer.

---

## Code Checkpoint 15

NumPy only; JAX may be used only to check item A. Each item asks you to report a number.

**A. An RNN and BPTT by hand.**
1. Implement the vanilla RNN forward pass and BPTT (§3) for scalar input and output and a Gaussian likelihood. Check all gradients against central finite differences on a sequence of length 20. **Report the maximum relative error.**
2. Train it with your Lecture 13 AdamW on $y(t)=\sin(\pi t)$, $\Delta t=0.1$, inputs $x_t=y_{t-1}$, with $H=8$. **Report the one-step test MSE**, and the MSE of a 100-step free-running rollout started from the true first value.
3. Repeat with the state reset to 0 every two steps (a two-lag window model). **Report both errors** again and explain the difference.

**B. How fast gradients vanish.** For $H=64$, draw $W$ with i.i.d. Gaussian entries and rescale it to spectral radius $\rho\in\{0.5,0.9,1.0,1.1,1.5\}$. For both a linear RNN ($h_t = Wh_{t-1}$) and a $\tanh$ RNN driven by small random inputs, compute $\lVert\partial h_T/\partial h_0\rVert$ for $T=1,\dots,100$ (as a product of Jacobians). Plot the log-norm against $T$. **Report the fitted slope** for each $\rho$ and compare it with $\log\rho$ for the linear case. In two sentences, explain why the $\tanh$ network does not explode for $\rho=1.5$ the way the linear one does.

**C. Forecasting chaos with an echo state network.**
1. Simulate Lorenz-63 ($\sigma=10$, $\rho=28$, $\beta=8/3$) with RK4 at $\Delta t=0.01$; discard a transient; standardize each coordinate.
2. Build a reservoir with $H=500$ units, sparse random $W$ rescaled to spectral radius 0.9, random input weights, and update $h_t=\tanh(Wh_{t-1}+Ux_t)$. Drive it with the training trajectory and fit the readout $\hat x_{t+1} = Vh_t$ by ridge regression (Lecture 5); choose the ridge penalty on a validation segment.
3. Close the loop: feed predictions back as inputs. Over 20 test initial conditions, **report the mean valid prediction time** — the first time the normalized error exceeds 0.4 — in Lyapunov times, using $\lambda_{\max}\approx0.906$.
4. **Report the one-step error** as well, and explain in two sentences why it is a poor guide to the rollout result.

---

## Quiz-eligible facts

1. Any sequence distribution factorizes as $p(y_{1:T})=\prod_t p(y_t\mid y_{<t})$; an RNN parameterizes each conditional through a fixed-size state $h_t$.
2. A $k$-th order Markov (windowed) model ignores everything before $k$ steps; an RNN's memory is unbounded in principle.
3. Vanilla RNN: $h_t=\tanh(Wh_{t-1}+Ux_t+b)$, $\hat y_t=Vh_t+c$, with weights shared across time; recurrence parameters $H^2+HD+H$.
4. BPTT is reverse-mode AD on the unrolled network; $\delta_t = V^\top\partial\ell_t/\partial\hat y_t + W^\top g_{t+1}$; parameter gradients sum over time; memory grows linearly in $T$, and time steps cannot be parallelized.
5. $\partial h_T/\partial h_k=\prod_{t}\operatorname{diag}(1-h_t^2)W$, with norm at most $(\gamma s_{\max})^{T-k}$: vanishing if $s_{\max}<1/\gamma$; $s_{\max}>1/\gamma$ is necessary for explosion.
6. For a linear RNN, eigendirections of $W$ scale like $\lambda_i^{T-k}$.
7. A correct model of chaotic dynamics has exponentially growing gradients over long horizons; this cannot be removed by architecture.
8. Clipping handles exploding gradients; gating handles vanishing ones.
9. LSTM: $c_t=f_t\odot c_{t-1}+i_t\odot\tilde c_t$, $h_t=o_t\odot\tanh(c_t)$; $\partial c_t/\partial c_{t-1}\approx\operatorname{diag}(f_t)$; parameters $4H(H+D+1)$; initialize the forget bias near 1.
10. GRU has two gates and $3H(H+D+1)$ parameters; it acts as a leaky integrator with a learned, input-dependent time constant.
11. Teacher forcing is maximum likelihood with true inputs; free-running rollout feeds back predictions, causing exposure bias and compounding error. Evaluate rollouts, not only one-step error.
12. The encoder–decoder bottleneck is a single fixed-size context vector; attention replaces it with a weighted sum $\sum_j\alpha_{t,j}h_j$ over all encoder states with softmax weights.
13. Self-attention has $O(1)$ sequential operations and path length versus $O(n)$ for recurrence, at $O(n^2)$ cost.

---

## Practice problems

**P1.** Count the parameters of a vanilla RNN with $D=3$ inputs, $H=64$ hidden units and a 3-dimensional linear readout. How many does an LSTM of the same width have?

*Answer.* Recurrence: $64^2+64\cdot3+64 = 4352$; readout: $3\cdot64+3=195$; total 4547. LSTM recurrence: $4\cdot64\,(64+3+1)=17{,}408$; with the readout, 17,603.

**P2.** A linear RNN has $W=\operatorname{diag}(0.9,\,1.1)$. Compute $\partial h_{50}/\partial h_0$ and interpret each entry.

*Answer.* $\operatorname{diag}(0.9^{50},\,1.1^{50})\approx\operatorname{diag}(0.005,\,117)$. Information in the first component is essentially forgotten after 50 steps (vanishing); perturbations in the second are amplified more than a hundredfold (exploding).

**P3.** Why does setting the LSTM forget-gate bias to 1 rather than 0 help learn long dependencies? Quantify over 20 steps.

*Answer.* The cell-state gradient path is approximately $\prod_t f_t$. At initialization $f_t\approx\operatorname{sigm}(b_f)$: about 0.5 with $b_f=0$, giving $0.5^{20}\approx10^{-6}$; about 0.73 with $b_f=1$, giving $0.73^{20}\approx2\times10^{-3}$, three orders of magnitude more gradient reaching 20 steps back.

**P4.** An RNN surrogate of a chaotic flow achieves 0.1% one-step error but diverges from the truth after about 5 Lyapunov times in rollout. Is this a training failure? What should you report?

*Answer.* Not necessarily. Small errors grow like $e^{\lambda t}$ in chaotic dynamics, so even a near-perfect one-step model has bounded predictability. Report rollout error against horizon and the valid prediction time in Lyapunov units, alongside statistical measures (spectra, invariant distributions) for longer horizons.

**P5.** Give the two properties of recurrent networks that self-attention removes, and the price it pays.

*Answer.* Sequential computation over time ($O(n)$ steps that cannot run in parallel) and long paths between distant positions ($O(n)$ Jacobian factors, hence vanishing gradients). Self-attention makes both $O(1)$, at a cost of $O(n^2)$ in sequence length.

---

## References

- Bahdanau, D., Cho, K., & Bengio, Y. (2015). Neural machine translation by jointly learning to align and translate. *ICLR*.
- Bengio, S., Vinyals, O., Jaitly, N., & Shazeer, N. (2015). Scheduled sampling for sequence prediction with recurrent neural networks. *NeurIPS*.
- Cho, K., van Merriënboer, B., Gulcehre, C., Bahdanau, D., Bougares, F., Schwenk, H., & Bengio, Y. (2014). Learning phrase representations using RNN encoder–decoder for statistical machine translation. *EMNLP*.
- Elman, J. L. (1990). Finding structure in time. *Cognitive Science*, 14(2), 179–211.
- Gers, F. A., Schmidhuber, J., & Cummins, F. (2000). Learning to forget: Continual prediction with LSTM. *Neural Computation*, 12(10), 2451–2471.
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press. Ch. 10.
- Gu, A., & Dao, T. (2023). Mamba: Linear-time sequence modeling with selective state spaces. arXiv:2312.00752.
- Gu, A., Goel, K., & Ré, C. (2022). Efficiently modeling long sequences with structured state spaces. *ICLR*.
- Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation*, 9(8), 1735–1780.
- Jozefowicz, R., Zaremba, W., & Sutskever, I. (2015). An empirical exploration of recurrent network architectures. *ICML*.
- Karpathy, A. (2015). The unreasonable effectiveness of recurrent neural networks. karpathy.github.io.
- Kratzert, F., Klotz, D., Shalev, G., Klambauer, G., Hochreiter, S., & Nearing, G. (2019). Towards learning universal, regional, and local hydrological behaviors via machine learning applied to large-sample datasets. *Hydrology and Earth System Sciences*, 23, 5089–5110.
- Le, Q. V., Jaitly, N., & Hinton, G. E. (2015). A simple way to initialize recurrent networks of rectified linear units. arXiv:1504.00941.
- Mikhaeil, J. M., Monfared, Z., & Durstewitz, D. (2022). On the difficulty of learning chaotic dynamics with RNNs. *NeurIPS*.
- Mozaffar, M., Bostanabad, R., Chen, W., Ehmann, K., Cao, J., & Bessa, M. A. (2019). Deep learning predicts path-dependent plasticity. *PNAS*, 116(52), 26414–26420.
- Murphy, K. P. (2022). *Probabilistic Machine Learning: An Introduction*. MIT Press. Ch. 15.
- Olah, C. (2015). Understanding LSTM networks. colah.github.io.
- Pascanu, R., Mikolov, T., & Bengio, Y. (2013). On the difficulty of training recurrent neural networks. *ICML*.
- Pathak, J., Hunt, B., Girvan, M., Lu, Z., & Ott, E. (2018). Model-free prediction of large spatiotemporally chaotic systems from data: A reservoir computing approach. *Physical Review Letters*, 120, 024102.
- Sanchez-Gonzalez, A., Godwin, J., Pfaff, T., Ying, R., Leskovec, J., & Battaglia, P. (2020). Learning to simulate complex physics with graph networks. *ICML*.
- Shi, X., Chen, Z., Wang, H., Yeung, D.-Y., Wong, W.-K., & Woo, W.-C. (2015). Convolutional LSTM network: A machine learning approach for precipitation nowcasting. *NeurIPS*.
- Sutskever, I., Vinyals, O., & Le, Q. V. (2014). Sequence to sequence learning with neural networks. *NeurIPS*.
- Vaswani, A., et al. (2017). Attention is all you need. *NeurIPS*.
