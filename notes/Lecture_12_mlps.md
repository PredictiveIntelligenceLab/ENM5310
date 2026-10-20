# Lecture 12 — From Chosen Features to Learned Features: The Multilayer Perceptron

**ENM 5310 — Data-driven Modeling and Probabilistic Scientific Computing**

**Builds on:** Lecture 1 (the $m^d$ cost of grids), Lecture 3 (Bernoulli and categorical distributions), Lectures 4–5 (maximum likelihood as minimizing KL to the empirical distribution; MAP; logistic regression and its gradient $\Phi^\top(\mu-y)$), Lecture 6 (gradient descent, SGD, Adam), Lecture 7 (fixed features $\phi(x)$; a Gaussian prior on weights is a Gaussian prior on functions with $k(x,x') = \tau^2\phi(x)^\top\phi(x')$; epistemic vs. aleatoric uncertainty), Lectures 8–9 (Gaussian processes, kernels, the $O(N^3)$ cost), Coding Assignment 1 (logistic regression on images).

**Reading:** Murphy I §13.1–13.2. Goodfellow, Bengio & Courville (2016), *Deep Learning*, §1.2 (historical trends). Olah (2014), *Neural Networks, Manifolds, and Topology* (blog post, colah.github.io). Optional: Prince (2023), *Understanding Deep Learning*, Ch. 3–4; Cybenko (1989), first two pages; Barron (1993), §I; Telgarsky (2016).

**What this lecture adds.** Every model in this course so far has had the form "a linear function of features we chose." Bayesian linear regression chose $\phi$ explicitly; a GP chose it implicitly through the kernel. This lecture asks what happens when the features themselves become parameters, learned by the same maximum likelihood you already know. The answer is the multilayer perceptron. We will see where the idea came from, what it looks like geometrically, what can be proved about its expressive power, and — just as important — what cannot. Lecture 13 then deals with the practical consequence: these models are much harder to train, and most of modern deep learning practice is about making training work.

---

### Notation

| Symbol | Meaning |
|---|---|
| $x\in\mathbb{R}^d$, $y$ | input, target |
| $\{(x_n,y_n)\}_{n=1}^N$ | training set |
| $q(x,y)$ | empirical distribution of the training set, $q(x,y)=\frac1N\sum_{n=1}^N\delta(x-x_n)\,\delta(y-y_n)$ (defined in §3) |
| $L$ | number of layers (hidden layers plus output layer) |
| $H_\ell$ | width of layer $\ell$; $H_0=d$, $H_L$ = output dimension |
| $W^{(\ell)}\in\mathbb{R}^{H_\ell\times H_{\ell-1}}$, $b^{(\ell)}\in\mathbb{R}^{H_\ell}$ | weights and biases of layer $\ell$ |
| $z^{(\ell)} = W^{(\ell)}h^{(\ell-1)}+b^{(\ell)}$ | pre-activation of layer $\ell$ |
| $h^{(\ell)} = \sigma(z^{(\ell)})$ | activation (hidden representation) of layer $\ell$; $h^{(0)}=x$ |
| $\sigma(\cdot)$ | activation function, applied elementwise |
| $f(x;\theta) = z^{(L)}$ | network output (no activation on the last layer) |
| $\theta$ | all weights and biases |
| $\operatorname{sigm}(a) = 1/(1+e^{-a})$ | logistic sigmoid (written $\operatorname{sigm}$ to avoid a clash with $\sigma$) |

Vectors are columns; one example at a time unless stated. For a minibatch $X\in\mathbb{R}^{B\times d}$ stacked as rows, the same layer reads $H^{(\ell)} = \sigma\big(H^{(\ell-1)}W^{(\ell)\top} + \mathbf{1}b^{(\ell)\top}\big)$.

---


**Learning objectives.** After this lecture you should be able to:

1. Explain the three ways fixed features fail (dimension, geometry, cost) and what learned features buy and cost.
2. Place the perceptron, backpropagation, universal approximation, AlexNet and the transformer on a timeline, and say what obstacle ended each of the first two waves.
3. Write down an MLP, count its parameters, and state its model, training objective and inference step, identifying which parts are inherited unchanged from logistic regression.
4. Explain why an MLP's training loss is nonconvex using the permutation symmetry of hidden units.
5. Describe a layer geometrically (SVD of the affine map, then an elementwise squash or fold) and explain why width-2 `tanh` networks cannot separate a disk from a surrounding annulus.
6. Build a one-hidden-layer ReLU network that interpolates a 1D function, and derive its $O(H^{-2})$ error.
7. State the universal approximation theorem precisely, and list the three things it does not guarantee.
8. Explain why the rate of approximation, not the possibility of it, is what distinguishes learned features from fixed ones, and give an example where depth beats width exponentially.
9. Show that a one-hidden-layer network with random weights becomes a GP as its width grows.

---

## 0. Where we are (4 min)

Look back at the models of Lectures 4–11 and they share a single template:

$$
p(y\mid x,w) = \text{(likelihood)}\big(y \,\big|\, w^\top\phi(x)\big), \qquad \phi \text{ chosen by you.}
$$

Linear regression used $\phi(x)=x$; Coding Assignment 1 used raw pixels, which is the same thing; Lecture 7 used polynomial or RBF features; Lectures 8–11 used a kernel, which Lecture 7 showed is $k(x,x')=\tau^2\phi(x)^\top\phi(x')$ for some feature map, possibly infinite-dimensional. Because the unknowns $w$ enter linearly, everything was tractable: closed-form or convex training, Gaussian posteriors, calibrated predictive variances.

The modeling skill in all of these was **choosing $\phi$** — or, equivalently, choosing the kernel and its length scales. Today we stop choosing.

---

## 1. Why learn features? (8 min)

Fixed features fail in three distinct ways, and it is worth keeping them apart because they have different remedies.

**Dimension.** A basis built by tensor products — $m$ functions per coordinate — has $m^d$ elements. This is the same count that killed quadrature grids in Lecture 1. For $d=2$ it is harmless; for a $28\times28$ image, $d=784$, and no tensor-product basis can even be written down. Kernels avoid the explicit count but not the underlying problem: a stationary kernel's notion of "nearby" is Euclidean distance in $\mathbb{R}^d$, and in high dimensions almost every pair of points is far apart.

**Geometry.** Even when the dimension is manageable, Euclidean distance in the raw input space may be the wrong notion of similarity. Shift a handwritten digit two pixels to the right and its pixel vector moves far from the original — often farther than a different digit drawn in the same place. A squared-exponential kernel on pixels therefore believes that a shifted "3" is a different object. Logistic regression on pixels, as in Coding Assignment 1, has an even more restrictive geometry: its decision boundary is a single hyperplane in pixel space. Whatever you did in that assignment to push accuracy up, you did by changing $\phi$.

**Cost.** GP training costs $O(N^3)$ and each prediction costs $O(N)$. With $N = 6\times10^4$ images, a single Cholesky factorization is of order $10^{14}$ floating-point operations, and modern datasets are larger by factors of $10^3$ or more.

**The proposal.** Make the features learnable:

$$
f(x;\theta) = w^\top\phi(x;\vartheta) + b, \qquad \theta = (w,b,\vartheta),
$$

and fit everything, readout and features together, by maximum likelihood. If $\phi$ is built from cheap, differentiable, composable pieces, training can use SGD (Lecture 6), whose cost per epoch is linear in $N$ and which runs on hardware built for matrix multiplication.

**The bill.** Be clear about what this gives up. The unknowns no longer enter linearly, so the loss is nonconvex and there is no closed form. There is no conjugate posterior, so the default output is a point estimate with no epistemic uncertainty. And the model has many more design choices — architecture, initialization, optimizer, schedule — each of which can silently break training. Lecture 13 is about paying this bill.

This trade is the one Sutton's *Bitter Lesson* (Week 0) argues has always won in the long run: general methods that scale with data and compute eventually beat methods that encode what we know. The scientific counterpoint (Welling, also Week 0) is that knowledge still matters — and over the coming weeks you will see it return, not as hand-chosen features, but as **architectural structure**: convolutions encode translation symmetry, operators encode resolution independence, attention encodes permutation structure.

---

## 2. A short history: three waves (12 min)

The ideas behind deep learning are old. What has changed, repeatedly, is whether they could be made to work. The standard periodization (Goodfellow et al., 2016, §1.2) has three waves.

| Wave | Years | Core idea | What made it possible | Why it stalled |
|---|---|---|---|---|
| Cybernetics | 1940s–1960s | Neurons as threshold units; learning as weight adjustment | Simple learning rules for single linear units | A single linear unit cannot represent XOR; no way to train multiple layers |
| Connectionism | 1980s–mid 1990s | Networks of many simple units; distributed representations | Backpropagation | Deep networks would not train; small datasets; convex kernel methods matched or beat them |
| Deep learning | 2006–present | Many layers, trained end to end | Large labeled datasets, GPUs, and a set of training techniques | (ongoing) |

**First wave: cybernetics.** McCulloch and Pitts (1943) modeled a neuron as a threshold on a weighted sum of binary inputs and showed networks of them could compute logical functions — but the weights were set by hand. Rosenblatt's **perceptron** (1958) added learning: update the weights whenever an example is misclassified. If the data are linearly separable, the rule provably converges in finitely many steps. At about the same time Widrow and Hoff's ADALINE (1960) trained a linear unit with the "least mean squares" rule, which is exactly SGD on squared error for linear regression — the algorithm of Lecture 6, sixty-five years earlier. Minsky and Papert's *Perceptrons* (1969) then made precise what a single layer cannot do: XOR is not linearly separable, so no single-layer perceptron computes it. Multilayer networks could, but nobody knew how to train their hidden weights. Funding and interest collapsed.

**Second wave: connectionism.** The missing algorithm was reverse-mode differentiation — computing the gradient of a scalar loss with respect to every weight in one backward sweep. It was described by Linnainmaa (1970) as a general automatic differentiation method and proposed for networks by Werbos (1974), but became widely known through Rumelhart, Hinton and Williams (1986), who showed that hidden units trained by **backpropagation** learn useful internal representations. A burst of results followed: LeCun and colleagues (1989) trained convolutional networks to read handwritten zip codes, later deployed to read bank checks; Cybenko (1989) and Hornik, Stinchcombe and White (1989) proved that one-hidden-layer networks are **universal approximators** (§5); Hochreiter and Schmidhuber (1997) introduced the LSTM.

Then the wave broke, for two reasons. The first was technical: Hochreiter (1991) and Bengio, Simard and Frasconi (1994) showed that gradients propagated through many layers tend to shrink or grow exponentially, so deep networks trained badly or not at all. The second was competitive: support vector machines and kernel methods (Cortes and Vapnik, 1995) and graphical models offered convex training, clean theory, and equal or better accuracy on the datasets of the day, which had thousands of examples, not millions. For roughly a decade, "neural networks" was not a phrase that helped a paper get accepted.

**Third wave: deep learning.** Hinton, Osindero and Teh (2006) showed that deep networks could be trained if each layer was first pretrained without labels, one at a time. That result revived interest, but its lasting contribution was to provoke the question: *why does plain backpropagation from a random initialization fail?* Glorot and Bengio (2010) — the paper at the center of Lecture 13 — traced much of the failure to how signals are scaled at initialization. Within a few years, layerwise pretraining was unnecessary: good initialization, non-saturating activations (ReLU), more data and faster hardware were enough.

The turning point was ImageNet. Deng et al. (2009) assembled a labeled image dataset two to three orders of magnitude larger than its predecessors, and the annual ILSVRC competition used 1.2 million training images in 1000 classes. In 2012, Krizhevsky, Sutskever and Hinton's **AlexNet** — a 60-million-parameter convolutional network with ReLUs and dropout, trained for about a week on two consumer GPUs — reached 15.3% top-5 test error against 26.2% for the next-best entry. Within three years the field had converged on deep networks for vision, speech and translation, and a stream of training techniques followed: Adam (2014), batch normalization and residual networks (2015), which allowed over a hundred layers.

**Since 2017.** The transformer (Vaswani et al., 2017) replaced recurrence with attention and turned out to scale: Kaplan et al. (2020) found that test loss falls as a smooth power law in model size, data and compute over many orders of magnitude. Some describe what followed — large pretrained "foundation" models, generative models of images and text, and now of molecules, materials and weather — as a fourth wave. In science, AlphaFold 2 (2020–21) predicted protein structures at near-experimental accuracy, and learned weather models such as GraphCast (2023) and Aurora (2025) now match or beat operational numerical forecasts on many metrics at a tiny fraction of the cost. The 2024 Nobel Prizes in Physics (Hopfield, Hinton) and Chemistry (Baker; Hassabis and Jumper) recognized both the foundations and the scientific payoff.

**Three lessons to carry forward.**

1. **The ideas are old; what changed was data, compute, and training technique.** Multilayer networks trained by gradient descent are a 1986 idea. The 2012 network differed from a 1989 one mainly in scale and in a handful of training choices — ReLU, careful initialization, dropout, GPUs. Those choices are Lecture 13.
2. **Each winter followed a real technical obstacle, not just hype.** The first was representational (linear separability). The second was optimization (vanishing gradients) compounded by small data.
3. **Expressiveness was never the bottleneck.** Universal approximation was proved in 1989. The field still stalled for twenty years. Whether a network *can* represent a function and whether gradient descent *will find* it from finite data are different questions — which is exactly how §5 is organized.

---

## 3. The MLP in three steps (16 min)

### 3.1 The architecture

A **multilayer perceptron** (MLP), or fully connected feedforward network, alternates affine maps with elementwise nonlinearities:

$$
h^{(0)} = x,\qquad
z^{(\ell)} = W^{(\ell)}h^{(\ell-1)} + b^{(\ell)},\qquad
h^{(\ell)} = \sigma\big(z^{(\ell)}\big),\quad \ell=1,\dots,L-1,
$$

$$
f(x;\theta) = z^{(L)} = W^{(L)}h^{(L-1)} + b^{(L)}.
$$

The last layer has no activation; it produces the quantity that the likelihood needs (§3.2). Layers $1,\dots,L-1$ are **hidden layers**. The parameter count is

$$
|\theta| = \sum_{\ell=1}^{L} H_\ell\,(H_{\ell-1}+1).
$$

A network with input $d=784$, two hidden layers of width 256, and 10 outputs has $256\cdot785 + 256\cdot257 + 10\cdot257 = 269{,}322$ parameters, compared with $7{,}850$ for multinomial logistic regression on the same pixels.

**Why the nonlinearity is essential.** Without $\sigma$, the network is $W^{(L)}\cdots W^{(1)}x + \text{const}$, a single affine map. Depth adds nothing to what a linear network can represent; only the nonlinearity lets composition create new functions.

**Read the network as a feature map plus a readout.** Write $\phi(x;\vartheta) = h^{(L-1)}(x)$. Then $f(x;\theta) = W^{(L)}\phi(x;\vartheta)+b^{(L)}$ is *exactly* the linear predictor of Lectures 4–7, applied to features computed by the first $L-1$ layers. Everything you know about the last layer still holds; what is new is that $\phi$ is trained.

**Activation functions.**

| Name | $\sigma(z)$ | $\sigma'(0)$ | Range | Notes |
|---|---|---|---|---|
| Logistic | $\operatorname{sigm}(z)$ | $1/4$ | $(0,1)$ | Saturates both sides; not zero-centered. Historical; rarely used in hidden layers now. |
| tanh | $\tanh z$ | $1$ | $(-1,1)$ | Zero-centered; saturates for $\lvert z\rvert \gtrsim 3$. |
| ReLU | $\max(0,z)$ | — | $[0,\infty)$ | No saturation for $z>0$; zero gradient for $z<0$ ("dead" units). Piecewise linear. |
| Leaky ReLU | $\max(\alpha z, z)$, $\alpha\approx0.01$ | — | $\mathbb{R}$ | Keeps a small gradient for $z<0$. |
| GELU | $z\,\Phi(z)$, $\Phi$ the standard normal CDF | $1/2$ | $\approx[-0.17,\infty)$ | Smooth ReLU; the default in transformers. |
| SiLU / swish | $z\,\operatorname{sigm}(z)$ | $1/2$ | $\approx[-0.28,\infty)$ | Smooth ReLU. |
| sine | $\sin z$ | $1$ | $[-1,1]$ | Used for implicit neural representations and some physics-informed models. |

One choice matters specifically in scientific computing: if you will differentiate the network with respect to its **inputs** — to evaluate a PDE residual, a force from an energy, a velocity from a stream function — the activation must be smooth enough. A ReLU network is piecewise linear, so its second derivative in $x$ is zero almost everywhere. Use `tanh`, SiLU, GELU or sine there.

### 3.2 Step 1 — The probabilistic model

Nothing about the likelihood changes. The network replaces the linear predictor $w^\top\phi(x)$, and the output layer's job is to produce whatever parameters the likelihood needs:

| Task | Model | Output layer produces | Negative log-likelihood per example |
|---|---|---|---|
| Regression | $y\sim\mathcal{N}\big(f(x;\theta),\,\sigma^2\big)$ | the mean | $\frac{1}{2\sigma^2}(y-f)^2 + \frac12\log 2\pi\sigma^2$ |
| Binary classification | $y\sim\operatorname{Ber}\big(\operatorname{sigm}(f(x;\theta))\big)$ | one logit | $-y\log\operatorname{sigm}(f) - (1-y)\log(1-\operatorname{sigm}(f))$ |
| $C$-class classification | $y\sim\operatorname{Cat}\big(\operatorname{softmax}(f(x;\theta))\big)$ | $C$ logits | $-f_y + \log\sum_{c=1}^C e^{f_c}$ |

These are the likelihoods of Lectures 4–5; the losses are squared error, binary cross-entropy and softmax cross-entropy. **The loss is chosen by the likelihood, not by the network.** An MLP with a softmax output and no hidden layers *is* multinomial logistic regression.

### 3.3 Step 2 — Training

Let $q(x,y) = \frac1N\sum_{n=1}^N\delta(x-x_n)\,\delta(y-y_n)$ be the **empirical distribution** of the training set — the distribution that puts mass $1/N$ on each observed pair. As in Lecture 4, maximum likelihood minimizes the expected negative log-likelihood under $q$, which is the same as minimizing the KL divergence from $q$ to the model:

$$
\hat\theta_{\text{MLE}} = \arg\min_\theta\; \mathbb{E}_{(x,y)\sim q}\big[-\log p(y\mid x,\theta)\big]
= \arg\min_\theta\; \frac1N\sum_{n=1}^N \ell_n(\theta), \qquad \ell_n(\theta) = -\log p(y_n\mid x_n,\theta).
$$

A Gaussian prior $\theta\sim\mathcal{N}(0,\tau^2 I)$ turns this into MAP, adding $\frac{1}{2N\tau^2}\lVert\theta\rVert^2$ exactly as ridge did in Lecture 5 (Lecture 13 §5.2).

Two things differ from logistic regression.

**The gradient is no longer something you derive by hand per model.** It is computed by **backpropagation**, a systematic application of the chain rule that costs about as much as one extra forward pass. Lecture 13 §1 derives it for the MLP.

**The objective is nonconvex.** Logistic regression's loss was convex (Lecture 5), so any stationary point was the global minimum. An MLP's is not, and the reason is a symmetry you can see without computing anything.

*Why the MLP loss is nonconvex.*

1. Let $\theta^\star$ minimize the training loss $\mathcal{L}$. Pick a hidden layer $\ell$ and a permutation $\pi$ of its $H_\ell$ units. Permute the rows of $W^{(\ell)}$ and entries of $b^{(\ell)}$ by $\pi$, and the columns of $W^{(\ell+1)}$ by the same $\pi$. Call the result $\pi\theta^\star$.
2. The network computes the same function: unit $j$ now sits in position $\pi(j)$, but it still receives the same inputs and its output is still multiplied by the same outgoing weights. So $\mathcal{L}(\pi\theta^\star)=\mathcal{L}(\theta^\star)$; every $\pi\theta^\star$ is a minimizer, and there are $H_\ell!$ of them.
3. If $\mathcal{L}$ were convex, its set of minimizers would be convex, so the average $\bar\theta = \frac{1}{H_\ell!}\sum_\pi \pi\theta^\star$ would also be a minimizer.
4. In $\bar\theta$, every row of $W^{(\ell)}$ equals the average row, and every column of $W^{(\ell+1)}$ equals the average column. All $H_\ell$ units compute the same function: layer $\ell$ has effectively width one.
5. A width-one layer generally fits worse than a width-$H_\ell$ one, contradicting step 3. Hence $\mathcal{L}$ is not convex.

The same argument applies with sign flips of `tanh` units and positive rescalings of ReLU units. The landscape is full of equivalent minima related by symmetry. That by itself is harmless — any of them will do — but it means we have lost every guarantee convexity gave us, and we will need empirical judgment where we used to have theorems.

### 3.4 Step 3 — Inference

Prediction at a new input $x^\ast$ is a single forward pass: $\hat y = f(x^\ast;\hat\theta)$, or $\operatorname{softmax}(f(x^\ast;\hat\theta))$ for classification. Its cost depends on the network size, not on $N$ — compare $O(N)$ per GP prediction after an $O(N^3)$ factorization.

What you get is a **point estimate**. The predictive distribution $p(y\mid x^\ast,\hat\theta)$ contains the aleatoric noise model and nothing else; in the language of Lecture 7, the epistemic term is missing entirely. Far from the training data, a GP reverts to its prior and says so; a ReLU network extrapolates linearly, with whatever confidence its logits happen to imply. We return to this later in the course (ensembles, Bayesian neural networks). For now, remember that a well-trained MLP knows nothing about what it does not know.

---

## 4. Geometry: a network warps space (18 min)

The fastest route to intuition is to watch what a network does to its input space, layer by layer. Two resources make this concrete: Olah's 2014 post, which these notes follow, and the TensorFlow Playground (playground.tensorflow.org), where you can train small networks on 2D datasets and watch the hidden representations live.

### 4.1 Anatomy of one layer

A layer applies $x\mapsto\sigma(Wx+b)$. Split it into its parts.

**The affine map.** By the SVD (Lecture 4), $W = U\Sigma V^\top$. Applying $W$ means: rotate (or reflect) by $V^\top$; stretch each axis by a singular value, and — if $H_\ell \ne H_{\ell-1}$ — either embed into a higher-dimensional space or discard directions; rotate again by $U$. Then $b$ translates. Affine maps keep straight lines straight and parallel lines parallel.

**The nonlinearity.** $\sigma$ acts on each coordinate separately, so it bends the coordinate axes' directions differently from everything else.

- `tanh` **squashes**: it is nearly linear near the origin and compresses everything far away into the cube $(-1,1)^{H_\ell}$. It is a continuous bijection with a continuous inverse, so it bends space without tearing or gluing it.
- ReLU **folds and clips**: it leaves the positive orthant untouched and projects every point with a negative coordinate onto the face where that coordinate is zero. It is not injective — distinct inputs can map to the same output — so information is discarded.

How nonlinear the layer is depends on the **scale** of $Wx+b$. If pre-activations stay within $|z|\lesssim 0.5$, `tanh` is nearly the identity and the layer is nearly affine. If they are large, every unit saturates at $\pm1$ and the layer maps most of space to the corners of a cube. The interesting regime is in between, and keeping the network there is the subject of initialization (Lecture 13 §3).

### 4.2 What a classifier does

For binary classification the last layer computes $\operatorname{sigm}\big(w^\top h^{(L-1)}(x)+b\big)$. The decision boundary is $\{x : w^\top h^{(L-1)}(x) + b = 0\}$: a **hyperplane in the last hidden space**, which pulled back through the earlier layers becomes a curved surface in input space.

This gives a clean way to say what the hidden layers are for: **they deform the input space until the classes are linearly separable, so that the logistic regression on top can finish the job.** In Coding Assignment 1 you were the deformation, choosing $\phi$. Here gradient descent chooses it.

### 4.3 What a narrow network cannot do

Take a 2D dataset: class A is the disk $\lVert x\rVert\le1$; class B is the annulus $2.5\le\lVert x\rVert\le3.5$ surrounding it. Try to classify it with a `tanh` network whose hidden layers all have width 2.

1. If $W\in\mathbb{R}^{2\times2}$ is invertible, $x\mapsto Wx+b$ is a homeomorphism of the plane (continuous, bijective, continuous inverse). Elementwise `tanh` is a homeomorphism from $\mathbb{R}^2$ onto the open square $(-1,1)^2$. So each layer is a homeomorphism onto its image, and so is any composition of them.
2. Homeomorphisms preserve the property "A is enclosed by B": the image of the disk still lies in a hole of the image of the annulus. Any path from a point of the disk's image to the outside of the image must cross the annulus's image.
3. The final layer separates the classes with a line. Pick a point $p$ in the disk's image and follow the ray from $p$ pointing away from the line. It stays on $p$'s side of the line, and it must cross the annulus's image on its way out. So some annulus point lies on the disk's side. No line separates them.
4. If some $W$ is singular, that layer collapses the plane onto a line, and every later layer sees only a single linear projection $a^\top x$. Along the line $a^\top x = a^\top 0$ through the disk's center, disk and annulus points share the same projection and therefore receive the same output. That does not help either.

So no width-2 `tanh` network can reach 100% training accuracy on this dataset, no matter how deep or how long it is trained. What you see in practice is a network that stretches a thin sliver of the annulus across the disk and misclassifies it.

**Add a third dimension and the problem disappears.** With three hidden units the network can lift the plane into $\mathbb{R}^3$ — think of pushing the disk down and the annulus up — where a plane separates them. The lesson: **width is room to untangle.** Topology constrains what a network of a given width can do, regardless of training. (Code Checkpoint B constructs such a separator by hand with three ReLUs.)

### 4.4 ReLU networks: linear regions and folding

A ReLU network computes a continuous **piecewise linear** function. Each first-layer unit $\max(0,w_j^\top x+b_j)$ switches on or off across the hyperplane $w_j^\top x + b_j = 0$, and the network is linear on every cell of the resulting arrangement.

For a single hidden layer of $H$ units in $\mathbb{R}^d$, the number of linear regions is at most $\sum_{i=0}^{d}\binom{H}{i}$, the number of cells in an arrangement of $H$ hyperplanes: polynomial in $H$, and in 1D just $H+1$.

Depth changes this count exponentially, because a later layer acts on the *already folded* space. The cleanest example is the **tent map** on $[0,1]$:

$$
t(x) = 2\operatorname{ReLU}(x) - 4\operatorname{ReLU}(x-\tfrac12) = \begin{cases}2x, & 0\le x\le\tfrac12\\ 2-2x, & \tfrac12<x\le1\end{cases}
$$

It folds $[0,1]$ in half and stretches it back onto $[0,1]$. Composing it $k$ times, $t^{(k)} = t\circ\cdots\circ t$, folds the interval $k$ times and produces a sawtooth with $2^{k-1}$ teeth and $2^k$ linear pieces, using a network with $k$ hidden layers of 2 units. A one-hidden-layer network has at most $H+1$ pieces in 1D, so it needs at least $2^k-1$ units to represent the same function. Telgarsky (2016) turned this into a theorem: there are functions computed by deep, narrow ReLU networks that shallow networks cannot even *approximate* without exponentially many units.

Geometrically: **a shallow network cuts space with hyperplanes; a deep network folds space and cuts the folded result, so every later cut is replicated across every fold.** Montúfar et al. (2014) show that the maximal number of linear regions grows exponentially with depth for fixed width.

Two cautions. The maximum is not the typical: randomly initialized and trained networks usually have far fewer regions than the bound. And "more regions" is not the same as "better" — it is a statement about what depth makes *possible*, which is the subject of the next section.

---

## 5. Universal approximation: what it says and what it doesn't (22 min)

### 5.1 A constructive warm-up in one dimension

Before stating any theorem, let us build an approximation by hand. Take $f\in C^2([0,1])$ and nodes $x_i = ih$, $i=0,\dots,m$, $h=1/m$.

1. Let $g$ be the piecewise linear interpolant of $f$ at the nodes, with slope $s_i = \big(f(x_i)-f(x_{i-1})\big)/h$ on $[x_{i-1},x_i]$.
2. Claim:
$$
g(x) = f(x_0) + s_1\operatorname{ReLU}(x-x_0) + \sum_{i=1}^{m-1}(s_{i+1}-s_i)\operatorname{ReLU}(x-x_i), \qquad x\in[0,1].
$$
3. Check: at $x=x_0$ every ReLU is zero, so the value is $f(x_0)$. On $[x_{k-1},x_k]$ the active units are those with $x_i\le x_{k-1}$, so the slope is $s_1 + \sum_{i=1}^{k-1}(s_{i+1}-s_i) = s_k$. A continuous function with the right starting value and the right slope on every interval is $g$.
4. This is a one-hidden-layer ReLU network with $H=m$ units: input weights $1$, biases $-x_{j-1}$, output weights $s_1$ and $s_{j+1}-s_j$, output bias $f(x_0)$.
5. The standard interpolation estimate gives
$$
\sup_{x\in[0,1]}|f(x)-g(x)| \le \frac{h^2}{8}\sup|f''| = \frac{\sup|f''|}{8H^2}.
$$

Two remarks for the engineers in the room. First, each finite element hat function is three ReLUs, $\varphi_i(x) = \big[\operatorname{ReLU}(x-x_{i-1}) - 2\operatorname{ReLU}(x-x_i) + \operatorname{ReLU}(x-x_{i+1})\big]/h$, so **a one-hidden-layer ReLU network contains the P1 finite element space**. Second, if the biases are trained rather than fixed, the knots $-b_j/w_j$ move: a 1D ReLU network is a P1 finite element space with **learnable node locations**, an adaptive mesh fitted by gradient descent.

### 5.2 The theorem

**Universal approximation theorem** (Cybenko 1989 for sigmoidal $\sigma$; Hornik, Stinchcombe & White 1989; in this general form Leshno, Lin, Pinkus & Schocken 1993). Let $\sigma:\mathbb{R}\to\mathbb{R}$ be continuous. Then for every compact $K\subset\mathbb{R}^d$, every continuous $f:K\to\mathbb{R}$, and every $\varepsilon>0$, there exist $H$ and parameters $a_j,b_j\in\mathbb{R}$, $w_j\in\mathbb{R}^d$ such that

$$
\sup_{x\in K}\Big|\,f(x) - \sum_{j=1}^{H}a_j\,\sigma(w_j^\top x+b_j)\Big| < \varepsilon
$$

**if and only if $\sigma$ is not a polynomial.**

The "only if" is easy and instructive. If $\sigma$ is a polynomial of degree $p$, then each $\sigma(w_j^\top x+b_j)$ is a polynomial in $x$ of degree at most $p$, and so is any sum of them. No matter how large $H$ is, the network can only represent polynomials of degree $\le p$, which are not dense in $C(K)$. Every activation in the table in §3.1 is non-polynomial, so all of them give universal approximators. The "if" direction is functional analysis: Cybenko's proof shows, via Hahn–Banach, that a measure orthogonal to every ridge function $\sigma(w^\top x + b)$ must be zero.

### 5.3 What the theorem does not say

The theorem is an existence statement about one function and an unspecified $H$. It is silent on three things, and each one corresponds to a different source of error. Decompose the excess risk of the network you actually end up with, $\tilde f_N$, relative to the best possible predictor $f^\star$ (Bottou & Bousquet, 2008):

$$
\underbrace{R(\tilde f_N)-R(f^\star)}_{\text{total excess risk}} =
\underbrace{R(f_{\mathcal F})-R(f^\star)}_{\text{approximation}} +
\underbrace{R(f_N)-R(f_{\mathcal F})}_{\text{estimation}} +
\underbrace{R(\tilde f_N)-R(f_N)}_{\text{optimization}},
$$

where $f_{\mathcal F}$ is the best network in the architecture class $\mathcal F$, $f_N$ is the empirical risk minimizer on $N$ samples, and $\tilde f_N$ is what your optimizer returned.

| Question | Error term | Does UAT answer it? |
|---|---|---|
| How many units are needed for accuracy $\varepsilon$? | approximation | No — only that some finite number suffices |
| Can the right parameters be learned from $N$ samples? | estimation | No |
| Will gradient descent find them? | optimization | No |

The history in §2 is a history of the last two rows. Expressiveness was settled in 1989; it took two decades to make estimation and optimization work in practice.

### 5.4 Rates, dimension, and why learned features win

The first row — how many units — is where the real argument for learned features lives.

**Fixed features suffer the curse of dimensionality.** The construction of §5.1 extends to $d$ dimensions on a grid of spacing $h$, which needs on the order of $h^{-d}$ units (plus a little extra depth to build $d$-dimensional hats). The error is still $O(h^2)$, so with $H$ units

$$
\text{error} \sim H^{-2/d}, \qquad\text{equivalently}\qquad H \sim \varepsilon^{-d/2}.
$$

For $\varepsilon = 10^{-2}$: $H\sim 10$ in one dimension, $H\sim 10^{10}$ in ten. This is the $m^d$ count of Lecture 1 again, and it afflicts *any* scheme that tiles the domain with fixed local functions.

**Learned features can escape it, for the right functions.** Barron (1993) considered functions whose Fourier transform $\hat f$ satisfies $C_f = \int|\omega|\,|\hat f(\omega)|\,d\omega < \infty$. For any such $f$ and any probability measure $\mu$ on the ball of radius $r$, there is a one-hidden-layer sigmoidal network $f_H$ with $H$ units such that

$$
\int\big(f(x)-f_H(x)\big)^2\,\mu(dx) \le \frac{(2rC_f)^2}{H}.
$$

The rate $H^{-1/2}$ in $L^2$ error does not depend on $d$. Barron also proved the matching negative result: for **any fixed** set of $H$ basis functions — polynomials, splines, Fourier modes, RBFs with fixed centers — the worst-case error over this same class is at least of order $H^{-1/d}$. The difference is that the network chooses its basis functions (the directions $w_j$ and offsets $b_j$) to fit the particular $f$; a fixed basis must be ready for every $f$ in the class at once.

This is the precise version of §1's argument: **the advantage of learned features is a rate, not a possibility.** Both fixed and learned bases are universal; only one of them avoids an exponent of $1/d$ for this class.

Be honest about the fine print. The dimension has not vanished; it has moved into the constant $C_f$ and into the choice of function class. Some functions have $C_f$ growing exponentially in $d$, and for those nothing is gained. The theorem says that if the function you care about is "simple" in Barron's sense, a network can exploit that without being told what the simplicity is.

**Depth helps for compositional functions.** One hidden layer is universal, so depth is never needed for *possibility*. It is needed for *efficiency* when the target is built by composition — and the sawtooth of §4.4 is the proof that the gap can be exponential. Many functions in science and engineering have this structure: a quantity computed from a quantity computed from local interactions. Poggio et al. (2017) show that for hierarchically compositional functions, deep networks avoid the curse of dimensionality where shallow ones do not.

**Summary of §5.** Shallow networks are universal; so are polynomials. The reason to use networks is that, for structured targets, they need far fewer parameters than any fixed basis, and depth multiplies that advantage for compositional structure. None of this says that training will find those parameters.

---

## 6. Infinitely wide networks are Gaussian processes (7 min)

Lecture 7 showed that a Gaussian prior on the weights of a fixed-feature model is a Gaussian prior on functions. A random neural network has random *features* too — so what prior over functions does it define? Neal (1996) answered this for one hidden layer.

Take $f(x) = b + \sum_{j=1}^H a_j\,\sigma(w_j^\top x+c_j)$, with all parameters independent, $(w_j,c_j)$ drawn i.i.d. from some distribution, $a_j\sim\mathcal{N}(0,\sigma_a^2/H)$ and $b\sim\mathcal{N}(0,\sigma_b^2)$. Note the $1/H$ in the output-weight variance.

1. Each term $a_j\sigma(w_j^\top x+c_j)$ has mean zero (since $a_j$ is zero-mean and independent of the rest) and variance $\frac{\sigma_a^2}{H}\,\mathbb{E}\big[\sigma(w^\top x+c)^2\big]$.
2. The $H$ terms are i.i.d., so $f(x)-b$ is a sum of $H$ i.i.d. zero-mean terms whose total variance, $\sigma_a^2\,\mathbb{E}[\sigma(w^\top x+c)^2]$, does not depend on $H$.
3. By the multivariate central limit theorem, for any finite set of inputs $x_1,\dots,x_n$ the vector $\big(f(x_1),\dots,f(x_n)\big)$ converges to a Gaussian as $H\to\infty$, with zero mean and covariance
$$
k(x,x') = \sigma_b^2 + \sigma_a^2\,\mathbb{E}_{w,c}\big[\sigma(w^\top x+c)\,\sigma(w^\top x'+c)\big].
$$
4. Since every finite collection is jointly Gaussian with this covariance, $f$ is a GP with kernel $k$ — the definition from Lecture 8.

Compare Lecture 7's $k(x,x')=\tau^2\phi(x)^\top\phi(x')$: here $\phi_j(x) = \sigma(w_j^\top x+c_j)/\sqrt H$ are random features, and the kernel is their average. For ReLU and Gaussian $w$, $k$ has a closed form (the arc-cosine kernel of Cho & Saul, 2009). The same limit holds layer by layer for deep networks (Lee et al., 2018; Matthews et al., 2018): exact Bayesian inference in an infinitely wide network is GP regression.

Now the twist. Jacot, Gabriel and Hongler (2018) showed that if such a network is *trained by gradient descent* in the same scaling, then as $H\to\infty$ its hidden features barely move, and training is equivalent to kernel regression with a fixed kernel (the neural tangent kernel). In other words, the infinite-width limit gives back exactly the fixed-feature model §1 set out to escape. Real networks earn their keep in a regime where the features do move during training, and the theory of that regime is still being written. Keep this in mind as a reality check on any argument that treats networks as kernels.

---

## 7. Summary and bridge (3 min)

An MLP is logistic or linear regression on top of a learned feature map built from alternating affine maps and elementwise nonlinearities. The likelihood and the loss are unchanged from Lectures 4–5; what changes is that the features are trained, which buys a much better rate of approximation for structured targets, at the price of a nonconvex objective and a point estimate with no epistemic uncertainty.

Geometrically, each layer rotates, stretches and shifts space, then squashes or folds it; the hidden layers exist to untangle the data until a hyperplane suffices. Width gives room to untangle; depth multiplies the number of folds.

Universal approximation says a large enough network *can* represent any continuous function. It does not say how large, whether the parameters can be estimated from finite data, or whether gradient descent will find them. The history of the field is the history of those last two questions — and Lecture 13 is about how practice answers them: how gradients are computed, how signals are kept at the right scale, how the optimizer is configured, and how overfitting is controlled.

---

## Code Checkpoint 12

NumPy and Matplotlib only. Each item asks you to *report a number*, not just produce a picture.

**A. ReLU networks as finite elements.**
1. Implement the network of §5.1: given nodes and values, return input weights, biases and output weights, and a function that evaluates the network on a batch of inputs.
2. For $f(x) = \sin(2\pi x) + 0.5x$ on $[0,1]$ and $H\in\{4,8,16,\dots,512\}$, compute the maximum error on a grid of $10^5$ points. Plot error against $H$ on log–log axes and **report the fitted slope**. Compare with the prediction of §5.1.
3. Repeat for $f(x)=\sqrt{|x-0.3|}$. Report the slope and explain in two sentences why it differs. (Hint: which assumption of §5.1 fails, and where?)

**B. Watching a network warp space.**
1. Implement `mlp_forward(params, X, activation)` that returns the list of all hidden activations $[h^{(1)},\dots,h^{(L-1)}]$ for a batch $X\in\mathbb{R}^{B\times d}$.
2. Build a random width-2, depth-4 `tanh` network with $W^{(\ell)}_{ij}\sim\mathcal{N}(0, s^2/H_{\ell-1})$ and zero biases, for $s\in\{0.3,1,3\}$. Push a square grid of lines on $[-2,2]^2$ through it and plot the image after each layer. In two sentences, describe how $s$ controls how nonlinear the map is. (You will meet $s$ again in Lecture 13.)
3. Generate the disk/annulus dataset of §4.3 (disk radius 1, annulus radii 2.5 to 3.5, 500 points each). Construct by hand a network $\mathbb{R}^2\to\mathbb{R}^3\to\mathbb{R}$ with ReLU hidden units $h_j = \operatorname{ReLU}(u_j^\top x - c)$, where $u_1,u_2,u_3$ are unit vectors at $120^\circ$ to one another, followed by a logistic readout. Find a value of $c$ for which the classification is perfect, **verify 100% accuracy**, and explain why your $c$ works. (Hint: what is $\max_j u_j^\top x$ for $x$ at radius $r$?) Plot the 3D hidden representation.

**C. Depth versus width.**
1. Implement the tent map $t$ of §4.4 as a one-hidden-layer ReLU network with 2 units, and compose it $k$ times for $k=1,\dots,10$.
2. Count the linear pieces numerically: evaluate on a fine grid, compute finite-difference slopes, and count the slope changes. Plot $\log_2(\text{pieces})$ against $k$ and **report the fitted slope**.
3. Report the minimum number of hidden units a one-hidden-layer ReLU network needs to represent $t^{(10)}$ exactly, and the total number of hidden units in your deep network. One sentence on what this ratio means.

---

## Quiz-eligible facts

1. An MLP without nonlinearities is an affine map, regardless of depth.
2. The last layer of an MLP classifier is logistic (or multinomial logistic) regression on the learned features $h^{(L-1)}$; the loss is the negative log-likelihood of the chosen likelihood, unchanged from Lectures 4–5.
3. The MLP training loss is nonconvex; the permutation symmetry of hidden units is enough to prove it.
4. An MLP gives a point estimate; its predictive distribution has no epistemic term.
5. Each layer is an affine map (rotate–stretch–rotate–shift, by the SVD) followed by an elementwise squash (`tanh`) or fold/clip (ReLU).
6. Width-2 `tanh` networks cannot perfectly separate a disk from a surrounding annulus, because their layers are homeomorphisms or projections; a third dimension makes it possible.
7. A one-hidden-layer ReLU network in 1D can represent any piecewise linear interpolant with one unit per interior node; its error for $C^2$ functions is $O(H^{-2})$.
8. Universal approximation holds for any continuous activation that is not a polynomial; it guarantees existence of an approximation, not its size, its learnability from finite data, or that gradient descent will find it.
9. Fixed bases on a grid need $H\sim\varepsilon^{-d/2}$ units (curse of dimensionality); for Barron-class functions, one-hidden-layer networks achieve $L^2$ error $O(H^{-1/2})$ independent of $d$, while any fixed basis has worst-case error at least of order $H^{-1/d}$.
10. Composing a 2-unit tent map $k$ times gives $2^k$ linear pieces; a shallow network needs exponentially many units to match it.
11. A one-hidden-layer network with i.i.d. random weights and output-weight variance $\propto 1/H$ converges to a GP as $H\to\infty$.
12. The three waves: cybernetics (perceptron; ended by the XOR/linear-separability limit), connectionism (backpropagation; ended by vanishing gradients, small data, and kernel methods), deep learning (data, GPUs, training technique; AlexNet 2012).

---

## Practice problems

**P1.** A colleague builds a 10-layer network with identity activations and claims it is "more expressive than linear regression because it is deep." Is it? Is its training loss convex in the weights?

*Answer.* No: $W^{(10)}\cdots W^{(1)}x$ is a single linear map, so the function class is that of linear regression. But the loss is not convex in the individual matrices. The scalar case shows it: $(w_2w_1-1)^2$ has a saddle at $w_1=w_2=0$ and two separate curves of minima. Depth changes the landscape without changing the function class.

**P2.** How many hidden units does a one-hidden-layer ReLU network need to represent the piecewise linear interpolant of $f$ at 101 equally spaced nodes on $[0,1]$? If $f\in C^2$, by roughly what factor does the maximum error fall when you double the number of nodes? What happens to the required number of units in $d$ dimensions at fixed accuracy?

*Answer.* 100 units (one per interval: $s_1$ for the first, then one per slope change). Doubling the nodes halves $h$, so the error falls by about 4. On a $d$-dimensional grid the count scales as $h^{-d}$, so $H\sim\varepsilon^{-d/2}$.

**P3.** You train a `tanh` network with three hidden layers, each of width 2, on the disk-inside-annulus dataset. After a very long training run, training accuracy is stuck at 97%. Is the optimizer to blame? What single change would you make?

*Answer.* No. Every layer is a homeomorphism onto its image (or a projection if $W$ is singular), so the disk's image stays enclosed by the annulus's image and no line in the last hidden space separates them; 100% is unreachable for any training procedure. Increase the hidden width to 3 or more.

**P4.** Is a one-hidden-layer network with activation $\sigma(z) = z^2$ a universal approximator?

*Answer.* No. Each $(w_j^\top x+b_j)^2$ is a quadratic polynomial, so any sum of them is a polynomial of degree at most 2, however many units are used. Universality requires a non-polynomial activation.

**P5.** In §6 the output weights have variance $\sigma_a^2/H$. What happens to $\operatorname{Var}[f(x)]$ as $H\to\infty$ if you use variance $\sigma_a^2$ instead?

*Answer.* $\operatorname{Var}[f(x)] = \sigma_b^2 + H\sigma_a^2\,\mathbb{E}[\sigma(w^\top x+c)^2]$ grows linearly in $H$ and diverges, so there is no limit. The $1/H$ scaling keeps the output variance fixed as the network widens — the same idea as the initialization scaling of Lecture 13.

---

## References

- Barron, A. R. (1993). Universal approximation bounds for superpositions of a sigmoidal function. *IEEE Trans. Information Theory*, 39(3), 930–945.
- Bengio, Y., Simard, P., & Frasconi, P. (1994). Learning long-term dependencies with gradient descent is difficult. *IEEE Trans. Neural Networks*, 5(2), 157–166.
- Bottou, L., & Bousquet, O. (2008). The tradeoffs of large scale learning. *NeurIPS*.
- Cho, Y., & Saul, L. K. (2009). Kernel methods for deep learning. *NeurIPS*.
- Cybenko, G. (1989). Approximation by superpositions of a sigmoidal function. *Mathematics of Control, Signals and Systems*, 2, 303–314.
- Deng, J., et al. (2009). ImageNet: A large-scale hierarchical image database. *CVPR*.
- Glorot, X., & Bengio, Y. (2010). Understanding the difficulty of training deep feedforward neural networks. *AISTATS*.
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press. Ch. 1.
- Hinton, G. E., Osindero, S., & Teh, Y.-W. (2006). A fast learning algorithm for deep belief nets. *Neural Computation*, 18(7), 1527–1554.
- Hornik, K., Stinchcombe, M., & White, H. (1989). Multilayer feedforward networks are universal approximators. *Neural Networks*, 2(5), 359–366.
- Jacot, A., Gabriel, F., & Hongler, C. (2018). Neural tangent kernel: Convergence and generalization in neural networks. *NeurIPS*.
- Kaplan, J., et al. (2020). Scaling laws for neural language models. arXiv:2001.08361.
- Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). ImageNet classification with deep convolutional neural networks. *NeurIPS*.
- LeCun, Y., et al. (1989). Backpropagation applied to handwritten zip code recognition. *Neural Computation*, 1(4), 541–551.
- Lee, J., et al. (2018). Deep neural networks as Gaussian processes. *ICLR*.
- Leshno, M., Lin, V. Y., Pinkus, A., & Schocken, S. (1993). Multilayer feedforward networks with a nonpolynomial activation function can approximate any function. *Neural Networks*, 6(6), 861–867.
- Matthews, A. G. de G., et al. (2018). Gaussian process behaviour in wide deep neural networks. *ICLR*.
- McCulloch, W. S., & Pitts, W. (1943). A logical calculus of the ideas immanent in nervous activity. *Bulletin of Mathematical Biophysics*, 5, 115–133.
- Minsky, M., & Papert, S. (1969). *Perceptrons*. MIT Press.
- Montúfar, G., Pascanu, R., Cho, K., & Bengio, Y. (2014). On the number of linear regions of deep neural networks. *NeurIPS*.
- Murphy, K. P. (2022). *Probabilistic Machine Learning: An Introduction*. MIT Press. Ch. 13.
- Neal, R. M. (1996). *Bayesian Learning for Neural Networks*. Springer. Ch. 2.
- Olah, C. (2014). Neural networks, manifolds, and topology. colah.github.io.
- Poggio, T., et al. (2017). Why and when can deep—but not shallow—networks avoid the curse of dimensionality. *Int. J. Automation and Computing*, 14, 503–519.
- Prince, S. J. D. (2023). *Understanding Deep Learning*. MIT Press. Ch. 3–4.
- Rosenblatt, F. (1958). The perceptron: A probabilistic model for information storage and organization in the brain. *Psychological Review*, 65(6), 386–408.
- Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). Learning representations by back-propagating errors. *Nature*, 323, 533–536.
- Telgarsky, M. (2016). Benefits of depth in neural networks. *COLT*.
- Vaswani, A., et al. (2017). Attention is all you need. *NeurIPS*.
- Widrow, B., & Hoff, M. E. (1960). Adaptive switching circuits. *IRE WESCON Convention Record*.
