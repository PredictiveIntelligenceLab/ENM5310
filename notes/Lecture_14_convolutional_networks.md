# Lecture 14 — Convolutional Neural Networks: Building Symmetry into the Architecture

**ENM 5310 — Data-driven Modeling and Probabilistic Scientific Computing**

**Builds on:** Lecture 4 (least squares, Toeplitz and structured matrices via the SVD view), Lecture 12 (the MLP; layers as warps of space; depth versus width), Lecture 13 (VJPs and backprop, initialization by fan-in, normalization layers, residual connections, data augmentation as an invariance assumption).

**Reading:** Murphy I Ch. 14 (§14.1–14.3). Goodfellow, Bengio & Courville (2016), *Deep Learning*, Ch. 9. Stanford CS231n notes, *Convolutional Neural Networks* (cs231n.github.io/convolutional-networks). Optional: Prince (2023), *Understanding Deep Learning*, Ch. 10–11; Olah, Mordvintsev & Schubert (2017), *Feature Visualization* (Distill).

**The point of this lecture.** What a trained network ends up computing depends on several choices: the architecture, the initialization, the data, the normalization, the objective, and the optimizer. Lecture 13 dealt with all of them except the first. Today is about the architecture: how to restrict an MLP so that it *assumes* what we know about the data before seeing any of it. For data on a grid, what we know is that the same local pattern means the same thing wherever it appears. The architecture that encodes exactly this is the convolutional network.

---

### Notation

| Symbol | Meaning |
|---|---|
| $x\in\mathbb{R}^{C\times H\times W}$ | input with $C$ channels on an $H\times W$ grid (RGB image: $C=3$) |
| $w\in\mathbb{R}^{K\times C\times F\times F}$, $b\in\mathbb{R}^K$ | $K$ filters of spatial size $F\times F$, and their biases |
| $s\in\mathbb{R}^{K\times H'\times W'}$ | output feature maps ("activation maps") |
| $P$, $S$ | zero-padding width, stride |
| $T_\tau$ | translation by $\tau$ grid points: $(T_\tau x)_i = x_{i-\tau}$ |
| $r_\ell$ | receptive field of a unit in layer $\ell$ |

**Learning objectives.** After this lecture you should be able to:

1. Write a convolution as a banded Toeplitz (or circulant) matrix, and count the parameters it saves over a dense layer.
2. Compute output shapes, parameter counts and receptive fields for a stack of convolutional and pooling layers.
3. Prove that convolution is translation equivariant, explain why it is the *only* linear map that is, and say which operations (zero padding, stride, pooling, flattening) break equivariance.
4. Distinguish equivariance from invariance, and architecture-level symmetry (convolution) from data-level symmetry (augmentation).
5. State the VJP of a convolution, and adapt initialization and normalization to convolutional layers.
6. Explain what each landmark architecture contributed (LeNet, AlexNet, VGG, Inception, ResNet, DenseNet, U-Net).
7. Interpret a convolution as a learnable finite-difference stencil and a residual CNN as a learned explicit time stepper, and explain why this makes CNNs resolution-dependent.

---

## 0. Why MLPs fail on images (8 min)

**Too many parameters.** Flatten a $32\times32\times3$ image into a vector of length 3072. A single hidden unit needs 3072 weights. A hidden layer of the same size as the input needs $3072^2\approx9.4$ million. For a $224\times224\times3$ image the first layer alone would have $150{,}528^2\approx2.3\times10^{10}$ weights. Data requirements, memory, and the risk of overfitting all scale accordingly.

**No useful inductive bias.** An MLP treats its input as an unordered list. Permute the pixels the same way in every image and an MLP learns equally well — which means it has no idea that neighboring pixels are related. Everything a person knows at a glance about images, it must learn from data. Lecture 12 made the same point geometrically: a shifted digit is far from the original in pixel space, so the MLP must learn "a 3 at the top-left" and "a 3 at the bottom-right" as separate facts.

**Spectral bias.** MLPs trained by gradient descent fit low-frequency components of the target long before high-frequency ones (Rahaman et al., 2019). Fine detail — edges, textures, sharp fronts — is learned last, if at all. This is a property of the training dynamics, not only of the architecture, and it will return when we discuss coordinate-based networks and PINNs.

**What we know about grid data.** Two things are true of images, time series, and fields sampled on a mesh. *Locality*: what happens at a point is mostly determined by its neighborhood. *Stationarity*: the same local pattern (an edge, a vortex, a shock) means the same thing wherever it appears. A convolutional network is an MLP whose weight matrices are forced to respect both.

---

## 1. Convolution as a constrained linear layer (15 min)

### 1.1 Definition

For a 1D signal $x$ and a kernel $w$ of width $F=2r+1$, deep learning uses the **cross-correlation**

$$
s_i = \sum_{k=-r}^{r} w_k\,x_{i+k}.
$$

Mathematical convolution flips the kernel, $\sum_k w_k x_{i-k}$. Since the kernel is learned, the flip is irrelevant, and everyone calls both "convolution." In 2D the same formula runs over a patch: $s_{i,j} = \sum_{a,b} w_{a,b}\,x_{i+a,j+b}$. The slide figure (Goodfellow et al., Fig. 9.1) shows a $2\times2$ kernel sliding over a $3\times4$ input to produce a $2\times3$ output, each entry a dot product of the kernel with one patch.

### 1.2 The matrix view

Convolution is linear in $x$, so $s = Wx$ for some matrix $W$. For a length-5 signal and a width-3 kernel, padded with zeros at both ends,

$$
W = \begin{pmatrix}
w_0 & w_1 & 0 & 0 & 0\\
w_{-1} & w_0 & w_1 & 0 & 0\\
0 & w_{-1} & w_0 & w_1 & 0\\
0 & 0 & w_{-1} & w_0 & w_1\\
0 & 0 & 0 & w_{-1} & w_0
\end{pmatrix}.
$$

Two constraints turn a dense matrix into this one:

1. **Locality (sparse connectivity).** $W_{ij}=0$ unless $|i-j|\le r$: the matrix is banded. Each output sees only $F$ inputs (its receptive field), and each input affects only $F$ outputs (slides: "sparse connectivity").
2. **Weight sharing.** $W_{ij} = w_{j-i}$ depends only on $j-i$: every diagonal is constant. The same $F$ numbers are reused at every position (slides: "parameter sharing").

A banded matrix with constant diagonals is a **Toeplitz** matrix. A dense layer on $n$ inputs has $n^2$ parameters; locality alone reduces this to $nF$; sharing reduces it to $F$, independent of $n$. The computational cost is $O(nF)$ rather than $O(n^2)$. The convolution can also be applied to an input of *any* length, which a dense layer cannot.

If we pad periodically instead of with zeros, the band wraps around the corners and $W$ becomes **circulant**. Every circulant matrix is diagonalized by the discrete Fourier transform, with eigenvalues equal to the DFT of the kernel, $\hat w$. This is the convolution theorem:

$$
w * x = \mathcal F^{-1}\big(\hat w\odot\hat x\big).
$$

A convolution acts on each Fourier mode independently, scaling mode $\omega$ by $\hat w(\omega)$. A small kernel has a smooth, low-resolution $\hat w$, so it can only shape the spectrum coarsely — a fact we will use in §7.

> **Remember.** A convolutional layer is a dense layer whose weight matrix is constrained to be banded (locality) and Toeplitz (sharing). Parameters fall from $n^2$ to $F$, cost from $O(n^2)$ to $O(nF)$, and the layer accepts inputs of any size.

---

## 2. The convolutional layer in practice (15 min)

### 2.1 Channels

Real inputs have several channels, and a layer produces several output feature maps. A filter spans *all* input channels: a $5\times5$ filter on an RGB image has shape $3\times5\times5$ and computes, at each position, a 75-dimensional dot product plus a bias (slides: "1 number"). Sliding it over the image gives one 2D activation map. With $K$ filters we get $K$ maps, stacked into a new "image" with $K$ channels:

$$
s_{k,i,j} = b_k + \sum_{c=1}^{C}\sum_{a=0}^{F-1}\sum_{b=0}^{F-1} w_{k,c,a,b}\;x_{c,\;Si+a-P,\;Sj+b-P}.
$$

Spatially the layer is a convolution; across channels it is a dense layer. Each filter is a learned detector, and each output channel is a map of where that feature occurs.

### 2.2 Padding, stride, and output size

Without padding, an $F\times F$ filter fits in $H-F+1$ positions, so every layer shrinks the map. **Zero padding** with $P$ pixels on each side restores the size; $P=(F-1)/2$ with stride 1 preserves it exactly ($F=3\Rightarrow P=1$, $F=5\Rightarrow P=2$). **Stride** $S$ moves the filter $S$ pixels at a time, downsampling by $S$. In general

$$
H' = \frac{H-F+2P}{S}+1, \qquad W' = \frac{W-F+2P}{S}+1, \qquad \text{channels out} = K,
$$

$$
\#\text{params} = K\,(C\,F^2+1), \qquad \#\text{multiply-adds} = K\,H'W'\,C\,F^2 .
$$

**Worked example (slides).** A $32\times32\times3$ input, 6 filters of size $5\times5$, $P=0$, $S=1$: output $28\times28\times6$; parameters $6(3\cdot25+1) = 456$. A dense layer producing the same $4704$ outputs from the same $3072$ inputs would need $3072\times4704\approx1.4\times10^7$ weights — about 30,000 times more.

Common settings: $F=3,S=1,P=1$ (the workhorse); $F=5,S=1,P=2$; $F=1,S=1,P=0$; and $K$ a power of two (32, 64, 128, ...).

### 2.3 $1\times1$ convolutions

A $1\times1$ convolution has no spatial extent. At each pixel it applies the same $K\times C$ matrix to the channel vector — an MLP layer applied independently at every position (Lin et al., 2013). It is used to mix channels and, above all, to reduce their number cheaply before an expensive $3\times3$ or $5\times5$ convolution (the Inception "bottleneck", §6).

### 2.4 Pooling

A pooling layer downsamples each channel independently with a fixed, parameter-free function over small windows. **Max pooling** with $F=2$, $S=2$ takes the maximum of each $2\times2$ block, halving height and width and keeping the depth: $224\times224\times64\to112\times112\times64$. Average pooling takes the mean instead. The output size is $(H-F)/S+1$, and there are no parameters. Pooling (or a strided convolution, which modern networks often use instead) reduces computation, enlarges receptive fields (§4), and builds in local invariance (§3).

### 2.5 The basic pattern

A classic CNN is a stack of [CONV → ReLU] blocks with occasional POOL layers, followed by one or more fully connected (FC) layers and a softmax:

$$
\text{INPUT}\;\to\;[\text{CONV}\to\text{ReLU}]\times n\;\to\;\text{POOL}\;\to\;\cdots\;\to\;\text{FC}\;\to\;\text{softmax}.
$$

As depth increases, spatial resolution falls and the number of channels rises: the representation trades *where* for *what*. The final FC layers are Lecture 12's MLP — logistic regression on learned features — applied to features that are now far better suited to images.

> **Remember.** $H'=(H-F+2P)/S+1$; parameters $K(CF^2+1)$; a filter spans all input channels; a $1\times1$ convolution is a per-pixel dense layer; pooling has no parameters.

---

## 3. Equivariance and invariance (14 min)

### 3.1 Definitions

A map $f$ is **equivariant** to a transformation $T$ if transforming the input transforms the output the same way, $f(Tx) = T f(x)$. It is **invariant** if transforming the input leaves the output unchanged, $f(Tx) = f(x)$. A segmentation map should be equivariant to translation: move the object and its mask moves with it. A class label should be invariant: a shifted cat is still a cat.

### 3.2 Convolution is translation equivariant

Let $(T_\tau x)_i = x_{i-\tau}$ shift a signal by $\tau$ (with periodic boundaries, so shifts are exact). Then

$$
\big(w*T_\tau x\big)_i = \sum_k w_k\,(T_\tau x)_{i+k} = \sum_k w_k\,x_{i+k-\tau} = (w*x)_{i-\tau} = \big(T_\tau(w*x)\big)_i .
$$

An elementwise nonlinearity also commutes with shifts, $\sigma(T_\tau z) = T_\tau\sigma(z)$, and a composition of equivariant maps is equivariant. So **an entire stack of stride-1 convolutions and activations is translation equivariant**: shift the input and every feature map shifts by the same amount. Detect an edge in one place and the same filter detects it everywhere.

### 3.3 Convolution is the *only* such linear map

The converse is the real reason convolution is used. Suppose a linear map $W$ on a periodic grid commutes with the unit shift, $WT = TW$.

1. Entrywise, $(TW)_{ij} = W_{i-1,j}$ and $(WT)_{ij} = W_{i,j+1}$, so $W_{i-1,j} = W_{i,j+1}$, i.e. $W_{i,j} = W_{i+1,j+1}$.
2. So every diagonal of $W$ is constant: $W_{ij} = w_{j-i}$, with indices taken modulo $n$.
3. That is exactly a circulant matrix — a convolution with kernel $w$.

**Any linear, translation-equivariant layer is a convolution.** Once we decide that the layer should treat every location the same way, the architecture is no longer a design choice. Locality (a small kernel) is a separate assumption, added on top.

### 3.4 From equivariance to invariance: pooling

Summing (or averaging) an equivariant feature map over all positions gives an invariant:

$$
g(x) = \sum_i f(x)_i \quad\Longrightarrow\quad g(T_\tau x) = \sum_i\big(T_\tau f(x)\big)_i = \sum_i f(x)_i = g(x).
$$

This is **global average pooling**, the standard way modern CNNs turn equivariant feature maps into an invariant prediction. *Local* max pooling is only approximately invariant: a shift smaller than the pooling window often leaves the maximum unchanged.

Pooling over *channels* rather than positions can produce invariance to other transformations. The slide example (Goodfellow et al., Fig. 9.9): three filters, each tuned to a "5" at a different orientation, followed by a max over the three. Whichever orientation appears, the pooled unit responds. Here the invariance is learned, not built in.

### 3.5 What breaks equivariance

The derivation in §3.2 assumed periodic boundaries and stride 1. Real networks violate both.

- **Zero padding** treats the boundary differently from the interior, so equivariance fails near the edges. (Networks can even use the padding to learn absolute position.)
- **Stride and pooling** subsample. After a stride-2 layer, a 1-pixel input shift does not correspond to any shift of the output grid; the network is equivariant only to shifts by multiples of the total stride. Subsampling without low-pass filtering also aliases high frequencies (Zhang, 2019).
- **Flattening into FC layers** discards spatial structure entirely. Global average pooling preserves invariance; flattening does not.
- **Rotation and scale** were never part of the deal. Convolution with an ordinary kernel is equivariant to translation only.

### 3.6 Two ways to encode a symmetry

Lecture 13 introduced data augmentation as a way to tell a model about an invariance: add transformed copies of the training data. Convolution tells the model through the architecture. The two are complementary.

| | Data augmentation | Architectural symmetry |
|---|---|---|
| How | Train on transformed copies of the data | Constrain the weights (sharing) |
| Guarantee | Approximate, learned; holds only near the training data | Exact, by construction, for every input |
| Cost | More training; must hold for the labels | Fewer parameters; must be derived per symmetry group |
| Typical use | Rotation, scale, lighting, elastic deformation, noise | Translation (convolution); rotations via group convolutions (Cohen & Welling, 2016) |

In practice CNNs use both: convolution for translation, augmentation for everything else. The slides list the standard image augmentations — rotation, translation, rescaling, shearing, mirroring, noise, lighting, elastic distortions. Two cautions from Lecture 13 apply. Augmentations that create new pixel regions (rotating by a non-multiple of $90^\circ$, translating) must fill them with something, and a constant fill the network can recognize becomes a spurious feature. And the transformation must preserve the label: mirroring is fine for cats, wrong for text, and for a physical vector field it requires transforming the vector components as well.

> **Remember.** Convolution is translation *equivariant*; global pooling makes it *invariant*. Any linear translation-equivariant map is a convolution. Zero padding, stride and flattening break equivariance. Architecture encodes symmetry exactly; augmentation encodes it approximately.

---

## 4. Receptive fields and hierarchical features (8 min)

A unit in the first layer sees an $F\times F$ patch. A unit in the second layer sees $F\times F$ units of the first, each of which sees a patch, so its **receptive field** is larger (slides: Goodfellow et al., Fig. 9.4). For a stack of stride-1 layers with kernel size $F$,

$$
r_L = 1 + L(F-1),
$$

and with strides $S_i$,

$$
r_\ell = r_{\ell-1} + (F_\ell-1)\prod_{i<\ell}S_i, \qquad r_0=1 .
$$

Strides and pooling make the receptive field grow geometrically rather than linearly, which is why deep CNNs reach global context in a modest number of layers. (The *effective* receptive field — the region that actually influences a unit appreciably — is much smaller than this bound and roughly Gaussian; Luo et al., 2016.)

**Why stacks of small kernels.** Two $3\times3$ layers have a $5\times5$ receptive field with $2\cdot9C^2=18C^2$ weights, against $25C^2$ for one $5\times5$ layer. Three $3\times3$ layers reach $7\times7$ with $27C^2$ weights against $49C^2$. The stack has fewer parameters *and* more nonlinearities. This is the argument behind VGG (§6), and it is Lecture 12's depth-versus-width story in a convolutional setting.

**What the layers learn.** Feature visualization — optimizing an input image to maximally excite a chosen channel — shows a consistent hierarchy in trained image classifiers (slides; Olah et al., 2017): oriented edges and color contrasts in the first layers, then textures, patterns, object parts, and finally whole objects. Each level is built by composing the level below over a larger receptive field. Lecture 12's argument that depth pays off for compositional functions is exactly what images reward.

---

## 5. Training CNNs: what changes from Lecture 13 (8 min)

Almost nothing. A CNN is a composition of linear maps and elementwise nonlinearities, and everything in Lecture 13 applies. Three adjustments follow from the convolutional structure.

**Backpropagation.** A convolution is $s = Wx$ with $W$ Toeplitz, so its VJPs follow from Lecture 13's linear-layer rule:

- *Input cotangent:* $\bar x = W^\top\bar s$. The transpose of a Toeplitz matrix is Toeplitz with the kernel reversed, so $\bar x$ is $\bar s$ convolved with the flipped kernel. This operation is called a **transposed convolution**, and run forward, it is also how decoders upsample (§6, U-Net).
- *Kernel cotangent:* $\bar w_k = \sum_i \bar s_i\,x_{i+k}$, the correlation of the input with the output cotangent. The sum runs over *every* position because the same weight is used at every position — sharing means each use contributes to the gradient, exactly as $\bar x_1$ accumulated contributions in Lecture 13's worked example.

**Initialization.** A unit's fan-in is now $n_{\text{in}} = C\,F^2$, the number of inputs to each dot product. He initialization for a ReLU CNN is $\operatorname{Var}[w] = 2/(CF^2)$.

**Normalization.** **Batch normalization** in a CNN computes one mean and variance *per channel*, pooled over the batch and all spatial positions, which keeps the layer translation equivariant. It was central to training deep CNNs (Ioffe & Szegedy, 2015). With small batches — common for large scientific fields that fit only a few samples per GPU — its statistics become noisy; **group normalization** (Wu & He, 2018) or layer normalization avoid the dependence on batch size.

---

## 6. The evolution of architectures (15 min)

The architectures on the slides read as a sequence of answers to two questions: how to go deeper, and how to spend parameters well.

| Architecture | Year | Size | Key idea |
|---|---|---|---|
| LeNet-5 (LeCun et al.) | 1998 | ~60k | CONV–POOL–CONV–POOL–FC for digits; the template |
| AlexNet (Krizhevsky et al.) | 2012 | ~60M | LeNet scaled up: ReLU, dropout, augmentation, GPUs; won ILSVRC 2012 |
| VGG-16 (Simonyan & Zisserman) | 2014 | ~138M | Only $3\times3$ convolutions, 16–19 layers; depth as the main lever |
| GoogLeNet / Inception (Szegedy et al.) | 2014 | ~12× fewer params than AlexNet | Parallel $1\times1$, $3\times3$, $5\times5$ branches; $1\times1$ bottlenecks; global average pooling |
| ResNet (He et al.) | 2015 | ResNet-50: ~25M | Residual connections; 152 layers; won ILSVRC 2015 |
| DenseNet (Huang et al.) | 2016–17 | small | Each layer receives *all* earlier feature maps, concatenated |
| U-Net (Ronneberger et al.) | 2015 | — | Encoder–decoder with skip connections for dense, per-pixel output |

**LeNet → AlexNet.** The architecture barely changed in fourteen years. What changed was everything from Lecture 13 — ReLU, dropout, initialization, augmentation — plus ImageNet-scale data and GPUs.

**VGG: where the parameters are.** VGG-16's convolutional layers hold only about 15M of its 138M parameters. The first FC layer alone maps $7\times7\times512 = 25{,}088$ inputs to 4096 outputs: $25{,}088\times4096\approx103$M weights. That is a clean illustration of §0: dense layers are where parameter counts explode.

**Inception: spend less, see more.** Each Inception module runs several filter sizes in parallel and concatenates the results, letting the network choose its scale. $1\times1$ convolutions reduce channels before the expensive branches. Replacing the final FC layers with **global average pooling** removed most of VGG's parameters and gave translation invariance for free (§3.4).

**ResNet: going deeper by learning corrections.** He et al. observed that a plain 56-layer CNN had higher *training* error than a 20-layer one. That is an optimization failure, not overfitting, since a deeper network could always copy the shallower one and set its extra layers to the identity. Residual blocks $h+F(h)$ make the identity the default. Lecture 13 showed why this fixes the gradient: the backward product $\prod(I+J_F^\top)$ always contains an identity path. Residual connections are now in essentially every deep architecture, transformers included.

**DenseNet** pushes the idea further by concatenating rather than adding. Every layer sees all earlier feature maps, which encourages feature reuse and keeps each layer narrow.

**U-Net: dense outputs.** Classification needs one label per image. Segmentation, denoising, super-resolution — and predicting a physical field from another field — need one output per pixel. U-Net pairs a downsampling encoder (large receptive field, coarse resolution) with an upsampling decoder built from transposed convolutions, and **skip connections** copy each encoder level's features to the matching decoder level, so that fine spatial detail lost in downsampling is restored. The structure closely resembles a multigrid V-cycle: smooth and restrict on the way down, prolong and correct on the way up. U-Nets are the backbone of most diffusion models and a strong baseline for PDE surrogates (Gupta & Brandstetter, 2022).

**Where things stand.** Since 2020, vision transformers have matched or overtaken CNNs on large-scale benchmarks. But CNNs modernized with transformer-era training recipes perform comparably (ConvNeXt; Liu et al., 2022), and when data are limited, convolution's built-in translation equivariance is a significant advantage. In scientific ML, convolutional backbones remain standard on regular grids.

---

## 7. CNNs in scientific computing (7 min)

**A convolution is a learnable stencil.** The five-point finite-difference Laplacian on a grid of spacing $h$ is a $3\times3$ convolution:

$$
\Delta u \approx \frac{1}{h^2}\begin{pmatrix}0&1&0\\1&-4&1\\0&1&0\end{pmatrix} * u .
$$

Every linear constant-coefficient finite-difference operator is a convolution. A convolutional layer is therefore a *learned* stencil, and a stack of them with nonlinearities is a learned nonlinear local operator.

**A residual CNN is a learned explicit time stepper.** For $\partial_t u = \mathcal N(u)$, forward Euler gives $u^{n+1} = u^n + \Delta t\,\mathcal N(u^n)$. A residual block computes $h + F(h)$. With convolutional $F$, a ResNet is a sequence of explicit steps of a learned local PDE, which is how PDE-Net (Long et al., 2018) and many learned simulators are built.

**Boundary conditions are padding choices.** Circular padding corresponds to periodic boundaries; zero or constant padding to Dirichlet conditions; reflection padding approximates zero-flux conditions. For periodic physics, use circular padding — otherwise the network sees an artificial boundary that is not there.

**The limitation: resolution dependence.** The weights of a $3\times3$ stencil encode the grid spacing ($1/h^2$ above). Train a CNN on a $64\times64$ grid and apply it to $128\times128$, and each filter now acts over half the physical distance; the learned operator changes. A CNN learns a map between *grids*, not between *functions*. And in Fourier terms (§1.2), a small kernel shapes the spectrum only coarsely, so long-range interactions require many layers. Both limitations motivate the operator-learning architectures later in the course — the Fourier neural operator, for instance, parameterizes $\hat w$ directly in Fourier space, which makes it global and largely independent of resolution (Li et al., 2021).

> **Remember.** Convolution = stencil; residual CNN = explicit time stepper; padding = boundary condition. CNNs are tied to the grid they were trained on, which is what neural operators set out to fix.

---

## Code Checkpoint 14

NumPy only. Each item asks you to report a number.

**A. Convolution from scratch.**
1. Implement `conv2d(x, w, b, stride, pad, mode)` for inputs of shape $(C,H,W)$ and filters $(K,C,F,F)$, with `mode` in {`zero`, `circular`}, first with explicit loops and then with an im2col reshaping followed by one matrix multiply. **Report the speedup** of the vectorized version on a $3\times64\times64$ input with 16 filters of size $3\times3$.
2. For a 1D input of length 8 and a width-3 kernel, build the explicit Toeplitz matrix (zero padding) and circulant matrix (circular padding) and verify they reproduce `conv2d`. **Report the eigenvalues** of the circulant matrix and confirm they equal the DFT of the zero-padded kernel.
3. Implement the input and kernel VJPs of §5 and check them against central finite differences. **Report the maximum relative error.**

**B. Testing equivariance.** Build a random 4-layer CNN with ReLU, 8 channels per layer, and $3\times3$ kernels. For a random $1\times32\times32$ input, measure the relative equivariance error $\lVert f(T_\tau x) - T_\tau f(x)\rVert/\lVert f(x)\rVert$ for $\tau=1$ pixel in four settings: (i) circular padding, stride 1; (ii) zero padding, stride 1; (iii) circular padding with one stride-2 layer, for $\tau=1$ and $\tau=2$ (compare the output with a shift of $\tau/2$); (iv) setting (i) followed by global average pooling, measuring the *invariance* error $|g(T_\tau x)-g(x)|/|g(x)|$. **Report the four numbers** and explain each in one sentence.

**C. Learning a stencil by least squares.** On a periodic $64\times64$ grid with $h = 1/64$, generate random smooth fields $u$ (sums of a few low-frequency Fourier modes) and compute $\Delta u$ exactly from their Fourier representation. Fit a single $3\times3$ kernel $w$ so that $w * u\approx\Delta u$. This is linear least squares in the 9 kernel entries (Lecture 4), solved with im2col and `np.linalg.lstsq`.
1. **Report the learned kernel multiplied by $h^2$**, and compare it with the five-point stencil.
2. Apply the learned kernel unchanged to fields on a $128\times128$ grid. **Report the relative error** and the factor by which the predictions are off, and explain it in one sentence.

---

## Quiz-eligible facts

1. A convolutional layer is a dense layer with a banded (locality) and Toeplitz (weight-sharing) weight matrix; with periodic padding the matrix is circulant and is diagonalized by the DFT.
2. Output size $H' = (H-F+2P)/S+1$; parameters $K(CF^2+1)$; each filter spans all input channels; $P=(F-1)/2$ with $S=1$ preserves the spatial size.
3. A $1\times1$ convolution is a dense layer applied independently at each pixel; it mixes or reduces channels.
4. Convolution with stride 1 and periodic boundaries is translation equivariant, $f(T x)=Tf(x)$; global average pooling of an equivariant map is translation invariant.
5. Every linear translation-equivariant map on a periodic grid is a convolution.
6. Zero padding, stride, pooling and flattening break exact translation equivariance; plain convolution is not equivariant to rotation or scale.
7. Stride-1 receptive field: $r_L = 1+L(F-1)$; strides make it grow geometrically. Two $3\times3$ layers see $5\times5$ with $18C^2$ weights versus $25C^2$.
8. The input VJP of a convolution is a transposed convolution (the flipped kernel); the kernel gradient sums contributions from every position because weights are shared.
9. He initialization for a ReLU conv layer uses $\operatorname{Var}[w] = 2/(CF^2)$; batch normalization in CNNs uses per-channel statistics over the batch and spatial positions.
10. Residual connections let networks of 100+ layers train by making the identity the default; U-Net's skip connections restore spatial detail lost in downsampling.
11. A linear finite-difference stencil is a convolution; a residual CNN resembles an explicit time integrator; CNN weights are tied to the training grid's resolution.

---

## Practice problems

**P1.** A layer takes a $64\times64\times16$ input and applies 32 filters of size $3\times3$ with stride 2 and padding 1. Give the output shape and the number of parameters.

*Answer.* $H' = \lfloor(64-3+2)/2\rfloor+1 = 31+1 = 32$, so the output is $32\times32\times32$. Parameters $32(16\cdot9+1) = 4640$.

**P2.** What is the receptive field of a unit after four stride-1 $3\times3$ convolutions? After the same four layers if the second has stride 2?

*Answer.* $1+4\cdot2 = 9$. With stride 2 at layer 2: $r_1=3$, $r_2=5$, $r_3 = 5+2\cdot2 = 9$, $r_4 = 9+2\cdot2 = 13$.

**P3.** A CNN classifier ends by flattening its last feature map into a dense layer. Is its prediction invariant to translations of the input? What single change would make it (approximately) so?

*Answer.* No: flattening ties each weight to a specific position, so shifting the input changes which weights see which feature. Replace flattening by global average pooling over positions before the final layer.

**P4.** Why does a linear layer that treats every location identically have to be a convolution?

*Answer.* Commuting with the shift forces $W_{i,j}=W_{i+1,j+1}$, so every diagonal is constant: $W$ is circulant (Toeplitz with periodic wrap-around), which is a convolution.

**P5.** A $3\times3$ CNN trained to predict $\Delta u$ on a $64\times64$ grid is applied to a $128\times128$ grid of the same domain. Without retraining, by roughly what factor will its predictions be off, and why?

*Answer.* About a factor of 4 too small. The learned kernel approximates $\frac{1}{h^2}\times$ the stencil with $h=1/64$; on the finer grid the correct factor is $1/h^2$ with $h=1/128$, four times larger. The CNN learned a grid operator, not a continuous one.

---

## References

- Cohen, T., & Welling, M. (2016). Group equivariant convolutional networks. *ICML*.
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press. Ch. 9.
- Gupta, J. K., & Brandstetter, J. (2022). Towards multi-spatiotemporal-scale generalized PDE modeling. arXiv:2209.15616.
- He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. *CVPR*.
- Huang, G., Liu, Z., van der Maaten, L., & Weinberger, K. Q. (2017). Densely connected convolutional networks. *CVPR*.
- Ioffe, S., & Szegedy, C. (2015). Batch normalization. *ICML*.
- Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). ImageNet classification with deep convolutional neural networks. *NeurIPS*.
- LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning applied to document recognition. *Proc. IEEE*, 86(11), 2278–2324.
- Li, Z., et al. (2021). Fourier neural operator for parametric partial differential equations. *ICLR*.
- Lin, M., Chen, Q., & Yan, S. (2013). Network in network. arXiv:1312.4400.
- Liu, Z., et al. (2022). A ConvNet for the 2020s. *CVPR*.
- Long, Z., Lu, Y., Ma, X., & Dong, B. (2018). PDE-Net: Learning PDEs from data. *ICML*.
- Luo, W., Li, Y., Urtasun, R., & Zemel, R. (2016). Understanding the effective receptive field in deep convolutional neural networks. *NeurIPS*.
- Murphy, K. P. (2022). *Probabilistic Machine Learning: An Introduction*. MIT Press. Ch. 14.
- Olah, C., Mordvintsev, A., & Schubert, L. (2017). Feature visualization. *Distill*.
- Prince, S. J. D. (2023). *Understanding Deep Learning*. MIT Press. Ch. 10–11.
- Rahaman, N., et al. (2019). On the spectral bias of neural networks. *ICML*.
- Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional networks for biomedical image segmentation. *MICCAI*.
- Simonyan, K., & Zisserman, A. (2015). Very deep convolutional networks for large-scale image recognition. *ICLR*.
- Szegedy, C., et al. (2015). Going deeper with convolutions. *CVPR*.
- Wu, Y., & He, K. (2018). Group normalization. *ECCV*.
- Zhang, R. (2019). Making convolutional networks shift-invariant again. *ICML*.
