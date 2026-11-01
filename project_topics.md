# ENM 5310 Final Project Topics

Six areas where deep learning is changing how science and engineering get done. Each area below has a short description, starting references, key papers, and the tools worth learning. Choose an area, start from its references, and build your project from there.

---

## For every project

- [A Recipe for Training Neural Networks](https://karpathy.github.io/2019/04/25/recipe/) (Karpathy) — how to debug a model that silently fails
- [Deep Learning Tuning Playbook](https://github.com/google-research/tuning_playbook) (Google) — how to run hyperparameter studies that mean something
- [Weak baselines and reporting biases lead to overoptimism in machine learning for fluid-related PDEs](https://arxiv.org/abs/2407.07218) — why your baseline matters as much as your method
- Benchmarks: [PDEBench](https://arxiv.org/abs/2210.07182), [The Well](https://github.com/PolymathicAI/the_well)

---

## 1. Transformers and Attention for Scientific Data

Transformers are the default architecture across language, vision, weather, and now PDE foundation models. Attention is a learned, input-dependent kernel, closely related to the integral operators and GP kernels from earlier in the course.

**Start here:**
- [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/)
- [Let's build GPT from scratch](https://www.youtube.com/watch?v=kCc8FmEb1nY) and [nanoGPT](https://github.com/karpathy/nanoGPT) (Karpathy)

**Key papers:**
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [An Image is Worth 16x16 Words (ViT)](https://arxiv.org/abs/2010.11929)
- [RoFormer: Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
- Scaling laws: [Kaplan et al.](https://arxiv.org/abs/2001.08361), [Hoffmann et al. (Chinchilla)](https://arxiv.org/abs/2203.15556)
- In science: [Multiple Physics Pretraining](https://arxiv.org/abs/2310.02994), [Poseidon](https://arxiv.org/abs/2405.19101), [Aurora](https://arxiv.org/abs/2405.13063)

**Tools:** PyTorch or JAX/Flax, einops, Hugging Face Hub, Weights & Biases.

---

## 2. Generative Modeling: Diffusion and Flow Matching

Diffusion and flow matching dominate image, video, and molecular generation, and they power probabilistic weather forecasting. They are density estimators, a direct continuation of the probabilistic modeling thread of this course, and they produce samples rather than point predictions.

**Start here:**
- [Step-by-Step Diffusion: An Elementary Tutorial](https://arxiv.org/abs/2406.08929)
- [Flow Matching Guide and Code](https://arxiv.org/abs/2412.06264)

**Key papers:**
- [Denoising Diffusion Probabilistic Models (DDPM)](https://arxiv.org/abs/2006.11239)
- [Score-Based Generative Modeling through SDEs](https://arxiv.org/abs/2011.13456)
- [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747)
- [Stochastic Interpolants: A Unifying Framework for Flows and Diffusions](https://arxiv.org/abs/2303.08797)
- In science: [GenCast](https://arxiv.org/abs/2312.15796), [PDE-Refiner](https://arxiv.org/abs/2308.05732), [Diffusion Posterior Sampling](https://arxiv.org/abs/2209.14687)

**Tools:** [flow_matching](https://github.com/facebookresearch/flow_matching), torchdiffeq or diffrax.

---

## 3. Neural Operators and Learned Simulators

Neural operators learn maps between functions, from initial conditions, coefficients, or geometries to PDE solutions, so one trained model answers many queries. They are the core of fast surrogate modeling for design, control, and uncertainty quantification.

**Start here:**
- [Neural Operator: Learning Maps Between Function Spaces](https://arxiv.org/abs/2108.08481)

**Key papers:**
- [Fourier Neural Operator (FNO)](https://arxiv.org/abs/2010.08895)
- [DeepONet](https://arxiv.org/abs/1910.03193) and its [physics-informed variant](https://www.science.org/doi/abs/10.1126/sciadv.abi8605)
- [CViT: Continuous Vision Transformer for Operator Learning](https://arxiv.org/abs/2405.13998)
- [Transolver: A Fast Transformer Solver for PDEs on General Geometries](https://arxiv.org/abs/2402.02366)

**Tools:** [neuraloperator](https://github.com/neuraloperator/neuraloperator), [the_well](https://github.com/PolymathicAI/the_well).

---

## 4. Physics-Informed and Hybrid Models

When data is scarce but the physics is partly known, that knowledge can be built into the loss or the model structure. The ways these methods fail are as instructive as the ways they succeed.

**Start here:**
- [Physics-informed neural networks](https://www.sciencedirect.com/science/article/pii/S0021999118307125)
- [An Expert's Guide to Training Physics-informed Neural Networks](https://arxiv.org/abs/2308.08468)

**Key papers:**
- [PirateNets: Physics-informed Deep Learning with Residual Adaptive Networks](https://arxiv.org/abs/2402.00326)
- [ConFIG: Conflict-free Training of Physics Informed Neural Networks](https://arxiv.org/abs/2408.11104)
- [Neural Ordinary Differential Equations](https://arxiv.org/abs/1806.07366)
- [Universal Differential Equations for Scientific Machine Learning](https://arxiv.org/abs/2001.04385)

**Tools:** JAX + Optax, diffrax or torchdiffeq, [PySR](https://github.com/MilesCranmer/PySR).

---

## 5. Symmetry, Geometry, and Graphs

Physical systems have symmetries (rotation, translation, permutation) and often live on meshes, molecules, or the sphere rather than on grids. Building symmetry into a model is one of the most reliable ways to get more out of less data.

**Start here:**
- [Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges](https://arxiv.org/abs/2104.13478)
- [E(n) Equivariant Graph Neural Networks](https://arxiv.org/abs/2102.09844)

**Key papers:**
- [SchNet: Continuous-filter Convolutions for Quantum Interactions](https://arxiv.org/abs/1706.08566)
- [Learning Mesh-Based Simulation with Graph Networks (MeshGraphNets)](https://arxiv.org/abs/2010.03409)
- [GraphCast: Learning Skillful Medium-Range Global Weather Forecasting](https://arxiv.org/abs/2212.12794)

**Tools:** [PyTorch Geometric](https://github.com/pyg-team/pytorch_geometric), [e3nn](https://github.com/e3nn/e3nn).

---

## 6. Uncertainty, Decisions, and Inference

A surrogate is only useful if you know when to trust it. Simulations and experiments are expensive, so deciding which one to run next is often the real problem. This area connects deep learning to the GP, Bayesian, and Bayesian optimization material from the first half of the course.

**Start here:**
- [A Gentle Introduction to Conformal Prediction](https://arxiv.org/abs/2107.07511)
- [A Tutorial on Bayesian Optimization](https://arxiv.org/abs/1807.02811)

**Key papers:**
- [Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles](https://arxiv.org/abs/1612.01474)
- [Composite Bayesian Optimization in Function Spaces](https://arxiv.org/abs/2404.03099)
- [The Frontier of Simulation-Based Inference](https://arxiv.org/abs/1911.01429)

**Tools:** [BoTorch](https://botorch.org), GPyTorch, [sbi](https://github.com/sbi-dev/sbi).
