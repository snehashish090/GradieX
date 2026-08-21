<div align="center">
   <pre>
 ██████  ██████   █████  ██████  ██ ███████ ██   ██ 
██       ██   ██ ██   ██ ██   ██ ██ ██       ██ ██  
██   ███ ██████  ███████ ██   ██ ██ █████     ███   
██    ██ ██   ██ ██   ██ ██   ██ ██ ██       ██ ██  
 ██████  ██   ██ ██   ██ ██████  ██ ███████ ██   ██ 
   </pre>
</div>

<div align="center">

**A neural network library in C++17 — no dependencies, no framework.**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![C++17](https://img.shields.io/badge/C%2B%2B-17-00599C.svg)](https://en.cppreference.com/w/cpp/17)
[![Tests](https://img.shields.io/badge/tests-112%20passing-brightgreen.svg)](tests/test_gradiex.cpp)

</div>

---

GradieX runs both forward and backward passes to calculate the loss gradient and
update its internal weights. It is a single translation unit with no dependencies
beyond the C++17 standard library, so a trained model can be saved to disk and
loaded back into any C++ program — useful for inference wherever a Python runtime
is not an option.

The backward pass is verified against central-difference numerical gradients across
ten network configurations, to roughly $10^{-11}$ absolute. That check is why the
library can be trusted, and it runs on every commit.

```bash
g++ -std=c++17 -O2 -Wall -Wextra -pedantic main.cpp -o gradiex && ./gradiex
```

*Written by Snehashish Laskar. MIT licensed.*

---

## Contents

- [Capabilities](#capabilities)
- [Build, Run, Test](#build-run-test)
- [Notation](#notation)
- [Activations and Losses](#activations-and-losses)
- [Architecture](#architecture)
- [Data Flow](#data-flow)
- [Training](#training)
- [Quick Start](#quick-start)
- [Model File Format](#model-file-format)
- [Benchmarks](#benchmarks)
- [Invariants](#invariants)
- [Not Supported](#not-supported)
- [Distribute Binary via the Web App](#distribute-binary-via-the-web-app)
- [Contributing](#contributing)
- [License](#license)

---

## Capabilities

- Dense feedforward network with configurable hidden/output activations
- Forward pass across multiple hidden layers; backward pass verified against numerical gradients
- Activations: `sigmoid`, `tanh`, `ReLU`, `linear`, `leaky_ReLU`, `softmax` (output only)
- Losses: `mean_squared_error`, `binary_cross_entropy`, `categorical_cross_entropy`
- Multi-class classification via fused softmax + categorical cross-entropy
- Mini-batch training with per-epoch shuffling
- L2 weight decay and inverted dropout on hidden layers
- Train/validation split with per-epoch loss history
- Seeded, fully reproducible runs
- Model save/load with bit-exact round-trip
- Enum-dispatched activations (no string comparison on the hot path)

---

## Build, Run, Test

Build and run the examples:

```bash
g++ -std=c++17 -O2 -Wall -Wextra -pedantic main.cpp -o gradiex
./gradiex
```

`main.cpp` demonstrates two examples: XOR with per-sample SGD, and a three-class
spiral classifier that trains with mini-batches, saves itself, reloads, and confirms
the reloaded model predicts identically.

Run the test suite:

```bash
g++ -std=c++17 -O2 -Wall -Wextra -pedantic tests/test_gradiex.cpp -o test_gradiex
./test_gradiex
```

112 checks across 17 groups; exits non-zero on any failure. The suite includes a
central-difference gradient check over ten network configurations, so any change that
breaks backpropagation fails loudly. It `#include`s `main.cpp` and renames its `main`,
so there is no duplicated source to keep in sync.

---

## Notation

Layers are indexed by $l$, with $\mathbf{a}^{(0)} = \mathbf{x}$ the input and
$L$ the output layer.

| Symbol | Meaning |
|---|---|
| $\mathbf{x}$ | the input vector |
| $\mathbf{z}^{(l)}$ | pre-activations of layer $l$, before the activation is applied |
| $\mathbf{a}^{(l)}$ | activations of layer $l$, $f^{(l)}\left(\mathbf{z}^{(l)}\right)$ — what layer $l+1$ receives |
| $\mathbf{p}$ | the network's predictions, $\mathbf{a}^{(L)}$ |
| $\mathbf{y}$ | the target / ground truth |
| $W^{(l)}, \mathbf{b}^{(l)}$ | weight matrix and bias vector of layer $l$ |
| $\delta_i^{(l)}$ | local gradient, $\partial L / \partial z_i^{(l)}$ |
| $n$ | number of elements in the vector under discussion |
| $\eta$ | learning rate |
| $\lambda$ | L2 coefficient |
| $B$ | batch size |

> The file header in `main.cpp` cannot render LaTeX, so it writes vectors in the
> form `:v:` — `:z:`, `:a:`, `:p:`, `:y:` correspond to $\mathbf{z}$, $\mathbf{a}$,
> $\mathbf{p}$, $\mathbf{y}$ here.

## Activations and Losses

### Activation functions

All activations except softmax are elementwise, so each is written as a scalar
function $f(x)$.

| Name | $f(x)$ | $f'(x)$ |
|---|---|---|
| `sigmoid` | $\sigma(x) = \dfrac{1}{1 + e^{-x}}$ | $\sigma(x)\left(1 - \sigma(x)\right)$ |
| `tanh` | $\dfrac{e^{x} - e^{-x}}{e^{x} + e^{-x}}$ | $1 - \tanh^{2}(x)$ |
| `ReLU` | $\max(0, x)$ | $1$ if $x > 0$, else $0$ |
| `linear` | $x$ | $1$ |
| `leaky_ReLU` | $x$ if $x > 0$, else $\alpha x$ | $1$ if $x > 0$, else $\alpha$ |

with $\alpha = 0.01$ for `leaky_ReLU`. Softmax is layer-wide rather than
elementwise:

$$
\mathrm{softmax}(\mathbf{z})_i \;=\; \frac{e^{z_i}}{\sum_{j=1}^{n} e^{z_j}}
\qquad\text{with Jacobian}\qquad
\frac{\partial p_i}{\partial z_j} \;=\; p_i\left(\delta_{ij} - p_j\right)
$$

where $\delta_{ij}$ is the Kronecker delta.

`sigmoid` branches on the sign of $x$, evaluating $\frac{1}{1+e^{-x}}$ for
$x \geq 0$ and $\frac{e^{x}}{1+e^{x}}$ otherwise, so neither exponential can
overflow. `softmax` subtracts $\max_j z_j$ before exponentiating, for the same
reason. Both are stable out to $\pm 10^{5}$.

**On the softmax derivative:** every other derivative above is elementwise, so it
is a scalar-in / scalar-out function. The softmax Jacobian is a dense
$n \times n$ matrix, which that signature cannot express. It is therefore never
evaluated on its own — it is fused with categorical cross-entropy (see
[Data Flow](#data-flow)). `Neuron::activation_derivative` throws if asked for it.

### Loss functions

| Name | $L$ | $\partial L / \partial p_i$ |
|---|---|---|
| `mean_squared_error` | $\dfrac{1}{n}\displaystyle\sum_{i=1}^{n}\left(y_i - p_i\right)^{2}$ | $\dfrac{2}{n}\left(p_i - y_i\right)$ |
| `binary_cross_entropy` | $-\dfrac{1}{n}\displaystyle\sum_{i=1}^{n}\Big[y_i \ln p_i + (1 - y_i)\ln(1 - p_i)\Big]$ | $\dfrac{p_i - y_i}{n\,p_i\left(1 - p_i\right)}$ |
| `categorical_cross_entropy` | $-\displaystyle\sum_{i=1}^{n} y_i \ln p_i$ | $-\dfrac{y_i}{p_i}$ |

MSE and BCE average over the $n$ outputs; CCE sums over the $n$ classes, which is
the standard definition for a one-hot target. Both cross entropies clamp $p_i$
into $[\varepsilon,\, 1 - \varepsilon]$ with $\varepsilon = 10^{-9}$, so a fully
confident wrong answer costs a large finite number rather than $\infty$.

The CCE derivative above is $\partial L / \partial p_i$. It is only reached if CCE
is somehow paired with a non-softmax output; the supported path never evaluates
it, because the fused form avoids the division by $p_i$ entirely.

> Mean Absolute Error is **not** implemented. Earlier revisions of the source
> header listed its derivative; it was never in the `Loss` enum.

## Architecture

| Type | Responsibility |
|---|---|
| `Neuron` | One weight vector, one bias. Caches the input it last saw, its pre-activation `z_i`, and its activation. Also holds this batch's accumulated gradients. |
| `Layer` | A vector of `Neuron`s sharing one activation. Owns the layer-wide steps: softmax (which needs every $\mathbf{z}$ before any $\mathbf{a}$ can be computed) and the dropout mask. |
| `TrainOptions` | Knobs for a training run. |
| `TrainHistory` | Per-epoch losses returned by `train()`. |
| `NeuralNetwork` | `hidden_layers` + one `output_layer`, a loss, and the RNG. Chains the layers, owns the training loop, and handles persistence. |

**Widths:** every hidden layer shares one width, fixed by the constructor. `load()` reads
each layer's shape independently, so a hand-written model file with varying widths will
load and train correctly — the uniform width is a constructor limitation, not an engine one.

---

## Data Flow

### Forward

`forward_pass()` carries one vector through the network, reassigning it at each
layer, so layer $l$'s output vector **is** layer $l+1$'s input vector:

$$
\mathbf{z}^{(l)} = W^{(l)}\mathbf{a}^{(l-1)} + \mathbf{b}^{(l)}
\qquad
\mathbf{a}^{(l)} = f^{(l)}\!\left(\mathbf{z}^{(l)}\right)
$$

starting from $\mathbf{a}^{(0)} = \mathbf{x}$ and ending at
$\mathbf{p} = \mathbf{a}^{(L)}$. Per neuron $i$ of layer $l$:

$$
z_i^{(l)} = \sum_j w_{ij}^{(l)} a_j^{(l-1)} + b_i^{(l)}
$$

Each neuron stores the input it saw in `input_cache`; the backward pass needs
it, and it is gone once the next forward pass runs.

### Backward

Gradients flow the other way. At each layer, three things happen:

$$
\textbf{1. local gradient}\qquad
\delta_i^{(l)} \;=\; \frac{\partial L}{\partial a_i^{(l)}} \cdot f'\!\left(z_i^{(l)}\right)
$$

$$
\textbf{2. weight gradient}\qquad
\frac{\partial L}{\partial w_{ij}^{(l)}} = \delta_i^{(l)} \, a_j^{(l-1)}
\qquad
\frac{\partial L}{\partial b_i^{(l)}} = \delta_i^{(l)}
$$

$$
\textbf{3. pass back}\qquad
\frac{\partial L}{\partial a_j^{(l-1)}} \;=\; \sum_i \delta_i^{(l)} \, w_{ij}^{(l)}
$$

The bias gradient equals the local gradient because a bias's input is always
$1$. Step 3 is the transpose of the weight matrix, and it must read
$w_{ij}^{(l)}$ **before** the weights are updated — see
[Invariants](#invariants).

#### The fused softmax path

Softmax + categorical cross-entropy skip step 1 entirely. Substituting the
softmax Jacobian into the CCE derivative:

$$
\frac{\partial L}{\partial z_j}
= \sum_i \frac{\partial L}{\partial p_i}\,\frac{\partial p_i}{\partial z_j}
= \sum_i \left(-\frac{y_i}{p_i}\right) p_i\left(\delta_{ij} - p_j\right)
= -\sum_i y_i\left(\delta_{ij} - p_j\right)
= p_j \sum_i y_i - y_j
$$

For a one-hot target $\sum_i y_i = 1$, so the whole chain collapses to

$$
\frac{\partial L}{\partial \mathbf{z}} = \mathbf{p} - \mathbf{y}
$$

The dense Jacobian and the $-y_i / p_i$ singularity cancel exactly. Fusing is
therefore not merely an optimization: the unfused route divides by $p_i$, which
approaches $0$ for confidently wrong predictions, while the fused form contains
no division at all.

That is why `Layer::backward_local()` exists — it takes gradients already with
respect to $\mathbf{z}$, letting the fused path inject $\mathbf{p} - \mathbf{y}$
directly. The constructor rejects softmax without CCE, CCE without softmax, and
softmax as a hidden activation, because none of those have a correct path
through this code.

## Training

| `TrainOptions` field | Default | Meaning |
|---|---|---|
| `epochs` | `100` | passes over the training data |
| `learning_rate` | `0.1` | step size $\eta$ |
| `batch_size` | `1` | $B$; $1$ reproduces pure per-sample SGD |
| `shuffle` | `true` | reshuffle order every epoch |
| `l2` | `0.0` | weight decay $\lambda$ (never applied to biases) |
| `dropout` | `0.0` | hidden layers only, training only |
| `validation_split` | `0.0` | fraction held out, taken after a shuffle |
| `log_interval` | `0` | 0 picks a readable interval |
| `verbose` | `true` | print per-epoch progress |

Gradients accumulate across a batch and apply once, so the update is

$$
w \;\leftarrow\; w - \eta\left(\frac{1}{B}\sum_{k=1}^{B}
\frac{\partial L_k}{\partial w} \;+\; \lambda w\right)
$$

with the $\lambda w$ term omitted for biases. A useful consequence: with zero
gradients, an L2 step is pure exponential shrinkage,
$w \leftarrow w\left(1 - \eta\lambda\right)$ — which is exactly what the test
suite asserts.

`TrainHistory` returns `train_loss` and `validation_loss` per epoch, plus
`best_validation_loss` and `best_epoch`. Nothing acts on those — there is no
early stopping — so a caller watching for overfitting must read them.

Dropout is **inverted**. During training each hidden unit draws
$m_i \sim \mathrm{Bernoulli}(1 - p)$ and its activation becomes

$$
\tilde{a}_i = \frac{m_i}{1 - p}\, a_i
$$

so $\mathbb{E}\left[\tilde{a}_i\right] = a_i$. Inference needs no rescaling, and
`predict()` is always deterministic.

The original five-argument
`train(inputs, targets, epochs, learning_rate, log_interval)` still exists and is
unchanged: one sample per update, in fixed order, no regularization.

Weights use **He initialization**,
$w \sim \mathcal{N}\!\left(0,\; 2 / n_{\text{in}}\right)$, and biases start at
$0$. Inputs are assumed to be roughly unit-scale — unnormalized features train
noticeably worse.

## Quick Start

```cpp
NeuralNetwork net(n_features, 2, 32, n_classes,
                  "ReLU", "softmax", "categorical_cross_entropy", 42);

TrainOptions o;
o.epochs = 300;  o.learning_rate = 0.05;  o.batch_size = 32;
o.l2 = 1e-4;     o.validation_split = 0.2;

TrainHistory h = net.train(X, Y, o);

std::cout << "best val loss " << h.best_validation_loss
          << " at epoch " << h.best_epoch << "\n";
std::cout << "accuracy " << net.accuracy(X, Y) << "\n";

net.save("model.gx");
NeuralNetwork back = NeuralNetwork::load("model.gx");   // predicts the same
```

### Activations and losses as enums

The string spellings are parsed once at construction. The enum overloads skip that step:

```cpp
NeuralNetwork net(2, 2, 24, 3, Activation::Tanh, Activation::Softmax,
                  Loss::CategoricalCrossEntropy, 20260821);
```

### Reproducibility

Pass a seed to fix weight initialization, shuffling, and dropout masks. Omit it (or pass
`-1`) to seed from `random_device`. The seed actually used is always readable as `net.seed`,
and is written into saved models.

---

## Model File Format

Plain text, versioned. Values carry 17 significant digits, which is enough to round-trip an
IEEE-754 double exactly, so a reloaded network predicts bit-for-bit identically to the one
that was saved.

```text
GRADIEX-MODEL 1
loss categorical_cross_entropy
seed 20260821
layers 3
layer 0 24 2 tanh          <- index, neurons, fan-in, activation
<bias> <w0> <w1> ...       <- one line per neuron
...
```

Bump the version if this layout changes; `load()` rejects versions it does not recognize,
along with truncated, malformed, and missing files.

---

## Benchmarks

### Handwriting recognition

Trained on the UCI *Optical Recognition of Handwritten Digits* set (3,823 train / 1,797
test, $8 \times 8$ greyscale). The test split was written by **different people** than the training
split, so this measures generalization rather than memorization.

| Model | Params | Train | Test | Time |
|---|---|---|---|---|
| `64 → 64×2 → 10`, ReLU | 8,970 | 99.69% | **96.10%** | 2.3s |
| `64 → 256×2 → 10`, ReLU + dropout 0.2 | 84,234 | 99.92% | **97.11%** | 21s |

Accuracy plateaus at 96–97% across configurations, with a persistent $\approx 3\%$ train/test gap
that regularization does not close. That gap is the writer shift plus the limit of a dense
network: an MLP treats pixel 0 and pixel 63 as unrelated coordinates and has no notion that
neighbouring pixels form strokes. For reference, k-NN reaches ~97.8% and SVM ~98.3% on this
dataset. The missing $\approx 2\%$ is convolution.

Errors are structured rather than random — the worst confusions are 7→5, 8→1, and 3→9,
which are the mistakes humans make on sloppy handwriting too.

Pixels must be scaled (`/16.0` → `[0,1]`) and labels one-hot encoded; the engine does
neither for you.

### Throughput

Single-threaded, scalar doubles, `-O2`:

| Architecture | Params | µs/update | 1k rows | 10k rows | 60k rows |
|---|---|---|---|---|---|
| `2→8×2→1` | 105 | 2.9 | 0.003s | 0.03s | 0.17s |
| `20→32×2→3` | 1,827 | 13.6 | 0.014s | 0.14s | 0.8s |
| `64→128×3→10` | 42,634 | 82 | 0.08s | 0.8s | 4.9s |
| `784→128×2→10` | 118,282 | 303 | 0.30s | 3.0s | 18s |
| `256→512×3→10` | 662,026 | 1,664 | 1.7s | 17s | 100s |

MNIST-scale work (784 features, 60k rows) runs at roughly 18s per epoch, so a 30-epoch run
takes about 9 minutes. Note that data is held in memory as `vector<vector<double>>` — 60k×784
doubles is ~360 MB before targets.

---

## Invariants

Changing these silently breaks correctness.

1. **Gradients accumulate; no weight moves until `apply_gradients()`.** This is what makes
   mini-batching exact, and what guarantees that step 3 of the backward pass,
   $\partial L / \partial a_j^{(l-1)} = \sum_i \delta_i^{(l)} w_{ij}^{(l)}$, reads
   $w_{ij}^{(l)}$ *before* it is updated. Updating a layer's weights before propagating
   through it corrupts every layer behind it.
2. **`input_cache` must hold the input from *this* forward pass.** Weight gradients are
   meaningless against a stale cache.
3. **`Neuron::output` is the activation *before* the dropout mask; `Layer::output_vector` is
   the value *after* it.** Sigmoid and tanh derivatives are written in terms of the unmasked
   output, so masking `Neuron::output` would corrupt them. With dropout off the two are equal.
4. **Softmax is layer-wide and fused with CCE.** Its Jacobian is a dense
   $n \times n$ matrix, which the scalar-in / scalar-out derivative signature cannot
   express, so `Neuron::activate` and `Neuron::activation_derivative` throw rather than
   return a wrong answer. The supported path uses
   $\partial L / \partial \mathbf{z} = \mathbf{p} - \mathbf{y}$.
5. **`Activation` and `Loss` are enums on the hot path.** Strings are parsed once at the API
   boundary; do not push a string comparison back into `Layer::forward` or the per-neuron loops.
6. **The five-argument `train()` is bit-identical** to a manual fixed-order,
   one-sample-at-a-time loop. A test asserts this.

---

## Not Supported

Optimizers other than plain SGD (no momentum, Adam, or LR schedule), per-layer widths or
dropout rates from the constructor, early stopping, convolution, recurrence, attention,
embeddings, skip connections, batch or layer normalization, sparse input, streaming datasets,
GPU, threading, or SIMD. The inner loops are scalar double arithmetic.

---

## Distribute Binary via the Web App

To make the Linux binary downloadable from the frontend app:

```bash
g++ -std=c++17 -O2 -Wall -Wextra -pedantic main.cpp -o gradiex-linux-x86_64
mkdir -p frontend/public/downloads
cp gradiex-linux-x86_64 frontend/public/downloads/gradiex-linux-x86_64
```

Then start/build the frontend:

```bash
cd frontend
npm install
npm run dev
```

The binary is served at `/downloads/gradiex-linux-x86_64`.

---

## Project Layout

```text
GradieX/
  main.cpp                  engine + two worked examples
  README.md
  LICENSE                   MIT
  CONTRIBUTING.md           build/test workflow, style, invariants
  CODE_OF_CONDUCT.md        Contributor Covenant 2.1
  CHANGELOG.md
  tests/
    test_gradiex.cpp        112 checks, includes main.cpp directly
  .github/
    workflows/ci.yml        build + test matrix, sanitizers, release binary
    ISSUE_TEMPLATE/
    PULL_REQUEST_TEMPLATE.md
  frontend/
    public/
    src/
```

---

## Contributing

Contributions are welcome. See **[CONTRIBUTING.md](CONTRIBUTING.md)** for the build
and test workflow, the code style, and — most importantly — the
[invariants](CONTRIBUTING.md#invariants) that must not be broken. Breaking one
produces a library that trains plausibly and is silently wrong.

If you touch the gradient path, the gradient check in Test 1 is what decides whether
you got it right.

By participating you agree to the [Code of Conduct](CODE_OF_CONDUCT.md).

## License

[MIT](LICENSE) &copy; 2026 Snehashish Laskar.

## Roadmap

- Momentum / Adam optimizers — plain SGD is the current ceiling on hard problems
- Per-layer widths and per-layer dropout rates from the constructor
- Early stopping driven by `TrainHistory`
- Vectorization: the inner loops are scalar, with no BLAS or SIMD
